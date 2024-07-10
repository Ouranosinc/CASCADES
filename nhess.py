import numpy as np
import xclim
import pandas as pd
import geopandas as gpd
import cartopy
import json
import xarray as xr
import xclim.core.units

import xscen as xs
import xskillscore as xss
import figanos.matplotlib as fg
from figanos.matplotlib.utils import add_cartopy_features
from cartopy import feature
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as cl
import os
from pathlib import Path

from copy import deepcopy

from xclim.sdba.utils import ecdf, rank

from distributed import Client
import logging
logger = logging.getLogger("distributed")
logger.setLevel(logging.WARNING)
logger2 = logging.getLogger("flox")
logger2.setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Qt5Agg")

# xscen setup
xs.load_config("paths.yml", "configs/cfg_nhess.yml", reset=True)

proj = cartopy.crs.Mercator()

# Rivers
shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
shp_portrait = shp_portrait.set_index("TRONCON")
shp_portrait = shp_portrait.to_crs(proj)

stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
stations_atlas = stations_atlas.loc[stations_atlas["MASQUE"] == 2]  # Only keep public stations

shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")
shp_zg = shp_zg.set_index("SIGLE")

# Open the list of reaches per ZGIEBV
with open(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_per_reach.json", 'r') as fp:
    reaches_per_zgiebv = json.load(fp)

extent = [-80.5, -62, 44.5, 53.5]


def main(todo: list):
    dc = xs.DataCatalog(xs.CONFIG["project"]["path"])
    raw_cat = xs.DataCatalog(xs.CONFIG["io"]["nhess_rawcat"])

    w_y = []
    for w in shp_zg.index:
        w_y.extend(reaches_per_zgiebv[w])
    w_y = list(set(w_y).intersection(stations_atlas["TRONCON_ID"]))

    # Map of the study region
    if "domain" in todo:
        watersheds = xs.CONFIG["analogs"]["targets"]["all"]["region"]
        w_y = []
        for w in watersheds:
            w_y.extend(reaches_per_zgiebv[w])
        w_y = list(set(w_y).intersection(stations_atlas["TRONCON_ID"]))
        shp_fig = shp_portrait.loc[w_y]

        f, ax = plt.subplots(1, 1, figsize=(10, 7.5), subplot_kw={"projection": proj})
        features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                    'states': {"edgecolor": "black", "linestyle": "dotted"}}
        add_cartopy_features(ax, features)
        # shp_zg.to_crs(proj).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=3.5)
        shp_fig.plot(ax=ax, color="#052946", linewidth=(shp_fig["SUPERFICIE"] / 1000).clip(1, 3))
        ax.set_extent([-80.5, -56, 44.5, 53.5], crs=cartopy.crs.PlateCarree())
        # Add the lat/lon grid
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.6)
        # Adjust label size and have ticks only on degrees
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 24}
        gl.ylabel_style = {"size": 24}
        gl.xlocator = matplotlib.ticker.FixedLocator(np.arange(-80, -55, 5.5))
        gl.ylocator = matplotlib.ticker.FixedLocator(np.arange(45, 55, 3.5))
        plt.tight_layout()
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / "domain.png", dpi=300, transparent=True)
        plt.close()

    # Comparison of the water budget between ERA5-Land and ClimEx
    if "water_budget" in todo:
        e5l = dc.search(source="ERA5-Land", variable=["pr", "evspsblpot"]).to_dataset()
        e5l = e5l.sel(time=slice(xs.CONFIG["storylines"]["ref_period"][0], xs.CONFIG["storylines"]["ref_period"][1]))
        e5l = xs.spatial.subset(e5l, method="shape", shape=shp_zg, tile_buffer=1.1)  # Remove grid cells outside the region
        e5l["water_budget"] = e5l["pr"] - e5l["evspsblpot"]
        e5l["water_budget"].attrs["units"] = "kg m-2 s-1"
        e5l["water_budget"] = xclim.units.convert_units_to(e5l["water_budget"], "mm s-1", context="hydro")
        e5l["water_budget"] = xclim.units.rate2amount(e5l["water_budget"], out_units="mm")
        e5l = xs.utils.unstack_dates(e5l[['water_budget']], new_dim="month")

        climex_dict = dc.search(source="CRCM5.*", variable=["pr", "evspsblpot"]).to_dataset_dict()
        climex = []
        for key, ds in climex_dict.items():
            years = xs.get_warming_level(realization=f"CMIP5_CanESM2_rcp85_{ds.attrs['cat:member']}", wl=0.91, window=30)
            ds = ds.assign_coords(realization=key.split(".")[0])
            ds = ds.sel(time=slice(years[0], years[1]))
            climex.extend([ds])
        climex = xr.concat(climex, dim="realization")
        climex = xs.spatial.subset(climex, method="shape", shape=shp_zg, tile_buffer=1.1)
        climex["water_budget"] = climex["pr"] - climex["evspsblpot"]
        climex["water_budget"].attrs["units"] = "kg m-2 s-1"
        climex["water_budget"] = xclim.units.convert_units_to(climex["water_budget"], "mm s-1", context="hydro")
        climex["water_budget"] = xclim.units.rate2amount(climex["water_budget"], out_units="mm")
        climex = xs.utils.unstack_dates(climex[['water_budget']], new_dim="month")

        # Compute anomalies
        e5l["anomaly"] = e5l["water_budget"] - e5l["water_budget"].mean(dim="time")
        climex["anomaly"] = climex["water_budget"] - climex["water_budget"].mean(dim="time")

        # Add attributes
        e5l = e5l.compute()
        e5l["water_budget"].attrs["long_name"] = "P - PET (mm)"
        e5l["anomaly"].attrs["long_name"] = "Anomalies (mm)"
        e5l["anomaly"].attrs["units"] = "mm"
        climex = climex.compute()
        climex["water_budget"].attrs["long_name"] = "P - PET (mm)"
        climex["anomaly"].attrs["long_name"] = "Anomalies (mm)"
        climex["anomaly"].attrs["units"] = "mm"

        # Plot
        f, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i, ax in enumerate(axes):
            if i == 0:
                v = "water_budget"
                title = "a)"
            else:
                v = "anomaly"
                title = "b)"

            # Prepare the data
            data_e5l = e5l[v].mean(dim=["lat", "lon"]).to_dataframe().reset_index()
            data_e5l["Dataset"] = "ERA5-Land"
            data_climex = climex[v].mean(dim=["lat", "lon"]).to_dataframe().reset_index()
            data_climex["Dataset"] = "ClimEx"
            data = pd.concat([data_e5l, data_climex]).reset_index()

            # Boxplots
            import seaborn as sns
            sns.boxplot(data=data, x="month", y=v, ax=ax, linewidth=1.5, hue="Dataset", palette=["#fdc414", "#ce153d"])

            # Adjust the labels
            ax.set_xlabel("")
            ax.set_ylabel(climex[v].attrs["long_name"] , fontsize=14)
            ax.set_yticklabels(ax.get_yticks(), fontsize=14)
            ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=14)

            # Add title
            ax.set_title(title, loc="left", fontweight="bold", pad=25, fontsize=18)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.2)

        # Save the figure
        folder = Path(xs.CONFIG["io"]["nhess_fig"])
        os.makedirs(folder, exist_ok=True)
        plt.savefig(folder / "water_budget.png", dpi=300, transparent=True)
        plt.close()

    # Schema of the search for analogs
    if "spei" in todo:
        warming_levels = xs.CONFIG["storylines"]["warming_levels"]
        target_years = xs.CONFIG["storylines"]["target_years"]
        region = {"name": "SouthernQC", "method": "shape", "shape": shp_zg, "tile_buffer": 1.1}

        for wl in warming_levels:
            for y in target_years:
                # Open ERA5-Land
                ref = dc.search(source="ERA5.*", variable=["spei3", "spei6"]).to_dask(xarray_open_kwargs={"chunks": {}})
                ref = xs.spatial.subset(ref, **region)

                # Open ClimEx
                climex_dict = dc.search(source="CRCM5.*", processing_level=f"indicators-warminglevel.*{wl}vs.*", variable=["spei3", "spei6"]).to_dataset_dict(xarray_open_kwargs={"chunks": {}})
                climex = xclim.ensembles.create_ensemble(climex_dict)
                climex = xs.utils.clean_up(climex, common_attrs_only=climex_dict)
                climex = climex.sortby("realization")
                climex = xs.spatial.subset(climex, **region)

                # If the RMSE file does not exist, compute the criteria
                if not os.path.isfile(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_{wl}.nc"):
                    with Client(n_workers=4, threads_per_worker=1, memory_limit="20GB",
                                local_directory=xs.CONFIG["dask"]["local_directory"]) as client:
                        _compute_criteria(ref, climex, target_year=y, **xs.CONFIG["analogs"]["compute_criteria"], warming_level=wl)

                # Open RMSE and weights
                rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_{wl}.nc", chunks={}).load()
                weights = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"weights_{y}_{wl}.nc", chunks={})
                criteria = xs.CONFIG["analogs"]["compute_criteria"]["criteria"]
                sorted_analogs = sort_analogs(rmse)

                # For the plots, it's prettier to have the criteria in a different order (SPEI6 in last)
                order = [f"SPEI3-{m}" for m in range(5, 12)] + [f"SPEI6-{m}" for m in [5, 10]]
                criteria = criteria[2:] + criteria[:2]
                rmse = rmse.sel(criteria=order)
                weights = weights.sel(criteria=order)

                params = {'mathtext.default': 'regular'}
                plt.rcParams.update(params)
                # First row: ERA5-Land
                f = plt.figure(figsize=(19, 2.5))
                for i in range(9):
                    ax = f.add_subplot(1, 9, i+1, projection=proj)
                    c = criteria[i]

                    title = f"{c[0][:-1].upper()}-{c[0][-1]}$_{{{xr.coding.cftime_offsets._MONTH_ABBREVIATIONS[c[1]]}}}$"
                    target = ref[c[0]].sel(time=f"{y}-{c[1]:02d}-01")
                    _spei_plot(target, ax=ax, cb=False)
                    plt.title(title, fontsize=14, fontweight="bold")
                plt.tight_layout()
                plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"spei_{y}_era5land.png", dpi=300, transparent=True)
                plt.close()

                # Second row: Weights
                f = plt.figure(figsize=(19, 2.5))
                for i in range(9):
                    ax = f.add_subplot(1, 9, i+1, projection=proj)
                    _weights_plot(weights.sel(criteria="-".join([str(cc).upper() for cc in criteria[i]])).squeeze(), ax=ax, cb=False)
                plt.tight_layout()
                plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"spei_{y}_weights.png", dpi=300, transparent=True)
                plt.close()

                # Best 2 analogs
                for j in range(2):
                    f = plt.figure(figsize=(19, 2.5))
                    sim = climex.sel(realization=sorted_analogs.isel(stacked=j).realization.values, time=climex.time.dt.year.isin([sorted_analogs.isel(stacked=j).time.dt.year.values]))
                    sim_rmse = rmse.sel(realization=sorted_analogs.isel(stacked=j).realization.values, time=rmse.time.dt.year.isin([sorted_analogs.isel(stacked=j).time.dt.year.values]))
                    # Prepare the title
                    total_rmse = str(np.round(sim_rmse["rmse"].sum().values, 2))
                    r = str(sorted_analogs.isel(stacked=j).realization.values).split(".")[0].split("rcp85_")[-1]
                    yy = str(sorted_analogs.isel(stacked=j).time.dt.year.values)
                    title = f"Analog #{j+1} - {r} @ {yy} | ΣRMSE = {total_rmse}        "
                    for i in range(9):
                        ax = f.add_subplot(1, 9, i + 1, projection=proj)
                        c = criteria[i]
                        da = sim[c[0]].sel(time=sim.time.dt.month == c[1], drop=True).squeeze()
                        crit_rmse = str(np.round(sim_rmse["rmse"].sel(criteria="-".join([str(cc).upper() for cc in c])).squeeze().values, 2))
                        _spei_plot(da, ax=ax, cb=False, rmse=crit_rmse)
                        if i == 1:
                            plt.title(title, fontsize=14, fontweight="bold", pad=10)
                    plt.tight_layout()
                    plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"spei_{y}_{wl}_bestanalog{j}.png", dpi=300, transparent=True)
                    plt.close()

                # Worst one
                j = int(sorted_analogs.argmax("stacked"))
                f = plt.figure(figsize=(19, 2.5))
                sim = climex.sel(realization=sorted_analogs.isel(stacked=j).realization.values,
                                 time=climex.time.dt.year.isin([sorted_analogs.isel(stacked=j).time.dt.year.values]))
                sim_rmse = rmse.sel(realization=sorted_analogs.isel(stacked=j).realization.values,
                                    time=rmse.time.dt.year.isin([sorted_analogs.isel(stacked=j).time.dt.year.values]))
                # Prepare the title
                total_rmse = str(np.round(sim_rmse["rmse"].sum().values, 2))
                r = str(sorted_analogs.isel(stacked=j).realization.values).split(".")[0].split("rcp85_")[-1]
                yy = str(sorted_analogs.isel(stacked=j).time.dt.year.values)
                title = f"Analog #1500 - {r} @ {yy} | ΣRMSE = {total_rmse}        "
                for i in range(9):
                    ax = f.add_subplot(1, 9, i + 1, projection=proj)
                    c = criteria[i]
                    da = sim[c[0]].sel(time=sim.time.dt.month == c[1], drop=True).squeeze()
                    crit_rmse = str(np.round(sim_rmse["rmse"].sel(criteria="-".join([str(cc).upper() for cc in c])).squeeze().values, 2))
                    _spei_plot(da, ax=ax, cb=False, rmse=crit_rmse)
                    if i == 1:
                        plt.title(title, fontsize=14, fontweight="bold", pad=10)
                plt.tight_layout()
                plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"spei_{y}_{wl}_worstanalog.png", dpi=300, transparent=True)
                plt.close()

    # Some hydrological indicators were never computed, so we need to recompute them
    def _recompute_hydro(r, y, wl):
        if not os.path.isfile(Path(xs.CONFIG["io"]["nhess_data"]) / f"indicators_{wl}_{y}_{r}.nc"):
            with Client(n_workers=4, threads_per_worker=1, memory_limit="20GB",
                        local_directory=xs.CONFIG["dask"]["local_directory"]) as client:
                ds_day = raw_cat.search(type="simulation-hydro", frequency="day", member=r, hydrology_member="MG24HQ").to_dask(xarray_open_kwargs={"chunks": {}})
                [ds_day[c].load() for c in ds_day.coords]
                ds_day = ds_day.where(ds_day.station.isin(w_y), drop=True)

                # Compute the indicators
                years_hist = xs.get_warming_level(f"CMIP5_CanESM2_rcp85_{r}", 0.91, window=30)

                if y == "all":
                    # Preload
                    ds_day = ds_day.sel(time=slice(f"{years_hist[0]}-01-01", f"{years_hist[1]}-12-31")).load()

                indicator_7qx = [xclim.core.indicator.Indicator.from_dict(
                    data={"base": "return_level",
                          "input": {"da": "streamflow"},
                          "parameters": {"mode": "min", "window": 7, "t": [2, 10], "dist": "lognorm", "indexer": {"month": [5, 6, 7, 8, 9, 10, 11]}}},
                    identifier="7qx",
                    module="hydro",
                )]
                base_7qx = xs.compute_indicators(ds_day.sel(time=slice(years_hist[0], years_hist[1])), indicator_7qx)["fx"]

                if y != "all":
                    da_subset = ds_day.sel(time=slice(f"{y}-01-01", f"{y}-12-31"))["streamflow"]
                else:
                    da_subset = ds_day.sel(time=slice(f"{years_hist[0]}-01-01", f"{years_hist[1]}-12-31"))["streamflow"]
                out = xr.Dataset()
                out["14qmax"] = da_subset.rolling(time=14, center=True).mean().groupby("time.year").max(dim="time") / da_subset.drainage_area
                # out["doy_14qmax"] = da_subset.rolling(time=14, center=True).mean().idxmax(dim="time").dt.dayofyear
                out["discharge_mean_mam"] = da_subset.sel(time=da_subset["time.month"].isin([3, 4, 5])).groupby("time.year").mean(dim="time") / da_subset.drainage_area
                out["discharge_mean_jja"] = da_subset.sel(time=da_subset["time.month"].isin([6, 7, 8])).groupby("time.year").mean(dim="time") / da_subset.drainage_area
                out["discharge_mean_son"] = da_subset.sel(time=da_subset["time.month"].isin([9, 10, 11])).groupby("time.year").mean(dim="time") / da_subset.drainage_area
                out["7qmin"] = da_subset.where(da_subset.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11])).rolling(time=7, center=True).mean().groupby("time.year").min(dim="time") / da_subset.drainage_area
                out["days_under_hist7q2"] = (da_subset.where(da_subset.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11])) < base_7qx["7qx"].sel(return_period=2)).groupby("time.year").sum(dim="time")
                out["days_under_hist7q10"] = (da_subset.where(da_subset.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11])) < base_7qx["7qx"].sel(return_period=10)).groupby("time.year").sum(dim="time")

                # Low-flow season
                histq_mean = ds_day["streamflow"].sel(time=slice(years_hist[0], years_hist[1])).mean(dim="time")
                thresh = histq_mean - (histq_mean - base_7qx["7qx"].sel(return_period=2).squeeze()) * 0.85
                bool_under_low = (da_subset < thresh).where(da_subset.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)

                out["season_histstart"] = xclim.indices.run_length.first_run(bool_under_low, window=7, coord="dayofyear", freq="YS-JAN").squeeze().groupby("time.year").mean(dim="time")
                out["season_histend"] = xclim.indices.run_length.last_run(bool_under_low, window=7, coord="dayofyear", freq="YS-JAN").squeeze().groupby("time.year").mean(dim="time")
                out["season_histduration"] = out["season_histend"] - out["season_histstart"]

                # Save
                out.to_netcdf(Path(xs.CONFIG["io"]["nhess_data"]) / f"indicators_{wl}_{y}_{r}.nc")

        # Load
        return xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"indicators_{wl}_{y}_{r}.nc", chunks={})

    # Prepare the hydrological data
    def _prep_hydro_ds(ds, rename, indicator_list=True):
        [ds[c].load() for c in ds.coords]
        if "station_id" in ds:
            ds = ds.swap_dims({"station": "station_id"}).drop_vars(["station"]).rename({"station_id": "station"})
        ds = ds.where(ds.station.isin(w_y), drop=True)
        if rename:
            ds = ds.rename({"season_start": "season_histstart", "season_end": "season_histend", "days_under_7q2": "days_under_hist7q2", "days_under_7q10": "days_under_hist7q10"})
        if indicator_list:
            ds["season_histduration"] = ds["season_histend"] - ds["season_histstart"]
            ds["season_histduration"].attrs["units"] = "d"
            ds = ds[indicators.keys()]
        for v in ds.data_vars:
            if ds[v].attrs["units"] in ["m^3 s-1", "m3 s-1"]:
                ds[v] = xclim.units.convert_units_to(ds[v], "m^3 s-1", context="hydro")
                ds[v] = ds[v] / ds.drainage_area
                ds[v].attrs["units"] = "m^3 s-1 km-2"
        return ds

    cmap_streamflow = cl.LinearSegmentedColormap.from_list("custom_cmap", ["#582c0eff", "wheat", "#a1dab4ff", "#41b6c4ff", "dodgerblue", "#2c7fb8ff", "#253494ff"])
    cmap_dates = cl.LinearSegmentedColormap.from_list("custom_cmap", ["indigo", "seagreen", "yellow", "orange", "indigo"])
    indicators = {
        # 'doy_14qmax': {"range": [1, 365], "cmap": "misc_seq_1", "name": "DOY($14Q_{max}$)"},
        '14qmax': {"range": [0, 0.3], "cmap": cmap_streamflow, "cmap_div": "temp_div_r", "name": "$14Q_{max}$"},
        'discharge_mean_mam': {"range": [0, 0.08], "cmap": cmap_streamflow, "cmap_div": "temp_div_r", "name": "$\\bar Q_{MAM}$", "unit": "L s$^{-1}$ km$^{-2}$"},
        'discharge_mean_jja': {"range": [0, 0.03], "cmap": cmap_streamflow, "cmap_div": "temp_div_r", "name": "$\\bar Q_{JJA}$", "unit": "L s$^{-1}$ km$^{-2}$"},
        'discharge_mean_son': {"range": [0, 0.03], "cmap": cmap_streamflow, "cmap_div": "temp_div_r", "name": "$\\bar Q_{SON}$", "unit": "L s$^{-1}$ km$^{-2}$"},
        '7qmin': {"range": [0, 0.01], "cmap": cmap_streamflow, "cmap_div": "temp_div_r", "name": "$7Q_{min}$", "unit": "L s$^{-1}$ km$^{-2}$"},
        'season_histstart': {"range": [1, 365], "cmap": cmap_dates, "cmap_div": "slev_div_r", "name": "$LWS_{start}$"},
        'season_histend': {"range": [1, 365], "cmap": cmap_dates, "cmap_div": "slev_div", "name": "$LWS_{end}$"},
        'season_histduration': {"range": [0, 200], "cmap": "temp_seq", "cmap_div": "slev_div", "name": "$LWS_{duration}$", "unit": "days"},
        'days_under_hist7q2': {"range": [0, 150], "cmap": "Reds", "cmap_div": "temp_div", "name": "$n_{days}$ < 7Q2", "unit": "days"},
        'days_under_hist7q10': {"range": [0, 100], "cmap": "Reds", "cmap_div": "temp_div", "name": "$n_{days}$ < 7Q10", "unit": "days"},
    }

    # Validation of the analogs
    if "hydro_validation" in todo:
        target_years = xs.CONFIG["storylines"]["target_years"]
        for y in target_years:
            # Open Portrait
            ref = dc.search(type="reconstruction-hydro", processing_level="indicators", xrfreq=["YS-JAN"]).to_dask(xarray_open_kwargs={"chunks": {}})
            ref = _prep_hydro_ds(ref, rename=True)
            ref = ref.compute()

            # Open RMSE
            rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_0.91.nc", chunks={}).load()
            sorted_analogs = sort_analogs(rmse)

            # Indicators: Best analogs in order
            blend = {v: [] for v in indicators}
            for j in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs.isel(stacked=j).realization.values).split(".")[0].split("rcp85_")[-1]
                sim = dc.search(type="simulation-hydro", processing_level=f"indicators-0.91", xrfreq=["YS-JAN"], member=realization)
                if len(sim) == 0:
                    sim = _recompute_hydro(r=realization, y=int(sorted_analogs.isel(stacked=j).time.dt.year.values), wl=0.91)
                elif len(sim) == 1:
                    sim = sim.to_dask(xarray_open_kwargs={"chunks": {}})
                    try:
                        sim = _prep_hydro_ds(sim, rename=True)
                        sim = sim.sel(time=sim.time.dt.year == sorted_analogs.isel(stacked=j).time.dt.year.values).squeeze()
                    except:
                        sim = _recompute_hydro(r=realization, y=int(sorted_analogs.isel(stacked=j).time.dt.year.values), wl=0.91)
                else:
                    raise ValueError("More than one simulation found.")
                if not all(variable in sim for variable in indicators) or any(sim[variable].isnull().all().values for variable in indicators):
                    sim = _recompute_hydro(r=realization, y=int(sorted_analogs.isel(stacked=j).time.dt.year.values), wl=0.91)

                for i in range(len(indicators)):
                    v = list(indicators.keys())[i]
                    da = sim[v]
                    blend[v].extend([da])

            blend_final = {v: xr.concat(blend[v], dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze() for v in indicators}
            blend_final = xr.Dataset(blend_final)

            for j in range(4):
                f = plt.figure(figsize=(19, 3))
                for i in range(len(indicators)):
                    ax = f.add_subplot(1, len(indicators), i + 1, projection=proj)
                    v = list(indicators.keys())[i]
                    if j == 0:
                        da = ref[v].sel(time=f"{y}-01-01").squeeze()
                    elif j == 1:
                        da = blend[v][0]
                    elif j == 2:
                        da = blend[v][1]
                    else:
                        da = blend_final[v]
                    if indicators[v]["range"][1] == 365:
                        ticks = np.array([1, 15, 32, 46, 60, 74, 91, 105, 121, 135, 152, 166, 182, 196, 213, 227, 244, 258, 274, 288, 305, 319, 335, 349, 365])
                        norm = cl.BoundaryNorm(ticks, ncolors=256)
                    else:
                        norm = None
                    _indicator_plot(da, ax=ax, vmin=indicators[v]["range"][0], vmax=indicators[v]["range"][1], levels=12, norm=norm, cmap=indicators[v]["cmap"], cb=False)
                    if j == 0:
                        plt.title(indicators[v]["name"], fontsize=14)
                plt.tight_layout()
                plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"hydro_validation_{y}_{j}.png", dpi=300, transparent=True)
                plt.close()

    # Analog construction
    if "analog_construction" in todo:
        # Open Portrait
        ref = dc.search(type="reconstruction-hydro", processing_level="indicators", xrfreq=["YS-JAN"]).to_dask(xarray_open_kwargs={"chunks": {}})
        ref = _prep_hydro_ds(ref, rename=True)
        ref = ref[['7qmin']]
        ref = ref.sel(time="2021-01-01").squeeze()

        # Open historical RMSE and construct the blend
        rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_0.91.nc", chunks={}).load()
        sorted_analogs_hist = sort_analogs(rmse)
        blend = []
        for i in range(xs.CONFIG["storylines"]["n_analogs"]):
            realization = str(sorted_analogs_hist.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
            analog_year = int(sorted_analogs_hist.isel(stacked=i).time.dt.year.values)
            sim = dc.search(type="simulation-hydro", processing_level=f"indicators-0.91", xrfreq=["YS-JAN"], member=realization)
            if len(sim) == 0:
                sim = _recompute_hydro(r=realization, y=analog_year, wl=0.91)
            elif len(sim) == 1:
                sim = sim.to_dask(xarray_open_kwargs={"chunks": {}})
                try:
                    sim = _prep_hydro_ds(sim, rename=True)
                    sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                except:
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=0.91)
            else:
                raise ValueError("More than one simulation found.")
            if not all(variable in sim for variable in indicators) or any(sim[variable].isnull().all().values for variable in indicators):
                sim = _recompute_hydro(r=realization, y=analog_year, wl=0.91)
            sim = sim[["7qmin"]]
            blend.extend([sim])
        blend_hist = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze().drop_vars(["time"])

        # Open future RMSE and construct the blend
        rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_3.nc", chunks={}).load()
        sorted_analogs_fut = sort_analogs(rmse)
        blend = []
        for i in range(xs.CONFIG["storylines"]["n_analogs"]):
            realization = str(sorted_analogs_fut.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
            analog_year = int(sorted_analogs_fut.isel(stacked=i).time.dt.year.values)
            sim = dc.search(type="simulation-hydro", processing_level=f"indicators-3", xrfreq=["YS-JAN"], member=realization)
            if len(sim) == 0:
                sim = _recompute_hydro(r=realization, y=analog_year, wl=3)
            elif len(sim) == 1:
                sim = sim.to_dask(xarray_open_kwargs={"chunks": {}})
                try:
                    sim = _prep_hydro_ds(sim, rename=True)
                    sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                except:
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=3)
            else:
                raise ValueError("More than one simulation found.")
            if not all(variable in sim for variable in indicators) or any(sim[variable].isnull().all().values for variable in indicators):
                sim = _recompute_hydro(r=realization, y=analog_year, wl=3)
            sim = sim[["7qmin"]]
            blend.extend([sim])
        blend_fut = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze().drop_vars(["time"])

        # First figure: Reference
        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(1, 1, 1, projection=proj)
        _indicator_plot(ref["7qmin"], ax, vmin=indicators["7qmin"]["range"][0], vmax=indicators["7qmin"]["range"][1], levels=12, cmap=indicators["7qmin"]["cmap"], cb=False, linewidth=2)
        plt.tight_layout()
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"construction_ref.png", dpi=300, transparent=True)
        plt.close()

        # Second figure: Blend historical
        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(1, 1, 1, projection=proj)
        _indicator_plot(blend_hist["7qmin"], ax, vmin=indicators["7qmin"]["range"][0], vmax=indicators["7qmin"]["range"][1], levels=12, cmap=indicators["7qmin"]["cmap"], cb=False, linewidth=2)
        plt.tight_layout()
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"construction_hist.png", dpi=300, transparent=True)
        plt.close()

        # Third figure: Blend future
        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(1, 1, 1, projection=proj)
        _indicator_plot(blend_fut["7qmin"], ax, vmin=indicators["7qmin"]["range"][0], vmax=indicators["7qmin"]["range"][1], levels=12, cmap=indicators["7qmin"]["cmap"], cb=False, linewidth=2)
        plt.tight_layout()
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"construction_fut.png", dpi=300, transparent=True)
        plt.close()

        # Fourth figure: Deltas
        deltas = xs.compute_deltas(blend_fut, blend_hist, kind="%", rename_variables=False)
        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(1, 1, 1, projection=proj)
        _indicator_plot(deltas["7qmin"], ax, vmin=-100, vmax=100, levels=12, cmap="temp_div_r", cb=False, linewidth=2)
        plt.tight_layout()
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"construction_deltas.png", dpi=300, transparent=True)
        plt.close()

        # Fifth figure: Analog 2021
        analog = ref + (deltas / 100) * ref
        f = plt.figure(figsize=(7, 5))
        ax = f.add_subplot(1, 1, 1, projection=proj)
        _indicator_plot(analog["7qmin"], ax, vmin=indicators["7qmin"]["range"][0], vmax=indicators["7qmin"]["range"][1], levels=12, cmap=indicators["7qmin"]["cmap"], cb=False, linewidth=2)
        plt.tight_layout()
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"construction_analog.png", dpi=300, transparent=True)
        plt.close()

    # Hydrological indicators
    if "indicators" in todo:
        target_years = xs.CONFIG["storylines"]["target_years"]

        # Open Portrait
        ref = dc.search(type="reconstruction-hydro", processing_level="indicators", xrfreq=["YS-JAN"]).to_dask(xarray_open_kwargs={"chunks": {}})
        ref = _prep_hydro_ds(ref, rename=True)
        for y in target_years:
            ref_y = ref.sel(time=f"{y}-01-01").squeeze().drop_vars(["time"])
            ref_y["season_histduration"] = xr.where(ref_y["season_histduration"].isnull(), 0, ref_y["season_histduration"])

            # Open historical RMSE and construct the blend
            rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_0.91.nc", chunks={}).load()
            sorted_analogs_hist = sort_analogs(rmse)
            blend = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_hist.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_hist.isel(stacked=i).time.dt.year.values)
                sim = dc.search(type="simulation-hydro", processing_level=f"indicators-0.91", xrfreq=["YS-JAN"], member=realization)
                if len(sim) == 0:
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=0.91)
                elif len(sim) == 1:
                    sim = sim.to_dask(xarray_open_kwargs={"chunks": {}})
                    try:
                        sim = _prep_hydro_ds(sim, rename=True)
                        sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                    except:
                        sim = _recompute_hydro(r=realization, y=analog_year, wl=0.91)
                else:
                    raise ValueError("More than one simulation found.")
                if not all(variable in sim for variable in indicators) or any(sim[variable].isnull().all().values for variable in indicators):
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=0.91)
                sim = sim[list(indicators)]
                blend.extend([sim])
            blend_hist = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze().drop_vars(["time"])

            # Open future RMSE and construct the blend
            rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_2.nc", chunks={}).load()
            sorted_analogs_fut2 = sort_analogs(rmse)
            blend = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_fut2.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_fut2.isel(stacked=i).time.dt.year.values)
                sim = dc.search(type="simulation-hydro", processing_level=f"indicators-2", xrfreq=["YS-JAN"], member=realization)
                if len(sim) == 0:
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=2)
                elif len(sim) == 1:
                    sim = sim.to_dask(xarray_open_kwargs={"chunks": {}})
                    try:
                        sim = _prep_hydro_ds(sim, rename=True)
                        sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                    except:
                        sim = _recompute_hydro(r=realization, y=analog_year, wl=2)
                else:
                    raise ValueError("More than one simulation found.")
                if not all(variable in sim for variable in indicators) or any(sim[variable].isnull().all().values for variable in indicators):
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=2)
                sim = sim[list(indicators)]
                blend.extend([sim])
            blend_fut2 = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze().drop_vars(["time"])

            rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_3.nc", chunks={}).load()
            sorted_analogs_fut3 = sort_analogs(rmse)
            blend = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_fut3.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_fut3.isel(stacked=i).time.dt.year.values)
                sim = dc.search(type="simulation-hydro", processing_level=f"indicators-3", xrfreq=["YS-JAN"], member=realization)
                if len(sim) == 0:
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=3)
                elif len(sim) == 1:
                    sim = sim.to_dask(xarray_open_kwargs={"chunks": {}})
                    try:
                        sim = _prep_hydro_ds(sim, rename=True)
                        sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                    except:
                        sim = _recompute_hydro(r=realization, y=analog_year, wl=3)
                else:
                    raise ValueError("More than one simulation found.")
                if not all(variable in sim for variable in indicators) or any(sim[variable].isnull().all().values for variable in indicators):
                    sim = _recompute_hydro(r=realization, y=analog_year, wl=3)
                sim = sim[list(indicators)]
                blend.extend([sim])
            blend_fut3 = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze().drop_vars(["time"])

            delta_kind = {v: "+" if v in ["doy_14qmax", "season_histstart", "season_histend", "season_histduration", "days_under_hist7q2", "days_under_hist7q10"] else "%" for v in indicators}
            deltas2 = xs.compute_deltas(blend_fut2, blend_hist, kind=delta_kind, rename_variables=False)
            deltas3 = xs.compute_deltas(blend_fut3, blend_hist, kind=delta_kind, rename_variables=False)

            analog2 = xr.Dataset()
            analog3 = xr.Dataset()
            for v in indicators:
                if delta_kind[v] == "%":
                    analog2[v] = ref_y[v] + (deltas2[v] / 100) * ref_y[v]
                    analog3[v] = ref_y[v] + (deltas3[v] / 100) * ref_y[v]
                else:
                    analog2[v] = ref_y[v] + deltas2[v]
                    analog3[v] = ref_y[v] + deltas3[v]
            analog2 = _fix_analogs(analog2)
            analog3 = _fix_analogs(analog3)

            for v in indicators:
                # Create a gridspec layout
                plt.figure(figsize=(20, 10))
                gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 0.2, 0.2])

                nlevels = 12
                norm = None
                if indicators[v]["range"][1] == 365:
                    ticks = np.array([1, 15, 32, 46, 60, 74, 91, 105, 121, 135, 152, 166, 182, 196, 213, 227, 244, 258, 274, 288, 305, 319, 335, 349, 365])
                    norm = cl.BoundaryNorm(ticks, ncolors=256)

                # First plot: Climatology
                if "days_under" in v:
                    clim = ref[v].sel(time=slice(f"1992-01-01", f"2021-12-31")).max(dim="time").squeeze()
                else:
                    clim = ref[v].sel(time=slice(f"1992-01-01", f"2021-12-31")).mean(dim="time").squeeze()

                ax1 = plt.subplot(gs[:, 0], projection=proj)
                _indicator_plot(clim, ax1, vmin=indicators[v]["range"][0], vmax=indicators[v]["range"][1], levels=nlevels, norm=norm, cmap=indicators[v]["cmap"], cb=False)
                if "days_under" in v:
                    ax1.set_title("Maximum \n(1992-2021)", fontsize=24, fontweight="bold", pad=20)
                else:
                    ax1.set_title("Climatology \n(1992-2021)", fontsize=24, fontweight="bold", pad=20)

                # First row: Absolute values
                data = {
                    0: ref[v].sel(time=f"{y}-01-01").squeeze(),
                    1: analog2[v],
                    2: analog3[v]
                }
                titles = {
                    0: f"{y}",
                    1: "Future (+2°C)",
                    2: "Future (+3°C)"
                }
                for j in range(1, 4):
                    ax = plt.subplot(gs[0, j], projection=proj)
                    _indicator_plot(data[j-1], ax, vmin=indicators[v]["range"][0], vmax=indicators[v]["range"][1], levels=nlevels, norm=norm, cmap=indicators[v]["cmap"], cb=False)
                    ax.set_title(titles[j-1], fontsize=24, fontweight="bold", pad=20)

                # Second row: Deltas vs. climatology
                if delta_kind[v] == "%":
                    data = {
                        0: (ref_y[v] - clim) / clim * 100,
                        1: (analog2[v] - ref_y[v]) / ref_y[v] * 100,
                        2: (analog3[v] - ref_y[v]) / ref_y[v]* 100
                    }
                    delta_range = [-100, 100]
                else:
                    data = {
                        0: ref_y[v] - clim if "days_under" not in v else ref_y[v] / clim * 100,
                        1: analog2[v] - ref_y[v],
                        2: analog3[v] - ref_y[v]
                    }
                    delta_range = [-90, 90]
                for j in range(1, 4):
                    ax = plt.subplot(gs[1, j], projection=proj)
                    if j == 1 and "days_under" in v:
                        _indicator_plot(data[j - 1], ax, vmin=0, vmax=100, levels=12, cmap=indicators[v]["cmap_div"], cb=False)
                    else:
                        _indicator_plot(data[j-1], ax, vmin=delta_range[0], vmax=delta_range[1], levels=12, cmap=indicators[v]["cmap_div"], cb=False)
                    if j == 1:
                        if "days_under" in v:
                            ax.set_title("% of Maximum", fontsize=24, fontweight="bold", pad=20)
                        else:
                            ax.set_title(f"vs. Climatology", fontsize=24, fontweight="bold", pad=20)
                    else:
                        ax.set_title(f"vs. {y}", fontsize=24, fontweight="bold", pad=20)

                # plt.suptitle(indicators[v]["name"], fontsize=20, fontweight="bold")
                plt.tight_layout()
                plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"indicators_{y}_{v}.png", dpi=300, transparent=True)
                plt.close()

    # Alert levels
    if "alert_levels" in todo:
        ref_q = dc.search(processing_level="indicators", xrfreq="MS", variable='discharge_mean_mon').to_dask(xarray_open_kwargs={"chunks": {}})
        ref_q = _prep_hydro_ds(ref_q, rename=False, indicator_list=False)
        ref_q = ref_q.sel(time=slice("1992-01-01", "2021-12-31"))
        # Reverse reaches_per_zgiebv
        reaches_per_zgiebv2 = {}
        for k in reaches_per_zgiebv:
            for v in reaches_per_zgiebv[k]:
                reaches_per_zgiebv2[v] = k
        reaches_per_zgiebv2 = {k: v for k, v in reaches_per_zgiebv2.items() if k in ref_q.station.values}
        new_coord = xr.DataArray(list(reaches_per_zgiebv2.values()), dims="station", coords={"station": list(reaches_per_zgiebv2.keys())})
        ref_q = ref_q.assign_coords({"SIGLE": new_coord})
        ref_q = ref_q.groupby("SIGLE").mean(dim="station")

        ref_pr = dc.search(processing_level="indicators", variable='precip_accumulation_mon').to_dask(xarray_open_kwargs={"chunks": {}})
        ref_pr = ref_pr.sel(time=slice("1992-01-01", "2021-12-31"))
        ref_pr = ref_pr.swap_dims({"geom": "SIGLE"})

        # Compute thresholds
        q_threshold = xs.climatological_op(ref_q, op="mean", rename_variables=False)
        q_threshold = q_threshold.where(q_threshold.time.dt.month.isin([6, 7, 8]), drop=True)['discharge_mean_mon'].min(dim="time")
        pr_threshold1 = xs.climatological_op(ref_pr, op="mean", rename_variables=False, horizons_as_dim=True).squeeze().drop_vars(["horizon"])["precip_accumulation_mon"]
        pr_threshold3 = xs.climatological_op(ref_pr.rolling({"time": 3}, center=False).sum(), op="mean", rename_variables=False, horizons_as_dim=True).squeeze().drop_vars(["horizon"])["precip_accumulation_mon"]

        def _compute_alert(da_pr, da_q):
            da_pr3 = da_pr.rolling({"month": 3}, center=False).sum()

            alert = None
            months = [v for k, v in xr.coding.cftime_offsets._MONTH_ABBREVIATIONS.items() if k in np.arange(3, 13)]
            for month in months:
                q_coeff = [1.0, 0.7, 0.5] if month in ["MAR", "APR", "MAY"] else [0.7, 0.5, 0.3]

                alert1 = xr.where((da_pr3.sel(month=month) < 0.8 * pr_threshold3.sel(month=month)) |
                                  (da_q.sel(month=month) < q_coeff[0] * q_threshold), 1, 0)
                alert2 = xr.where((da_pr3.sel(month=month) < 0.6 * pr_threshold3.sel(month=month)) |
                                  (da_pr.sel(month=month) < 0.6 * pr_threshold1.sel(month=month)) |
                                  (da_q.sel(month=month) < q_coeff[1] * q_threshold), 2, alert1)
                alert3 = xr.where((da_pr3.sel(month=month) < 0.4 * pr_threshold3.sel(month=month)) |
                                  (da_pr.sel(month=month) < 0.4 * pr_threshold1.sel(month=month)) |
                                  (da_q.sel(month=month) < q_coeff[2] * q_threshold), 3, alert2)

                if alert is None:
                    alert = alert1
                    alert = alert.expand_dims("month")
                else:
                    alert_prev = alert.isel(month=-1).drop_vars(["month"])
                    alert_tmp = xr.where(alert_prev == 2, alert3, xr.where(alert_prev == 1, alert2, xr.where(alert_prev == 0, alert1, 0)))
                    alert_tmp = alert_tmp.expand_dims("month")
                    alert = xr.concat([alert, alert_tmp], dim="month")

            alert.name = "alert"
            return alert

        for y in xs.CONFIG["storylines"]["target_years"]:
            if not os.path.exists(Path(xs.CONFIG["io"]["nhess_data"]) / f"alert_{y}.csv"):
                # Reference
                da_pr = xs.utils.unstack_dates(ref_pr.sel(time=ref_pr.time.dt.year == y)['precip_accumulation_mon'], new_dim="month").squeeze().drop_vars(["time"])
                da_q = xs.utils.unstack_dates(ref_q.sel(time=ref_q.time.dt.year == y)['discharge_mean_mon'], new_dim="month").squeeze().drop_vars(["time"])

                # Open historical RMSE and construct the blend
                rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_0.91.nc", chunks={}).load()
                sorted_analogs_hist = sort_analogs(rmse)
                blend_pr = []
                blend_q = []
                for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                    realization = str(sorted_analogs_hist.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                    analog_year = int(sorted_analogs_hist.isel(stacked=i).time.dt.year.values)
                    sim_search = dc.search(type="simulation", xrfreq=["MS"], variable="pr", member=realization)
                    with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                        sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                        sim = xs.spatial_mean(sim, method="xesmf", region={"name": "zgiebv", "tile_buffer": 1.1, "method": "shape", "shape": shp_zg}, simplify_tolerance=0.01, kwargs={"skipna": True})
                        sim = sim.rename({"geom": "SIGLE"})
                        sim["pr"] = xclim.core.units.rate2amount(sim["pr"], out_units="mm")
                        sim = xs.utils.unstack_dates(sim, new_dim="month").squeeze().drop_vars(["time"])["pr"]
                        blend_pr.extend([sim])

                    sim_search = raw_cat.search(type="simulation-hydro", xrfreq=["D"], member=realization, hydrology_member="MG24HQ")
                    with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                        sim = _prep_hydro_ds(sim, rename=False, indicator_list=False)
                        sim = sim.sel(time=sim.time.dt.year == analog_year).resample(time="MS").mean()
                        sim = sim.assign_coords({"SIGLE": new_coord})
                        sim = sim.groupby("SIGLE").mean(dim="station")
                        sim = xs.utils.unstack_dates(sim, new_dim="month").squeeze().drop_vars(["time"])["streamflow"]
                        blend_q.extend([sim])

                blend_hist_pr = xr.concat(blend_pr, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()
                blend_hist_q = xr.concat(blend_q, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()

                # Open +2 RMSE and construct the blend
                rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_2.nc", chunks={}).load()
                sorted_analogs_fut2 = sort_analogs(rmse)
                blend_pr = []
                blend_q = []
                for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                    realization = str(sorted_analogs_fut2.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                    analog_year = int(sorted_analogs_fut2.isel(stacked=i).time.dt.year.values)
                    sim_search = dc.search(type="simulation", xrfreq=["MS"], variable="pr", member=realization)
                    with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                        sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                        sim = xs.spatial_mean(sim, method="xesmf", region={"name": "zgiebv", "tile_buffer": 1.1, "method": "shape", "shape": shp_zg}, simplify_tolerance=0.01, kwargs={"skipna": True})
                        sim = sim.rename({"geom": "SIGLE"})
                        sim["pr"] = xclim.core.units.rate2amount(sim["pr"], out_units="mm")
                        sim = xs.utils.unstack_dates(sim, new_dim="month").squeeze().drop_vars(["time"])["pr"]
                        blend_pr.extend([sim])

                    sim_search = raw_cat.search(type="simulation-hydro", xrfreq=["D"], member=realization, hydrology_member="MG24HQ")
                    with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                        sim = _prep_hydro_ds(sim, rename=False, indicator_list=False)
                        sim = sim.sel(time=sim.time.dt.year == analog_year).resample(time="MS").mean()
                        sim = sim.assign_coords({"SIGLE": new_coord})
                        sim = sim.groupby("SIGLE").mean(dim="station")
                        sim = xs.utils.unstack_dates(sim, new_dim="month").squeeze().drop_vars(["time"])["streamflow"]
                        blend_q.extend([sim])

                blend_fut2_pr = xr.concat(blend_pr, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()
                blend_fut2_q = xr.concat(blend_q, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()

                # Open +3 RMSE and construct the blend
                rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_3.nc", chunks={}).load()
                sorted_analogs_fut3 = sort_analogs(rmse)
                blend_pr = []
                blend_q = []
                for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                    realization = str(sorted_analogs_fut3.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                    analog_year = int(sorted_analogs_fut3.isel(stacked=i).time.dt.year.values)
                    sim_search = dc.search(type="simulation", xrfreq=["MS"], variable="pr", member=realization)
                    with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                        sim = sim.sel(time=sim.time.dt.year == analog_year).squeeze()
                        sim = xs.spatial_mean(sim, method="xesmf", region={"name": "zgiebv", "tile_buffer": 1.1, "method": "shape", "shape": shp_zg}, simplify_tolerance=0.01, kwargs={"skipna": True})
                        sim = sim.rename({"geom": "SIGLE"})
                        sim["pr"] = xclim.core.units.rate2amount(sim["pr"], out_units="mm")
                        sim = xs.utils.unstack_dates(sim, new_dim="month").squeeze().drop_vars(["time"])["pr"]
                        blend_pr.extend([sim])

                    sim_search = raw_cat.search(type="simulation-hydro", xrfreq=["D"], member=realization, hydrology_member="MG24HQ")
                    with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                        sim = _prep_hydro_ds(sim, rename=False, indicator_list=False)
                        sim = sim.sel(time=sim.time.dt.year == analog_year).resample(time="MS").mean()
                        sim = sim.assign_coords({"SIGLE": new_coord})
                        sim = sim.groupby("SIGLE").mean(dim="station")
                        sim = xs.utils.unstack_dates(sim, new_dim="month").squeeze().drop_vars(["time"])["streamflow"]
                        blend_q.extend([sim])

                blend_fut3_pr = xr.concat(blend_pr, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()
                blend_fut3_q = xr.concat(blend_q, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()

                # Compute deltas and create the analogs
                blend_hist_pr.name = "pr"
                blend_hist_q.name = "q"
                blend_fut2_pr.name = "pr"
                blend_fut2_q.name = "q"
                blend_fut3_pr.name = "pr"
                blend_fut3_q.name = "q"
                deltas2_pr = xs.compute_deltas(blend_fut2_pr.to_dataset(), blend_hist_pr.to_dataset(), kind="%", rename_variables=False)["pr"]
                deltas2_q = xs.compute_deltas(blend_fut2_q.to_dataset(), blend_hist_q.to_dataset(), kind="%", rename_variables=False)["q"]
                deltas3_pr = xs.compute_deltas(blend_fut3_pr.to_dataset(), blend_hist_pr.to_dataset(), kind="%", rename_variables=False)["pr"]
                deltas3_q = xs.compute_deltas(blend_fut3_q.to_dataset(), blend_hist_q.to_dataset(), kind="%", rename_variables=False)["q"]

                analog2_pr = da_pr + (deltas2_pr / 100) * da_pr
                analog2_q = da_q + (deltas2_q / 100) * da_q
                analog3_pr = da_pr + (deltas3_pr / 100) * da_pr
                analog3_q = da_q + (deltas3_q / 100) * da_q

                # Compute alert levels
                alert_ref = _compute_alert(da_pr, da_q)
                alert2 = _compute_alert(analog2_pr, analog2_q)
                alert3 = _compute_alert(analog3_pr, analog3_q)

                # Update the dataframe
                alert_ref.name = f"alert"
                alert2.name = f"alert_2"
                alert3.name = f"alert_3"

                df = pd.concat([alert_ref.sel(month=["JUN", "JUL", "AUG", "SEP", "OCT"]).max(dim="month").to_dataframe(),
                                alert2.sel(month=["JUN", "JUL", "AUG", "SEP", "OCT"]).max(dim="month").to_dataframe(),
                                alert3.sel(month=["JUN", "JUL", "AUG", "SEP", "OCT"]).max(dim="month").to_dataframe()], axis=1)[[f"alert", f"alert_2", f"alert_3"]]

                # Save the dataframe
                df.to_csv(Path(xs.CONFIG["io"]["nhess_data"]) / f"alert_{y}.csv")

            df = pd.read_csv(Path(xs.CONFIG["io"]["nhess_data"]) / f"alert_{y}.csv", index_col=0)
            alert_ref = xr.DataArray(df["alert"].values, dims="SIGLE", coords={"SIGLE": df.index})
            alert_ref.name = "alert"
            alert2 = xr.DataArray(df["alert_2"].values, dims="SIGLE", coords={"SIGLE": df.index})
            alert2.name = "alert_2"
            alert3 = xr.DataArray(df["alert_3"].values, dims="SIGLE", coords={"SIGLE": df.index})
            alert3.name = "alert_3"

            if y == 2021:
                # Add the number of consequences
                cons_ref = xr.DataArray(df["cons"].values, dims="SIGLE", coords={"SIGLE": df.index})
                cons_ref.name = "cons"
                cons2 = xr.DataArray(df["cons_2"].values, dims="SIGLE", coords={"SIGLE": df.index})
                cons2.name = "cons_2"
                cons3 = xr.DataArray(df["cons_3"].values, dims="SIGLE", coords={"SIGLE": df.index})
                cons3.name = "cons_3"

                # Plot
                f = plt.figure(figsize=(18, 10))
                for i in range(3):
                    ax = f.add_subplot(2, 3, i + 1, projection=proj)
                    if i == 0:
                        _alert_plot(alert_ref, ax)
                        plt.title(f"{y}", fontsize=20, fontweight="bold", pad=20)
                    elif i == 1:
                        _alert_plot(alert2, ax)
                        plt.title(f"Future (+2°C)", fontsize=20, fontweight="bold", pad=20)
                    else:
                        _alert_plot(alert3, ax)
                        plt.title(f"Future (+3°C)", fontsize=20, fontweight="bold", pad=20)
                for i in range(3):
                    ax = f.add_subplot(2, 3, i + 4, projection=proj)
                    if i == 0:
                        _indicator_plot(cons_ref, ax, vmin=0, vmax=29, levels=12, cmap="Reds", cb=False, shp="zg")
                        shp_zg.to_crs(proj).boundary.plot(ax=ax, linewidth=1, color="black")
                    elif i == 1:
                        _indicator_plot(cons2, ax, vmin=0, vmax=29, levels=12, cmap="Reds", cb=False, shp="zg")
                        shp_zg.to_crs(proj).boundary.plot(ax=ax, linewidth=1, color="black")
                    else:
                        _indicator_plot(cons3, ax, vmin=0, vmax=29, levels=12, cmap="Reds", cb=False, shp="zg")
                        shp_zg.to_crs(proj).boundary.plot(ax=ax, linewidth=1, color="black")

            else:
                # Plot
                f = plt.figure(figsize=(18, 5.5))
                for i in range(3):
                    ax = f.add_subplot(1, 3, i + 1, projection=proj)
                    if i == 0:
                        _alert_plot(alert_ref, ax)
                        plt.title(f"{y}", fontsize=20, fontweight="bold", pad=20)
                    elif i == 1:
                        _alert_plot(alert2, ax)
                        plt.title(f"Future (+2°C)", fontsize=20, fontweight="bold", pad=20)
                    else:
                        _alert_plot(alert3, ax)
                        plt.title(f"Future (+3°C)", fontsize=20, fontweight="bold", pad=20)

            plt.tight_layout()
            plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"alert_levels_{y}.png", dpi=300, transparent=True)
            plt.close()

    # T&P anomalies
    if "temp_precip_anom" in todo:
        region = {"name": "SouthernQC", "method": "shape", "shape": shp_zg, "tile_buffer": 1.1}

        # Open and subset the data
        ref = dc.search(source="ERA5-Land", variable=['evspsblpot', 'pr']).to_dask(xarray_open_kwargs={"chunks": {}})
        ref = ref.sel(time=slice("1992-01-01", "2021-12-31"))
        ref = xs.spatial.subset(ref, **region)

        # Rate to amount
        ref["pr"] = xclim.core.units.rate2amount(ref["pr"], out_units="mm")
        ref["evspsblpot"] = xclim.core.units.rate2amount(xclim.core.units.convert_units_to(ref["evspsblpot"], "mm s-1", context="hydro"), out_units="mm")
        # Moving sums and climatology
        ref["pr6"] = ref["pr"].rolling({"time": 6}, center=False).sum(keep_attrs=True)
        ref["evspsblpot6"] = ref["evspsblpot"].rolling({"time": 6}, center=False).sum(keep_attrs=True)

        clim = xs.climatological_op(ref, op="mean", rename_variables=False, min_periods=29)
        stdev = xs.utils.unstack_dates(ref, new_dim="month").std(dim="time")

        for y in xs.CONFIG["storylines"]["target_years"]:
            # Open historical RMSE and construct the blend
            rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_0.91.nc", chunks={}).load()
            sorted_analogs_hist = sort_analogs(rmse)
            blend = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_hist.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_hist.isel(stacked=i).time.dt.year.values)
                sim_search = dc.search(type="simulation", xrfreq=["MS"], variable=['pr', 'evspsblpot'], member=realization)
                with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                    sim = xs.spatial.subset(sim, **region)

                    sim["pr"] = xclim.core.units.rate2amount(sim["pr"], out_units="mm")
                    sim["evspsblpot"] = xclim.core.units.rate2amount(xclim.core.units.convert_units_to(sim["evspsblpot"], "mm s-1", context="hydro"), out_units="mm")
                    sim["pr6"] = sim["pr"].rolling({"time": 6}, center=False).sum(keep_attrs=True)
                    sim["evspsblpot6"] = sim["evspsblpot"].rolling({"time": 6}, center=False).sum(keep_attrs=True)

                    sim = xs.utils.unstack_dates(sim.sel(time=sim.time.dt.year == analog_year).squeeze(), new_dim="month").squeeze().drop_vars(["time"])
                    blend.extend([sim])

            blend_hist = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()

            # Open +2 RMSE and construct the blend
            rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_2.nc", chunks={}).load()
            sorted_analogs_fut2 = sort_analogs(rmse)
            blend = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_fut2.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_fut2.isel(stacked=i).time.dt.year.values)
                sim_search = dc.search(type="simulation", xrfreq=["MS"], variable=['pr', 'evspsblpot'], member=realization)
                with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                    sim = xs.spatial.subset(sim, **region)

                    sim["pr"] = xclim.core.units.rate2amount(sim["pr"], out_units="mm")
                    sim["evspsblpot"] = xclim.core.units.rate2amount(xclim.core.units.convert_units_to(sim["evspsblpot"], "mm s-1", context="hydro"), out_units="mm")
                    sim["pr6"] = sim["pr"].rolling({"time": 6}, center=False).sum(keep_attrs=True)
                    sim["evspsblpot6"] = sim["evspsblpot"].rolling({"time": 6}, center=False).sum(keep_attrs=True)

                    sim = xs.utils.unstack_dates(sim.sel(time=sim.time.dt.year == analog_year).squeeze(), new_dim="month").squeeze().drop_vars(["time"])
                    blend.extend([sim])

            blend_fut2 = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()

            # Open +3 RMSE and construct the blend
            rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{y}_3.nc", chunks={}).load()
            sorted_analogs_fut3 = sort_analogs(rmse)
            blend = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_fut3.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_fut3.isel(stacked=i).time.dt.year.values)
                sim_search = dc.search(type="simulation", xrfreq=["MS"], variable=['pr', 'evspsblpot'], member=realization)
                with sim_search.to_dask(xarray_open_kwargs={"chunks": {}}) as sim:
                    sim = xs.spatial.subset(sim, **region)

                    sim["pr"] = xclim.core.units.rate2amount(sim["pr"], out_units="mm")
                    sim["evspsblpot"] = xclim.core.units.rate2amount(xclim.core.units.convert_units_to(sim["evspsblpot"], "mm s-1", context="hydro"), out_units="mm")
                    sim["pr6"] = sim["pr"].rolling({"time": 6}, center=False).sum(keep_attrs=True)
                    sim["evspsblpot6"] = sim["evspsblpot"].rolling({"time": 6}, center=False).sum(keep_attrs=True)

                    sim = xs.utils.unstack_dates(sim.sel(time=sim.time.dt.year == analog_year).squeeze(), new_dim="month").squeeze().drop_vars(["time"])
                    blend.extend([sim])

            blend_fut3 = xr.concat(blend, dim="realization", coords="minimal", compat="override").mean(dim="realization").squeeze()

            # Compute deltas and create the analogs
            deltasref = xs.utils.unstack_dates(xs.compute_deltas(ref.sel(time=f"{y}"), clim, kind="+", rename_variables=False), new_dim="month").squeeze().drop_vars(["time"])
            deltas2 = xs.compute_deltas(blend_fut2, blend_hist, kind="+", rename_variables=False)
            deltas3 = xs.compute_deltas(blend_fut3, blend_hist, kind="+", rename_variables=False)

            # Plot #1: Ref
            f = plt.figure(figsize=(12, 8))
            i = 1
            for v in ["pr6", "evspsblpot6"]:
                cmap = "" if v == "pr6" else "_r"

                for ii in range(2):
                    month = "MAY" if ii == 0 else "OCT"

                    da = deltasref[v].sel(month=month) / stdev[v].sel(month=month)

                    ax = f.add_subplot(2, 2, i, projection=proj)
                    features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                                'states': {"edgecolor": "black", "linestyle": "dotted"}}
                    fg.gridmap(da, ax=ax, cmap=f"prec_div" + cmap, frame=True, features=features, plot_kw={"vmin": -3, "vmax": 3, "levels": 13, 'add_colorbar': False})
                    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
                    i += 1

            plt.tight_layout()

            fig_name = f"pev_precip_anom_{y}_ref"
            plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"{fig_name}.png", dpi=300, transparent=True)
            plt.close()


            # Plot #2: Future

            alldata = [deltas2, deltas3]
            for d in range(2):
                data = alldata[d]

                f = plt.figure(figsize=(12, 8))
                i = 1
                for v in ["pr6", "evspsblpot6"]:
                    cmap = "" if v == "pr6" else "_r"
                    for ii in range(2):
                        month = "MAY" if ii == 0 else "OCT"

                        da = data[v].sel(month=month)
                        lim = 100 if v == "pr6" else 30

                        ax = f.add_subplot(2, 2, i, projection=proj)
                        features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                                    'states': {"edgecolor": "black", "linestyle": "dotted"}}
                        fg.gridmap(da, ax=ax, cmap=f"prec_div" + cmap, frame=True, features=features, plot_kw={"vmin": -lim, "vmax": lim, "levels": 13, 'add_colorbar': False})
                        ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
                        i += 1

                plt.tight_layout()

                fig_name = f"pev_precip_anom_{y}_{d+2}"
                plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"{fig_name}.png", dpi=300, transparent=True)
                plt.close()

                # Plot diff
                f = plt.figure(figsize=(12, 4))
                i = 1
                for ii in range(2):
                    month = "MAY" if ii == 0 else "OCT"

                    da = data["pr6"].sel(month=month) - data["evspsblpot6"].sel(month=month)

                    ax = f.add_subplot(1, 2, i, projection=proj)
                    features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                                'states': {"edgecolor": "black", "linestyle": "dotted"}}
                    fg.gridmap(da, ax=ax, cmap="prec_div", frame=True, features=features,
                               plot_kw={"vmin": -60, "vmax": 60, "levels": 13, 'add_colorbar': False})
                    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
                    i += 1

                plt.tight_layout()

                fig_name = f"pev_precip_anom_{y}_{d + 2}diff"
                plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"{fig_name}.png", dpi=300, transparent=True)
                plt.close()

    # Colorbars
    if "colorbars" in todo:
        # delta_kind = {v: "+" if v in ["doy_14qmax", "season_histstart", "season_histend", "season_histduration", "days_under_hist7q2",
        #                               "days_under_hist7q10"] else "%" for v in indicators}
        # for v in indicators:
        #     # Create the colormap
        #     if isinstance(indicators[v]["cmap"], str):
        #         if indicators[v]["cmap"] != "Reds":
        #             original_cmap = fg.utils.create_cmap(filename=indicators[v]["cmap"])
        #         else:
        #             original_cmap = plt.cm.get_cmap("Reds")
        #     else:
        #         original_cmap = indicators[v]["cmap"]
        #     cbrange = indicators[v]["range"]
        #     if any(substring in v for substring in ["7qmin", "_jja", "_mam", "_son", "14qmax"]):
        #         cbrange = [cbrange[0] * 1000, cbrange[1] * 1000]
        #     colors = original_cmap(np.linspace(0, 1, 12 if indicators[v]["range"][1] != 365 else 24))
        #     cmap = plt.cm.colors.ListedColormap(colors)
        #
        #     # Create the colorbar
        #     fig = plt.figure(figsize=(5, 10))
        #     ax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
        #     cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=cmap)
        #     cb.outline.set_visible(False)
        #     if indicators[v]["range"][1] == 365:
        #         ticks = np.array([15, 46, 76, 106, 137, 167, 198, 228, 258, 289, 320, 350]) / 365  # Fake dates to get the right spacing
        #         cb.ax.set_yticks(ticks)
        #         cb.ax.set_yticklabels(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], fontsize=36, fontweight="bold")
        #     else:
        #         cb.ax.set_yticks(np.linspace(0, 1, 13)[::2])
        #         ticks = [f"{i:.0f}" for i in np.linspace(cbrange[0], cbrange[1], 13)[::2]]
        #         cb.ax.set_yticklabels(ticks, fontsize=36, fontweight="bold")
        #
        #     # Save
        #     plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"colorbar_{v}.png", dpi=300, transparent=True)
        #     plt.close()
        #
        #     # Horizontal
        #     fig = plt.figure(figsize=(10, 5))
        #     ax = fig.add_axes([0.05, 0.3, 0.9, 0.2])
        #     cb = matplotlib.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap)
        #     cb.outline.set_visible(False)
        #     if indicators[v]["range"][1] == 365:
        #         ticks = np.array([15, 46, 76, 106, 137, 167, 198, 228, 258, 289, 320, 350]) / 365  # Fake dates to get the right spacing
        #         cb.ax.set_xticks(ticks)
        #         cb.ax.set_xticklabels(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], fontsize=24, fontweight="bold", rotation=45)
        #     else:
        #         cb.ax.set_xticks(np.linspace(0, 1, 13)[::2])
        #         ticks = [f"{i:.0f}" for i in np.linspace(cbrange[0], cbrange[1], 13)[::2]]
        #         cb.ax.set_xticklabels(ticks, fontsize=24, fontweight="bold")
        #
        #     # Save
        #     plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"colorbar_{v}_horizontal.png", dpi=300, transparent=True)
        #     plt.close()
        #
        #     # Deltas
        #     if isinstance(indicators[v]["cmap_div"], str):
        #         original_cmap = fg.utils.create_cmap(filename=indicators[v]["cmap_div"])
        #     else:
        #         original_cmap = indicators[v]["cmap"]
        #     colors = original_cmap(np.linspace(0, 1, 12))
        #     cmap = plt.cm.colors.ListedColormap(colors)
        #
        #     # Create the colorbar
        #     fig = plt.figure(figsize=(5, 10))
        #     ax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
        #     cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=cmap)
        #     cb.outline.set_visible(False)
        #     cb.ax.set_yticks(np.linspace(0, 1, 13)[::2])
        #     minmax = [-100, 100] if delta_kind[v] == "%" else [-90, 90]
        #     ticks = [f"{i:.0f}" for i in np.linspace(minmax[0], minmax[1], 13)[::2]]
        #
        #     cb.ax.set_yticklabels(ticks, fontsize=36, fontweight="bold")
        #
        #     # Save
        #     plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"colorbar_{v}_delta.png", dpi=300, transparent=True)
        #     plt.close()

        # # Alert levels
        # original_cmap = plt.cm.get_cmap("hot_r")
        # colors = original_cmap(np.linspace(0, 0.6, 4))
        # cmap = plt.cm.colors.ListedColormap(colors)
        #
        # # Create the colorbar
        # fig = plt.figure(figsize=(5, 10))
        # ax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
        # cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=cmap)
        # # cb.outline.set_visible(False)
        #
        # cb.ax.set_yticks(np.linspace(1/8, 7/8, 4))
        # ticks = ["0", "1", "2", "3"]
        # cb.ax.set_yticklabels(ticks, fontsize=36, fontweight="bold")
        #
        # # Save
        # plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"colorbar_alert_levels.png", dpi=300, transparent=True)
        # plt.close()
        #
        # # Consequences
        # original_cmap = plt.cm.get_cmap("Reds")
        # cbrange = [0, 30]
        # colors = original_cmap(np.linspace(0, 1, 12))
        # cmap = plt.cm.colors.ListedColormap(colors)
        #
        # # Create the colorbar
        # fig = plt.figure(figsize=(5, 10))
        # ax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
        # cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=cmap)
        # cb.outline.set_visible(False)
        # cb.ax.set_yticks(np.linspace(0, 1, 13)[::2])
        # ticks = [f"{i:.0f}" for i in np.linspace(cbrange[0], cbrange[1], 13)[::2]]
        # cb.ax.set_yticklabels(ticks, fontsize=36, fontweight="bold")
        #
        # # Save
        # plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"colorbar_alert_cons.png", dpi=300, transparent=True)
        # plt.close()

        # # PEV
        # for cc in ["", "_r"]:
        #     cmap = fg.utils.create_cmap(filename=f"prec_div" + cc)
        #     colors_list = cmap(np.linspace(0, 1, 12))
        #
        #     cmap = cl.ListedColormap(colors_list)
        #
        #     bounds = [3, 100 if cc == "" else 30, 60]
        #     for i in range(3):
        #         fig = plt.figure(figsize=(5, 10))
        #         ax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
        #         cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=cmap)
        #         cb.outline.set_visible(False)
        #         cb.ax.set_yticks(np.linspace(0, 1, 13)[::2])
        #         ticks = [f"{i:.0f}" for i in np.linspace(-bounds[i], bounds[i], 13)[::2]]
        #         cb.ax.set_yticklabels(ticks, fontsize=36, fontweight="bold")
        #
        #         plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"colorbar_pev_pr{cc}_{i}.png", dpi=300, transparent=True)
        #         plt.close()

        # Weights
        original_cmap = plt.cm.get_cmap("Greys")
        colors = original_cmap(np.linspace(0, 1, 9))
        cmap = plt.cm.colors.ListedColormap(colors)

        # Create the colorbar
        fig = plt.figure(figsize=(5, 10))
        ax = fig.add_axes([0.05, 0.05, 0.2, 0.9])
        cb = matplotlib.colorbar.ColorbarBase(ax, orientation='vertical', cmap=cmap)
        cb.outline.set_visible(False)
        cb.ax.set_yticks(np.linspace(0, 1, 10)[1::2])
        ticks = [f"{i:.0f}" for i in np.linspace(1, 5, 10)[::2]]
        cb.ax.set_yticklabels(ticks, fontsize=36, fontweight="bold")

        # Save
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"colorbar_weights.png", dpi=300, transparent=True)
        plt.close()

    if "spei_vs_7qmin" in todo:
        region = {"name": "SouthernQC", "method": "shape", "shape": shp_zg, "tile_buffer": 1.1}

        # Open Portrait
        ref = dc.search(type="reconstruction-hydro", processing_level="indicators", xrfreq=["YS-JAN"]).to_dask(xarray_open_kwargs={"chunks": {}})
        ref = _prep_hydro_ds(ref, rename=True)
        ref = ref.compute()
        ref = rank(ref["7qmin"], dim="time")

        f = plt.figure(figsize=(18, 10))
        years = [2021, 2018, 2012, 2010]

        for i, y in enumerate(years):
            ax = f.add_subplot(2, 2, i + 1, projection=proj)

            weights = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"weights_{y}_0.91.nc", chunks={})
            weights = xs.spatial.subset(weights, **region)
            _weights_plot(weights["weights"].sum(dim="criteria"), ax=ax, vmin=1, vmax=5, levels=9, cb=False, cmap="Greys")

            # Rank of 7Qmin
            _indicator_plot(ref.where(ref <= 3).sel(time=f"{y}-01-01").squeeze(), ax=ax, vmin=0, vmax=30, levels=13, cmap="temp_div_r", cb=False, linewidth=1)

            ax.set_title(f"{y}", fontsize=24, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / "spei_vs_7Qmin.png", dpi=300, transparent=True)
        plt.close()


def _compute_criteria(ref, sim,
                      target_year: int,
                      criteria: list,
                      warming_level: str,
                      weights_2_1: list,
                      ):

    watersheds = xs.CONFIG["analogs"]["targets"]["all"]["region"]
    shp_region = shp_zg.loc[watersheds]
    region = {"name": "SouthernQC", "method": "shape", "shape": shp_region, "tile_buffer": 1.1}

    # Loop on each criterion
    all_rmse = []
    all_weights = []
    for c in criteria:
        target = ref[c[0]].sel(time=f"{target_year}-{c[1]:02d}-01").chunk({"lon": -1})
        candidates = sim[c[0]].where(sim.time.dt.month == c[1], drop=True).chunk({"lon": -1})
        candidates["time"] = pd.to_datetime(candidates["time"].dt.strftime("%Y-01-01"))

        weights = xr.where(target <= -2, weights_2_1[0], xr.where(target <= -1, weights_2_1[1], weights_2_1[2]))
        weights_tmp = xs.spatial.subset(weights, **region).interp_like(weights, method="zero")
        weights = weights_tmp
        weights = weights.assign_coords({"criteria": "-".join([str(cc).upper() for cc in c])})

        rmse = xss.rmse(candidates, target, dim=["lon", "lat"], weights=weights, skipna=True)
        rmse = rmse.assign_coords({"criteria": "-".join([str(cc).upper() for cc in c])})
        all_weights.extend([weights])
        all_rmse.extend([rmse])

    all_rmse = xr.concat(all_rmse, dim="criteria")
    all_rmse.name = "rmse"
    all_rmse.attrs = {
        "long_name": "RMSE",
        "description": f"RMSEs for various SPEIs",
        "units": ""
    }
    os.makedirs(Path(xs.CONFIG["io"]["nhess_data"]), exist_ok=True)
    all_rmse.to_netcdf(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_{target_year}_{warming_level}.nc")

    all_weights = xr.concat(all_weights, dim="criteria")
    all_weights.name = "weights"
    all_weights.attrs = {
        "long_name": "Weights",
        "description": f"Weights for various SPEIs",
        "units": ""
    }
    all_weights = all_weights.to_dataset()
    all_weights.to_netcdf(Path(xs.CONFIG["io"]["nhess_data"]) / f"weights_{target_year}_{warming_level}.nc")


def _spei_plot(da, title="", ax=None, cb=True, rmse=None):
    features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                'states': {"edgecolor": "black", "linestyle": "dotted"}}

    fg.gridmap(da, ax=ax, contourf=True, features=features, cmap="prec_div", frame=True,
               plot_kw={"vmin": -3, "vmax": 3, "levels": 13, 'add_colorbar': cb})
    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())

    # if cb:
    #     # Bigger colorbar ticks
    #     cb_ax = fig.axes[1]
    #     cb_ax.tick_params(labelsize=18)
    #     # Colorbar label
    #     cb_ax.set_ylabel("Standardized Precipitation-Evapotranspiration Index", fontsize=20, labelpad=20)

    plt.title(title, fontsize=20)

    if rmse is not None:
        ax.text(0.01, 0.985, rmse, transform=ax.transAxes, fontsize=18, color="black", ha="left", va="top", bbox=dict(facecolor='white', edgecolor='black'))


def _weights_plot(da, title="", ax=None, cb=False, vmin=0, vmax=1, cmap="Reds", levels=13):
    features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                'states': {"edgecolor": "black", "linestyle": "dotted"}}

    fg.gridmap(da, contourf=False, ax=ax, features=features, cmap=cmap, frame=True,
               plot_kw={"vmin": vmin, "vmax": vmax+1e-15, "levels": levels, 'add_colorbar': cb})
    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())
    plt.title(title, fontsize=20)


def _indicator_plot(da, ax, title="", vmin=None, vmax=None, levels=None, norm=None, cmap="temp_seq", cb=False, linewidth=1.5, shp="Portrait"):
    if shp == "Portrait":
        shp = deepcopy(shp_portrait)
    else:
        shp = deepcopy(shp_zg)
    da2 = da.to_dataframe()
    j = da2.index.intersection(shp.index)
    shp[da.name] = da2.loc[j][da.name]

    if isinstance(linewidth, str):
        # Format divisor_min_max
        linewidth = (shp["SUPERFICIE"] / float(linewidth.split("_")[0])).clip(float(linewidth.split("_")[1]), float(linewidth.split("_")[2]))

    features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                'states': {"edgecolor": "black", "linestyle": "dotted"}}

    if norm is None:
        fg.gdfmap(df=shp, df_col=da.name, ax=ax, cmap=cmap, cbar=cb, frame=True, features=features, levels=levels, plot_kw={"vmin": vmin, "vmax": vmax, "linewidth": linewidth})
    else:
        fg.gdfmap(df=shp, df_col=da.name, ax=ax, cmap=cmap, cbar=cb, frame=True, features=features, plot_kw={"norm": norm, "linewidth": linewidth})

    # Ocean in white in foreground
    ax.add_feature(feature.OCEAN, zorder=100, facecolor="#ffffff")

    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())

    # if cb:
    #     # Bigger colorbar ticks
    #     cb_ax = fig.axes[1]
    #     cb_ax.tick_params(labelsize=18)
    #     cb_ax.set_ylabel(f"{da.attrs.get('long_name','').replace('.', '')} ({da.attrs.get('units','')})", fontsize=20, labelpad=20)

    plt.title(title, fontsize=20)


def _alert_plot(da, ax):
    shp = deepcopy(shp_zg)
    da2 = da.to_dataframe()
    j = da2.index.intersection(shp.index)
    shp[da.name] = da2.loc[j][da.name]

    features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                'states': {"edgecolor": "black", "linestyle": "dotted"}}

    fg.gdfmap(df=shp, df_col=da.name, ax=ax, cmap="hot_r", cbar=False, levels=list(np.arange(0, 6, 1)), frame=True, plot_kw={'legend_kwds': {'orientation': 'vertical'}, 'edgecolor': 'black', 'linewidth': 1})
    add_cartopy_features(ax, features)

    # Ocean in white in foreground
    ax.add_feature(feature.OCEAN, zorder=100, facecolor="#ffffff")

    ax.set_extent(extent, crs=cartopy.crs.PlateCarree())


def _fix_analogs(ds):
    if "doy_14qmax" in ds:
        # Negative DOY
        ds["doy_14qmax"] = xr.where(ds["doy_14qmax"] < 0, ds["doy_14qmax"] + 365, ds["doy_14qmax"])

    # Number of days
    for v in ["days_under_hist7q2", "days_under_hist7q10"]:
        ds[v] = ds[v].clip(min=0, max=365)

    # Season start and end
    ds["season_histstart"] = xr.where(ds["season_histstart"] > 273, np.nan, ds["season_histstart"])
    ds["season_histend"] = xr.where(ds["season_histend"] > 365, np.nan, ds["season_histend"])
    ds["season_histstart"] = xr.where(ds["season_histend"] < ds["season_histstart"], np.nan, ds["season_histstart"])
    ds["season_histend"] = xr.where(ds["season_histstart"].isnull(), np.nan, ds["season_histend"])
    # ds["season_histduration"] = xr.where(ds["season_histstart"].isnull(), 0, ds["season_histend"] - ds["season_histstart"])

    return ds


def sort_analogs(da):
    if isinstance(da, xr.Dataset):
        da = da["rmse"]
    if "criteria" in da.dims:
        da = da.sum(dim="criteria", skipna=False)

    if "time" in da.dims:
        da = da.stack({"stacked": ["time", "realization"]})
    else:
        da = da.stack({"stacked": ["year", "realization"]})
    da = da.sortby(da)

    return da


if __name__ == '__main__':
    figures = ["spei_vs_7qmin"]

    main(todo=figures)
