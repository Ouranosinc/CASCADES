import numpy as np
import xclim
import pandas as pd
import geopandas as gpd
import cartopy
import json
import xarray as xr
import xclim.core.units

import xscen as xs
import matplotlib.pyplot as plt
from pathlib import Path

import logging
logger = logging.getLogger("distributed")
logger.setLevel(logging.WARNING)
logger2 = logging.getLogger("flox")
logger2.setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Qt5Agg")
plt.ion()

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
    # Patch the paths, since data has been moved
    dc.esmcat.df["path"] = dc.esmcat.df["path"].str.replace("/exec/rondeau/projets/", "/jarre/rondeau/")

    raw_cat = xs.DataCatalog(xs.CONFIG["io"]["nhess_rawcat"])

    watersheds = ['OBV Yamaska', 'COGESAF', 'COPERNIC', 'COBARIC', 'ABV des 7', 'OBVRLY', 'OBV-Capitale', 'OBVLSJ']

    if "annual_cycles" in todo:
        ds_ref = raw_cat.search(variable=["streamflow"], type="reconstruction-hydro", periods=["1992", "2021"]).to_dataset(xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})
        ds_ref = ds_ref.sel(time=slice("1992", "2021"))

        hist_years = xs.get_period_from_warming_level("CMIP5_CanESM2_rcp85_r1-r1i1p1", 0.91, window=30)
        ds_sim = raw_cat.search(variable=["streamflow"], type="simulation-hydro", activity="ClimEx", hydrology_member="MG24HQ", periods=hist_years).to_dataset(create_ensemble_on=["member"], xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})
        ds_sim = ds_sim.sel(time=slice(hist_years[0], hist_years[1]))

        for i, w in enumerate(watersheds):
            # Get the reaches in the ZGIEBV
            w_y = []
            w_y.extend(reaches_per_zgiebv[w])
            w_y = list(set(w_y).intersection(stations_atlas["TRONCON_ID"]))

            ds_ref_w = ds_ref.sel(station=w_y).compute()
            ds_sim_w = ds_sim.sel(station=w_y).compute()

            # Compute the specific streamflow
            ds_ref_w["q"] = ds_ref_w["streamflow"] / ds_ref_w["drainage_area"]
            ds_ref_w["q"].attrs["units"] = "m3 s-1 km-2"
            ds_ref_w["q"] = xclim.core.units.convert_units_to(ds_ref_w["q"], "L s-1 km-2", context="hydro")
            ds_sim_w["q"] = ds_sim_w["streamflow"] / ds_sim_w["drainage_area"]
            ds_sim_w["q"].attrs["units"] = "m3 s-1 km-2"
            ds_sim_w["q"] = xclim.core.units.convert_units_to(ds_sim_w["q"], "L s-1 km-2", context="hydro")

            # Compute the annual cycles
            ds_ref_w = ds_ref_w.convert_calendar("365_day")
            ds_ref_w = ds_ref_w.sel(percentile=50.0)
            ds_ref_w_doy = ds_ref_w.groupby("time.dayofyear").quantile([0.1, 0.5, 0.9], dim=["time", "station"])
            ds_sim_w = ds_sim_w.convert_calendar("365_day")
            ds_sim_w_doy = ds_sim_w.groupby("time.dayofyear").quantile([0.1, 0.5, 0.9], dim=["time", "realization", "station"])

            f, ax = plt.subplots(1, 1, figsize=(10, 6))

            ax.fill_between(ds_ref_w_doy.dayofyear, ds_ref_w_doy["q"].sel(quantile=0.9), ds_ref_w_doy["q"].sel(quantile=0.1), color="k", alpha=0.2)
            # ds_ref_w_doy["q"].sel(quantile=0.1).plot(color="k", ax=ax, linestyle="--")
            ds_ref_w_doy["q"].sel(quantile=0.5).plot(color="k", ax=ax)
            # ds_ref_w_doy["q"].sel(quantile=0.9).plot(color="k", ax=ax, linestyle="--")
            ds_sim_w_doy["q"].sel(quantile=0.1).plot(color="r", ax=ax, linestyle="--")
            ds_sim_w_doy["q"].sel(quantile=0.5).plot(color="r", ax=ax)
            ds_sim_w_doy["q"].sel(quantile=0.9).plot(color="r", ax=ax, linestyle="--")

            plt.ylabel("Specific discharge [L s$^{-1}$ km$^{-2}$]", fontsize=14)
            ax.tick_params(axis='y', labelsize=14)
            plt.xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=14)
            plt.xlabel("")
            plt.title(w, fontsize=14, fontweight="bold")
            plt.tight_layout()
            plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"supplements_ClimEx_validation_{w}.png", dpi=300, transparent=True)
            plt.close()

    # Timeseries of P, PET and specific streamflow at a few key ZGIEBV
    if "timeseries" in todo:
        ds_meteo_ref = dc.search(variable=["evspsblpot", "pr"], xrfreq="MS", source="ERA5-Land").to_dataset()
        ds_meteo_ref = ds_meteo_ref.sel(time=slice("1992", "2021"))
        ds_hydro_ref = raw_cat.search(variable=["streamflow"], type="reconstruction-hydro", periods=["1992", "2021"]).to_dataset(xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})
        ds_hydro_ref = ds_hydro_ref.sel(time=slice("1992", "2021"))

        hist_years = xs.get_period_from_warming_level("CMIP5_CanESM2_rcp85_r1-r1i1p1", 0.91, window=30)
        ds_meteo_sim = dc.search(variable=["evspsblpot", "pr"], xrfreq="MS", source="CRCM5-Ouranos").to_dataset(create_ensemble_on=["member"])
        ds_hydro_sim = raw_cat.search(variable=["streamflow"], type="simulation-hydro", activity="ClimEx", hydrology_member="MG24HQ", periods=hist_years).to_dataset(create_ensemble_on=["member"], xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})

        # Open RMSE
        rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_0.91.nc", chunks={}).load()
        sorted_analogs_hist = sort_analogs(rmse)
        rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_2.nc", chunks={}).load()
        sorted_analogs_2 = sort_analogs(rmse)
        rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_3.nc", chunks={}).load()
        sorted_analogs_3 = sort_analogs(rmse)

        for ii, w in enumerate(watersheds):
            region = {"name": "ZGIEBV", "method": "shape", "shape": shp_zg.loc[[w]]}
            # Get the reaches in the ZGIEBV
            w_y = []
            w_y.extend(reaches_per_zgiebv[w])
            w_y = list(set(w_y).intersection(stations_atlas["TRONCON_ID"]))

            ds_meteo_ref_w = xs.spatial_mean(ds_meteo_ref, method="xesmf", region=region, simplify_tolerance=0.01).compute()
            ds_hydro_ref_w = ds_hydro_ref.sel(station=w_y, percentile=50.0).compute()
            ds_hydro_ref_w["q"] = ds_hydro_ref_w["streamflow"] / ds_hydro_ref_w["drainage_area"]
            ds_hydro_ref_w["q"].attrs["units"] = "m3 s-1 km-2"
            ds_hydro_ref_w["q"] = xclim.core.units.convert_units_to(ds_hydro_ref_w["q"], "L s-1 km-2", context="hydro")

            ds_meteo_ref_w_avg = ds_meteo_ref_w.sel(time=slice("1992", "2021")).groupby("time.month").mean()
            ds_meteo_ref_w_2021 = ds_meteo_ref_w.sel(time=slice("2021", "2021")).groupby("time.month").mean()
            ds_meteo_ref_w_2021_anom = (ds_meteo_ref_w_2021 - ds_meteo_ref_w_avg) / ds_meteo_ref_w_avg * 100
            ds_hydro_ref_w_avg = ds_hydro_ref_w.sel(time=slice("1992", "2021")).groupby("time.month").mean(dim=["time", "station"])
            ds_hydro_ref_w_2021 = ds_hydro_ref_w.sel(time=slice("2021", "2021")).groupby("time.month").mean(dim=["time", "station"])
            ds_hydro_ref_w_2021_anom = (ds_hydro_ref_w_2021 - ds_hydro_ref_w_avg) / ds_hydro_ref_w_avg * 100

            ds_meteo_sim_w = xs.spatial_mean(ds_meteo_sim, method="xesmf", region=region, simplify_tolerance=0.01).compute()
            ds_hydro_sim_w = ds_hydro_sim.sel(station=w_y).compute()
            ds_hydro_sim_w["q"] = ds_hydro_sim_w["streamflow"] / ds_hydro_sim_w["drainage_area"]
            ds_hydro_sim_w["q"].attrs["units"] = "m3 s-1 km-2"
            ds_hydro_sim_w["q"] = xclim.core.units.convert_units_to(ds_hydro_sim_w["q"], "L s-1 km-2", context="hydro")

            ds_meteo_sim_w_avg = ds_meteo_sim_w.sel(time=slice(hist_years[0], hist_years[1])).groupby("time.month").mean(dim=["time", "realization"])
            ds_hydro_sim_w_avg = ds_hydro_sim_w.sel(time=slice(hist_years[0], hist_years[1])).groupby("time.month").mean(dim=["time", "realization", "station"])
            # Historical
            blend = []
            blend_hydro = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_hist.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_hist.isel(stacked=i).time.dt.year.values)
                ds_analog = ds_meteo_sim_w.sel(realization=realization, time=slice(f"{analog_year}", f"{analog_year}")).groupby("time.month").mean()
                # ds_analog_anom = (ds_analog - ds_meteo_sim_w_avg) / ds_meteo_sim_w_avg * 100
                blend.append(ds_analog)
                ds_hydro_analog = ds_hydro_sim_w.sel(realization=realization, time=slice(f"{analog_year}", f"{analog_year}")).groupby("time.month").mean()
                # ds_hydro_analog_anom = (ds_hydro_analog - ds_hydro_sim_w_avg) / ds_hydro_sim_w_avg * 100
                blend_hydro.append(ds_hydro_analog)
            ds_meteo_sim_w_analog = (xr.concat(blend, dim="analog").mean(dim="analog") - ds_meteo_sim_w_avg) / ds_meteo_sim_w_avg * 100
            ds_hydro_sim_w_analog = (xr.concat(blend_hydro, dim="analog").mean(dim=["analog", "station"]) - ds_hydro_sim_w_avg) / ds_hydro_sim_w_avg * 100

            # +2C
            blend = []
            blend_hydro = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_2.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_2.isel(stacked=i).time.dt.year.values)
                ds_analog = ds_meteo_sim_w.sel(realization=realization, time=slice(f"{analog_year}", f"{analog_year}")).groupby("time.month").mean()
                # ds_analog_anom = (ds_analog - ds_meteo_sim_w_avg) / ds_meteo_sim_w_avg * 100
                blend.append(ds_analog)
                ds_hydro_analog = ds_hydro_sim_w.sel(realization=realization, time=slice(f"{analog_year}", f"{analog_year}")).groupby("time.month").mean()
                # ds_hydro_analog_anom = (ds_hydro_analog - ds_hydro_sim_w_avg) / ds_hydro_sim_w_avg * 100
                blend_hydro.append(ds_hydro_analog)
            ds_meteo_sim_w_analog2 = (xr.concat(blend, dim="analog").mean(dim="analog") - ds_meteo_sim_w_avg) / ds_meteo_sim_w_avg * 100
            # ds_meteo_sim_w_analog2 = (ds_meteo_ref_w_2021 * ((100 + ds_meteo_sim_w_analog2) / 100) - ds_meteo_ref_w_avg) / ds_meteo_ref_w_avg * 100
            ds_hydro_sim_w_analog2 = (xr.concat(blend_hydro, dim="analog").mean(dim=["analog", "station"]) - ds_hydro_sim_w_avg) / ds_hydro_sim_w_avg * 100

            # +3C
            blend = []
            blend_hydro = []
            for i in range(xs.CONFIG["storylines"]["n_analogs"]):
                realization = str(sorted_analogs_3.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(sorted_analogs_3.isel(stacked=i).time.dt.year.values)
                ds_analog = ds_meteo_sim_w.sel(realization=realization, time=slice(f"{analog_year}", f"{analog_year}")).groupby("time.month").mean()
                # ds_analog_anom = (ds_analog - ds_meteo_sim_w_avg) / ds_meteo_sim_w_avg * 100
                blend.append(ds_analog)
                ds_hydro_analog = ds_hydro_sim_w.sel(realization=realization, time=slice(f"{analog_year}", f"{analog_year}")).groupby("time.month").mean()
                # ds_hydro_analog_anom = (ds_hydro_analog - ds_hydro_sim_w_avg) / ds_hydro_sim_w_avg * 100
                blend_hydro.append(ds_hydro_analog)
            ds_meteo_sim_w_analog3 = (xr.concat(blend, dim="analog").mean(dim="analog") - ds_meteo_sim_w_avg) / ds_meteo_sim_w_avg * 100
            # ds_meteo_sim_w_analog3 = (ds_meteo_ref_w_2021 * ((100 + ds_meteo_sim_w_analog3) / 100) - ds_meteo_ref_w_avg) / ds_meteo_ref_w_avg * 100
            ds_hydro_sim_w_analog3 = (xr.concat(blend_hydro, dim="analog").mean(dim=["analog", "station"]) - ds_hydro_sim_w_avg) / ds_hydro_sim_w_avg * 100

            # Define the number of bars and their positions
            n_groups = 11
            index = np.arange(n_groups)
            bar_width = 0.2

            # Plot #1: Validation
            f, axes = plt.subplots(1, 3, figsize=(18, 5))
            variables = ["pr", "evspsblpot", "q"]

            for iii, v in enumerate(variables):
                ax = plt.subplot(1, 3, iii + 1)
                data = [ds_meteo_ref_w_2021_anom, ds_meteo_sim_w_analog, ds_meteo_sim_w_analog2, ds_meteo_sim_w_analog3] if v != "q" else \
                    [ds_hydro_ref_w_2021_anom, ds_hydro_sim_w_analog, ds_hydro_sim_w_analog2, ds_hydro_sim_w_analog3]

                bar1 = ax.bar(index, data[0][v].isel(month=slice(0, -1)), bar_width, label='2021 Anom', color="k")
                bar2 = ax.bar(index + bar_width, data[1][v].isel(month=slice(0, -1)), bar_width, label='Analog', color="gray")
                # bar3 = ax.bar(index + 2 * bar_width, data[2][v].isel(month=slice(0, -1)), bar_width, label='Analog 2', color="orange")
                # bar4 = ax.bar(index + 3 * bar_width, data[3][v].isel(month=slice(0, -1)), bar_width, label='Analog 3', color="red")

                # Add labels, title, and legend
                if iii == 0:
                    ax.set_ylabel('Difference vs. Historical Avg (%)', fontsize=14)
                ax.set_title(f"{v.replace('evspsblpot', 'PET').replace('pr', 'Precipitation').replace('q', 'Specific Streamflow')}", fontsize=14, fontweight="bold")
                ax.set_xticks(index + 1.5 * bar_width)
                ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N"], fontsize=14)
                plt.ylim([-100, 50])
                ax.tick_params(axis='y', labelsize=14)
                plt.axhline(0, color='black', lw=0.5)
                # ax.legend()
            plt.suptitle(w, fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"supplements_meteo_validation_{w}.png", dpi=300, transparent=True)
            plt.close()

            # Plot #2: Future
            f, axes = plt.subplots(1, 3, figsize=(18, 5))
            variables = ["pr", "evspsblpot", "q"]

            for iii, v in enumerate(variables):
                ax = plt.subplot(1, 3, iii + 1)
                data = [ds_meteo_ref_w_2021_anom, ds_meteo_sim_w_analog, ds_meteo_sim_w_analog2, ds_meteo_sim_w_analog3] if v != "q" else \
                    [ds_hydro_ref_w_2021_anom, ds_hydro_sim_w_analog, ds_hydro_sim_w_analog2, ds_hydro_sim_w_analog3]

                # bar1 = ax.bar(index, data[0][v].isel(month=slice(0, -1)), bar_width, label='2021 Anom', color="k")
                bar2 = ax.bar(index + bar_width, data[1][v].isel(month=slice(0, -1)), bar_width, label='Analog', color="gray")
                bar3 = ax.bar(index + 2 * bar_width, data[2][v].isel(month=slice(0, -1)), bar_width, label='Analog 2', color="orange")
                bar4 = ax.bar(index + 3 * bar_width, data[3][v].isel(month=slice(0, -1)), bar_width, label='Analog 3', color="red")

                # Add labels, title, and legend
                if iii == 0:
                    ax.set_ylabel('Difference vs. Historical Avg (%)', fontsize=14)
                ax.set_title(f"{v.replace('evspsblpot', 'PET').replace('pr', 'Precipitation').replace('q', 'Specific Streamflow')}", fontsize=14,
                             fontweight="bold")
                ax.set_xticks(index + 1.5 * bar_width)
                ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N"], fontsize=14)
                plt.ylim([-100, 50])
                ax.tick_params(axis='y', labelsize=14)
                plt.axhline(0, color='black', lw=0.5)
                # ax.legend()
            plt.suptitle(w, fontsize=16, fontweight="bold")
            plt.tight_layout()
            plt.savefig(Path(xs.CONFIG["io"]["nhess_fig"]) / f"supplements_meteo_{w}.png", dpi=300, transparent=True)
            plt.close()


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
    figures = ["annual_cycles", "timeseries"]

    main(todo=figures)
