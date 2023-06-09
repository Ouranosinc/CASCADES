import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy
import json
import xarray as xr
import xscen as xs
import spirograph.matplotlib as sp
import matplotlib.pyplot as plt
import os

from copy import deepcopy

from xclim.sdba.utils import ecdf
from utils import sort_analogs

import matplotlib
matplotlib.use("Qt5Agg")

# xscen setup
xs.load_config("paths.yml", "configs/cfg_analogs.yml", reset=True)


def main(todo: list):
    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

    # Open the ZGIEBV
    shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")
    shp_zg = shp_zg.set_index("SIGLE")
    # Open the list of reaches per ZGIEBV
    with open(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_per_reach.json", 'r') as fp:
        tmp = json.load(fp)
    regions = {}
    for key, vals in tmp.items():
        for val in vals:
            regions[val] = key

    # Portrait shapefiles
    shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
    shp_portrait = shp_portrait.set_index("TRONCON")
    stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
    stations_atlas = stations_atlas.loc[stations_atlas["MASQUE"] == 0]  # Remove invalid/fake stations

    # Projection setup for the figures
    proj = cartopy.crs.Mercator()
    shp_zg = shp_zg.to_crs(proj)
    shp_portrait = shp_portrait.to_crs(proj)

    if "nice_figures" in todo:
        shp_portrait_bv = gpd.read_file(f"{xs.CONFIG['gis']}A22_contour_bv_troncon/contour_bv_troncon_complet_wsg84.shp")
        shp_portrait_bv = shp_portrait_bv.set_index("Num").sort_index(ascending=True)
        shp_portrait_bv = shp_portrait_bv.to_crs(proj)

    lon_bnds = [-79.75, -58.]
    lat_bnds = [44.75, 54.]

    # Show the SPEI and 7Qmin for the 5 chosen years
    if "annexe_figure1" in todo:
        features = {'land': {"color": "#f0f0f0"}, 'ocean': {}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                    'states': {"edgecolor": "black", "linestyle": "dotted"}}

        ref_spei = pcat.search(processing_level="indicators", source="ERA5-Land", variable=["spei3", "spei6"]).to_dask()
        ref_reaches = pcat.search(processing_level="indicators",  variable=["7qmin"]).to_dask()
        ref_reaches = ref_reaches.where(~ref_reaches.station_id.isin(list(stations_atlas["TRONCON_ID"])))

        for y in [2021, 2018, 2012, 2010, 1995]:
            # Sum of negative SPEI-3
            da = ref_spei.sel(time=slice(f"{y}-05-01", f"{y}-11-01"))
            da = da.where(da <= 0)
            da = da["spei3"].sum(dim="time")
            da = da.where(da <= -3)

            plt.figure(figsize=(15, 15))
            ax = plt.subplot(1, 1, 1, projection=proj)
            sp.gridmap(da, contourf=True, ax=ax, features=features, cmap="prec_div", frame=True, plot_kw={"vmin": -12, "vmax": 12, "levels": 13, 'cbar_kwargs': {'orientation': 'horizontal'}})

            # ---
            # Empirical quantile of 7Qmin
            da = ref_reaches.sel(time=slice(f"{y}-01-01", f"{y}-01-01")).squeeze()
            da = ecdf(ref_reaches["7qmin"], da["7qmin"])

            drain = [0, 100, 1000, 100000]
            width = [0.33, 0.85, 1.5]
            for i in range(len(drain) - 1):
                shp = deepcopy(shp_portrait)
                da2 = da.where((da.drainage_area > drain[i]) & (da.drainage_area <= drain[i+1]) & (da <= 0.15)).to_dataframe().set_index("station_id")
                j = da2.index.intersection(shp.index)
                shp[da.name] = da2.loc[j][da.name]

                sp.gdfmap(df=shp, df_col="7qmin", ax=ax, cmap="temp_div_r", levels=9, divergent=0.5, cbar=True if i == 0 else False,
                          frame=True, plot_kw={"vmin": 0, "vmax": 1, "linewidth": width[i]})

            shp_zg.plot(ax=ax, facecolor="None", edgecolor="k", linewidth=0.5)
            ax.set_extent([lon_bnds[0], lon_bnds[1], lat_bnds[0], lat_bnds[1]], crs=cartopy.crs.PlateCarree())

            os.makedirs(xs.CONFIG["io"]["phase1"], exist_ok=True)
            plt.savefig(f"{xs.CONFIG['io']['phase1']}Annexe_Fig1_{y}.png")
            plt.close()

    # Show the construction of an analog, step-by-step, Part #1
    if "cwra_spei" in todo:
        features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                    'states': {"edgecolor": "black", "linestyle": "dotted"}}

        ref = pcat.search(processing_level="indicators", source="ERA5-Land", variable=["spei3", "spei6"]).to_dask()
        ref = ref.sel(time=slice("2021", "2021"))
        data = [ref]

        perf = sort_analogs(pcat.search(processing_level=f"2-performance-vs-2021").to_dask().rmse)
        for i in [0, 1, 49]:
            member = str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
            analog_year = int(perf.isel(stacked=i).time.dt.year.values)
            ds = pcat.search(activity="ClimEx", processing_level="indicators-warminglevel-2.*", variable=["spei3", "spei6"], member=member).to_dask()
            [ds[c].load() for c in ds.coords]
            data.extend([ds.sel(time=slice(str(analog_year), str(analog_year)))])

        titles = ["ref", "analog1", "analog2", "analog50"]

        for i in range(len(data)):
            for v in ["spei3", "spei6"]:
                months = [5, 11] if v == "spei3" else [5, 10]
                for m in months:
                    da = data[i][v].where(data[i].time.dt.month == m, drop=True).squeeze()

                    plt.figure(figsize=(15, 15))
                    ax = plt.subplot(1, 1, 1, projection=proj)
                    sp.gridmap(da, contourf=True, ax=ax, features=features, cmap="prec_div", frame=True,
                               plot_kw={"vmin": -3, "vmax": 3, "levels": 13, 'cbar_kwargs': {'orientation': 'horizontal'}})

                    ax.set_extent([lon_bnds[0], lon_bnds[1], lat_bnds[0], lat_bnds[1]], crs=cartopy.crs.PlateCarree())

                    os.makedirs(xs.CONFIG["io"]["phase1"], exist_ok=True)
                    plt.savefig(f"{xs.CONFIG['io']['phase1']}CWRA_{v}-{m}_{titles[i]}.png")
                    plt.close()

    # Show the construction of an analog, step-by-step, Part #2
    if "annexe_figure2" in todo:
        features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                    'states': {"edgecolor": "black", "linestyle": "dotted"}}

        ref = pcat.search(processing_level="indicators", variable=["season_start", "season_end"]).to_dask()
        ref["season_length"] = ref["season_end"] - ref["season_start"]
        ref = ref.where(~ref.station_id.isin(list(stations_atlas["TRONCON_ID"])))
        ref = ref.sel(time="2021-01-01").squeeze()

        # Analogs were not saved and need to be reconstructed
        perf = sort_analogs(pcat.search(processing_level=f"0.91-performance-vs-2021").to_dask().rmse)
        analog_hist = []
        for i in range(5):
            member = str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
            analog_year = int(perf.isel(stacked=i).time.dt.year.values)
            ds = pcat.search(type="simulation-hydro", activity="ClimEx", processing_level="indicators-0.91", variable=["season_start", "season_end"], member=member, xrfreq="AS-JAN").to_dask()
            ds["season_length"] = (ds["season_end"] - ds["season_start"])
            ds["season_length"].attrs["units"] = "d"
            [ds[c].load() for c in ds.coords]
            analog_hist.extend([ds.sel(time=slice(str(analog_year), str(analog_year))).squeeze().drop_vars(["time"])])
        hist = xr.concat(analog_hist, dim="realization").mean(dim="realization").squeeze()
        hist = hist.where(~hist.station_id.isin(list(stations_atlas["TRONCON_ID"])))

        perf = sort_analogs(pcat.search(processing_level=f"2-performance-vs-2021").to_dask().rmse)
        analog_fut = []
        for i in range(5):
            member = str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
            analog_year = int(perf.isel(stacked=i).time.dt.year.values)
            ds = pcat.search(type="simulation-hydro", activity="ClimEx", processing_level="indicators-2", variable=["season_start", "season_end"], member=member, xrfreq="AS-JAN").to_dask()
            ds["season_length"] = (ds["season_end"] - ds["season_start"])
            ds["season_length"].attrs["units"] = "d"
            [ds[c].load() for c in ds.coords]
            analog_fut.extend([ds.sel(time=slice(str(analog_year), str(analog_year))).squeeze().drop_vars(["time"])])
        fut = xr.concat(analog_fut, dim="realization").mean(dim="realization").squeeze()
        fut = fut.where(~fut.station_id.isin(list(stations_atlas["TRONCON_ID"])))

        deltas = xs.compute_deltas(fut, hist, kind="+", rename_variables=False)
        analog = ref + deltas

        data = [ref, hist, fut, analog, deltas]
        titles = ["ref", "hist", "fut", "analog", "deltas"]

        v = "season_length"

        vmin = 0
        vmax = np.max([np.max(d[v]).values for d in data[:-2]])
        steps = np.round(((vmax - vmin) / 9) / 5) * 5
        levels = list(np.arange(vmin, vmax + steps, steps))
        for i in range(len(data)):
            da = data[i][v]

            if i == 4:
                vmin = -60
                vmax = 60
                steps = np.round(((vmax - vmin) / 9) / 5) * 5
                levels = list(np.arange(vmin, vmax + steps, steps))

            plt.figure(figsize=(15, 15))
            ax = plt.subplot(1, 1, 1, projection=proj)

            dazg = da.assign_coords({"SIGLE": da.station_id.to_series().map(regions)}).squeeze()
            dazg = dazg.groupby(dazg.SIGLE).mean(dim="station")
            shp = deepcopy(shp_zg)
            da2 = dazg.to_dataframe()
            j = da2.index.intersection(shp.index)
            shp[dazg.name] = da2.loc[j][dazg.name]

            sp.gdfmap(df=shp, df_col=v, ax=ax, cmap="temp_seq" if i != 4 else "temp_div", cbar=True, levels=levels,
                      frame=True, features=features, divergent=None if i != 4 else 0)

            shp_zg.plot(ax=ax, facecolor="None", edgecolor="k", linewidth=0.5)
            ax.set_extent([lon_bnds[0], lon_bnds[1], lat_bnds[0], lat_bnds[1]], crs=cartopy.crs.PlateCarree())

            os.makedirs(xs.CONFIG["io"]["phase1"], exist_ok=True)
            plt.savefig(f"{xs.CONFIG['io']['phase1']}Annexe_Fig2_{titles[i]}.png")
            plt.close()

    # Blend of years
    if "annexe_figure3" in todo:
        features = {'land': {"color": "#f0f0f0"}, 'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                    'states': {"edgecolor": "black", "linestyle": "dotted"}}

        shp = deepcopy(shp_zg)
        shp["blend_yr"] = pd.Series()
        for y in [1995, 2010, 2018, 2021]:
            shp.loc[xs.CONFIG["analogs"]["targets"][y]["region"], "blend_yr"] = y

        plt.figure(figsize=(15, 15))
        ax = plt.subplot(1, 1, 1, projection=proj)
        sp.gdfmap(df=shp, df_col="blend_yr", ax=ax, cmap="viridis", cbar=False, frame=True, features=features)

        shp_zg.plot(ax=ax, facecolor="None", edgecolor="k", linewidth=0.5)
        ax.set_extent([lon_bnds[0], lon_bnds[1], lat_bnds[0], lat_bnds[1]], crs=cartopy.crs.PlateCarree())

        plt.text(0.83, 0.85, '1995', transform=ax.transAxes, fontsize=20, color="w", backgroundcolor="black", horizontalalignment='center', verticalalignment="center")
        plt.text(0.13, 0.5, '2010', transform=ax.transAxes, fontsize=20, color="w", backgroundcolor="black", horizontalalignment='center', verticalalignment="center")
        plt.text(0.67, 0.31, '2018', transform=ax.transAxes, fontsize=20, color="w", backgroundcolor="black", horizontalalignment='center', verticalalignment="center")
        plt.text(0.45, 0.07, '2021', transform=ax.transAxes, fontsize=20, color="w", backgroundcolor="black", horizontalalignment='center', verticalalignment="center")

        os.makedirs(xs.CONFIG["io"]["phase1"], exist_ok=True)
        plt.savefig(f"{xs.CONFIG['io']['phase1']}Annexe_Fig3.png")
        plt.close()

    # Nice indicator figures for the report itself
    if "nice_figures" in todo:
        features = {'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                    'states': {"edgecolor": "black", "linestyle": "dotted"}, 'ocean': {}, 'land': {"color": "#f0f0f0"}}
        allv = ["season_start", "season_end", "season_length", "days_under_7q2", "days_under_7q10", "7qmin", "discharge_mean_jja", "discharge_mean_son", "doy_14qmax"]

        ref = pcat.search(processing_level="indicators", variable=allv).to_dask()
        ref["season_length"] = ref["season_end"] - ref["season_start"]
        ref = ref.where(~ref.station_id.isin(list(stations_atlas["TRONCON_ID"])))
        ref_mean = ref.mean(dim="time")
        for v in ["days_under_7q2", "days_under_7q10"]:
            ref_mean[v] = ref[v].where(ref[v] > 0).mean(dim="time")

        ref_blend = pcat.search(processing_level="temporalblend-indicators", variable=allv).to_dask()
        ref_blend = ref_blend.where(~ref_blend.station_id.isin(list(stations_atlas["TRONCON_ID"])))

        data = [ref_mean, ref_blend]
        titles = ["ref_mean", "ref_blend"]
        for wl in ["1.5", "2", "3", "4"]:
            analog_blend = pcat.search(processing_level=f"temporalblend-analog-2021-{wl}", variable=allv).to_dask()
            analog_blend = analog_blend.where(~analog_blend.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            data.extend([analog_blend])
            titles.extend([f"analog_blend_{wl}"])

        # Specific discharge
        for v in ["7qmin", "discharge_mean_jja", "discharge_mean_son"]:
            for i in range(len(data)):
                attrs = data[i][v].attrs
                data[i][v] = (data[i][v] / data[i]["drainage_area"]) * 1000
                data[i][v].attrs = attrs
                data[i][v].attrs["units"] = "L s-1 km-2"

        figure_types = ["raw", "dabs", "dpct"]

        for ftype in figure_types:
            data2 = deepcopy(data)
            if ftype == "raw":
                new_data = data2
                new_titles = deepcopy(titles)
            elif ftype == "dabs":
                new_data = []
                for d in data2[2:]:
                    new_data.extend([xs.compute_deltas(d, data2[1], kind="+", rename_variables=False)])
                new_titles = deepcopy(titles[2:])
            elif ftype == "dpct":
                new_data = []
                for d in data2[2:]:
                    new_data.extend([xs.compute_deltas(d, data2[1], kind="%", rename_variables=False)])
                new_titles = deepcopy(titles[2:])
            else:
                raise ValueError

            for v in allv:
                if ftype == "raw":
                    vmin = np.min([np.min(d[v]).values for d in new_data]).clip(min=0)
                    vmax = np.max([np.max(d[v]).values for d in new_data])
                    steps = np.round(((vmax - vmin) / 9) / 5) * 5 if vmax > 20 else np.round(((vmax - vmin) / 9) / 2) * 2
                    levels = list(np.arange(vmin, vmax + steps, steps))
                else:
                    vmin = np.min([np.min(d[v]).values for d in new_data])
                    vmax = np.max([np.max(d[v]).values for d in new_data])
                    steps = np.round(((vmax - vmin) / 15) / 5) * 5
                    levels = list(np.arange(vmin, vmax + steps, steps))

                for i in range(len(new_data)):
                    da = new_data[i][v]

                    plt.figure(figsize=(15, 15))
                    ax = plt.subplot(1, 1, 1, projection=proj)

                    shp = deepcopy(shp_portrait_bv)
                    da2 = da.to_dataframe().set_index("station_id")
                    j = da2.index.intersection(shp.index)
                    shp[da.name] = da2.loc[j][da.name]
                    #
                    # sp.gdfmap(df=shp, df_col=v, ax=ax, cmap="temp_seq", cbar=True, levels=levels,
                    #           frame=True, features=features)

                    # dazg = da.assign_coords({"SIGLE": da.station_id.to_series().map(regions)}).squeeze()
                    # dazg = dazg.groupby(dazg.SIGLE).mean(dim="station")
                    # shp = deepcopy(shp_zg)
                    # da2 = dazg.to_dataframe()
                    # j = da2.index.intersection(shp.index)
                    # shp[dazg.name] = da2.loc[j][dazg.name]

                    if ftype == "raw":
                        cmap = "temp_seq" if v in ["season_start", "season_end", "season_length", "days_under_7q2", "days_under_7q10"] else "prec_seq" if v in ["7qmin", "discharge_mean_jja", "discharge_mean_son"] else "nipy_spectral"
                        sp.gdfmap(df=shp, df_col=v, ax=ax, cmap=cmap, cbar=True, levels=levels, frame=False, features=features)
                    else:
                        if ftype == "dabs":
                            sp.gdfmap(df=shp, df_col=v, ax=ax, cmap="temp_div_r", cbar=True, levels=levels, divergent=0, frame=False, features=features)
                        else:
                            sp.gdfmap(df=shp, df_col=v, ax=ax, cmap="temp_div_r", cbar=True, levels=15, divergent=0, frame=False, features=features, plot_kw={"vmin": vmin, "vmax": vmax})

                    shp_zg.plot(ax=ax, facecolor="None", edgecolor="k", linewidth=1)
                    ax.set_extent([lon_bnds[0], lon_bnds[1], lat_bnds[0], lat_bnds[1]], crs=cartopy.crs.PlateCarree())

                    os.makedirs(xs.CONFIG["io"]["phase1"], exist_ok=True)
                    plt.savefig(f"{xs.CONFIG['io']['phase1']}Indicators_{v}-{new_titles[i]}-{ftype}.png")
                    plt.close()

    # Ratios from Nada Conseils
    if "nada" in todo:
        features = {'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"},
                    'states': {"edgecolor": "black", "linestyle": "dotted"}, 'ocean': {}, 'land': {"color": "#f0f0f0"}}
        allv = ["q7min_Hist", "q7min_2deg", "q7min_3deg"]

        data = gpd.read_file(f"{xs.CONFIG['nada']}polygones_conso_radeau/polygones_conso_radeau.shp")

        for v in allv:
            fig = plt.figure(figsize=(15, 15))
            ax = plt.subplot(1, 1, 1, projection=proj)

            sp.gdfmap(df=data, df_col=v, ax=ax, cmap="prec_seq", cbar=True, levels=15, divergent=50, frame=True, features=features, plot_kw={"vmin": 0, "vmax": 100})
            cb_ax = fig.axes[1]
            cb_ax.tick_params(labelsize=20)

            os.makedirs(xs.CONFIG["io"]["phase1"], exist_ok=True)
            plt.savefig(f"{xs.CONFIG['io']['phase1']}nada_{v}.png")
            plt.close()


if __name__ == '__main__':
    figures = ["nada"]

    main(todo=figures)
