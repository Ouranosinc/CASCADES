import numpy as np
import pandas as pd
import xarray as xr
import xclim
import xclim.indices
from xclim.core.calendar import convert_calendar
import xscen as xs
import xskillscore as xss
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy
import matplotlib
import os
from distributed import Client
from xclim.sdba.utils import ecdf

import figures
from utils import get_target_region, get_stations_within_target_region, sort_analogs, atlas_radeau_common, fix_infocrue

xs.load_config("configs/cfg_analogs.yml", "paths.yml")


def main():
    # matplotlib.use("QtAgg")

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"])

    if xs.CONFIG["tasks"]["compute"]:
        for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
            for target_year in xs.CONFIG["analogs"]["targets"]:
                for v in xs.CONFIG["analogs"]["compute_criteria"]:
                    if not pcat.exists_in_cat(activity="ClimEx", processing_level=f"{warming_level}-performance-vs-{target_year}-{v}"):
                        # Open ClimEx
                        hist_dict = pcat.search(source="CRCM5.*", processing_level=f"indicators-warminglevel.*{warming_level}vs.*").to_dataset_dict()
                        hist = xclim.ensembles.create_ensemble(hist_dict)
                        hist = xs.utils.clean_up(hist, common_attrs_only=hist_dict)

                        # Open ERA5-Land
                        ref = pcat.search(source="ERA5.*", variable=["spei3", "spei6", "spei9"]).to_dask()
                        # ClimEx was bias adjusted on a smaller domain. Use the same region.
                        ref = ref.where(~np.isnan(hist["spei3"].isel(realization=0)).all(dim="time"))

                        perf = compute_criteria(ref, hist, target_year=target_year, **xs.CONFIG["analogs"]["compute_criteria"][v])
                        perf.attrs["cat:processing_level"] = f"{warming_level}-{perf.attrs['cat:processing_level']}-{v}"

                        if "time" not in perf.dims:
                            perf = perf.drop_vars(["time"])
                            perf.attrs["cat:xrfreq"] = "fx"
                            perf.attrs["cat:frequency"] = "fx"

                        filename = f"{xs.CONFIG['io']['stats']}{perf.attrs['cat:id']}_{perf.attrs['cat:processing_level']}.zarr"
                        xs.save_to_zarr(perf, filename=filename)
                        pcat.update_from_ds(perf, path=filename)

    if xs.CONFIG["tasks"]["figure_spei"]:
        dcat = xs.DataCatalog(xs.CONFIG["project"]["path"])
        levels = np.arange(-3, 3.5, 0.5)
        cmap = figures.utils.make_cmap("BrWhGr", 25)
        proj = cartopy.crs.PlateCarree()

        for v in xs.CONFIG["analogs"]["compute_criteria"]:
            criteria = xs.CONFIG["analogs"]["compute_criteria"]["v1"]["criteria"]

            for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
                for target_year in xs.CONFIG["analogs"]["targets"]:
                    perf = dcat.search(processing_level=f"{warming_level}-performance-vs-{target_year}-{v}").to_dask()
                    analogs = sort_analogs(perf.rmse)

                    # Open the reference
                    ref = dcat.search(source="ERA5-Land", processing_level=f"indicators.*", variable=tuple(np.unique([c[0] for c in criteria]))).to_dask()

                    # Highlight the region used for the criteria
                    region_perf = xs.extract.clisops_subset(xr.ones_like(ref.spei3.isel(time=0)), {"method": "shape",
                                                                                                   "shape": {"shape": get_target_region(target_year),
                                                                                                             "buffer": 0.1}})
                    lon_bnds = [-80., -64.]
                    lat_bnds = [44.75, 54.]
                    # lon_bnds = [region_perf.lon.min() - 1, region_perf.lon.max() + 1]
                    # lat_bnds = [region_perf.lat.min() - 1, region_perf.lat.max() + 1]
                    region_perf = region_perf.interp_like(ref.spei3.isel(time=0)).fillna(0)

                    blend = {f"{k[0]}-{k[1]}": [] for k in criteria}
                    for j in [0, 5]:
                        # Plot
                        plt.subplots(6, len(criteria), figsize=(35, 15))
                        plt.suptitle(f"Analogues de l'année {target_year} - +{warming_level}°C vs pré-industriel - {v}")

                        ii = 1
                        for c in criteria:
                            ax = plt.subplot(6, len(criteria), ii, projection=proj)
                            figures.templates.cartopy_map(ax, ref[c[0]].sel(time=f"{target_year}-{c[1]:02d}-01"), highlight=region_perf, hide_labels=True,
                                                          lon_bnds=lon_bnds, lat_bnds=lat_bnds, levels=levels, cmap=cmap, add_colorbar=False)

                            plt.title(f"{c[0].upper()}-{c[1]}")
                            if ii == 1:
                                ax.set_yticks([])
                                ax.set_ylabel("ERA5-Land")

                            ii = ii + 1

                        for i in range(j, j+5, 1):
                            ds = dcat.search(activity="ClimEx", processing_level=f"indicators.*{warming_level}vs", member=str(analogs.isel(stacked=i).realization.values).split(".")[0].split("_")[-1],
                                             variable=tuple(np.unique([c[0] for c in criteria]))).to_dask()

                            for c in criteria:
                                ax = plt.subplot(6, len(criteria), ii, projection=proj)
                                if "time" in perf.dims:
                                    da = ds[c[0]].sel(time=f"{str(analogs.isel(stacked=i).time.dt.year.values)}-{c[1]:02d}-01").squeeze()
                                else:
                                    da = ds[c[0]].sel(time=f"{str(analogs.isel(stacked=i).year.values)}-{c[1]:02d}-01").squeeze()
                                blend[f"{c[0]}-{c[1]}"].extend([da])
                                figures.templates.cartopy_map(ax, da,
                                                              highlight=region_perf, hide_labels=True, lon_bnds=lon_bnds, lat_bnds=lat_bnds,
                                                              levels=levels, cmap=cmap, add_colorbar=False)

                                plt.title("")
                                if c == criteria[0]:
                                    ax.set_yticks([])
                                    if "time" in perf.dims:
                                        ax.set_ylabel(f"{str(analogs.isel(stacked=i).realization.values).split('.')[0].split('_')[-1]} | "
                                                      f"{str(analogs.isel(stacked=i).time.dt.year.values)}\nsum(RMSE) = {np.round(analogs.isel(stacked=i).values, 2)}")
                                    else:
                                        ax.set_ylabel(f"{str(analogs.isel(stacked=i).realization.values).split('.')[0].split('_')[-1]} | "
                                                      f"{str(analogs.isel(stacked=i).year.values)}\nsum(RMSE) = {np.round(analogs.isel(stacked=i).values, 2)}")

                                ii = ii + 1

                        plt.tight_layout()
                        plt.subplots_adjust(right=0.9)
                        cax = plt.axes([0.925, 0.1, 0.025, 0.8])

                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=len(levels)*2+1))
                        sm._A = []
                        plt.colorbar(sm, cax=cax, extend="both")

                        plt.tight_layout()
                        plt.subplots_adjust(right=0.9)

                        os.makedirs(xs.CONFIG['io']['figures'], exist_ok=True)
                        plt.savefig(f"{xs.CONFIG['io']['figures']}SPEI-analogs-{target_year}_{warming_level}degC-{v}-{j}.png")
                        plt.close()

                    # BLEND
                    # Plot
                    plt.subplots(4, len(criteria), figsize=(35, 15))
                    plt.suptitle(f"Analogues de l'année {target_year} - +{warming_level}°C vs pré-industriel - {v}")

                    ii = 1
                    for c in criteria:
                        ax = plt.subplot(4, len(criteria), ii, projection=proj)
                        figures.templates.cartopy_map(ax, ref[c[0]].sel(time=f"{target_year}-{c[1]:02d}-01"), highlight=region_perf,
                                                      hide_labels=True,
                                                      lon_bnds=lon_bnds, lat_bnds=lat_bnds, levels=levels, cmap=cmap, add_colorbar=False)

                        plt.title(f"{c[0].upper()}-{c[1]}")
                        if ii == 1:
                            ax.set_yticks([])
                            ax.set_ylabel("ERA5-Land")

                        ii = ii + 1

                    for j in [1, 5, 10]:
                        for c in list(blend.keys()):
                            ax = plt.subplot(4, len(criteria), ii, projection=proj)
                            da = xr.concat(blend[c][0:j], dim="realization").mean(dim="realization", keep_attrs=True)
                            figures.templates.cartopy_map(ax, da,
                                                          highlight=region_perf, hide_labels=True, lon_bnds=lon_bnds, lat_bnds=lat_bnds,
                                                          levels=levels, cmap=cmap, add_colorbar=False)

                            plt.title("")
                            if c == list(blend.keys())[0]:
                                ax.set_yticks([])
                                ax.set_ylabel(f"Blend of the best {j} analogs")
                            ii = ii + 1

                    plt.tight_layout()
                    plt.subplots_adjust(right=0.9)
                    cax = plt.axes([0.925, 0.1, 0.025, 0.8])

                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=len(levels) * 2 + 1))
                    sm._A = []
                    plt.colorbar(sm, cax=cax, extend="both")

                    plt.tight_layout()
                    plt.subplots_adjust(right=0.9)

                    os.makedirs(xs.CONFIG['io']['figures'], exist_ok=True)
                    plt.savefig(f"{xs.CONFIG['io']['figures']}SPEI-analogs-{target_year}_{warming_level}degC-{v}-blend.png")
                    plt.close()

    if xs.CONFIG["tasks"]["figure_hydro"]:
        proj = cartopy.crs.PlateCarree()

        # Portrait shapefiles
        shp = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
        shp = shp.set_index("TRONCON")

        # Open the ZGIEBV
        shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")

        for v in xs.CONFIG["analogs"]["compute_criteria"]:
            for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
                for target_year in xs.CONFIG["analogs"]["targets"]:
                    perf = pcat.search(processing_level=f"{warming_level}-performance-vs-{target_year}-{v}").to_dask()
                    analogs = sort_analogs(perf.rmse)

                    # Highlight the region used for the criteria
                    ref_clim = pcat.search(source="ERA5-Land", processing_level=f"indicators.*").to_dask()
                    region_perf = xs.extract.clisops_subset(xr.ones_like(ref_clim.spei3.isel(time=0)), {"method": "shape",
                                                                                                        "shape": {"shape": get_target_region(target_year),
                                                                                                                  "buffer": 0.1}})
                    lon_bnds = [region_perf.lon.min(), region_perf.lon.max()]
                    lat_bnds = [region_perf.lat.min(), region_perf.lat.max()]
                    # lon_bnds = [-80., -58.]
                    # lat_bnds = [44.75, 54.]

                    ref = {k: [] for k in xs.CONFIG["analogs"]["hydro"]}
                    blend = {k: [] for k in xs.CONFIG["analogs"]["hydro"]}
                    for j in [0]:
                        # Open the reference
                        stats_ref = pcat.search(type="reconstruction-hydro", processing_level=f"indicators", xrfreq="AS-JAN").to_dask()
                        stats_ref["season_length"] = stats_ref["season_end"] - stats_ref["season_start"]
                        [stats_ref[c].load() for c in stats_ref.coords]

                        within_region = get_stations_within_target_region(stats_ref, get_target_region(target_year))

                        # Plot
                        plt.subplots(6, len(stats_ref.data_vars), figsize=(35, 15))
                        plt.suptitle(f"Analogues de l'année {target_year} - +{warming_level}°C vs pré-industriel - {v}")

                        ii = 1
                        for vv in xs.CONFIG["analogs"]["hydro"]:
                            # Empirical CDF
                            da = stats_ref[vv].sel(time=slice("1992-01-01", "2021-01-01"))
                            if vv in ['days_under_7q2', 'max_consecutive_days_under_7q2']:
                                da = da.where(da > 0)
                                da.name = "days_under"
                            da = ecdf(da, da.sel(time=slice(f"{target_year}-01-01", f"{target_year}-01-01")).squeeze())
                            da = da.where(da > 0)
                            ref[vv] = da

                            # with xr.set_options(keep_attrs=True):
                            #     if "days_under_7q2" in vv:
                            #         data = stats_ref[vv].sel(time=slice(str(target_year), str(target_year))).dt.days
                            #         data.attrs["units"] = "days"
                            #     elif vv == "7qmin":
                            #         data = ((stats_ref[vv].sel(time=slice(str(target_year), str(target_year))) - stats_ref_fx["7q2"]) / stats_ref_fx["7q2"] * 100).squeeze()
                            #         data.name = vv
                            #     elif stats_ref[vv].attrs["units"] == "dayofyear":
                            #         data = stats_ref[vv].sel(time=slice(str(target_year), str(target_year))) - stats_ref[vv].mean(dim="time")
                            #     else:
                            #         data = (stats_ref[vv].sel(time=slice(str(target_year), str(target_year))) - stats_ref[vv].mean(dim="time")) / stats_ref[vv].mean(dim="time") * 100

                            # bounds = np.linspace(**xs.CONFIG["figures"][vv]["bnds"])
                            # norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                            # cmap = xs.CONFIG["figures"][vv]["cmap"]

                            bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
                            norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                            cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2', 'max_consecutive_days_under_7q2'] else 'RdBu_r'

                            ax = plt.subplot(6, len(stats_ref.data_vars), ii, projection=proj)
                            figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, linewidth=0.25, norm=norm, cmap=cmap)
                            figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                            figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                            shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")

                            # figures.templates.map_hydro(ax, data, shp=shp,
                            #                             lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25,
                            #                             linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                            # shp_zg.plot(ax=ax, facecolor="None", edgecolor="k")

                            plt.title(f"{vv}")
                            if ii == 1:
                                ax.set_yticks([])
                                ax.set_ylabel("Portrait")

                            ii = ii + 1

                        for i in range(j, j+5, 1):
                            ds = pcat.search(activity="ClimEx", processing_level=f"indicators-{warming_level}",
                                             member=str(analogs.isel(stacked=i).realization.values).split(".")[0].split("_")[-1], xrfreq="AS-JAN").to_dask()
                            ds["season_length"] = ds["season_end"] - ds["season_start"]
                            [ds[c].load() for c in ds.coords]
                            analog_year = int(analogs.isel(stacked=i).time.dt.year.values)

                            rmse = {"weighted": [], "unweighted": []}

                            for vv in xs.CONFIG["analogs"]["hydro"]:
                                # Empirical CDF
                                da = ds[vv]
                                if vv in ['days_under_7q2', 'max_consecutive_days_under_7q2']:
                                    da = da.where(da > 0)
                                    da.name = "days_under"
                                if vv in ['doy_14qmax', 'season_start', 'season_end', 'season_length']:
                                    da = ecdf(stats_ref[vv].sel(time=slice("1992-01-01", "2021-01-01")), da.sel(time=slice(f"{analog_year}-01-01", f"{analog_year}-01-01")).squeeze())
                                else:
                                    da = ecdf(da, da.sel(time=slice(f"{analog_year}-01-01", f"{analog_year}-01-01")).squeeze())
                                da = da.where(da > 0)

                                blend[vv].extend([da])

                                # bounds = np.linspace(**xs.CONFIG["figures"][vv]["bnds"])
                                # norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                                # cmap = xs.CONFIG["figures"][vv]["cmap"]
                                #
                                # ax = plt.subplot(6, len(stats_ref.data_vars), ii, projection=proj)
                                # figures.templates.map_hydro(ax, ds[vv], shp=shp,
                                #                             lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25,
                                #                             linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                                # shp_zg.plot(ax=ax, facecolor="None", edgecolor="k")

                                bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
                                norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                                cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2',
                                                            'max_consecutive_days_under_7q2'] else 'RdBu_r'

                                ax = plt.subplot(6, len(stats_ref.data_vars), ii, projection=proj)
                                figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, linewidth=0.25, norm=norm, cmap=cmap)
                                figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                                figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                                shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")

                                rmse["weighted"].extend([np.round(xss.rmse(da.compute().where(da.station_id.isin(within_region)),
                                                                           ref[vv].compute().where(ref[vv].station_id.isin(within_region)), dim=["station"], weights=xr.where(da.drainage_area>1000, 1, xr.where(da.drainage_area>150, 0.5, 0.33)), skipna=True).values, 3)])
                                rmse["unweighted"].extend([np.round(xss.rmse(da.compute().where(da.station_id.isin(within_region)),
                                                                             ref[vv].compute().where(ref[vv].station_id.isin(within_region)), dim=["station"], skipna=True).values, 3)])
                                plt.title(f"{rmse['unweighted'][-1]} | {rmse['weighted'][-1]}")
                                if vv == list(xs.CONFIG["analogs"]["hydro"])[0]:
                                    ax.set_yticks([])
                                    ax.set_ylabel(f"{str(analogs.isel(stacked=i).realization.values).split('.')[0].split('_')[-1]} | "
                                                  f"{str(analogs.isel(stacked=i).time.dt.year.values)}")
                                if vv == list(xs.CONFIG["analogs"]["hydro"])[-1]:
                                    ax.yaxis.set_label_position("right")
                                    ax.set_yticks([])
                                    ax.set_ylabel(f"{np.round(np.mean(rmse['unweighted']), 3)} | {np.round(np.mean(rmse['weighted']), 3)}")

                                ii = ii + 1

                        plt.tight_layout()

                        os.makedirs(xs.CONFIG['io']['figures'], exist_ok=True)
                        plt.savefig(f"{xs.CONFIG['io']['figures']}hydro-analogs-{target_year}_{warming_level}degC-{v}-{j}.png")
                        plt.close()

                    # BLEND
                    plt.subplots(3, len(stats_ref.data_vars), figsize=(35, 15))
                    plt.suptitle(f"Analogues de l'année {target_year} - +{warming_level}°C vs pré-industriel - {v}")

                    ii = 1
                    for vv in xs.CONFIG["analogs"]["hydro"]:
                        da = ref[vv]

                        bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
                        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                        cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2',
                                                    'max_consecutive_days_under_7q2'] else 'RdBu_r'

                        ax = plt.subplot(3, len(stats_ref.data_vars), ii, projection=proj)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, linewidth=0.25, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                        shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")

                        plt.title(f"{vv}")
                        if ii == 1:
                            ax.set_yticks([])
                            ax.set_ylabel("Portrait")

                        ii = ii + 1

                    for j in [1, 5]:
                        rmse = {"weighted": [], "unweighted": []}

                        for vv in xs.CONFIG["analogs"]["hydro"]:
                            da = xr.concat(blend[vv][0:j], dim="realization").mean(dim="realization", keep_attrs=True)

                            bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
                            norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                            cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2',
                                                        'max_consecutive_days_under_7q2'] else 'RdBu_r'

                            ax = plt.subplot(3, len(stats_ref.data_vars), ii, projection=proj)
                            figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, linewidth=0.25, norm=norm, cmap=cmap)
                            figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                            figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                            shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")

                            rmse["weighted"].extend([np.round(xss.rmse(da.compute().where(da.station_id.isin(within_region)), ref[vv].compute().where(ref[vv].station_id.isin(within_region)),
                                                                       dim=["station"], weights=xr.where(da.drainage_area>1000, 1, xr.where(da.drainage_area>150, 0.5, 0.33)), skipna=True).values, 3)])
                            rmse["unweighted"].extend([np.round(xss.rmse(da.compute().where(da.station_id.isin(within_region)), ref[vv].compute().where(ref[vv].station_id.isin(within_region)),
                                                                         dim=["station"], skipna=True).values, 3)])
                            plt.title(f"{rmse['unweighted'][-1]} | {rmse['weighted'][-1]}")
                            if vv == list(xs.CONFIG["analogs"]["hydro"])[0]:
                                ax.set_yticks([])
                                ax.set_ylabel(f"Blend of the best {j} analogs")
                            if vv == list(xs.CONFIG["analogs"]["hydro"])[-1]:
                                ax.yaxis.set_label_position("right")
                                ax.set_yticks([])
                                ax.set_ylabel(f"{np.round(np.mean(rmse['unweighted']), 3)} | {np.round(np.mean(rmse['weighted']), 3)}")

                            ii = ii + 1

                    plt.tight_layout()

                    os.makedirs(xs.CONFIG['io']['figures'], exist_ok=True)
                    plt.savefig(f"{xs.CONFIG['io']['figures']}hydro-analogs-{target_year}_{warming_level}degC-{v}-blend.png")
                    plt.close()


def compute_criteria(ref, hist,
                     target_year: int,
                     criteria: list,
                     *,
                     weights_close_far: list = None,
                     weights_2_1: list = None,
                     normalise: bool = True,
                     to_level: str = None):

    # Prepare weights
    target_region = get_target_region(target_year)
    full_domain = ref[list(ref.data_vars)[0]].sel(time=f"{target_year}-06-01").chunk({"lon": -1})
    region_close = xs.extract.clisops_subset(xr.ones_like(full_domain), {"method": "shape", "shape": {"shape": target_region, "buffer": 0.1}}).interp_like(full_domain)
    region_far = xs.extract.clisops_subset(xr.ones_like(full_domain), {"method": "shape", "shape": {"shape": target_region, "buffer": 0.5}}).interp_like(full_domain)

    # Loop on each criterion
    rsme_sum = []
    for c in criteria:
        if isinstance(c, list):
            target = ref[c[0]].sel(time=f"{target_year}-{c[1]:02d}-01").chunk({"lon": -1})
            candidates = hist[c[0]].where(hist.time.dt.month == c[1], drop=True).chunk({"lon": -1})
            candidates["time"] = pd.to_datetime(candidates["time"].dt.strftime("%Y-01-01"))
        else:
            target = ref[c].sel(time=slice(str(target_year), str(target_year))).mean(dim="time").chunk({"lon": -1})
            candidates = hist[c].groupby("time.year").mean(dim="time").chunk({"lon": -1})

        weights = xr.where(region_close == 1, weights_close_far[0], xr.where(region_far == 1, weights_close_far[1], weights_close_far[2])) * \
                  xr.where(target <= -2, weights_2_1[0], xr.where(target <= -1, weights_2_1[1], weights_2_1[2]))

        rmse = xss.rmse(candidates, target, dim=["lon", "lat"], weights=weights, skipna=True)
        # Normalise
        if normalise:
            rmse = (rmse - rmse.min()) / (rmse.max() - rmse.min())

        rsme_sum.extend([rmse])

    r = sum(rsme_sum)
    r.name = "rmse"
    r.attrs = {
        "long_name": "RMSE",
        "description": f"Sum of RMSEs for {criteria}",
        "units": "",
        "weights_close_far": str(weights_close_far),
        "weights_2_1": str(weights_2_1),
        "normalise": str(normalise)
    }

    out = r.to_dataset()
    out.attrs = hist.attrs
    out.attrs["cat:processing_level"] = f"performance-vs-{target_year}" if not to_level else to_level

    return out


if __name__ == '__main__':
    main()
