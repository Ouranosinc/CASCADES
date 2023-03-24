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

import figures
from utils import get_target_region, sort_analogs, atlas_radeau_common, fix_infocrue

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

                        filename = f"{xs.CONFIG['io']['stats']}{perf.attrs['cat:id']}_{perf.attrs['cat:processing_level']}.zarr"
                        xs.save_to_zarr(perf, filename=filename)
                        pcat.update_from_ds(perf, path=filename)

    if xs.CONFIG["tasks"]["hydro_stats"]:
        stations = atlas_radeau_common()["TRONCON_ID"]

        hcat = xs.DataCatalog(xs.CONFIG["dpphc"]["atlas2022"])
        for target_year in xs.CONFIG["analogs"]["targets"]:
            if not pcat.exists_in_cat(type="reconstruction-hydro", processing_level=f"{target_year}-stats"):
                # Open the reference
                ref = hcat.search(type="reconstruction-hydro", processing_level="raw")
                if not os.path.isdir(f"{xs.CONFIG['tmp_rechunk']}{ref.unique('id')[0]}.zarr"):
                    fix_infocrue(ref)

                ref = xr.open_zarr(f"{xs.CONFIG['tmp_rechunk']}{ref.unique('id')[0]}.zarr")
                [ref[c].load() for c in ref.coords]  # load coordinates
                ref = ref.sel(time=slice(xs.CONFIG["storylines"]["ref_period"][0], xs.CONFIG["storylines"]["ref_period"][1]), percentile=50)

                ref = ref.where(ref.station_id.isin(stations), drop=True)
                ref = ref.chunk({"station": 500})
                with Client(**xs.CONFIG["dask"]) as c:
                    ref_stats = streamflow_stats(ref.discharge, target_year, ds=ref)
                    [ref_stats[c].load() for c in ref_stats.coords]  # load coordinates

                    filename = f"{xs.CONFIG['io']['stats']}{ref_stats.attrs['cat:id']}_{ref_stats.attrs['cat:processing_level']}.zarr"
                    xs.save_to_zarr(ref_stats, filename)
                    pcat.update_from_ds(ref_stats, path=filename)

            for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
                for v in xs.CONFIG["analogs"]["compute_criteria"]:
                    perf = pcat.search(processing_level=f"{warming_level}-performance-vs-{target_year}-{v}").to_dask()
                    analogs = sort_analogs(perf.rmse)

                    for a in analogs[0:10]:
                        # Open the simulation
                        sim = hcat.search(type="simulation-hydro", processing_level="raw", member=str(a.realization.values).split(".")[0].split("_")[-1])
                        if not pcat.exists_in_cat(type="simulation-hydro", processing_level=f"{target_year}-stats-{warming_level}", member=str(a.realization.values).split(".")[0].split("_")[-1]):
                            if not os.path.isdir(f"{xs.CONFIG['tmp_rechunk']}{sim.unique('id')[0]}.zarr"):
                                fix_infocrue(sim)

                            sim = xr.open_zarr(f"{xs.CONFIG['tmp_rechunk']}{sim.unique('id')[0]}.zarr")
                            [sim[c].load() for c in sim.coords]  # load coordinates
                            sim.attrs["cat:driving_model"] = "CanESM2"
                            sim = xs.subset_warming_level(sim, wl=warming_level).squeeze()
                            sim.attrs["cat:driving_model"] = "CCCma-CanESM2"

                            sim = sim.where(sim.station_id.isin(stations), drop=True)
                            sim = sim.chunk({"station": 500})
                            with Client(**xs.CONFIG["dask"]) as c:
                                sim_stats = streamflow_stats(sim.discharge, target_year=int(a["time"].dt.year.values), ds=sim, to_level=f"{target_year}-stats-{warming_level}")
                                [sim_stats[c].load() for c in sim_stats.coords]  # load coordinates

                                filename = f"{xs.CONFIG['io']['stats']}{sim_stats.attrs['cat:id']}_{sim_stats.attrs['cat:processing_level']}.zarr"
                                xs.save_to_zarr(sim_stats, filename)
                                pcat.update_from_ds(sim_stats, path=filename)

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
                    lon_bnds = [region_perf.lon.min() - 1, region_perf.lon.max() + 1]
                    lat_bnds = [region_perf.lat.min() - 1, region_perf.lat.max() + 1]
                    region_perf = region_perf.interp_like(ref.spei3.isel(time=0)).fillna(0)

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
                                figures.templates.cartopy_map(ax, ds[c[0]].sel(time=f"{str(analogs.isel(stacked=i).time.dt.year.values)}-{c[1]:02d}-01").squeeze(),
                                                              highlight=region_perf, hide_labels=True, lon_bnds=lon_bnds, lat_bnds=lat_bnds,
                                                              levels=levels, cmap=cmap, add_colorbar=False)

                                plt.title("")
                                if c == criteria[0]:
                                    ax.set_yticks([])
                                    ax.set_ylabel(f"{str(analogs.isel(stacked=i).realization.values).split('.')[0].split('_')[-1]} | "
                                               f"{str(analogs.isel(stacked=i).time.dt.year.values)}\nsum(RMSE) = {np.round(analogs.isel(stacked=i).values, 2)}")

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

    if xs.CONFIG["tasks"]["figure_hydro"]:
        proj = cartopy.crs.PlateCarree()

        # Open the RADEAU shapefiles
        stations = atlas_radeau_common()
        cv = dict(zip(stations["ATLAS2018"], stations["TRONCON_ID"]))
        shp = gpd.read_file(f"{xs.CONFIG['gis']}RADEAU/CONSOM_SURF_BV_CF1_WGS84.shp")
        shp["BV_ID"] = shp["BV_ID"].str[0:3].str.cat(shp["BV_ID"].str[4:])
        shp["BV_ID"] = shp["BV_ID"].map(cv)
        shp = shp.dropna(subset=["BV_ID"])
        shp = shp.set_index("BV_ID")
        shp = shp.sort_values("Shape_Area", ascending=False)

        # Open the ZGIEBV
        shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")

        for v in xs.CONFIG["analogs"]["compute_criteria"]:
            for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
                for target_year in xs.CONFIG["analogs"]["targets"]:
                    # Highlight the region used for the criteria
                    ref_clim = pcat.search(source="ERA5-Land", processing_level=f"indicators.*").to_dask()
                    region_perf = xs.extract.clisops_subset(xr.ones_like(ref_clim.spei3.isel(time=0)), {"method": "shape",
                                                                                                        "shape": {"shape": get_target_region(target_year),
                                                                                                                  "buffer": 0.1}})
                    lon_bnds = [region_perf.lon.min(), region_perf.lon.max()]
                    lat_bnds = [region_perf.lat.min(), region_perf.lat.max()]

                    blend = {k: [] for k in xs.CONFIG["figures"]}
                    for j in [0, 5]:
                        # Open the reference
                        stats_ref = pcat.search(type="reconstruction-hydro", processing_level=f"indicators", xrfreq="AS-DEC").to_dask()
                        stats_ref_fx = pcat.search(type="reconstruction-hydro", processing_level=f"indicators", xrfreq="fx").to_dask()
                        [stats_ref[c].load() for c in stats_ref.coords]

                        # Plot
                        plt.subplots(6, len(xs.CONFIG["figures"]), figsize=(35, 15))
                        plt.suptitle(f"Analogues de l'année {target_year} - +{warming_level}°C vs pré-industriel - {v}")

                        ii = 1
                        for vv in xs.CONFIG["figures"]:
                            with xr.set_options(keep_attrs=True):
                                if "days_under_7q2" in vv:
                                    data = stats_ref[vv].sel(time=slice(str(target_year), str(target_year))).dt.days
                                    data.attrs["units"] = "days"
                                elif vv == "7qmin":
                                    data = ((stats_ref[vv].sel(time=slice(str(target_year), str(target_year))) - stats_ref_fx["7q2"]) / stats_ref_fx["7q2"] * 100).squeeze()
                                    data.name = vv
                                elif stats_ref[vv].attrs["units"] == "dayofyear":
                                    data = stats_ref[vv].sel(time=slice(str(target_year), str(target_year))) - stats_ref[vv].mean(dim="time")
                                else:
                                    data = (stats_ref[vv].sel(time=slice(str(target_year), str(target_year))) - stats_ref[vv].mean(dim="time")) / stats_ref[vv].mean(dim="time") * 100

                            bounds = np.linspace(**xs.CONFIG["figures"][vv]["bnds"])
                            norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                            cmap = xs.CONFIG["figures"][vv]["cmap"]

                            ax = plt.subplot(6, len(xs.CONFIG["figures"]), ii, projection=proj)
                            figures.templates.map_hydro(ax, data, shp=shp,
                                                        lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25,
                                                        linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                            shp_zg.plot(ax=ax, facecolor="None", edgecolor="k")

                            plt.title(f"{vv} ({data.attrs['units']})")
                            if ii == 1:
                                ax.set_yticks([])
                                ax.set_ylabel("Portrait")

                            ii = ii + 1

                        perf = pcat.search(processing_level=f"{warming_level}-performance-vs-{target_year}-{v}").to_dask()
                        analogs = sort_analogs(perf.rmse)
                        for i in range(j, j+5, 1):
                            ds = pcat.search(activity="ClimEx", processing_level=f"{target_year}-stats-{warming_level}", member=str(analogs.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]).to_dask()

                            for vv in xs.CONFIG["figures"]:
                                if ds[vv].dtype == '<m8[ns]':
                                    ds[vv] = ds[vv].dt.days
                                    ds[vv].attrs["units"] = "days"
                                if vv == "season_end":
                                    ds[vv] = ds[vv] - 244  # vs. Sept. 1st
                                    ds[vv].attrs["units"] = "days"
                                if vv == "season_start":
                                    ds[vv] = ds[vv] - 152  # vs. June  1st
                                    ds[vv].attrs["units"] = "days"

                                blend[vv].extend([ds[vv]])

                                bounds = np.linspace(**xs.CONFIG["figures"][vv]["bnds"])
                                norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                                cmap = xs.CONFIG["figures"][vv]["cmap"]

                                ax = plt.subplot(6, len(stats_ref.data_vars), ii, projection=proj)
                                figures.templates.map_hydro(ax, ds[vv], shp=shp,
                                                            lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25,
                                                            linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                                shp_zg.plot(ax=ax, facecolor="None", edgecolor="k")

                                plt.title("")
                                if vv == list(xs.CONFIG["figures"])[0]:
                                    ax.set_yticks([])
                                    ax.set_ylabel(f"{str(analogs.isel(stacked=i).realization.values).split('.')[0].split('_')[-1]} | "
                                               f"{str(analogs.isel(stacked=i).time.dt.year.values)}\nsum(RMSE) = {np.round(analogs.isel(stacked=i).values, 2)}")

                                ii = ii + 1

                        plt.tight_layout()

                        os.makedirs(xs.CONFIG['io']['figures'], exist_ok=True)
                        plt.savefig(f"{xs.CONFIG['io']['figures']}hydro-analogs-{target_year}_{warming_level}degC-{v}-{j}.png")
                        plt.close()

                    # BLEND
                    # Open the reference
                    stats_ref = pcat.search(type="reconstruction-hydro", processing_level=f"{target_year}-stats").to_dask()
                    [stats_ref[c].load() for c in stats_ref.coords]

                    # Plot
                    plt.subplots(4, len(stats_ref.data_vars), figsize=(35, 15))
                    plt.suptitle(f"Analogues de l'année {target_year} - +{warming_level}°C vs pré-industriel - {v}")

                    ii = 1
                    for vv in xs.CONFIG["figures"]:
                        if stats_ref[vv].dtype == '<m8[ns]':
                            stats_ref[vv] = stats_ref[vv].dt.days
                            stats_ref[vv].attrs["units"] = "days"
                        if vv == "season_end":
                            stats_ref[vv] = stats_ref[vv] - 244  # vs. Sept. 1st
                            stats_ref[vv].attrs["units"] = "days"
                        if vv == "season_start":
                            stats_ref[vv] = stats_ref[vv] - 152  # vs. June  1st
                            stats_ref[vv].attrs["units"] = "days"

                        bounds = np.linspace(**xs.CONFIG["figures"][vv]["bnds"])
                        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                        cmap = xs.CONFIG["figures"][vv]["cmap"]

                        ax = plt.subplot(4, len(stats_ref.data_vars), ii, projection=proj)
                        figures.templates.map_hydro(ax, stats_ref[vv], shp=shp,
                                                    lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25,
                                                    linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                        shp_zg.plot(ax=ax, facecolor="None", edgecolor="k")

                        plt.title(f"{vv} ({stats_ref[vv].attrs['units']})")
                        if ii == 1:
                            ax.set_yticks([])
                            ax.set_ylabel("Portrait")

                        ii = ii + 1

                    for j in [1, 5, 10]:
                        for vv in xs.CONFIG["figures"]:
                            da = xr.concat(blend[vv][0:j], dim="realization").mean(dim="realization", keep_attrs=True)

                            bounds = np.linspace(**xs.CONFIG["figures"][vv]["bnds"])
                            norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                            cmap = xs.CONFIG["figures"][vv]["cmap"]

                            ax = plt.subplot(4, len(stats_ref.data_vars), ii, projection=proj)
                            figures.templates.map_hydro(ax, da, shp=shp,
                                                        lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25,
                                                        linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                            shp_zg.plot(ax=ax, facecolor="None", edgecolor="k")

                            plt.title("")
                            if vv == list(xs.CONFIG["figures"])[0]:
                                ax.set_yticks([])
                                ax.set_ylabel(f"Blend of the best {j} analogs")

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
    full_domain = ref[criteria[0][0]].sel(time=f"{target_year}-06-01").chunk({"lon": -1})

    region_close = xs.extract.clisops_subset(xr.ones_like(full_domain), {"method": "shape", "shape": {"shape": target_region, "buffer": 0.1}}).interp_like(full_domain)
    region_far = xs.extract.clisops_subset(xr.ones_like(full_domain), {"method": "shape", "shape": {"shape": target_region, "buffer": 0.5}}).interp_like(full_domain)

    # Loop on each criterion
    rsme_sum = []
    for c in criteria:
        target = ref[c[0]].sel(time=f"{target_year}-{c[1]:02d}-01").chunk({"lon": -1})
        candidates = hist[c[0]].where(hist.time.dt.month == c[1], drop=True).chunk({"lon": -1})
        candidates["time"] = pd.to_datetime(candidates["time"].dt.strftime("%Y-01-01"))

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


def streamflow_stats(da, target_year, ds = None, to_level: str = None):

    da = convert_calendar(da, 'noleap').chunk({"time": -1})

    out = xr.Dataset()

    q_summer = da.sel(time=slice(f"{target_year}-04", f"{target_year}-11"))
    # Days under the ecological discharge
    q_eco = xclim.land.freq_analysis(da, mode="min", window=7, t=2, dist="lognorm", **{"month": [5, 6, 7, 8, 9, 10, 11]}).squeeze()
    bool_under_eco = q_summer < q_eco
    out["sum_under_eco"] = bool_under_eco.sum(dim="time")
    out["sum_under_eco"].attrs = {"long_name": "Number of days between April and November that are below the 7Q2", "units": "days"}
    out["longest_under_eco"] = xclim.indices.run_length.longest_run(bool_under_eco)
    out["longest_under_eco"].attrs = {"long_name": "Maximum consecutive days between April and November that are below the 7Q2", "units": "days"}

    # Minimum volume relative to the ecological discharge
    qmin = q_summer.rolling({"time": 7}).mean().min(dim="time")
    out["lowest_volume_vs_eco"] = (qmin - q_eco) / q_eco * 100
    out["lowest_volume_vs_eco"].attrs = {"long_name": "Difference between the minimum 7-day discharge and the 7Q2", "units": "%"}

    # First and last date of lowflows
    thresh = da.mean(dim="time") - (da.mean(dim="time") - da.quantile(0.05, dim="time")) * 0.9
    bool_under_low = q_summer < thresh
    out["season_start"] = xclim.indices.run_length.first_run(bool_under_low, window=7, coord="dayofyear")
    out["season_start"].attrs = {"long_name": "First dayofyear where the discharge is below 10% of the mean annual flow for 7 consecutive days", "units": "dayofyear"}
    out["season_end"] = xclim.indices.run_length.last_run(bool_under_low, window=7, coord="dayofyear")
    out["season_end"].attrs = {"long_name": "Last dayofyear where the 7-day discharge is below 10% of the mean annual flow for 7 consecutive days", "units": "dayofyear"}

    # Freshet
    q_spring = da.sel(time=slice(f"{target_year}-02", f"{target_year}-07")).rolling({"time": 14}, center=True).mean()
    usual_freshet = da.groupby("time.dayofyear").quantile(0.5, dim="time").rolling({"dayofyear": 14}, center=True).mean()
    out["freshet_earliness"] = q_spring.idxmax(dim="time").dt.dayofyear - usual_freshet.idxmax(dim="dayofyear")
    out["freshet_earliness"].attrs = {"long_name": "Difference between the date of maximum discharge and the climatological average", "units": "days"}
    out["freshet_volume"] = (q_spring.max(dim="time") - usual_freshet.max(dim="dayofyear")) / usual_freshet.max(dim="dayofyear") * 100
    out["freshet_volume"].attrs = {"long_name": "Ratio between the maximum 7-day discharge and the climatological average", "units": "%"}

    if ds:
        out.attrs = ds.attrs
        out.attrs["cat:frequency"] = "fx"
        out.attrs["cat:xrfreq"] = "fx"

    if not to_level:
        out.attrs["cat:processing_level"] = f"{target_year}-stats"
    else:
        out.attrs["cat:processing_level"] = to_level

    return out


if __name__ == '__main__':
    main()
