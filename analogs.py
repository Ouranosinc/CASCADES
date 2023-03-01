import os.path

import numpy as np
import pandas as pd
import xarray as xr
import xscen as xs
import xskillscore as xss
import xclim.ensembles
import matplotlib.pyplot as plt
import geopandas as gpd

xs.load_config("project.yml", "paths.yml")


def main():

    # FIXME
    if ~os.path.isfile("a"):
        pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"])

        # FIXME
        for warming_level in xs.CONFIG["storylines"]["warming_levels"]:

            # ClimEx
            hist_dict = pcat.search(source="CRCM5.*", processing_level=f".*{warming_level}vs.*").to_dataset_dict()
            hist = xclim.ensembles.create_ensemble(hist_dict)
            hist = xs.utils.clean_up(hist, common_attrs_only=hist_dict)

            # ERA5-Land
            ref = pcat.search(source="ERA5.*", variable=["spei3", "spei6", "spei9"]).to_dask()
            # ClimEx was bias adjusted on a smaller domain. Use the same region.
            ref = ref.where(~np.isnan(hist["spei3"].isel(realization=0)).all(dim="time"))

            for target in xs.CONFIG["analogs"]["targets"]:
                if not pcat.exists_in_cat(activity="ClimEx", processing_level=f"{warming_level}-performance-vs-{target}"):
                    perf = compute_criteria(ref, hist, target_year=target,
                                            criteria=xs.CONFIG["analogs"]["criteria"],
                                            to_level=f"{warming_level}-performance-vs-{target}")

                    filename = f"{xs.CONFIG['io']['stats']}{perf.attrs['cat:id']}_{perf.attrs['cat:processing_level']}.zarr"

                    xs.save_to_zarr(perf, filename=filename)
                    pcat.update_from_ds(perf, path=filename)
                    perf = xr.open_zarr(filename)
                else:
                    perf = pcat.search(activity="ClimEx", processing_level=f"{warming_level}-performance-vs-{target}").to_dask()

                analogs = sort_analogs(perf.rmse)

                compare_streamflow(target, analogs, warming_level=warming_level)


                criteria = xs.CONFIG["analogs"]["criteria"]
                # Plot
                plt.subplots(5, len(criteria))
                ii = 1
                for c in criteria:
                    ax = plt.subplot(5, len(criteria), ii)
                    ref[c[0]].sel(time=f"{target}-{c[1]:02d}-01").plot(vmin=-3.09, vmax=3.09, add_colorbar=False)
                    plt.title(f"{c[0].upper()}-{c[1]}")
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    plt.xlabel("")
                    plt.ylabel("")
                    if ii == 1:
                        plt.ylabel("ERA5-Land")

                    ii = ii + 1

                for i in range(4):
                    for c in criteria:
                        ax = plt.subplot(5, len(criteria), ii)
                        hist[c[0]].sel(realization=analogs.isel(stacked=i).realization,
                                       time=f"{str(analogs.isel(stacked=i).time.dt.year.values)}-{c[1]:02d}-01").plot(vmin=-3.09,
                                                                                                                      vmax=3.09,
                                                                                                                      add_colorbar=False)
                        plt.title("")
                        ax.set_xticklabels([])
                        ax.set_yticklabels([])
                        plt.xlabel("")
                        plt.ylabel("")
                        if c == criteria[0]:
                            plt.ylabel(f"{str(analogs.isel(stacked=i).realization.values).split('.')[0].split('_')[-1]} | "
                                       f"{str(analogs.isel(stacked=i).time.dt.year.values)}\nsum(RMSE) = {np.round(analogs.isel(stacked=i).values, 2)}")

                        ii = ii + 1

                plt.tight_layout()


def compute_criteria(ref, hist,
                     target_year: dict,
                     criteria: list,
                     *,
                     to_level: str = None):

    # Prepare weights
    shp = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")
    target_region = shp.loc[shp["SIGLE"].isin(xs.CONFIG["analogs"]["targets"][target_year])]
    full_domain = ref[criteria[0][0]].sel(time=f"{target_year}-06-01").chunk({"lon": -1})

    region_close = xs.extract.clisops_subset(xr.ones_like(full_domain), {"method": "shape", "shape": {"shape": target_region, "buffer": 0.1}}).interp_like(full_domain)
    region_far = xs.extract.clisops_subset(xr.ones_like(full_domain), {"method": "shape", "shape": {"shape": target_region, "buffer": 0.5}}).interp_like(
        full_domain)

    # Loop on each criterion
    rsme_sum = []
    for c in criteria:
        target = ref[c[0]].sel(time=f"{target_year}-{c[1]:02d}-01").chunk({"lon": -1})
        candidates = hist[c[0]].where(hist.time.dt.month == c[1], drop=True).chunk({"lon": -1})
        candidates["time"] = pd.to_datetime(candidates["time"].dt.strftime("%Y-01-01"))

        weights = xr.where(region_close == 1, 1, xr.where(region_far == 1, 0.33, 0)) * xr.where(target <= -2, 1, xr.where(target <= -1, 0.5, 0.33))

        rmse = xss.rmse(candidates, target, dim=["lon", "lat"], weights=weights, skipna=True)
        # Normalise
        rmse = (rmse - rmse.min()) / (rmse.max() - rmse.min())

        rsme_sum.extend([rmse])

    r = sum(rsme_sum)
    r.name = "rmse"
    r.attrs = {
        "long_name": "RMSE",
        "description": f"Sum of RMSEs for {criteria}",
        "units": ""
    }

    out = r.to_dataset()
    out.attrs = hist.attrs
    out.attrs["cat:processing_level"] = f"performance-vs-{target_year}" if not to_level else to_level

    return out


def sort_analogs(da):

    da = da.stack({"stacked": ["time", "realization"]})
    da = da.sortby(da)

    return da


def compare_streamflow(target_year: str,
                       analogs,
                       warming_level='0.91',
                       station="SLSO00941"
                       ):

    # Open Reference data
    portrait = xr.open_dataset(f"{xs.CONFIG['dpphc']['portrait']}PORTRAIT_2020_light.nc")
    portrait = portrait.sel(time=slice('1992', '2021')).sel(percentile=50)
    portrait["station_id"] = portrait.station_id.astype(str).str.join(dim="nchar_station_id")

    # Select the station
    portrait_station = portrait.where(portrait.station_id == station, drop=True).squeeze()
    portrait_station_7q2 = xclim.land.freq_analysis(portrait_station.Dis, mode="min", window=7, t=2, dist="lognorm", **{"month": [5, 6, 7, 8, 9, 10, 11]})
    portrait_station_qt = portrait_station.Dis.groupby("time.dayofyear").quantile(q=[0.10, 0.25, 0.50, 0.75, 0.90])

    # Plot
    plt.figure(figsize=(35, 15))
    plt.fill_between(portrait_station_qt.dayofyear, portrait_station.Dis.groupby("time.dayofyear").min(dim="time"), portrait_station.Dis.groupby("time.dayofyear").max(dim="time"), facecolor="#c6dfe7")
    plt.fill_between(portrait_station_qt.dayofyear, portrait_station_qt.sel(quantile=0.10), portrait_station_qt.sel(quantile=0.90), facecolor="#80b0c8")
    plt.fill_between(portrait_station_qt.dayofyear, portrait_station_qt.sel(quantile=0.25), portrait_station_qt.sel(quantile=0.75), facecolor="#003366")
    plt.plot(portrait_station_qt.dayofyear, portrait_station_qt.sel(quantile=0.50), c="k")
    plt.plot(portrait_station.sel(time=slice(str(target_year), str(target_year))).Dis, c='r')
    plt.hlines(portrait_station_7q2, 1, 365, linestyle="--")
    plt.ylim([0, portrait_station.Dis.max()])

    # Stats
    portrait_under_thresh = portrait_station.sel(time=slice(f"{target_year}-04", f"{target_year}-11")).Dis < portrait_station_7q2
    portrait_under_thresh.sum()
    consecutive = portrait_under_thresh.cumsum(dim='time') - portrait_under_thresh.cumsum(dim='time').where(portrait_under_thresh.values == 0).ffill(dim='time').fillna(0)
    consecutive.max()
    consecutive.where(consecutive > 7, drop=True).time.min()
    consecutive.where(consecutive > 7, drop=True).time.max()

    q_spring = portrait_station.sel(time=slice(f"{target_year}-02", f"{target_year}-07")).Dis
    qmax = q_spring.where(q_spring >= 0.5 * q_spring.max(), drop=True)
    qmax.idxmax()
    qsub = q_spring.sel(time=slice(str(qmax.idxmax().values.astype(str)), f"{target_year}-07")).where(q_spring < 0.1 * qmax.max().values)
    qsub.dropna(dim="time").time.min()

    q_spring_qt = portrait_station_qt.sel(quantile=.50).rolling({"dayofyear": 7}, center=True).mean()
    qmax_qt = q_spring_qt.where(q_spring_qt >= 0.8 * q_spring_qt.max(), drop=True)
    tmp = xr.DataArray(pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31"), coords={"time": pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31")})
    qmax_qt = tmp.where(tmp.dt.dayofyear.isin(qmax_qt.dayofyear), drop=True)
    qmax_qt.idxmin()
    qmax_qt.idxmax()
    qmax_qt.idxmax().dt.dayofyear - qmax.idxmax().dt.dayofyear

    #TODO: volume minimal vs 7q2


    # Open Analog and subset the 30-year data
    atlas2022 = xs.DataCatalog(xs.CONFIG["dpphc"]["atlas2022"])

    an = []
    for i in range(5):
        a = analogs.isel(stacked=i)
        ds = atlas2022.search(id=f".*MG24HQ.*{'_'.join(str(a.realization.values).split('.')[0].split('_')[0:3])}.*{'_'.join(str(a.realization.values).split('.')[0].split('_')[3:])}.*").to_dask()
        ds["station_id"] = ds.station_id.astype(str)
        ds.attrs["cat:mip_era"] = "CMIP5"
        ds = xs.subset_warming_level(ds, wl=float(warming_level), window=30)

        ds_station = ds.where(ds.station_id.compute() == station, drop=True).squeeze().compute()
        an.extend([ds_station.sel(time=slice(str(a.time.dt.year.values), str(a.time.dt.year.values))).Dis])
        if i == 0:
            ds_station_7q2 = xclim.land.freq_analysis(ds_station.Dis, mode="min", window=7, t=2, dist="lognorm", **{"month": [5, 6, 7, 8, 9, 10, 11]})
            ds_station_qt = ds_station.Dis.groupby("time.dayofyear").quantile(q=[0.10, 0.25, 0.50, 0.75, 0.90])

            # Plot
            plt.figure()
            plt.fill_between(ds_station_qt.dayofyear, ds_station.Dis.groupby("time.dayofyear").min(dim="time"), ds_station.Dis.groupby("time.dayofyear").max(dim="time"), facecolor="#c6dfe7")
            plt.fill_between(ds_station_qt.dayofyear, ds_station_qt.sel(quantile=0.10), ds_station_qt.sel(quantile=0.90), facecolor="#80b0c8")
            plt.fill_between(ds_station_qt.dayofyear, ds_station_qt.sel(quantile=0.25), ds_station_qt.sel(quantile=0.75), facecolor="#003366")
            plt.plot(ds_station_qt.dayofyear, ds_station_qt.sel(quantile=0.50), c="k")
            plt.hlines(ds_station_7q2, 1, 365, linestyle="--")
    # for i in range(len(an)):
    #     plt.plot(an[i], c='r', linewidth=0.5)
    for i in range(len(an)):
        an[i]["time"] = an[i]["time"].dt.dayofyear
    # plt.plot(xr.concat(an, dim="realization").median(dim="realization"), c='g', linewidth=3)
    plt.plot(xr.concat(an[0:3], dim="realization").median(dim="realization"), c='g', linewidth=2)



if __name__ == '__main__':
    main()
