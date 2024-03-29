import numpy as np
import pandas as pd
import xarray as xr
import xclim
import xclim.indices
import xscen as xs
import xskillscore as xss
from copy import deepcopy
import geopandas as gpd
import os
import json

from utils import get_target_region, get_stations_within_target_region, sort_analogs

xs.load_config("configs/cfg_analogs.yml", "paths.yml")


def main():
    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"])

    if xs.CONFIG["tasks"]["identify"]:
        for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
            for target_year in xs.CONFIG["analogs"]["targets"]:
                if not pcat.exists_in_cat(activity="ClimEx", processing_level=f"{warming_level}-performance-vs-{target_year}"):
                    # Open ClimEx
                    hist_dict = pcat.search(source="CRCM5.*", processing_level=f"indicators-warminglevel.*{warming_level}vs.*").to_dataset_dict()
                    hist = xclim.ensembles.create_ensemble(hist_dict)
                    hist = xs.utils.clean_up(hist, common_attrs_only=hist_dict)

                    # Open ERA5-Land
                    ref = pcat.search(source="ERA5.*", variable=["spei3", "spei6", "spei9"]).to_dask()
                    # ClimEx was bias adjusted on a smaller domain. Use the same region.
                    ref = ref.where(~np.isnan(hist["spei3"].isel(realization=0)).all(dim="time"))

                    perf = compute_criteria(ref, hist, target_year=target_year, **xs.CONFIG["analogs"]["compute_criteria"])
                    perf.attrs["cat:processing_level"] = f"{warming_level}-{perf.attrs['cat:processing_level']}"

                    filename = f"{xs.CONFIG['io']['stats']}{perf.attrs['cat:id']}_{perf.attrs['cat:processing_level']}.zarr"
                    xs.save_to_zarr(perf, filename=filename, mode='o')
                    pcat.update_from_ds(perf, path=filename)

    if xs.CONFIG["tasks"]["construct_hydro"]:
        for target_year in xs.CONFIG["analogs"]["targets"]:
            for xrfreq in pcat.search(type="reconstruction-hydro", processing_level=f"indicators").unique("xrfreq"):
                if xrfreq != "fx":
                    # Reference
                    ds = pcat.search(type="reconstruction-hydro", processing_level=f"indicators", xrfreq=xrfreq).to_dask()
                    if xrfreq == "AS-JAN":
                        ds["season_length"] = (ds["season_end"] - ds["season_start"])
                        ds["season_length"].attrs["units"] = "d"
                    [ds[c].load() for c in ds.coords]
                    if xrfreq == "MS":
                        target_ref = ds.sel(time=slice(f"{target_year-1}-12", f"{target_year}-11"))
                    elif xrfreq =="AS-DEC":
                        target_ref = ds.sel(time=slice(f"{target_year - 1}", f"{target_year - 1}")).squeeze().drop_vars(["time"])
                    else:
                        target_ref = ds.sel(time=slice(str(target_year), str(target_year))).squeeze().drop_vars(["time"])

                    # Historical analog
                    perf = sort_analogs(pcat.search(processing_level=f"0.91-performance-vs-{target_year}").to_dask().rmse)
                    analog_hist = []
                    for i in range(5):
                        member = str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                        analog_year = int(perf.isel(stacked=i).time.dt.year.values)
                        ds = pcat.search(type="simulation-hydro", activity="ClimEx", processing_level="indicators-0.91", member=member, xrfreq=xrfreq).to_dask()
                        if xrfreq == "AS-JAN":
                            ds["season_length"] = (ds["season_end"] - ds["season_start"])
                            ds["season_length"].attrs["units"] = "d"
                        [ds[c].load() for c in ds.coords]
                        if xrfreq == "MS":
                            da = ds.sel(time=slice(f"{analog_year-1}-12", f"{analog_year}-11"))
                            da["time"] = target_ref["time"]
                            analog_hist.extend([da])
                        elif xrfreq == "AS-DEC":
                            analog_hist.extend([ds.sel(time=slice(str(analog_year - 1), str(analog_year - 1))).squeeze().drop_vars(["time"])])
                        else:
                            analog_hist.extend([ds.sel(time=slice(str(analog_year), str(analog_year))).squeeze().drop_vars(["time"])])

                    for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
                        if warming_level != 0.91:
                            # Analogs
                            perf = sort_analogs(pcat.search(processing_level=f"{warming_level}-performance-vs-{target_year}").to_dask().rmse)
                            analog_fut = []
                            for i in range(5):
                                member = str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                                analog_year = int(perf.isel(stacked=i).time.dt.year.values)
                                ds = pcat.search(type="simulation-hydro", activity="ClimEx", processing_level=f"indicators-{warming_level}", member=member, xrfreq=xrfreq).to_dask()
                                if xrfreq == "AS-JAN":
                                    ds["season_length"] = (ds["season_end"] - ds["season_start"])
                                    ds["season_length"].attrs["units"] = "d"
                                    ds["season_histlength"] = (ds["season_histend"] - ds["season_histstart"])
                                    ds["season_histlength"].attrs["units"] = "d"
                                    # Replace some indicators with the ones calculated from historical thresholds
                                    for v in ['days_under_hist7q2', 'max_consecutive_days_under_hist7q2', 'days_under_hist7q10', 'max_consecutive_days_under_hist7q10', 'season_histstart', 'season_histend', 'season_histlength']:
                                        ds[v.replace("hist", "")] = ds[v]
                                        ds = ds.drop_vars([v])
                                [ds[c].load() for c in ds.coords]
                                if xrfreq == "MS":
                                    da = ds.sel(time=slice(f"{analog_year-1}-12", f"{analog_year}-11"))
                                    da["time"] = target_ref["time"]
                                    analog_fut.extend([da])
                                elif xrfreq == "AS-DEC":
                                    analog_fut.extend([ds.sel(time=slice(str(analog_year - 1), str(analog_year - 1))).squeeze().drop_vars(["time"])])
                                else:
                                    analog_fut.extend([ds.sel(time=slice(str(analog_year), str(analog_year))).squeeze().drop_vars(["time"])])

                            kind = {v: "%" if v not in ["doy_14qmax", "season_start", "season_end", 'season_length', 'days_under_7q2',
                                                        'max_consecutive_days_under_7q2', 'days_under_7q10', 'max_consecutive_days_under_7q10'] else "+" for v in analog_fut[0].data_vars}

                            with xr.set_options(keep_attrs=True):
                                deltas = xs.compute_deltas(xr.concat(analog_fut, dim="realization").mean(dim="realization"),
                                                           xr.concat(analog_hist, dim="realization").mean(dim="realization"), kind=kind, rename_variables=False, to_level=f"deltas-{target_year}-{warming_level}")
                                deltas.attrs["cat:member"] = "5member-mean"
                                deltas.attrs["cat:id"] = xs.catalog.generate_id(deltas)[0]

                                analog_reconstruct = xr.Dataset()
                                for v in deltas.data_vars:
                                    analog_reconstruct[v] = target_ref[v] + target_ref[v] * deltas[v] / 100 if deltas[v].attrs["delta_kind"] == "percentage" else target_ref[v] + deltas[v]
                                    if v in ['days_under_7q2', 'max_consecutive_days_under_7q2', 'days_under_7q10', 'max_consecutive_days_under_7q10']:
                                        analog_reconstruct[v] = analog_reconstruct[v].clip(min=0)
                                analog_reconstruct.attrs = target_ref.attrs
                                analog_reconstruct.attrs["cat:processing_level"] = f"analog-{target_year}-{warming_level}"

                            filename = f"{xs.CONFIG['io']['analogs']}{deltas.attrs['cat:id']}_{deltas.attrs['cat:processing_level']}_{deltas.attrs['cat:xrfreq']}.zarr"
                            xs.save_to_zarr(deltas, filename=filename, mode="a")
                            pcat.update_from_ds(deltas, path=filename)
                            filename = f"{xs.CONFIG['io']['analogs']}{analog_reconstruct.attrs['cat:id']}_{analog_reconstruct.attrs['cat:processing_level']}_{analog_reconstruct.attrs['cat:xrfreq']}.zarr"
                            xs.save_to_zarr(analog_reconstruct, filename=filename, mode="a")
                            pcat.update_from_ds(analog_reconstruct, path=filename)
                            if 'days_under_7q10' in deltas.data_vars:
                                filename = f"{xs.CONFIG['io']['analogs']}{deltas.attrs['cat:id']}_{deltas.attrs['cat:processing_level']}_{deltas.attrs['cat:xrfreq']}.zarr"
                                xs.save_to_zarr(deltas[["days_under_7q10", "max_consecutive_days_under_7q10"]], filename=filename, mode="o")
                                filename = f"{xs.CONFIG['io']['analogs']}{analog_reconstruct.attrs['cat:id']}_{analog_reconstruct.attrs['cat:processing_level']}_{analog_reconstruct.attrs['cat:xrfreq']}.zarr"
                                xs.save_to_zarr(analog_reconstruct[["days_under_7q10", "max_consecutive_days_under_7q10"]], filename=filename, mode="o")


    if xs.CONFIG["tasks"]["construct_climate"]:
        for target_year in xs.CONFIG["analogs"]["targets"]:
            for xrfreq in pcat.search(type="reconstruction", domain="ZGIEBV").unique("xrfreq"):
                if xrfreq != "fx":
                    # Reference
                    ds = pcat.search(type="reconstruction", domain="ZGIEBV", processing_level=["extracted", "indicators"], xrfreq=xrfreq).to_dask()
                    [ds[c].load() for c in ds.coords]
                    if xrfreq == "MS":
                        pr_ref = pcat.search(source="CHIRPS2.0", xrfreq=xrfreq).to_dask()
                        new_precip = deepcopy(pr_ref["pr_mon"].fillna(ds["precip_accumulation_mon"]))
                        ds["precip_accumulation_mon"] = new_precip
                    elif xrfreq == "AS-DEC":
                        pr_ref = pcat.search(source="CHIRPS2.0", xrfreq=xrfreq).to_dask()
                        new_precip = deepcopy(pr_ref["pr_yr"].fillna(ds["precip_accumulation_yr"]))
                        ds["precip_accumulation_yr"] = new_precip
                    if xrfreq in ["MS", "D"]:
                        target_ref = ds.sel(time=slice(f"{target_year-1}-12", f"{target_year}-11"))
                        if xrfreq == "D":
                            target_ref = xs.clean_up(target_ref, convert_calendar_kwargs={"target": "noleap"})
                    elif xrfreq == "AS-DEC":
                        target_ref = ds.sel(time=slice(f"{target_year - 1}", f"{target_year - 1}")).squeeze().drop_vars(["time"])
                    else:
                        raise ValueError

                    # Historical analog
                    perf = sort_analogs(pcat.search(processing_level=f"0.91-performance-vs-{target_year}").to_dask().rmse)
                    analog_hist = []
                    for i in range(5):
                        member = str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                        analog_year = int(perf.isel(stacked=i).time.dt.year.values)
                        ds = pcat.search(type="simulation", activity="ClimEx", processing_level="indicators-0.91" if xrfreq != "D" else "extracted-0.91", member=member, xrfreq=xrfreq).to_dask()
                        [ds[c].load() for c in ds.coords]
                        if xrfreq in ["MS", "D"]:
                            da = ds.sel(time=slice(f"{analog_year-1}-12", f"{analog_year}-11"))
                            da["time"] = target_ref["time"]
                            analog_hist.extend([da])
                        elif xrfreq == "AS-DEC":
                            analog_hist.extend([ds.sel(time=slice(str(analog_year - 1), str(analog_year - 1))).squeeze().drop_vars(["time"])])
                        else:
                            raise ValueError

                    for warming_level in xs.CONFIG["storylines"]["warming_levels"]:
                        if warming_level != 0.91:
                            # Analogs
                            perf = sort_analogs(pcat.search(processing_level=f"{warming_level}-performance-vs-{target_year}").to_dask().rmse)
                            analog_fut = []
                            for i in range(5):
                                member = str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                                analog_year = int(perf.isel(stacked=i).time.dt.year.values)
                                ds = pcat.search(type="simulation", activity="ClimEx", processing_level=f"indicators-{warming_level}" if xrfreq != "D" else f"extracted-{warming_level}", member=member, xrfreq=xrfreq).to_dask()
                                [ds[c].load() for c in ds.coords]
                                if xrfreq in ["MS", "D"]:
                                    da = ds.sel(time=slice(f"{analog_year-1}-12", f"{analog_year}-11"))
                                    da["time"] = target_ref["time"]
                                    analog_fut.extend([da])
                                elif xrfreq == "AS-DEC":
                                    analog_fut.extend([ds.sel(time=slice(str(analog_year - 1), str(analog_year - 1))).squeeze().drop_vars(["time"])])
                                else:
                                    analog_fut.extend([ds.sel(time=slice(str(analog_year), str(analog_year))).squeeze().drop_vars(["time"])])

                            kind = {v: "%" if v not in ["tg_mean_mon", "tg_mean_yr", "tasmax"] else "+" for v in analog_fut[0].data_vars}

                            with xr.set_options(keep_attrs=True):
                                if xrfreq != "D":
                                    deltas = xs.compute_deltas(xr.concat(analog_fut, dim="realization").mean(dim="realization"),
                                                               xr.concat(analog_hist, dim="realization").mean(dim="realization"), kind=kind, rename_variables=False, to_level=f"deltas-{target_year}-{warming_level}")
                                else:
                                    with xr.set_options(keep_attrs=True):
                                        deltas = xr.concat(analog_fut, dim="realization").mean(dim="realization") - xr.concat(analog_hist, dim="realization").mean(dim="realization")
                                        deltas = deltas.rolling(dim={"time": 30}, min_periods=1, center=True).mean()
                                    deltas.attrs = analog_hist[0].attrs
                                    deltas.attrs["cat:processing_level"] = f"deltas-{target_year}-{warming_level}"
                                    deltas["tasmax"].attrs["delta_kind"] = "absolute"
                                deltas.attrs["cat:member"] = "5member-mean"
                                deltas.attrs["cat:domain"] = "ZGIEBV"
                                deltas.attrs["cat:id"] = xs.catalog.generate_id(deltas)[0]

                                analog_reconstruct = xr.Dataset()
                                for v in deltas.data_vars:
                                    analog_reconstruct[v] = target_ref[v] + target_ref[v] * deltas[v] / 100 if deltas[v].attrs["delta_kind"] == "percentage" else target_ref[v] + deltas[v]
                                analog_reconstruct.attrs = target_ref.attrs
                                analog_reconstruct.attrs["cat:processing_level"] = f"analog-{target_year}-{warming_level}"

                            if xrfreq == "D":
                                deltas = xs.clean_up(deltas.chunk({"time": -1}), convert_calendar_kwargs={"target": "default"}, missing_by_var={"tasmax": 'interpolate'})
                                analog_reconstruct = xs.clean_up(analog_reconstruct.chunk({"time": -1}), convert_calendar_kwargs={"target": "default"})

                            filename = f"{xs.CONFIG['io']['analogs']}{deltas.attrs['cat:id']}_{deltas.attrs['cat:processing_level']}_{deltas.attrs['cat:domain']}_{deltas.attrs['cat:xrfreq']}.zarr"
                            xs.save_to_zarr(deltas, filename=filename, mode="o")
                            pcat.update_from_ds(deltas, path=filename)
                            filename = f"{xs.CONFIG['io']['analogs']}{analog_reconstruct.attrs['cat:id']}_{analog_reconstruct.attrs['cat:processing_level']}_{analog_reconstruct.attrs['cat:domain']}_{analog_reconstruct.attrs['cat:xrfreq']}.zarr"
                            xs.save_to_zarr(analog_reconstruct, filename=filename, mode="o")
                            pcat.update_from_ds(analog_reconstruct, path=filename)

    if xs.CONFIG["tasks"]["construct_blend"]:
        if not os.path.isfile(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_per_reach.json"):
            shp = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp").set_index("SIGLE")
            dummy = pcat.search(type="reconstruction-hydro", processing_level=f"indicators", xrfreq="AS-DEC").to_dask()
            regions = {}
            for index in shp.index:
                region = shp.loc[[index]]
                regions[index] = get_stations_within_target_region(dummy, region)
            with open(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_per_reach.json", 'w') as fp:
                json.dump(regions, fp)
        else:
            with open(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_per_reach.json", 'r') as fp:
                regions = json.load(fp)

        for xrfreq in pcat.search(type="reconstruction-hydro", processing_level=f"indicators").unique("xrfreq"):
            if xrfreq != "fx":
                for dataset in ["ref", "analog-1.5", "analog-2", "analog-3", "analog-4"]:
                    if dataset == "ref":
                        ds = pcat.search(type="reconstruction-hydro", processing_level=f"indicators", xrfreq=xrfreq).to_dask()
                        if xrfreq == "AS-JAN":
                            ds["season_length"] = (ds["season_end"] - ds["season_start"])
                            ds["season_length"].attrs["units"] = "d"
                        [ds[c].load() for c in ds.coords]
                    else:
                        ds_dict = pcat.search(type="reconstruction-hydro", processing_level=f"^analog-.*-{dataset.split('-')[1]}", xrfreq=xrfreq).to_dataset_dict()
                        [[ds[c].load() for c in ds.coords] for ds in ds_dict.values()]

                    r = []
                    for target_year in xs.CONFIG["analogs"]["targets"]:
                        if dataset != "ref":
                            k = [a for a in ds_dict.keys() if str(target_year) in a][0]
                            ds = ds_dict[k]

                        if xrfreq == "MS":
                            target = ds.sel(time=slice(f"{target_year-1}-12", f"{target_year}-11")).groupby("time.month").mean(dim="time")
                        elif xrfreq == "AS-DEC":
                            target = ds.sel(time=slice(f"{target_year - 1}", f"{target_year - 1}")).squeeze().drop_vars(["time"]) if "time" in ds else ds
                        else:
                            target = ds.sel(time=slice(str(target_year), str(target_year))).squeeze().drop_vars(["time"]) if "time" in ds else ds

                        for i in range(len(xs.CONFIG["analogs"]["targets"][target_year]["region"])):
                            if (xs.CONFIG["analogs"]["targets"][target_year]["region"][i] not in r):# or (target_year >= 2010):
                                r.extend([xs.CONFIG["analogs"]["targets"][target_year]["region"][i]])
                                stations = regions[xs.CONFIG["analogs"]["targets"][target_year]["region"][i]]

                                if (target_year == list(xs.CONFIG["analogs"]["targets"])[0]) and (i == 0):
                                    out = target.where(target.station_id.isin(stations))
                                else:
                                    out = out.where(~out.station_id.isin(stations))
                                    out = out.fillna(target.where(target.station_id.isin(stations)))

                    out.attrs["cat:processing_level"] = f"temporalblend-{out.attrs['cat:processing_level']}"

                    filename = f"{xs.CONFIG['io']['analogs']}{out.attrs['cat:id']}_{out.attrs['cat:processing_level']}_{out.attrs['cat:domain']}_{out.attrs['cat:xrfreq']}.zarr"
                    xs.save_to_zarr(out, filename=filename, mode="o")
                    pcat.update_from_ds(out, path=filename)





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


# def compute_stats(ds, year, ref=None, hist=None):
#
#     out_stats = xr.Dataset()
#
#     # Manage indicators with multiple 0s
#     for v in ['days_under_7q2', 'max_consecutive_days_under_7q2', 'days_under_hist7q2', 'max_consecutive_days_under_hist7q2']:
#         if v in ds:
#             ds[v] = ds[v].where(ds[v] > 0)
#
#     # Climatology
#     ds = ds.chunk({"time": -1})
#     out_clim = ds.quantile(0.5, dim="time")
#     out_clim.attrs = ds.attrs
#     out_clim.attrs["cat:processing_level"] = "climatology"
#     out_clim.attrs["cat:frequency"] = "fx"
#     out_clim.attrs["cat:xrfreq"] = "fx"
#
#     # Some indicators were only computed for a single year
#     if ref is not None:
#         ref = ref.chunk({"time": -1})
#         ref_clim = ref.quantile(0.5, dim="time")
#         for vv in ['doy_14qmax', 'season_start', 'season_end', 'season_length']:
#             out_clim[vv] = ref_clim[vv]
#
#     if (hist is not None) and (ref is not None):
#         hist = hist.chunk({"time": -1})
#         hist_clim = hist.quantile(0.5, dim="time")
#         for vv in ['doy_14qmax', 'season_start', 'season_end', 'season_length', 'season_histstart', 'season_histend', 'season_histlength']:
#             hist[vv] = ref[vv.replace('hist', '')]
#             hist_clim[vv] = ref_clim[vv.replace('hist', '')]
#         for vv in ['days_under_hist7q2', 'max_consecutive_days_under_hist7q2']:
#             hist[vv] = hist[vv.replace('hist', '')]
#             hist_clim[vv] = hist_clim[vv.replace('hist', '')]
#
#     for vv in ds.data_vars:
#         da = ds[vv].sel(time=slice(f"{year}-01-01", f"{year}-01-01")).squeeze()
#         if "time" in da.coords:
#             da = da.drop_vars(["time"])
#         out_stats[f"{vv}_ecdf"] = ecdf(ref[vv] if ((ref is not None) and (vv in ['doy_14qmax', 'season_start', 'season_end', 'season_length'])) else ds[vv], da)
#         out_stats[f"{vv}_ecdf"] = out_stats[f"{vv}_ecdf"].where(~da.isnull())
#         out_stats[f"{vv}_delta-abs"] = xs.compute_deltas(da.to_dataset(), out_clim[vv].to_dataset(), kind="+", rename_variables=False)[vv]
#         out_stats[f"{vv}_delta-pct"] = xs.compute_deltas(da.to_dataset(), out_clim[vv].to_dataset(), kind="%", rename_variables=False)[vv]
#         if (hist is not None) and (ref is not None):
#             out_stats[f"{vv}_ecdf_vs_hist"] = ecdf(ref[vv.replace('hist', '')] if vv in ['doy_14qmax', 'season_start', 'season_end', 'season_length', 'season_histstart', 'season_histend', 'season_histlength'] else hist[vv], da)
#             out_stats[f"{vv}_ecdf_vs_hist"] = out_stats[f"{vv}_ecdf_vs_hist"].where(~da.isnull())
#             out_stats[f"{vv}_delta-abs_vs_hist"] = xs.compute_deltas(da.to_dataset(), hist_clim[vv].to_dataset(), kind="+", rename_variables=False)[vv]
#             out_stats[f"{vv}_delta-pct_vs_hist"] = xs.compute_deltas(da.to_dataset(), hist_clim[vv].to_dataset(), kind="%", rename_variables=False)[vv]
#
#         proj = cartopy.crs.PlateCarree()
#         shp = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
#         shp = shp.set_index("TRONCON")
#         bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
#         norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
#         cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2', 'max_consecutive_days_under_7q2',
#                                     'season_histend', 'season_histlength', 'days_under_hist7q2', 'max_consecutive_days_under_hist7q2'] else 'RdBu_r'
#
#         m = 3
#         plt.figure(figsize=(35, 15))
#         ax = plt.subplot(m, 3, 1, projection=proj)
#         out_clim[vv] = out_clim[vv].compute()
#         figures.templates.map_hydro(ax, out_clim[vv], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, vmin=out_clim[vv].quantile(.10), vmax=out_clim[vv].quantile(.90))
#         plt.title(f"climatology")
#         ax = plt.subplot(m, 3, 2, projection=proj)
#         figures.templates.map_hydro(ax, da, shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, vmin=out_clim[vv].quantile(.10), vmax=out_clim[vv].quantile(.90))
#         plt.title(f"{year}")
#         if hist is not None:
#             ax = plt.subplot(m, 3, 3, projection=proj)
#             hist_clim[vv] = hist_clim[vv].compute()
#             figures.templates.map_hydro(ax, hist_clim[vv], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, vmin=out_clim[vv].quantile(.10), vmax=out_clim[vv].quantile(.90))
#             plt.title(f"hist_climatology")
#         ax = plt.subplot(m, 3, 4, projection=proj)
#         vm = np.abs(out_stats[f"{vv}_delta-abs"]).compute().quantile(0.9)
#         figures.templates.map_hydro(ax, out_stats[f"{vv}_delta-abs"], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, vmin=-vm, vmax=vm, cmap=cmap)
#         plt.title(f"abs-delta")
#         ax = plt.subplot(m, 3, 5, projection=proj)
#         vm = np.abs(out_stats[f"{vv}_delta-pct"]).compute().quantile(0.9)
#         figures.templates.map_hydro(ax, out_stats[f"{vv}_delta-pct"], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, vmin=-vm, vmax=vm, cmap=cmap)
#         plt.title(f"pct-delta")
#         ax = plt.subplot(m, 3, 6, projection=proj)
#         figures.templates.map_hydro(ax, out_stats[f"{vv}_ecdf"], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, norm=norm, cmap=cmap)
#         plt.title(f"ecdf")
#         if hist is not None:
#             ax = plt.subplot(m, 3, 7, projection=proj)
#             vm = np.abs(out_stats[f"{vv}_delta-abs_vs_hist"]).compute().quantile(0.9)
#             figures.templates.map_hydro(ax, out_stats[f"{vv}_delta-abs_vs_hist"], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, vmin=-vm, vmax=vm, cmap=cmap)
#             plt.title(f"abs-delta_vs_hist")
#             ax = plt.subplot(m, 3, 8, projection=proj)
#             vm = np.abs(out_stats[f"{vv}_delta-pct_vs_hist"]).compute().quantile(0.9)
#             figures.templates.map_hydro(ax, out_stats[f"{vv}_delta-pct_vs_hist"], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, vmin=-vm, vmax=vm, cmap=cmap)
#             plt.title(f"pct-delta_vs_hist")
#             ax = plt.subplot(m, 3, 9, projection=proj)
#             figures.templates.map_hydro(ax, out_stats[f"{vv}_ecdf_vs_hist"], shp=shp, lon_bnds=[-80, -58], lat_bnds=[44.75, 54.], legend=True, linewidth=1, norm=norm, cmap=cmap)
#             plt.title(f"ecdf_vs_hist")
#
#         plt.suptitle(f"{vv}")
#         plt.tight_layout()
#         os.makedirs(f"{xs.CONFIG['io']['figures']}stats/", exist_ok=True)
#         plt.savefig(f"{xs.CONFIG['io']['figures']}stats/{vv}-{year}-{ds.attrs['cat:id']}.png")
#
#         plt.close()
#
#     out_stats.attrs = ds.attrs
#     out_stats.attrs["cat:processing_level"] = f"deltas-{year}"
#     out_stats.attrs["cat:frequency"] = "fx"
#     out_stats.attrs["cat:xrfreq"] = "fx"
#
#     return out_clim, out_stats


if __name__ == '__main__':
    main()
