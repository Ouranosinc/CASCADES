from distributed import Client
import os
import xarray as xr

import xclim
import xscen as xs
import pandas as pd
import xclim.indices.run_length as xcrl
import xclim.indicators.land as xcl
from xclim.core.calendar import convert_calendar
import json
from shapely.geometry import Point
import geopandas as gpd
from collections import defaultdict
from copy import deepcopy

from utils import fix_infocrue, atlas_radeau_common, sort_analogs

xs.load_config("paths.yml", "configs/cfg_hydro.yml")

def main():

    def _extract_func(wl=None, analog=None):
        analog = analog or []
        ds = xr.open_zarr(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}.zarr")

        # RADEAU
        stations = atlas_radeau_common()
        stations["ATLAS2018"] = stations["ATLAS2018"].str[0:3].str.cat(["0"] * len(stations["ATLAS2018"])).str.cat(
            stations["ATLAS2018"].str[3:])  # The RADEAU shapefile has 1 too many 0s
        cv = dict(zip(stations["TRONCON_ID"], stations["ATLAS2018"]))

        # load coordinates and subset
        [ds[c].load() for c in ds.coords]
        if "percentile" in ds:
            ds = ds.sel(time=slice(xs.CONFIG["extract"]["periods"][0], xs.CONFIG["extract"]["periods"][1]), percentile=50)
        else:
            ds.attrs["cat:driving_model"] = "CanESM2"
            ds_hist = xs.subset_warming_level(ds, wl=0.91, window=30).squeeze()
            ds = xs.subset_warming_level(ds, wl=wl, window=30).squeeze()
        ds = ds.where(ds.station_id.isin(stations["TRONCON_ID"].to_list()) | (ds.drainage_area > 25), drop=True)
        ds = ds.chunk({"station": 500, "time": -1})

        # Add the Atlas2018 name
        ds = ds.assign_coords({"atlas2018": ds.station_id.to_series().map(cv)})

        # Indicators
        if xs.CONFIG["tasks"]["indicators"]:
            ds = convert_calendar(ds, '365_day')
            ind_dict = xs.compute_indicators(ds, indicators="configs/indicators_hydro.yml")
            if ("AS-JAN" not in ind_dict.keys()) and xs.CONFIG["tasks"]["additional_indicators"]:
                ind_dict["AS-JAN"] = xr.Dataset()
                ind_dict["AS-JAN"].attrs = ds.attrs
                ind_dict["AS-JAN"].attrs["cat:processing_level"] = "indicators"
                ind_dict["AS-JAN"].attrs["cat:frequency"] = "yr"
                ind_dict["AS-JAN"].attrs["cat:xrfreq"] = "AS-JAN"

            if wl not in [None, 0.91]:
                ind_hist = xs.compute_indicators(ds_hist, indicators="configs/indicators_hydro.yml")["fx"]

            overwrite = {"doy_14qmax": False, "season_start": False, "season_end": False, "season_histstart": False, "season_histend": False}
            if xs.CONFIG["tasks"]["additional_indicators"]:
                if "days_under_7qx" in xs.CONFIG["additional_indicators"]:
                    # 7Q2
                    bool_under_eco = (ds.discharge < ind_dict["fx"]["7qx"].sel(return_period=2).squeeze()).where(ds.discharge.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]),
                                                                                            other=False)
                    ind_dict['AS-JAN']["days_under_7q2"] = bool_under_eco.resample({"time": "AS-JAN"}).sum(dim="time")
                    ind_dict['AS-JAN']["days_under_7q2"].attrs = {"long_name": "Number of days below the 7Q2.",
                                                                  "description": "Streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                  "units": "d"}
                    ind_dict['AS-JAN']["max_consecutive_days_under_7q2"] = xcrl.longest_run(bool_under_eco, freq="AS-JAN")
                    ind_dict['AS-JAN']["max_consecutive_days_under_7q2"].attrs = {"long_name": "Maximum consecutive number of days below the 7Q2.",
                                                                                  "description": "Maximum consecutive streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                                  "units": "d"}

                    if wl not in [None, 0.91]:
                        bool_under_eco = (ds.discharge < ind_hist["7qx"].sel(return_period=2).squeeze()).where(ds.discharge.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)
                        ind_dict['AS-JAN']["days_under_hist7q2"] = bool_under_eco.resample({"time": "AS-JAN"}).sum(dim="time")
                        ind_dict['AS-JAN']["days_under_hist7q2"].attrs = {"long_name": "Number of days below the 7Q2.",
                                                                          "description": "Streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                          "units": "d"}
                        ind_dict['AS-JAN']["max_consecutive_days_under_hist7q2"] = xcrl.longest_run(bool_under_eco, freq="AS-JAN")
                        ind_dict['AS-JAN']["max_consecutive_days_under_hist7q2"].attrs = {
                            "long_name": "Maximum consecutive number of days below the 7Q2.",
                            "description": "Maximum consecutive streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                            "units": "d"}

                    # 7Q10
                    bool_under_eco = (ds.discharge < ind_dict["fx"]["7qx"].sel(return_period=10).squeeze()).where(ds.discharge.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]),
                                                                                             other=False)
                    ind_dict['AS-JAN']["days_under_7q10"] = bool_under_eco.resample({"time": "AS-JAN"}).sum(dim="time")
                    ind_dict['AS-JAN']["days_under_7q10"].attrs = {"long_name": "Number of days below the 7Q10.",
                                                                   "description": "Streamflow under 7Q10 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                   "units": "d"}
                    ind_dict['AS-JAN']["max_consecutive_days_under_7q10"] = xcrl.longest_run(bool_under_eco, freq="AS-JAN")
                    ind_dict['AS-JAN']["max_consecutive_days_under_7q10"].attrs = {"long_name": "Maximum consecutive number of days below the 7Q10.",
                                                                                   "description": "Maximum consecutive streamflow under 7Q10 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                                   "units": "d"}

                    if wl not in [None, 0.91]:
                        bool_under_eco = (ds.discharge < ind_hist["7qx"].sel(return_period=10).squeeze()).where(ds.discharge.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)
                        ind_dict['AS-JAN']["days_under_hist7q10"] = bool_under_eco.resample({"time": "AS-JAN"}).sum(dim="time")
                        ind_dict['AS-JAN']["days_under_hist7q10"].attrs = {"long_name": "Number of days below the 7Q10.",
                                                                           "description": "Streamflow under 7Q10 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                           "units": "d"}
                        ind_dict['AS-JAN']["max_consecutive_days_under_hist7q10"] = xcrl.longest_run(bool_under_eco, freq="AS-JAN")
                        ind_dict['AS-JAN']["max_consecutive_days_under_hist7q10"].attrs = {
                            "long_name": "Maximum consecutive number of days below the 7Q2.",
                            "description": "Maximum consecutive streamflow under 7Q10 for month[5, 6, 7, 8, 9, 10, 11].",
                            "units": "d"}

                if "7qmin" in xs.CONFIG["additional_indicators"]:
                    ind_dict['AS-JAN']["7qmin"] = ds.discharge.rolling({"time": 7}, center=True).mean(keep_attrs=True) \
                        .where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11])).resample({"time": "AS-JAN"}).min(dim="time")
                    ind_dict['AS-JAN']["7qmin"].attrs["long_name"] = "Minimum 7-day discharge"
                    ind_dict['AS-JAN']["7qmin"].attrs["description"] = "Minimum 7-day discharge for month[5, 6, 7, 8, 9, 10, 11]."

                if "freshet" in xs.CONFIG["additional_indicators"]:
                    q14 = ds.discharge.rolling({"time": 14}, center=True).mean(keep_attrs=True).where(ds.time.dt.month.isin([2, 3, 4, 5, 6]), other=0)
                    ind_dict['AS-JAN']["14qmax"] = q14.resample({"time": "AS-JAN"}).max(dim="time")
                    ind_dict['AS-JAN']["14qmax"].attrs["long_name"] = "Maximum 14-day discharge"
                    ind_dict['AS-JAN']["14qmax"].attrs["description"] = "Maximum 14-day discharge for month[2, 3, 4, 5, 6]"
                    if wl is None:
                        ind_dict['AS-JAN']["doy_14qmax"] = xcl.doy_qmax(q14, freq="AS-JAN")
                        ind_dict['AS-JAN']["doy_14qmax"].attrs["long_name"] = "Dayofyear of the maximum 14-day discharge"
                        ind_dict['AS-JAN']["doy_14qmax"].attrs["description"] = "Dayofyear of the maximum 14-day discharge for month[2, 3, 4, 5, 6]"
                        ind_dict['AS-JAN']["doy_14qmax"].attrs["units"] = "dayofyear"
                    else:
                        already_exists = pcat.search(id=ds.attrs["cat:id"], processing_level=f"indicators-{wl}", variable="doy_14qmax")
                        if (len(already_exists) == 0) & (len(analog) > 0):
                            q14_a = q14.sel(time=q14.time.dt.year.isin(analog))
                            with xclim.set_options(data_validation="log"):
                                ind_dict['AS-JAN']["doy_14qmax"] = xcl.doy_qmax(q14_a, freq="AS-JAN")
                            ind_dict['AS-JAN']["doy_14qmax"].attrs["long_name"] = "Dayofyear of the maximum 14-day discharge"
                            ind_dict['AS-JAN']["doy_14qmax"].attrs[
                                "description"] = "Dayofyear of the maximum 14-day discharge for month[2, 3, 4, 5, 6]"
                            ind_dict['AS-JAN']["doy_14qmax"].attrs["units"] = "dayofyear"

                        elif len(already_exists) == 1:
                            already_exists = already_exists.to_dask()["doy_14qmax"].dropna(dim="time", how="all")
                            to_calc = list(set(analog).difference(set(already_exists.time.dt.year.values)))
                            if len(to_calc) > 0:
                                overwrite["doy_14qmax"] = True
                                q14_a = q14.sel(time=q14.time.dt.year.isin(to_calc))
                                with xclim.set_options(data_validation="log"):
                                    doy_14qmax = xcl.doy_qmax(q14_a, freq="AS-JAN")
                                ind_dict['AS-JAN']["doy_14qmax"] = xr.concat((deepcopy(already_exists), doy_14qmax), dim="time")
                                ind_dict['AS-JAN']["doy_14qmax"].attrs["long_name"] = "Dayofyear of the maximum 14-day discharge"
                                ind_dict['AS-JAN']["doy_14qmax"].attrs[
                                    "description"] = "Dayofyear of the maximum 14-day discharge for month[2, 3, 4, 5, 6]"
                                ind_dict['AS-JAN']["doy_14qmax"].attrs["units"] = "dayofyear"

                if "lowflow_season" in xs.CONFIG["additional_indicators"]:
                    thresh = ds.discharge.mean(dim="time") - (ds.discharge.mean(dim="time") - ind_dict["fx"]["7q2"].squeeze()) * 0.85
                    bool_under_low = (ds.discharge < thresh).where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)
                    if wl is None:
                        ind_dict['AS-JAN']["season_start"] = xcrl.first_run(bool_under_low, window=7, coord="dayofyear", freq="AS-JAN")
                        ind_dict['AS-JAN']["season_start"].attrs = {
                            "long_name": "First dayofyear where the discharge is below 15% of the mean annual flow for 7 consecutive days",
                            "units": "dayofyear"}

                        ind_dict['AS-JAN']["season_end"] = bool_under_low.resample({"time": "AS-JAN"}).map(xcrl.last_run, window=7, coord="dayofyear")
                        ind_dict['AS-JAN']["season_end"].attrs = {
                            "long_name": "Last dayofyear where the 7-day discharge is below 15% of the mean annual flow for 7 consecutive days",
                            "units": "dayofyear"}
                    else:
                        already_exists = pcat.search(id=ds.attrs["cat:id"], processing_level=f"indicators-{wl}", variable=["season_start", "season_end"])
                        if (len(already_exists) == 0) & (len(analog) > 0):
                            bool_under_low_yr = bool_under_low.sel(time=bool_under_low.time.dt.year.isin(analog))
                            with xclim.set_options(data_validation="log"):
                                ind_dict['AS-JAN']["season_start"] = xcrl.first_run(bool_under_low_yr, window=7, coord="dayofyear", freq="AS-JAN")
                            ind_dict['AS-JAN']["season_start"].attrs = {
                                "long_name": "First dayofyear where the discharge is below 15% of the mean annual flow for 7 consecutive days",
                                "units": "dayofyear"}

                            ind_dict['AS-JAN']["season_end"] = bool_under_low_yr.resample({"time": "AS-JAN"}).map(xcrl.last_run, window=7, coord="dayofyear")
                            ind_dict['AS-JAN']["season_end"].attrs = {
                                "long_name": "Last dayofyear where the 7-day discharge is below 15% of the mean annual flow for 7 consecutive days",
                                "units": "dayofyear"}

                        elif len(already_exists) == 1:
                            already_exists_start = already_exists.to_dask()["season_start"].dropna(dim="time", how="all")
                            to_calc = list(set(analog).difference(set(already_exists_start.time.dt.year.values)))
                            if len(to_calc) > 0:
                                overwrite["season_start"] = True
                                bool_under_low_yr = bool_under_low.sel(time=bool_under_low.time.dt.year.isin(to_calc))
                                with xclim.set_options(data_validation="log"):
                                    season_start = xcrl.first_run(bool_under_low_yr, window=7, coord="dayofyear", freq="AS-JAN")
                                ind_dict['AS-JAN']["season_start"] = xr.concat((deepcopy(already_exists_start), season_start), dim="time")
                                ind_dict['AS-JAN']["season_start"].attrs = {
                                    "long_name": "First dayofyear where the discharge is below 15% of the mean annual flow for 7 consecutive days",
                                    "units": "dayofyear"}
                            already_exists_end = already_exists.to_dask()["season_end"].dropna(dim="time", how="all")
                            to_calc = list(set(analog).difference(set(already_exists_end.time.dt.year.values)))
                            if len(to_calc) > 0:
                                overwrite["season_end"] = True
                                bool_under_low_yr = bool_under_low.sel(time=bool_under_low.time.dt.year.isin(to_calc))
                                season_end = bool_under_low_yr.resample({"time": "AS-JAN"}).map(xcrl.last_run, window=7, coord="dayofyear")
                                ind_dict['AS-JAN']["season_end"] = xr.concat((deepcopy(already_exists_end), season_end), dim="time")
                                ind_dict['AS-JAN']["season_end"].attrs = {
                                    "long_name": "Last dayofyear where the 7-day discharge is below 15% of the mean annual flow for 7 consecutive days",
                                    "units": "dayofyear"}

                        if wl not in [None, 0.91]:
                            thresh = ds.discharge.mean(dim="time") - (ds.discharge.mean(dim="time") - ind_hist["7q2"].squeeze()) * 0.85
                            bool_under_low = (ds.discharge < thresh).where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)

                            already_exists = pcat.search(id=ds.attrs["cat:id"], processing_level=f"indicators-{wl}",  variable=["season_histstart", "season_histend"])
                            if (len(already_exists) == 0) & (len(analog) > 0):
                                bool_under_low_yr = bool_under_low.sel(time=bool_under_low.time.dt.year.isin(analog))
                                with xclim.set_options(data_validation="log"):
                                    ind_dict['AS-JAN']["season_histstart"] = xcrl.first_run(bool_under_low_yr, window=7, coord="dayofyear", freq="AS-JAN")
                                ind_dict['AS-JAN']["season_histstart"].attrs = {
                                    "long_name": "First dayofyear where the discharge is below 15% of the mean annual flow for 7 consecutive days",
                                    "units": "dayofyear"}

                                ind_dict['AS-JAN']["season_histend"] = bool_under_low_yr.resample({"time": "AS-JAN"}).map(xcrl.last_run, window=7,
                                                                                                                      coord="dayofyear")
                                ind_dict['AS-JAN']["season_histend"].attrs = {
                                    "long_name": "Last dayofyear where the 7-day discharge is below 15% of the mean annual flow for 7 consecutive days",
                                    "units": "dayofyear"}

                            elif len(already_exists) == 1:
                                already_exists_start = already_exists.to_dask()["season_histstart"].dropna(dim="time", how="all")
                                to_calc = list(set(analog).difference(set(already_exists_start.time.dt.year.values)))
                                if len(to_calc) > 0:
                                    overwrite["season_histstart"] = True
                                    bool_under_low_yr = bool_under_low.sel(time=bool_under_low.time.dt.year.isin(to_calc))
                                    with xclim.set_options(data_validation="log"):
                                        season_start = xcrl.first_run(bool_under_low_yr, window=7, coord="dayofyear", freq="AS-JAN")
                                    ind_dict['AS-JAN']["season_histstart"] = xr.concat((deepcopy(already_exists_start), season_start), dim="time")
                                    ind_dict['AS-JAN']["season_histstart"].attrs = {
                                        "long_name": "First dayofyear where the discharge is below 15% of the mean annual flow for 7 consecutive days",
                                        "units": "dayofyear"}
                                already_exists_end = already_exists.to_dask()["season_histend"].dropna(dim="time", how="all")
                                to_calc = list(set(analog).difference(set(already_exists_end.time.dt.year.values)))
                                if len(to_calc) > 0:
                                    overwrite["season_histend"] = True
                                    bool_under_low_yr = bool_under_low.sel(time=bool_under_low.time.dt.year.isin(to_calc))
                                    season_end = bool_under_low_yr.resample({"time": "AS-JAN"}).map(xcrl.last_run, window=7, coord="dayofyear")
                                    ind_dict['AS-JAN']["season_histend"] = xr.concat((deepcopy(already_exists_end), season_end), dim="time")
                                    ind_dict['AS-JAN']["season_histend"].attrs = {
                                        "long_name": "Last dayofyear where the 7-day discharge is below 15% of the mean annual flow for 7 consecutive days",
                                        "units": "dayofyear"}

        else:
            ind_dict = {"D": ds}

        for freq, out in ind_dict.items():
            if (freq != "fx") & (wl is None):
                out = out.sel(time=slice(f"{xs.CONFIG['storylines']['ref_period'][0]}-01-01", f"{xs.CONFIG['storylines']['ref_period'][1]}-12-31"))

            if wl is not None:
                out.attrs["cat:processing_level"] = f"{out.attrs['cat:processing_level']}-{wl}"

            [out[c].load() for c in out.coords]
            filename = f"{xs.CONFIG['io']['stats']}{out.attrs['cat:id']}_{out.attrs['cat:processing_level']}_{out.attrs['cat:xrfreq']}.zarr"
            xs.save_to_zarr(out, filename, mode="a", rechunk={"station": 500, "time": -1})
            pcat.update_from_ds(out, filename)

            if freq == "AS-JAN":
                to_ov = [vv for vv in overwrite.keys() if ((overwrite[vv] is True) and (vv in out))]
                if len(to_ov) > 0:
                    out2 = out[to_ov]
                    xs.save_to_zarr(out2, filename, mode="o", rechunk={"station": 500, "time": -1})
                    pcat.update_from_ds(out2, filename)

    hcat = xs.DataCatalog(xs.CONFIG["dpphc"]["atlas2022"])
    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"])

    if xs.CONFIG["tasks"]["extract"]:
        if "ref" in xs.CONFIG["datasets"]:
            # if not pcat.exists_in_cat(type="reconstruction-hydro", processing_level="indicators"):
            # Open the reference
            cat = hcat.search(type="reconstruction-hydro", processing_level="raw")
            if not os.path.isdir(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}.zarr"):
                fix_infocrue(cat)

            with Client(**xs.CONFIG["dask"]) as c:
                _extract_func()

        if len(set(xs.CONFIG["datasets"]).difference(["ref"])) > 0:
            for wl in xs.CONFIG["storylines"]["warming_levels"]:
                analogs = defaultdict(list)
                for d in set(xs.CONFIG["datasets"]).difference(["ref"]):
                    dparts = d.split("-")
                    if float(dparts[0]) == wl:
                        perf = pcat.search(processing_level=f"{dparts[0]}-performance-vs-{dparts[1]}").to_dask()
                        best = sort_analogs(perf.rmse)[0:5]
                        for r, y in zip(best.realization, best.time):
                            analogs[str(r.realization.values).split(".")[0]].extend([int(y.dt.year.values)])

                if len(analogs) > 0:
                    for a in analogs.keys():
                        # Open the simulation
                        cat = hcat.search(type="simulation-hydro", processing_level="raw", member=a.split("_")[-1], impact_model_member="MG24HQ")
                        if not os.path.isdir(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}.zarr"):
                            fix_infocrue(cat)

                        with Client(**xs.CONFIG["dask"]) as c:
                            if wl != 0.91:
                                # 0.91 is required to compute deltas later on
                                _extract_func(wl=0.91)
                            _extract_func(wl=wl, analog=analogs[a])

    if xs.CONFIG["tasks"]["extract_radeau"]:
        for dataset in xs.CONFIG["datasets"]:
            if dataset == "ref":
                ds_dict = pcat.search(id=".*Portrait.*", processing_level="indicators").to_dataset_dict()
            else:
                ds_dict = pcat.search(type=".*hydro.*", processing_level=f"{dataset}.*").to_dataset_dict()

            for key, ds in ds_dict.items():
                if (ds.attrs["cat:xrfreq"] == "AS-JAN") and (dataset == "ref"):
                    ds["season_length"] = ds["season_end"] - ds["season_start"]
                    ds["season_length"].attrs["units"] = "d"
                # load coordinates and subset
                [ds[c].load() for c in ds.coords]
                stations = atlas_radeau_common()
                stations["ATLAS2018"] = stations["ATLAS2018"].str[0:3].str.cat(["0"] * len(stations["ATLAS2018"])).str.cat(
                    stations["ATLAS2018"].str[3:])  # The RADEAU shapefile has 1 too many 0s
                cv = dict(zip(stations["TRONCON_ID"], stations["ATLAS2018"]))

                ds = ds.where(ds.station_id.isin(stations["TRONCON_ID"]), drop=True).squeeze()
                # Add the Atlas2018 name
                ds = ds.assign_coords({"atlas2018": ds.station_id.to_series().map(cv)})

                # Cleanup
                ds = xs.clean_up(ds, change_attr_prefix="dataset:", attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})

                # Prepare CSV
                if dataset == "ref":
                    filename = f"{ds.attrs['dataset:impact_model_source']}_{ds.attrs['dataset:impact_model_project']}"
                elif "analog" in dataset:
                    filename = f"{ds.attrs['dataset:impact_model_source']}_{ds.attrs['dataset:impact_model_project']}_{ds.attrs['dataset:processing_level']}degC"
                else:
                    filename = f"{ds.attrs['dataset:activity']}_{ds.attrs['dataset:impact_model_project']}_{ds.attrs['dataset:processing_level']}degC"

                # Write some metadata
                os.makedirs(xs.CONFIG['io']['livraison'], exist_ok=True)
                with open(f"{xs.CONFIG['io']['livraison']}dataset-metadata_{filename}.json", 'w') as fp:
                    json.dump(ds.attrs, fp)

                for v in ds.data_vars:
                    df = ds[v].swap_dims({"station": "atlas2018"}).to_pandas()
                    if isinstance(df, pd.Series):
                        df = df.to_frame(name=v)
                    if df.index.name != "atlas2018":
                        df = df.T
                    df.to_csv(f"{xs.CONFIG['io']['livraison']}{v}_{filename}_RADEAU.csv")

                    with open(f"{xs.CONFIG['io']['livraison']}variable-metadata_{v}_{filename}.json", 'w') as fp:
                        json.dump(ds[v].attrs, fp)

    if xs.CONFIG["tasks"]["extract_dams"]:
        def closest_line(lines, point):
            # get distances
            distance_list = [line.distance(point) for line in lines]
            shortest_distance = min(distance_list)  # find the line closest to the point
            return distance_list.index(shortest_distance)

        repertoire_barrages = pd.read_csv(f"{xs.CONFIG['gis']}repertoire_des_barrages.csv")
        # Remove bad matches
        # North & Cascades Savard
        repertoire_barrages = repertoire_barrages.loc[repertoire_barrages["Latitude"] < 51]
        # Riviere des Prairies
        repertoire_barrages = repertoire_barrages.loc[~((repertoire_barrages["Longitude"] > -73.7) & (repertoire_barrages["Longitude"] < -73.6) &
                                                        (repertoire_barrages["Latitude"] > 45.5) * (repertoire_barrages["Latitude"] < 45.6))]
        # Beauharnois / Les Cèdres
        repertoire_barrages = repertoire_barrages.loc[~((repertoire_barrages["Longitude"] > -74.2) & (repertoire_barrages["Longitude"] < -73.8) &
                                                        (repertoire_barrages["Latitude"] > 45.3) * (repertoire_barrages["Latitude"] < 45.4))]
        # Rapide des Iles/Quinze
        repertoire_barrages = repertoire_barrages.loc[~((repertoire_barrages["Longitude"] > -79.4) & (repertoire_barrages["Longitude"] < -79.2) &
                                                        (repertoire_barrages["Latitude"] > 47.5) * (repertoire_barrages["Latitude"] < 47.6))]

        shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
        shp_portrait = shp_portrait.set_index("TRONCON")

        troncons_all = {r["Centrale"]: shp_portrait.iloc[closest_line(shp_portrait.geometry, Point(r["Longitude"], r["Latitude"]))].name for _, r in repertoire_barrages.iterrows()}
        troncons = {v: k for k, v in troncons_all.items()}

        # PLOT
        # import matplotlib.pyplot as plt
        # import cartopy.crs
        # import cartopy.feature
        # plt.figure()
        # ax = plt.subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
        # crs_proj4 = ax.projection.proj4_init
        # shp_portrait.to_crs(crs_proj4).plot(ax=ax, color="k")
        # ax.add_feature(cartopy.feature.RIVERS)
        # ax.add_feature(cartopy.feature.LAKES)
        # ax.add_feature(cartopy.feature.LAND, color="#f0f0f0", zorder=0)
        # ax.add_feature(cartopy.feature.OCEAN, zorder=0)
        # shp_portrait.loc[troncons].to_crs(crs_proj4).plot(color="red", ax=ax)
        # for _, r in repertoire_barrages.iterrows():
        #     plt.scatter(r["Longitude"], r["Latitude"], color="green")

        for dataset in xs.CONFIG["datasets"]:
            if dataset == "ref":
                ds_dict = pcat.search(id=".*Portrait.*", processing_level="indicators").to_dataset_dict()
            else:
                ds_dict = pcat.search(type=".*hydro.*", processing_level=f"{dataset}.*").to_dataset_dict()

            for key, ds in ds_dict.items():
                if (ds.attrs["cat:xrfreq"] == "AS-JAN") and (dataset == "ref"):
                    ds["season_length"] = ds["season_end"] - ds["season_start"]
                    ds["season_length"].attrs["units"] = "d"
                # load coordinates and subset
                [ds[c].load() for c in ds.coords]
                ds = ds.assign_coords({"centrale": ds.station_id.to_series().map(troncons)})
                ds = ds.where(~ds["centrale"].isnull(), drop=True).squeeze()

                # Cleanup
                ds = xs.clean_up(ds, change_attr_prefix="dataset:", attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})

                # Prepare CSV
                if dataset == "ref":
                    filename = f"{ds.attrs['dataset:impact_model_source']}_{ds.attrs['dataset:impact_model_project']}"
                elif "analog" in dataset:
                    filename = f"{ds.attrs['dataset:impact_model_source']}_{ds.attrs['dataset:impact_model_project']}_{ds.attrs['dataset:processing_level']}degC"
                else:
                    filename = f"{ds.attrs['dataset:activity']}_{ds.attrs['dataset:impact_model_project']}_{ds.attrs['dataset:processing_level']}degC"

                # Write some metadata
                os.makedirs(xs.CONFIG['io']['livraison'], exist_ok=True)
                with open(f"{xs.CONFIG['io']['livraison']}dataset-metadata_{filename}.json", 'w') as fp:
                    json.dump(ds.attrs, fp)

                for v in ds.data_vars:
                    df = ds[v].swap_dims({"station": "centrale"}).to_pandas()
                    if isinstance(df, pd.Series):
                        df = df.to_frame(name=v)
                    if df.index.name != "centrale":
                        df = df.T

                    # Add dams that are on the same reach as another
                    for t in set(troncons_all).difference(list(troncons.values())):
                        if troncons_all[t] in list(troncons.keys()):
                            centrale = ds.where(ds["station_id"] == troncons_all[t], drop=True).centrale
                            if len(centrale != 0):
                                data = df.loc[[str(centrale.isel(station=0).values)]]
                                if len(data.index) > 1:
                                    data = data.iloc[[0, ]]
                                df = pd.concat((df, pd.DataFrame(data.values, index=[t], columns=df.columns)))

                    df.to_csv(f"{xs.CONFIG['io']['livraison']}{v}_{filename}_BARRAGES.csv")

                    with open(f"{xs.CONFIG['io']['livraison']}variable-metadata_{v}_{filename}.json", 'w') as fp:
                        json.dump(ds[v].attrs, fp)

    if xs.CONFIG["tasks"]["extract_all"]:
        with open(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_per_reach.json", 'r') as fp:
            tmp = json.load(fp)
        regions = {}
        for k, v in tmp.items():
            for vv in v:
                regions[vv] = k
        for dataset in xs.CONFIG["datasets"]:
            if "ref" in dataset:
                ds_dict = pcat.search(id=".*Portrait.*", processing_level="indicators" if "temporalblend" not in dataset else "temporalblend-indicators").to_dataset_dict()
            else:
                ds_dict = pcat.search(type=".*hydro.*", processing_level=f"{dataset}.*").to_dataset_dict()

            for key, ds in ds_dict.items():
                if (ds.attrs["cat:xrfreq"] == "AS-JAN") and (dataset == "ref"):
                    ds["season_length"] = ds["season_end"] - ds["season_start"]
                    ds["season_length"].attrs["units"] = "d"
                # load coordinates and subset
                [ds[c].load() for c in ds.coords]
                stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
                stations_atlas = stations_atlas.loc[stations_atlas["MASQUE"] != 0]  # Remove invalid/fake stations
                ds = ds.where(ds.station_id.isin(list(stations_atlas["TRONCON_ID"])), drop=True)

                ds = ds.assign_coords({"SIGLE": ds.station_id.to_series().map(regions)}).squeeze()
                # ds_zg = ds.groupby(ds.SIGLE).mean(dim="station")

                # data = [ds, ds_zg]
                data = [ds]
                for i in range(len(data)):

                    # Cleanup
                    out = xs.clean_up(data[i], change_attr_prefix="dataset:", attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})
                    out.attrs['dataset:processing_level'] = out.attrs['dataset:processing_level'].replace("temporalblend-analog-2021-", "temporalblend-analog-")

                    # Prepare CSV
                    if "ref" in dataset:
                        filename = f"{out.attrs['dataset:impact_model_source']}_{out.attrs['dataset:impact_model_project']}_{out.attrs['dataset:processing_level']}"
                    else:
                        filename = f"{out.attrs['dataset:impact_model_source']}_{out.attrs['dataset:impact_model_project']}_{out.attrs['dataset:processing_level']}degC"

                    # Write some metadata
                    os.makedirs(xs.CONFIG['io']['livraison'], exist_ok=True)
                    with open(f"{xs.CONFIG['io']['livraison']}dataset-metadata_{filename}.json", 'w') as fp:
                        json.dump(out.attrs, fp)

                    for v in out.data_vars:
                        if i == 0:
                            df = out[v].swap_dims({"station": "station_id"}).to_pandas()
                        else:
                            df = out[v].to_pandas()
                        if isinstance(df, pd.Series):
                            df = df.to_frame(name=v)
                        if df.index.name != ("station_id" if i == 0 else "SIGLE"):
                            df = df.T
                        df.to_csv(f"{xs.CONFIG['io']['livraison']}{v}_{filename}_{'ATLAS-PRIO2021' if i == 0 else 'ZGIEBV'}.csv")

                        with open(f"{xs.CONFIG['io']['livraison']}variable-metadata_{v}_{filename}.json", 'w') as fp:
                            json.dump(out[v].attrs, fp)


if __name__ == '__main__':
    main()
