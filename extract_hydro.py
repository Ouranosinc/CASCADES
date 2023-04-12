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

from utils import fix_infocrue, atlas_radeau_common, sort_analogs

xs.load_config("paths.yml", "configs/cfg_hydro.yml")

def main():

    def _extract_func(wl=None, analog=None):
        ds = xr.open_zarr(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}.zarr")

        # RADEAU
        stations = atlas_radeau_common()
        stations["ATLAS2018"] = stations["ATLAS2018"].str[0:3].str.cat(["0"] * len(stations["ATLAS2018"])).str.cat(
            stations["ATLAS2018"].str[3:])  # The RADEAU shapefile has 1 too many 0s
        cv = dict(zip(stations["TRONCON_ID"], stations["ATLAS2018"]))

        # Official Atlas2022
        # atlas_meta = pd.read_csv(xs.CONFIG["dpphc"]["meta_atlas2022"], encoding='latin-1')

        # load coordinates and subset
        [ds[c].load() for c in ds.coords]
        if "percentile" in ds:
            ds = ds.sel(time=slice(xs.CONFIG["extract"]["periods"][0], xs.CONFIG["extract"]["periods"][1]), percentile=50)
        else:
            ds.attrs["cat:driving_model"] = "CanESM2"
            # ref7q2 = xs.subset_warming_level(ds, wl=0.91, window=30).squeeze()
            ds = xs.subset_warming_level(ds, wl=wl, window=30).squeeze()
        ds = ds.where(ds.station_id.isin(stations["TRONCON_ID"].to_list()) | (ds.drainage_area > 25), drop=True)
        # ds = ds.where(ds.station_id.isin(stations["TRONCON_ID"].to_list() + atlas_meta.loc[atlas_meta["MASQUE"] == 2]["TRONCON_ID"].to_list()) | (ds.drainage_area > 1000), drop=True)
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
            # if wl not in [None, 0.91]:
            #     ind_dict_ref = dict()
            #     ind_dict_ref["fx"] = pcat.search(id=ds["cat:id"], processing_level="indicators-0.91").to_dask()

            if xs.CONFIG["tasks"]["additional_indicators"]:
                if "days_under_7q2" in xs.CONFIG["additional_indicators"]:
                    bool_under_eco = (ds.discharge < ind_dict["fx"]["7q2"].squeeze()).where(ds.discharge.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]),
                                                                                            other=False)
                    ind_dict['AS-JAN']["days_under_7q2"] = bool_under_eco.resample({"time": "AS-JAN"}).sum(dim="time")
                    ind_dict['AS-JAN']["days_under_7q2"].attrs = {"long_name": "Number of days below the 7Q2.",
                                                                  "description": "Streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                  "units": "d"}
                    ind_dict['AS-JAN']["max_consecutive_days_under_7q2"] = xcrl.longest_run(bool_under_eco, freq="AS-JAN")
                    ind_dict['AS-JAN']["max_consecutive_days_under_7q2"].attrs = {"long_name": "Maximum consecutive number of days below the 7Q2.",
                                                                                  "description": "Maximum consecutive streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                                  "units": "d"}

                    # if wl not in [None, 0.91]:
                    #     bool_under_eco = (ds.discharge < ind_dict_ref["fx"]["7q2"].squeeze()).where(ds.discharge.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)
                    #     ind_dict['AS-JAN']["days_under_ref7q2"] = bool_under_eco.resample({"time": "AS-JAN"}).sum(dim="time")
                    #     ind_dict['AS-JAN']["days_under_ref7q2"].attrs = {"long_name": "Number of days below the 7Q2.",
                    #                                                      "description": "Streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                    #                                                      "units": "d"}
                    #     ind_dict['AS-JAN']["max_consecutive_days_under_ref7q2"] = xcrl.longest_run(bool_under_eco, freq="AS-JAN")
                    #     ind_dict['AS-JAN']["max_consecutive_days_under_ref7q2"].attrs = {
                    #         "long_name": "Maximum consecutive number of days below the 7Q2.",
                    #         "description": "Maximum consecutive streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                    #         "units": "d"}

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
                        q14_a = q14.sel(time=q14.time.dt.year.isin(analog))
                        with xclim.set_options(data_validation="log"):
                            ind_dict['AS-JAN']["doy_14qmax"] = xcl.doy_qmax(q14_a, freq="AS-JAN")
                        ind_dict['AS-JAN']["doy_14qmax"].attrs["long_name"] = "Dayofyear of the maximum 14-day discharge"
                        ind_dict['AS-JAN']["doy_14qmax"].attrs["description"] = "Dayofyear of the maximum 14-day discharge for month[2, 3, 4, 5, 6]"
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

                    # if wl not in [None, 0.91]:
                    #     thresh = ds.discharge.mean(dim="time") - (ds.discharge.mean(dim="time") - ind_dict_ref["fx"]["7q2"].squeeze()) * 0.85
                    #     bool_under_low = (ds.discharge < thresh).where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)
                    #     ind_dict['AS-JAN']["season_start_ref"] = xcrl.first_run(bool_under_low, window=7, coord="dayofyear", freq="AS-JAN")
                    #     ind_dict['AS-JAN']["season_start_ref"].attrs = {
                    #         "long_name": "First dayofyear where the discharge is below 15% of the mean annual flow for 7 consecutive days",
                    #         "units": "dayofyear"}
                    #
                    #     ind_dict['AS-JAN']["season_end_ref"] = bool_under_low.resample({"time": "AS-JAN"}).map(xcrl.last_run, window=7, coord="dayofyear")
                    #     ind_dict['AS-JAN']["season_end_ref"].attrs = {
                    #         "long_name": "Last dayofyear where the 7-day discharge is below 15% of the mean annual flow for 7 consecutive days",
                    #         "units": "dayofyear"}

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

    hcat = xs.DataCatalog(xs.CONFIG["dpphc"]["atlas2022"])
    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"])

    if xs.CONFIG["tasks"]["extract"]:
        if "ref" in xs.CONFIG["datasets"]:
            if not pcat.exists_in_cat(type="reconstruction-hydro", processing_level="indicators"):
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
                        perf = pcat.search(processing_level=f"{dparts[0]}-performance-vs-{dparts[1]}-{dparts[2]}").to_dask()
                        best = sort_analogs(perf.rmse)[0:5]
                        for r, y in zip(best.realization, best.time):
                            analogs[str(r.realization.values).split(".")[0]].extend([int(y.dt.year.values)])

                if len(analogs) > 0:
                    for a in analogs.keys():
                        if not pcat.exists_in_cat(type="simulation-hydro", processing_level=f"indicators-{wl}", member=a.split("_")[-1]):
                            # Open the simulation
                            cat = hcat.search(type="simulation-hydro", processing_level="raw", member=a.split("_")[-1])
                            if not os.path.isdir(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}.zarr"):
                                fix_infocrue(cat)

                            with Client(**xs.CONFIG["dask"]) as c:
                                _extract_func(wl=wl, analog=analogs[a])

    if xs.CONFIG["tasks"]["extract_radeau"]:
        if "ref" in xs.CONFIG["datasets"]:
            ds_dict = pcat.search(id=".*Portrait.*", processing_level="indicators").to_dataset_dict()

            for key, ds in ds_dict.items():
                # load coordinates and subset
                [ds[c].load() for c in ds.coords]
                ds = ds.where(ds["atlas2018"] != '', drop=True).squeeze()

                # Cleanup
                ds = xs.clean_up(ds, change_attr_prefix="dataset:", attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})

                # Prepare CSV
                filename = f"{xs.CONFIG['io']['livraison']}{ds.attrs['dataset:id']}_RADEAU_{ds.attrs['dataset:processing_level']}_{ds.attrs['dataset:frequency']}"

                # Write some metadata
                os.makedirs(xs.CONFIG['io']['livraison'], exist_ok=True)
                with open(f"{filename}_metadata.json", 'w') as fp:
                    json.dump(ds.attrs, fp)

                for v in ds.data_vars:
                    df = ds[v].swap_dims({"station": "atlas2018"}).to_pandas()
                    if df.index.name != "atlas2018":
                        df = df.T
                    df.to_csv(f"{filename}_{v}.csv")

                    with open(f"{filename}_{v}_metadata.json", 'w') as fp:
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

        if "ref" in xs.CONFIG["datasets"]:
            ds_dict = pcat.search(id=".*Portrait.*", processing_level="indicators").to_dataset_dict()

            for key, ds in ds_dict.items():
                # load coordinates and subset
                [ds[c].load() for c in ds.coords]
                ds = ds.assign_coords({"centrale": ds.station_id.to_series().map(troncons)})
                ds = ds.where(~ds["centrale"].isnull(), drop=True).squeeze()

                # Cleanup
                ds = xs.clean_up(ds, change_attr_prefix="dataset:", attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})

                # Prepare CSV
                filename = f"{xs.CONFIG['io']['livraison']}{ds.attrs['dataset:id']}_BARRAGES_{ds.attrs['dataset:processing_level']}_{ds.attrs['dataset:frequency']}"

                # Write some metadata
                os.makedirs(xs.CONFIG['io']['livraison'], exist_ok=True)
                with open(f"{filename}_metadata.json", 'w') as fp:
                    json.dump(ds.attrs, fp)

                for v in ds.data_vars:
                    df = ds[v].swap_dims({"station": "centrale"}).to_pandas()
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
                    df.to_csv(f"{filename}_{v}.csv")

                    with open(f"{filename}_{v}_metadata.json", 'w') as fp:
                        json.dump(ds[v].attrs, fp)


if __name__ == '__main__':
    main()
