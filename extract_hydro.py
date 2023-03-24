from distributed import Client
import os
import xarray as xr
import xscen as xs
import pandas as pd
import xclim.indices.run_length as xcrl
import xclim.indicators.land as xcl
from xclim.core.calendar import convert_calendar
import json

from utils import fix_infocrue, atlas_radeau_common

xs.load_config("paths.yml", "configs/cfg_radeau.yml")

def main():

    hcat = xs.DataCatalog(xs.CONFIG["dpphc"]["atlas2022"])
    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"])

    if xs.CONFIG["tasks"]["extract"]:
        if "ref" in xs.CONFIG["datasets"]:
            # Open the reference
            cat = hcat.search(type="reconstruction-hydro", processing_level="raw")
            if not os.path.isdir(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}.zarr"):
                fix_infocrue(cat)

            with Client(**xs.CONFIG["dask"]) as c:
                ds = xr.open_zarr(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}.zarr")

                # RADEAU
                stations = atlas_radeau_common()
                stations["ATLAS2018"] = stations["ATLAS2018"].str[0:3].str.cat(["0"] * len(stations["ATLAS2018"])).str.cat(stations["ATLAS2018"].str[3:])  # The RADEAU shapefile has 1 too many 0s
                cv = dict(zip(stations["TRONCON_ID"], stations["ATLAS2018"]))

                # Official Atlas2022
                atlas_meta = pd.read_csv(xs.CONFIG["dpphc"]["meta_atlas2022"], encoding='latin-1')

                # load coordinates and subset
                [ds[c].load() for c in ds.coords]
                ds = ds.sel(time=slice(xs.CONFIG["extract"]["periods"][0], xs.CONFIG["extract"]["periods"][1]), percentile=50)
                ds = ds.where(ds.station_id.isin(stations["TRONCON_ID"].to_list() + atlas_meta.loc[atlas_meta["MASQUE"] == 2]["TRONCON_ID"].to_list()), drop=True)
                ds = ds.chunk({"station": 500, "time": -1})

                # Add the Atlas2018 name
                ds = ds.assign_coords({"atlas2018": ds.station_id.to_series().map(cv)})

                # Indicators
                if xs.CONFIG["tasks"]["indicators"]:
                    ds = convert_calendar(ds, '365_day')
                    ind_dict = xs.compute_indicators(ds, indicators="configs/indicators_radeau.yml")

                    if xs.CONFIG["tasks"]["additional_indicators"]:
                        if "days_under_7q2" in xs.CONFIG["additional_indicators"]:
                            bool_under_eco = (ds.discharge < ind_dict["fx"]["7q2"].squeeze()).where(ds.discharge.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)
                            ind_dict['AS-DEC']["days_under_7q2"] = bool_under_eco.resample({"time": "AS-DEC"}).sum(dim="time")
                            ind_dict['AS-DEC']["days_under_7q2"].attrs = {"long_name": "Number of days below the 7Q2.",
                                                                          "description": "Streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                          "units": "d"}
                            ind_dict['AS-DEC']["max_consecutive_days_under_7q2"] = xcrl.longest_run(bool_under_eco, freq="AS-DEC")
                            ind_dict['AS-DEC']["max_consecutive_days_under_7q2"].attrs = {"long_name": "Maximum consecutive number of days below the 7Q2.",
                                                                          "description": "Maximum consecutive streamflow under 7Q2 for month[5, 6, 7, 8, 9, 10, 11].",
                                                                          "units": "d"}

                        if "7qmin" in xs.CONFIG["additional_indicators"]:
                            ind_dict['AS-DEC']["7qmin"] = ds.discharge.rolling({"time": 7}, center=True).mean(keep_attrs=True)\
                                .where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11])).resample({"time": "AS-DEC"}).min(dim="time")
                            ind_dict['AS-DEC']["7qmin"].attrs["long_name"] = "Minimum 7-day discharge"
                            ind_dict['AS-DEC']["7qmin"].attrs["description"] = "Minimum 7-day discharge for month[5, 6, 7, 8, 9, 10, 11]."

                        if "freshet" in xs.CONFIG["additional_indicators"]:
                            q14 = ds.discharge.rolling({"time": 14}, center=True).mean(keep_attrs=True).where(ds.time.dt.month.isin([2, 3, 4, 5, 6]), other=0)
                            ind_dict['AS-DEC']["14qmax"] = q14.resample({"time": "AS-DEC"}).max(dim="time")
                            ind_dict['AS-DEC']["14qmax"].attrs["long_name"] = "Maximum 14-day discharge"
                            ind_dict['AS-DEC']["14qmax"].attrs["description"] = "Maximum 14-day discharge for month[2, 3, 4, 5, 6]"
                            ind_dict['AS-DEC']["doy_14qmax"] = xcl.doy_qmax(q14, freq="AS-DEC")
                            ind_dict['AS-DEC']["doy_14qmax"].attrs["long_name"] = "Dayofyear of the maximum 14-day discharge"
                            ind_dict['AS-DEC']["doy_14qmax"].attrs["description"] = "Dayofyear of the maximum 14-day discharge for month[2, 3, 4, 5, 6]"
                            ind_dict['AS-DEC']["doy_14qmax"].attrs["units"] = "dayofyear"

                        if "lowflow_season" in xs.CONFIG["additional_indicators"]:
                            thresh = ds.discharge.mean(dim="time") - (ds.discharge.mean(dim="time") - ind_dict["fx"]["7q2"].squeeze()) * 0.85
                            bool_under_low = (ds.discharge < thresh).where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]), other=False)
                            ind_dict['AS-DEC']["season_start"] = xcrl.first_run(bool_under_low, window=7, coord="dayofyear", freq="AS-DEC")
                            ind_dict['AS-DEC']["season_start"].attrs = {
                                "long_name": "First dayofyear where the discharge is below 15% of the mean annual flow for 7 consecutive days",
                                "units": "dayofyear"}

                            ind_dict['AS-DEC']["season_end"] = bool_under_low.resample({"time": "AS-DEC"}).map(xcrl.last_run, window=7, coord="dayofyear")
                            ind_dict['AS-DEC']["season_end"].attrs = {
                                "long_name": "Last dayofyear where the 7-day discharge is below 15% of the mean annual flow for 7 consecutive days",
                                "units": "dayofyear"}

                else:
                    ind_dict = {"D": ds}

                for freq, out in ind_dict.items():
                    if freq == "AS-DEC":
                        out["time"] = xr.DataArray(pd.date_range(start=str(int(xs.CONFIG['extract']['periods'][0]) + 1),
                                                                 end=str(int(xs.CONFIG['extract']['periods'][1]) + 1), freq="YS"),
                                                   coords={"time": pd.date_range(start=str(int(xs.CONFIG['extract']['periods'][0]) + 1),
                                                                                 end=str(int(xs.CONFIG['extract']['periods'][1]) + 1), freq="YS")})
                    if freq != "fx":
                        out = out.sel(time=slice(f"{xs.CONFIG['storylines']['ref_period'][0]}-01-01", f"{xs.CONFIG['storylines']['ref_period'][1]}-12-31"))

                    [out[c].load() for c in out.coords]
                    # Prepare output
                    filename = f"{xs.CONFIG['io']['stats']}{out.attrs['cat:id']}_{out.attrs['cat:processing_level']}_{out.attrs['cat:frequency']}.zarr"
                    xs.save_to_zarr(out, filename, mode="a")
                    pcat.update_from_ds(out, filename)

                    if (xs.CONFIG["tasks"]["deltas"]) and freq != "fx":
                        out = xr.open_zarr(filename).astype('float32')
                        out = out.drop_vars([v for v in out.data_vars if "delta" in v])

                        ref = xs.climatological_mean(out, min_periods=20)
                        deltas = xs.compute_deltas(ds=out, reference_horizon=ref, kind="+")
                        for v in deltas.data_vars:
                            out[v] = deltas[v]

                        # Prepare output
                        xs.save_to_zarr(out, filename, mode="a")
                        pcat.update_from_ds(out, filename)

    # ds_dict = pcat.search(id=".*Portrait.*", processing_level="indicators", xrfreq=["AS-DEC"]).to_dataset_dict()
    # for key, ds in ds_dict.items():
    #     out = xr.Dataset()
    #     ds = ds.drop_vars([v for v in ds.data_vars if "delta" in v])
    #     for v in ds.data_vars:
    #         if ds[v].dtype == '<m8[ns]':
    #             attrs = ds[v].attrs
    #             ds[v] = ds[v].dt.days
    #             ds[v].attrs = attrs
    #             ds[v].attrs["units"] = "d"
    #             out[v] = ds[v]
    #     filename = ds_dict[key].attrs["cat:path"]
    #     ref = xs.climatological_mean(ds, min_periods=20)
    #     deltas = xs.compute_deltas(ds=ds, reference_horizon=ref, kind="+")
    #     for v in ref.data_vars:
    #         if "delta" not in v:
    #             out[f"{v}_delta_1992_2021"] = deltas[f"{v}_delta_1992_2021"]
    #     out.attrs = ds.attrs
    #     xs.save_to_zarr(out, filename, mode="o")


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


if __name__ == '__main__':
    main()
