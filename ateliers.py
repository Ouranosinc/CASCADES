import xclim
import geopandas as gpd
import xarray as xr
import xclim.core.units

import xscen as xs
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
plt.ion()

# xscen setup
xs.load_config("paths.yml", "configs/cfg_nhess.yml", reset=True)

shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")
shp_zg = shp_zg.set_index("SIGLE")


def main(todo):
    rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_0.91.nc", chunks={}).load()
    sorted_analogs_hist = sort_analogs(rmse)
    rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_2.nc", chunks={}).load()
    sorted_analogs_2 = sort_analogs(rmse)
    rmse = xr.open_dataset(Path(xs.CONFIG["io"]["nhess_data"]) / f"rmse_2021_3.nc", chunks={}).load()
    sorted_analogs_3 = sort_analogs(rmse)

    if "hydro" in todo:
        dc_hydro = xs.DataCatalog(xs.CONFIG["io"]["nhess_rawcat"])
        reaches = ["OUTV03263", "OUTV03331", "OUTV03291", "SLSO01106", "SLSO00958", "SLSO00969"]
        stations = ["040110", "040129", "040122", "030101", "030103", "030106"]

        # Open the datasets
        ds_ref = dc_hydro.search(variable=["streamflow"], type="reconstruction-hydro", periods=["1992", "2021"]).to_dataset(xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})
        ds_ref = ds_ref.sel(time=slice("1992", "2021"), station=reaches, percentile=50.0).compute()
        ds_ref = ds_ref.assign_coords({"station_id": xr.DataArray(stations, dims="station")})

        if not (Path(xs.CONFIG["io"]["ateliers"]) / "streamflow_ClimEx.nc").exists():
            ds_sim = dc_hydro.search(variable=["streamflow"], type="simulation-hydro", activity="ClimEx", hydrology_member="MG24HQ").to_dataset(create_ensemble_on=["member"], xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})
            ds_sim = ds_sim.sel(station=reaches)
            Path(xs.CONFIG["io"]["ateliers"]).mkdir(parents=True, exist_ok=True)
            ds_sim.to_netcdf(Path(xs.CONFIG["io"]["ateliers"]) / "streamflow_ClimEx.nc")
        ds_sim = xr.open_dataset(Path(xs.CONFIG["io"]["ateliers"]) / "streamflow_ClimEx.nc", chunks={})
        ds_sim = ds_sim.compute()

        # Get the analogs
        ds_hist_period, ds_hist_analog = prep_blend(ds_sim, sorted_analogs_hist, 0.91)
        ds_2_period, ds_2_analog = prep_blend(ds_sim, sorted_analogs_2, 2.0)
        ds_3_period, ds_3_analog = prep_blend(ds_sim, sorted_analogs_3, 3.0)

        # Compute the indicators
        ds_ref_ind = calc_ind_hydro(ds_ref, ds_ref.sel(time=slice("2021", "2021")))
        ds_hist_ind = calc_ind_hydro(ds_hist_period, ds_hist_analog, calc3=[ds_ref, ds_hist_period])
        ds_2_ind = calc_ind_hydro(ds_2_period, ds_2_analog, calchist=ds_hist_period, calc3=[ds_ref, ds_hist_period])
        ds_3_ind = calc_ind_hydro(ds_3_period, ds_3_analog, calchist=ds_hist_period, calc3=[ds_ref, ds_hist_period])

        # Construct the blend
        ind_rel = ['qmoy', 'qmin', '7qmin', '14qmax']
        ind_abs = ['ndays_lt_7q2', 'ndays_lt_3', 'season_start', 'season_end', 'season_duration']

        blend2_rel = ds_ref_ind[ind_rel] * (1 + (ds_2_ind[ind_rel].mean(dim="member") - ds_hist_ind[ind_rel].mean(dim="member")) / ds_hist_ind[ind_rel].mean(dim="member"))
        for v in ind_rel:
            blend2_rel[v].attrs = ds_ref_ind[v].attrs
            blend2_rel[f"{v}_anom"] = (blend2_rel[v] - ds_ref_ind[f"{v}_clim"]) / ds_ref_ind[f"{v}_clim"] * 100
            blend2_rel[f"{v}_anom"].attrs["units"] = "%"
        blend2_abs = ds_ref_ind[ind_abs] + (ds_2_ind[ind_abs].mean(dim="member") - ds_hist_ind[ind_abs].mean(dim="member"))
        for v in ind_abs:
            blend2_abs[v].attrs = ds_ref_ind[v].attrs
            blend2_abs[f"{v}_anom"] = blend2_abs[v] - ds_ref_ind[f"{v}_clim"]
            blend2_abs[f"{v}_anom"].attrs["units"] = ds_ref_ind[v].attrs.get("units", "")
        blend2 = xr.merge([blend2_rel, blend2_abs])

        blend3_rel = ds_ref_ind[ind_rel] * (1 + (ds_3_ind[ind_rel].mean(dim="member") - ds_hist_ind[ind_rel].mean(dim="member")) / ds_hist_ind[ind_rel].mean(dim="member"))
        for v in ind_rel:
            blend3_rel[v].attrs = ds_ref_ind[v].attrs
            blend3_rel[f"{v}_anom"] = (blend3_rel[v] - ds_ref_ind[f"{v}_clim"]) / ds_ref_ind[f"{v}_clim"] * 100
            blend3_rel[f"{v}_anom"].attrs["units"] = "%"
        blend3_abs = ds_ref_ind[ind_abs] + (ds_3_ind[ind_abs].mean(dim="member") - ds_hist_ind[ind_abs].mean(dim="member"))
        for v in ind_abs:
            blend3_abs[v].attrs = ds_ref_ind[v].attrs
            blend3_abs[f"{v}_anom"] = blend3_abs[v] - ds_ref_ind[f"{v}_clim"]
            blend3_abs[f"{v}_anom"].attrs["units"] = ds_ref_ind[v].attrs.get("units", "")
        blend3 = xr.merge([blend3_rel, blend3_abs])

        # Clean the indicators
        outh = xr.Dataset()
        for v in ds_ref_ind.data_vars:
            if "clim" in v:
                outh[f"{v.replace('_clim', '_1992-2021clim')}"] = ds_ref_ind[v]
            else:
                outh[f"{v}_2021"] = ds_ref_ind[v]
        for v in blend2.data_vars:
            outh[f"{v}_2021p2"] = blend2[v]
        for v in blend3.data_vars:
            outh[f"{v}_2021p3"] = blend3[v]
        outh = outh[[v for v in sorted(outh.data_vars)]]

        # Save
        outh = outh.swap_dims({"station": "station_id"})
        outh = outh.reset_coords(drop=True)

        for station in stations:
            df = outh.sel(station_id=station).to_dataframe()

            for v in df.columns:
                if "month" not in outh[v].dims:
                    # Put NaNs after the first line
                    df[v] = df[v].where(df.index.isin(["JAN"]), other=float("nan"))

                # Add units to the dataframe
                df = df.rename(columns={v: f"{v} ({outh[v].attrs.get('units', '')})"})
            df.to_csv(Path(xs.CONFIG["io"]["ateliers"]) / f"{station}.csv")

    if "climat" in todo:
        dc_reconstruction = xs.DataCatalog(xs.CONFIG["io"]["nhess_rawcat"].replace("INFO-Crue-hydro", "reconstruction"))
        dc_sim = xs.DataCatalog(xs.CONFIG["io"]["nhess_rawcat"].replace("INFO-Crue-hydro", "INFO-Crue-climat"))
        watersheds = ['ABRINORD', 'COPERNIC']
        for w in watersheds:
            # Open the datasets
            ds_ref = dc_reconstruction.search(variable=["pr", "tasmax", "tasmin", "snw"], source="ERA5-Land", xrfreq="D", periods=["1992", "2021"]).to_dataset(xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})
            ds_ref = ds_ref.sel(time=slice("1992", "2021"))
            ds_ref = xs.spatial.subset(ds_ref, method="shape", tile_buffer=0.5, shape=gpd.GeoDataFrame(shp_zg.loc[[w]], geometry="geometry"))
            ds_ref = ds_ref.assign_coords({"ZGIEBV": w})
            ds_ref = ds_ref.compute()

            if not (Path(xs.CONFIG["io"]["ateliers"]) / f"{w}_ClimEx.nc").exists():
                dc_subset = dc_sim.search(variable=["tasmin", "tasmax", "pr"], type="simulation", activity="ClimEx", processing_level="biasadjusted")
                ds_sim = []
                for member in dc_subset.unique("member"):
                    ds = dc_subset.search(member=member).to_dataset(xarray_open_kwargs={"chunks": {}}, xarray_combine_kwargs={"coords": "minimal", "compat": "override"})
                    ds = xs.spatial.subset(ds, method="shape", tile_buffer=0.5, shape=gpd.GeoDataFrame(shp_zg.loc[[w]], geometry="geometry"))
                    ds = ds.assign_coords({"ZGIEBV": w, "member": member})
                    ds_sim.append(ds)
                ds_sim = xr.concat(ds_sim, dim="realization")
                Path(xs.CONFIG["io"]["ateliers"]).mkdir(parents=True, exist_ok=True)
                ds_sim.to_netcdf(Path(xs.CONFIG["io"]["ateliers"]) / f"{w}_ClimEx.nc")
            ds_sim = xr.open_dataset(Path(xs.CONFIG["io"]["ateliers"]) / f"{w}_ClimEx.nc", chunks={})
            ds_sim = ds_sim.compute()
            ds_sim = ds_sim.swap_dims({"realization": "member"}).rename({"member": "realization"})

            # Get the analogs
            ds_hist_period, ds_hist_analog = prep_blend(ds_sim, sorted_analogs_hist, 0.91)
            ds_2_period, ds_2_analog = prep_blend(ds_sim, sorted_analogs_2, 2.0)
            ds_3_period, ds_3_analog = prep_blend(ds_sim, sorted_analogs_3, 3.0)

            # Compute the indicators
            ds_ref_ind = calc_ind_clim(ds_ref, ds_ref.sel(time=slice("2021", "2021"))).squeeze()
            ds_hist_ind = calc_ind_clim(ds_hist_period, ds_hist_analog).squeeze()
            ds_2_ind = calc_ind_clim(ds_2_period, ds_2_analog).squeeze()
            ds_3_ind = calc_ind_clim(ds_3_period, ds_3_analog).squeeze()

            # Spatial mean
            if w == "ABRINORD":
                additional = {"SaintJerome": [-74.00786803028535, 45.78039453225813]}
            elif w == "COPERNIC":
                additional = {"Nicolet": [-72.60847836142717, 46.22229159121031], "Victoriaville": [-71.94290220979131, 46.04867449784397]}

            def _aggregate_mean(ds, additional):
                ds_mean = xs.aggregate.spatial_mean(ds, method="xesmf",
                                                    region={'method': 'shape', 'shape': gpd.GeoDataFrame(shp_zg.loc[[w]], geometry="geometry")},
                                                    kwargs={'skipna': True}, simplify_tolerance=0.01)
                ds_mean = ds_mean.expand_dims({"region": [w]})

                for k, v in additional.items():
                    tmp = xs.aggregate.spatial_mean(ds, method="xesmf",
                                                    region={'method': 'bbox', 'lon_bnds': [v[0]-0.1, v[0]+0.1], 'lat_bnds': [v[1]-0.1, v[1]+0.1]},
                                                    kwargs={'skipna': True}, simplify_tolerance=0.01)
                    tmp = tmp.expand_dims({"region": [k]})
                    ds_mean = xr.concat([ds_mean, tmp], dim="region")
                return ds_mean
            ds_ref_ind = _aggregate_mean(ds_ref_ind, additional)
            ds_hist_ind = _aggregate_mean(ds_hist_ind, additional)
            ds_2_ind = _aggregate_mean(ds_2_ind, additional)
            ds_3_ind = _aggregate_mean(ds_3_ind, additional)

            # Construct the blend
            ind_rel = ['pr_sum', 'evap_sum']
            ind_abs = ['tasmax_mean', 'ndays_tasmax_gt_30', 'last_frost_date', 'gdd5']

            blend2_rel = ds_ref_ind[ind_rel] * (
                        1 + (ds_2_ind[ind_rel].mean(dim="member") - ds_hist_ind[ind_rel].mean(dim="member")) / ds_hist_ind[ind_rel].mean(
                dim="member"))
            # Add SWE
            with xr.set_options(keep_attrs=True):
                blend2_rel["swemax"] = ds_ref_ind["swemax"] * (1 - 0.075)  # Assuming a 7.5% decrease in SWE
            for v in ind_rel + ["swemax"]:
                blend2_rel[v].attrs = ds_ref_ind[v].attrs
                ds_ref_ind[f"{v}_anom"] = (ds_ref_ind[v] - ds_ref_ind[f"{v}_clim"]) / ds_ref_ind[f"{v}_clim"] * 100  # Recompute the anomalies as the climatology is now spatially averaged
                ds_ref_ind[f"{v}_anom"].attrs["units"] = "%"
                blend2_rel[f"{v}_anom"] = (blend2_rel[v] - ds_ref_ind[f"{v}_clim"]) / ds_ref_ind[f"{v}_clim"] * 100
                blend2_rel[f"{v}_anom"].attrs["units"] = "%"
            blend2_abs = ds_ref_ind[ind_abs] + (ds_2_ind[ind_abs].mean(dim="member") - ds_hist_ind[ind_abs].mean(dim="member"))
            for v in ind_abs:
                blend2_abs[v].attrs = ds_ref_ind[v].attrs
                ds_ref_ind[f"{v}_anom"] = ds_ref_ind[v] - ds_ref_ind[f"{v}_clim"]  # Recompute the anomalies as the climatology is now spatially averaged
                ds_ref_ind[f"{v}_anom"].attrs["units"] = ds_ref_ind[v].attrs["units"]
                blend2_abs[f"{v}_anom"] = blend2_abs[v] - ds_ref_ind[f"{v}_clim"]
                blend2_abs[f"{v}_anom"].attrs["units"] = ds_ref_ind[v].attrs["units"]
            blend2 = xr.merge([blend2_rel, blend2_abs])

            blend3_rel = ds_ref_ind[ind_rel] * (
                        1 + (ds_3_ind[ind_rel].mean(dim="member") - ds_hist_ind[ind_rel].mean(dim="member")) / ds_hist_ind[ind_rel].mean(
                    dim="member"))
            # Add SWE
            with xr.set_options(keep_attrs=True):
                blend3_rel["swemax"] = ds_ref_ind["swemax"] * (1 - 0.15)  # Assuming a 15% decrease in SWE
            for v in ind_rel + ["swemax"]:
                blend3_rel[v].attrs = ds_ref_ind[v].attrs
                blend3_rel[f"{v}_anom"] = (blend3_rel[v] - ds_ref_ind[f"{v}_clim"]) / ds_ref_ind[f"{v}_clim"] * 100
                blend3_rel[f"{v}_anom"].attrs["units"] = "%"
            blend3_abs = ds_ref_ind[ind_abs] + (ds_3_ind[ind_abs].mean(dim="member") - ds_hist_ind[ind_abs].mean(dim="member"))
            for v in ind_abs:
                blend3_abs[v].attrs = ds_ref_ind[v].attrs
                blend3_abs[f"{v}_anom"] = blend3_abs[v] - ds_ref_ind[f"{v}_clim"]
                blend3_abs[f"{v}_anom"].attrs["units"] = ds_ref_ind[v].attrs["units"]
            blend3 = xr.merge([blend3_rel, blend3_abs])

            # Clean the indicators
            outc = xr.Dataset()
            for v in ds_ref_ind.data_vars:
                if "clim" in v:
                    outc[f"{v.replace('_clim', '_1992-2021clim')}"] = ds_ref_ind[v]
                else:
                    outc[f"{v}_2021"] = ds_ref_ind[v]
            for v in blend2.data_vars:
                outc[f"{v}_2021p2"] = blend2[v]
            for v in blend3.data_vars:
                outc[f"{v}_2021p3"] = blend3[v]
            outc = outc[[v for v in sorted(outc.data_vars)]]

            # Save
            outc = outc.reset_coords(drop=True)

            for region in outc.region:
                df = outc.sel(region=region).to_dataframe()

                for v in df.columns:
                    if "month" not in outc[v].dims:
                        # Put NaNs after the first line
                        df[v] = df[v].where(df.index.isin(["JAN"]), other=float("nan"))

                    # Add units to the dataframe
                    df = df.rename(columns={v: f"{v} ({outc[v].attrs.get('units', '')})"})
                df.to_csv(Path(xs.CONFIG["io"]["ateliers"]) / f"{region.values}.csv")


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


def prep_blend(ds, analogs, warming_level):
    blend_period = []
    blend_analog = []
    for i in range(xs.CONFIG["storylines"]["n_analogs"]):
        realization = str(analogs.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
        analog_year = int(analogs.isel(stacked=i).time.dt.year.values)
        ds_period = ds.sel(realization=realization)
        ds_period.attrs["cat:member"] = realization
        ds_period = xs.subset_warming_level(ds_period, wl=warming_level, window=30).squeeze()
        ds_period = ds_period.convert_calendar("noleap")
        ds_period = ds_period.assign_coords({"member": i})

        # Get the analog year
        ds_analog = ds_period.sel(time=slice(f"{analog_year}", f"{analog_year}"))

        # Use the same years for the time dimension
        ds_period["time"] = xr.date_range(start=f"1992-01-01", end=f"2021-12-31", freq="D", calendar="noleap", use_cftime=True)
        ds_analog["time"] = xr.date_range(start=f"2021-01-01", end=f"2021-12-31", freq="D", calendar="noleap", use_cftime=True)

        blend_period.append(ds_period)
        blend_analog.append(ds_analog)

    ds_blend_period = xr.concat(blend_period, dim="member")
    ds_blend_analog = xr.concat(blend_analog, dim="member")
    return ds_blend_period, ds_blend_analog


def calc_ind_hydro(ds, ds_analog, calchist=None, calc3=None):
    out = xr.Dataset()

    # Monthly mean
    out["qmoy"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds_analog.streamflow, freq="MS", op="mean").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymean_clim_mean"]
    out["qmoy_clim"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds.streamflow, freq="MS", op="mean").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymean_clim_mean"]
    out["qmoy_anom"] = (out["qmoy"] - out["qmoy_clim"]) / out["qmoy_clim"] * 100
    out["qmoy_anom"].attrs["units"] = "%"

    # Monthly min
    out["qmin"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds_analog.streamflow, freq="MS", op="min").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymin_clim_mean"]
    out["qmin_clim"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds.streamflow, freq="MS", op="min").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymin_clim_mean"]
    out["qmin_anom"] = (out["qmin"] - out["qmin_clim"]) / out["qmin_clim"] * 100
    out["qmin_anom"].attrs["units"] = "%"

    # Monthly 7d min
    out["7qmin"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds_analog.streamflow.rolling(time=7, center=False).mean(), freq="MS", op="min").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymin_clim_mean"]
    out["7qmin_clim"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds.streamflow.rolling(time=7, center=False).mean(), freq="MS", op="min").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymin_clim_mean"]
    out["7qmin_anom"] = (out["7qmin"] - out["7qmin_clim"]) / out["7qmin_clim"] * 100
    out["7qmin_anom"].attrs["units"] = "%"

    # Monthly 14d max
    out["14qmax"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds_analog.streamflow.rolling(time=14, center=False).mean(), freq="MS", op="max").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymax_clim_mean"]
    out["14qmax_clim"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds.streamflow.rolling(time=14, center=False).mean(), freq="MS", op="max").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymax_clim_mean"]
    out["14qmax_anom"] = (out["14qmax"] - out["14qmax_clim"]) / out["14qmax_clim"] * 100
    out["14qmax_anom"].attrs["units"] = "%"

    # Number of days with streamflow < 7q2
    if calchist is None:
        ds4q2 = ds
    else:
        ds4q2 = calchist
    q2 = ds4q2.streamflow.rolling(time=7, center=False).mean()
    q2 = q2.where(q2.time.dt.month.isin([5, 6, 7, 8, 9, 10])).resample(time="1Y").min(dim="time").median(dim="time")
    out["ndays_lt_7q2"] = ds_analog.streamflow.where(ds_analog.time.dt.month.isin([5, 6, 7, 8, 9, 10]) & (ds_analog.streamflow < q2)).count(dim="time")
    out["ndays_lt_7q2"].attrs["units"] = "d"
    out["ndays_lt_7q2_clim"] = ds.streamflow.where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10]) & (ds.streamflow < q2)).resample(time="YS").count(dim="time")
    out["ndays_lt_7q2_clim"] = out["ndays_lt_7q2_clim"].where(out["ndays_lt_7q2_clim"] > 0).mean(dim="time")
    out["ndays_lt_7q2_clim"].attrs["units"] = "d"
    out["ndays_lt_7q2_anom"] = out["ndays_lt_7q2"] - out["ndays_lt_7q2_clim"]
    out["ndays_lt_7q2_anom"].attrs["units"] = "d"

    # Number of days with streamflow < 3 m3/s
    if calc3 is None:
        thresh = 3
    else:
        thresh = 3 * (1 + (calc3[1].streamflow.where(calc3[1].time.dt.month.isin([5, 6, 7, 8, 9, 10])).mean(dim="time") - calc3[0].streamflow.where(calc3[0].time.dt.month.isin([5, 6, 7, 8, 9, 10])).mean(dim="time")) / calc3[0].streamflow.where(calc3[0].time.dt.month.isin([5, 6, 7, 8, 9, 10])).mean(dim="time"))

    out["ndays_lt_3"] = ds_analog.streamflow.where(ds_analog.time.dt.month.isin([5, 6, 7, 8, 9, 10]) & (ds_analog.streamflow < thresh)).count(dim="time")
    out["ndays_lt_3"].attrs["units"] = "d"
    out["ndays_lt_3_clim"] = ds.streamflow.where(ds.time.dt.month.isin([5, 6, 7, 8, 9, 10]) & (ds.streamflow < thresh)).resample(time="YS").count(dim="time").mean(dim="time")
    out["ndays_lt_3_clim"].attrs["units"] = "d"
    out["ndays_lt_3_anom"] = out["ndays_lt_3"] - out["ndays_lt_3_clim"]
    out["ndays_lt_3_anom"].attrs["units"] = "d"

    # Season start, end and duration
    if calchist is None:
        ds4season = ds
    else:
        ds4season = calchist
    thresh = ds4season.streamflow.where(ds.streamflow.time.dt.month.isin([6, 7, 8])).median(dim="time")
    qroll = ds_analog.streamflow.rolling(time=7, center=False).mean()
    qroll = qroll.time.dt.dayofyear.where(qroll.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]) & (qroll < thresh))
    out["season_start"] = qroll.min(dim="time")
    out["season_end"] = qroll.max(dim="time")
    out["season_duration"] = out["season_end"] - out["season_start"]
    qroll = ds.streamflow.rolling(time=7, center=False).mean()
    qroll = qroll.time.dt.dayofyear.where(qroll.time.dt.month.isin([5, 6, 7, 8, 9, 10, 11]) & (qroll < thresh))
    out["season_start_clim"] = qroll.resample(time="YS").min().mean(dim="time")
    out["season_end_clim"] = qroll.resample(time="YS").max().mean(dim="time")
    out["season_duration_clim"] = out["season_end_clim"] - out["season_start_clim"]
    out["season_start_anom"] = out["season_start"] - out["season_start_clim"]
    out["season_end_anom"] = out["season_end"] - out["season_end_clim"]
    out["season_duration_anom"] = out["season_duration"] - out["season_duration_clim"]

    return out


def calc_ind_clim(ds, ds_analog):
    ds = xs.utils.change_units(ds, {"pr": "mm", "tasmax": "degC", "tasmin": "degC"})
    ds_analog = xs.utils.change_units(ds_analog, {"pr": "mm", "tasmax": "degC", "tasmin": "degC"})
    if "snw" in ds_analog.data_vars:
        ds_analog["snw"] = xclim.units.convert_units_to(ds_analog["snw"], "mm", context="hydro")
        ds["snw"] = xclim.units.convert_units_to(ds["snw"], "mm", context="hydro")

    out = xr.Dataset()

    # Monthly mean pr
    out["pr_sum"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds_analog.pr, freq="MS", op="sum").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlysum_clim_mean"]
    out["pr_sum_clim"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds.pr, freq="MS", op="sum").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlysum_clim_mean"]
    out["pr_sum_anom"] = (out["pr_sum"] - out["pr_sum_clim"]) / out["pr_sum_clim"] * 100

    # Monthly mean tasmax
    out["tasmax_mean"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds_analog.tasmax, freq="MS", op="mean").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymean_clim_mean"]
    out["tasmax_mean_clim"] = xs.climatological_op(
        xclim.indicators.generic.stats(ds.tasmax, freq="MS", op="mean").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlymean_clim_mean"]
    out["tasmax_mean_anom"] = out["tasmax_mean"] - out["tasmax_mean_clim"]

    # days with tasmax > 30 degC
    out["ndays_tasmax_gt_30"] = xclim.indicators.atmos.tx_days_above(ds_analog.tasmax, thresh="30 degC")
    out["ndays_tasmax_gt_30"].attrs["units"] = "d"
    out["ndays_tasmax_gt_30_clim"] = xclim.indicators.atmos.tx_days_above(ds.tasmax, thresh="30 degC").mean(dim="time")
    out["ndays_tasmax_gt_30_clim"].attrs["units"] = "d"
    out["ndays_tasmax_gt_30_anom"] = out["ndays_tasmax_gt_30"] - out["ndays_tasmax_gt_30_clim"]
    out["ndays_tasmax_gt_30_anom"].attrs["units"] = "d"

    # Last frost date
    out["last_frost_date"] = xclim.indicators.atmos.last_spring_frost(ds_analog.tasmin)
    out["last_frost_date_clim"] = xclim.indicators.atmos.last_spring_frost(ds.tasmin).mean(dim="time")
    out["last_frost_date_anom"] = out["last_frost_date"] - out["last_frost_date_clim"]

    # Evap
    evap_a = xclim.indicators.atmos.potential_evapotranspiration(tasmin=ds_analog.tasmin, tasmax=ds_analog.tasmax, lat=ds_analog.lat, method="mcguinnessbordne05")
    evap_a = xclim.units.rate2amount(evap_a, out_units="kg m-2")
    evap_a = xclim.units.convert_units_to(evap_a, "mm", context="hydro")
    evap = xclim.indicators.atmos.potential_evapotranspiration(tasmin=ds.tasmin, tasmax=ds.tasmax, lat=ds.lat, method="mcguinnessbordne05")
    evap = xclim.units.rate2amount(evap, out_units="kg m-2")
    evap = xclim.units.convert_units_to(evap, "mm", context="hydro")

    out["evap_sum"] = xs.climatological_op(
        xclim.indicators.generic.stats(evap_a, freq="MS", op="sum").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlysum_clim_mean"]
    out["evap_sum_clim"] = xs.climatological_op(
        xclim.indicators.generic.stats(evap, freq="MS", op="sum").to_dataset(),
        op="mean", horizons_as_dim=True, min_periods=0).squeeze()["stat_monthlysum_clim_mean"]
    out["evap_sum_anom"] = (out["evap_sum"] - out["evap_sum_clim"]) / out["evap_sum_clim"] * 100

    # Growing degree days
    tas_a = (ds_analog.tasmax + ds_analog.tasmin) / 2
    tas_a.attrs["units"] = "degC"
    tas = (ds.tasmax + ds.tasmin) / 2
    tas.attrs["units"] = "degC"
    out["gdd5"] = xclim.indicators.atmos.growing_degree_days(tas=tas_a, thresh='5 degC')
    out["gdd5_clim"] = xclim.indicators.atmos.growing_degree_days(tas=tas, thresh='5 degC').mean(dim="time")
    out["gdd5_anom"] = out["gdd5"] - out["gdd5_clim"]

    # SWE
    if "snw" in ds_analog.data_vars:
        out["swemax"] = ds_analog.snw.where(ds_analog.time.dt.month.isin([1, 2, 3, 4, 5])).max(dim="time")
        out["swemax"].attrs["units"] = "mm"
        out["swemax_clim"] = ds.snw.where(ds.time.dt.month.isin([1, 2, 3, 4, 5])).resample(time="YS").max(dim="time").mean(dim="time")
        out["swemax_anom"] = (out["swemax"] - out["swemax_clim"]) / out["swemax_clim"] * 100

    return out


if __name__ == '__main__':
    figures = ["hydro"]

    main(todo=figures)
