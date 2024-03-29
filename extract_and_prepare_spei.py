from distributed import Client
import os
import shutil
import numpy as np
import xarray as xr
import xscen as xs
import xclim
import xclim.indicators as xci

xs.load_config("paths.yml", "configs/cfg_spei.yml")

def main():

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=True, project=xs.CONFIG["project"], overwrite=False)

    if xs.CONFIG["tasks"]["extract"]:
        # Search
        cat_dict = xs.search_data_catalogs(**xs.CONFIG["extract"]["search_data_catalogs_ref"])
        cat_dict_sim = xs.search_data_catalogs(**xs.CONFIG["extract"]["search_data_catalogs_sim"])
        cat_dict |= cat_dict_sim  # Merge the two search results

        for key, cat in cat_dict.items():
            if not (pcat.exists_in_cat(source=cat.unique("source")[0], member=cat.unique("member")[0], processing_level="extracted") if "ClimEx" in key else \
                    pcat.exists_in_cat(id=key, processing_level="extracted")):

                with Client(**xs.CONFIG["dask"]) as c:

                    # Extract tas and pr
                    if "ClimEx" in key:
                        ds_dict = xs.extract_dataset(catalog=cat, region=xs.CONFIG["region"])
                        ds = ds_dict["D"]

                        # Regrid to ERA5-Land. Very important to do this here, otherwise evspsblpot makes dask explode in RAM
                        ds_grid = pcat.search(source="ERA5-Land", processing_level="extracted").to_dask()
                        chunks = xs.io.estimate_chunks(ds, ["time"], target_mb=125)
                        ds = ds.chunk(chunks)
                        ds = xs.regrid_dataset(ds, ds_grid, weights_location=f"{xs.CONFIG['io']['extract']}weights/", to_level="extracted")
                        ds["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}

                        if os.path.isdir(f"{xs.CONFIG['io']['extract']}tmp.zarr"):
                            shutil.rmtree(f"{xs.CONFIG['io']['extract']}tmp.zarr")
                        xs.save_to_zarr(ds, f"{xs.CONFIG['io']['extract']}tmp.zarr")
                        ds = xr.open_zarr(f"{xs.CONFIG['io']['extract']}tmp.zarr")
                    else:
                        ds_dict = xs.extract_dataset(catalog=cat, region=xs.CONFIG["region"])
                        ds = ds_dict["D"]

                    # Compute evspsblpot and wb (separated steps, because wb requires evspsbltot)
                    module = xs.indicators.load_xclim_module(xs.CONFIG["indicatorsWB"])

                    tmp = xs.compute_indicators(ds, [module.evspsblpot])
                    ds["evspsblpot"] = tmp["D"]["evspsblpot"]
                    # Append to temporary file
                    if "ClimEx" in key:
                        xs.save_to_zarr(ds, f"{xs.CONFIG['io']['extract']}tmp.zarr", mode="a")
                        ds = xr.open_zarr(f"{xs.CONFIG['io']['extract']}tmp.zarr")

                    tmp = xs.compute_indicators(ds, [module.water_budget])
                    ds["wb"] = tmp["D"]["wb"]
                    # Append to temporary file
                    if "ClimEx" in key:
                        xs.save_to_zarr(ds, f"{xs.CONFIG['io']['extract']}tmp.zarr", mode="a")
                        ds = xr.open_zarr(f"{xs.CONFIG['io']['extract']}tmp.zarr")

                    # Drop tas
                    ds = ds.drop_vars("tas")

                    # Ajust attributes
                    ds.attrs["cat:variable"] = tuple(v for v in ds.data_vars)
                    if "ClimEx" in key:
                        ds.attrs["cat:id"] = xs.catalog.generate_id(ds, id_columns=["activity", "driving_model", "source", "experiment", "member"]).values[0]

                    # Resample
                    ds = ds.resample({"time": "MS"}).mean(keep_attrs=True)
                    ds.attrs["cat:xrfreq"] = "MS"
                    ds.attrs["cat:frequency"] = "mon"

                    # Save
                    filename = f"{xs.CONFIG['io']['extract']}{ds.attrs['cat:id']}_{ds.attrs['cat:domain']}_{ds.attrs['cat:processing_level']}_{ds.attrs['cat:frequency']}.zarr"
                    chunks = xs.io.estimate_chunks(ds, ["lon"], target_mb=125)
                    xs.save_to_zarr(ds, filename, rechunk=chunks)
                    pcat.update_from_ds(ds, filename)

    if xs.CONFIG["tasks"]["indicators"]:
        def wrapped_spi(da, cal_period=None):
            if (da.time.dt.month[0] in good_months):
                with xclim.set_options(check_missing="skip", data_validation="log"):
                    return xci.atmos.standardized_precipitation_evapotranspiration_index(wb=da, wb_cal=da if cal_period is None else da.sel(time=slice(cal_period[0], cal_period[1])),
                                                                                         freq="MS", window=1,
                                                                                         method="ML", dist="fisk").clip(min=-3.09, max=3.09, keep_attrs=True)
            else:
                return da * np.NaN

        # Search
        ds_dict = pcat.search(processing_level="extracted", variable="wb").to_dataset_dict()
        for key, ds in ds_dict.items():
            for wl in xs.CONFIG["spei"]["cal_period"] if "ERA5" in key.split(".")[0] else xs.CONFIG["spei"]["warming_levels"]:

                if not pcat.exists_in_cat(id=key.split(".")[0], processing_level="indicators" if "ERA5" in key.split(".")[0] else f'indicators-warminglevel-{wl}vs1850-1900'):

                    with Client(**xs.CONFIG["dask"]) as c:
                        # Compute SPEI per desired warming level

                            if "ERA5" in key.split(".")[0]:
                                ds_wl = ds
                                cal_period = wl
                            else:
                                ds.attrs["cat:mip_era"] = "CMIP5"
                                ds_wl = xs.subset_warming_level(ds, wl=wl, window=30).squeeze()
                                cal_period = None

                            window = xs.CONFIG["spei"]["windows"][0]
                            good_months = [m for m in xs.CONFIG["spei"]["good_months"] if m - window >= -1]
                            with xr.set_options(keep_attrs=True):
                                da_wl = xr.where(ds_wl["time.month"].isin(good_months), ds_wl["wb"].rolling({"time": window}).mean(), np.NaN)

                            ind = da_wl.groupby("time.month").map(lambda x: wrapped_spi(x, cal_period=cal_period))
                            ind.attrs = {
                                'units': '',
                                'calibration_period': cal_period if "ERA5" in key.split(".")[0] else [str(ind.time.dt.year[0].values), str(ind.time.dt.year[-1].values)],
                                'cell_methods': 'pr: time: mean evspsblpot: tasmin: time:  minimum tasmax: time:  maximum time: mean within days',
                                'history': f"{ds['wb'].attrs['history']}\n[2023-01-xx] spei: SPEI(wb=month, wb_cal=month, freq='MS', window={window}, dist='fisk', method='ML') with options check_missing=skip - xclim version: 0.40.0",
                                'standard_name': 'spei',
                                'long_name': 'Standardized precipitation evapotranspiration index (spei)',
                                'description': f'Water budget (precipitation minus evapotranspiration) over a moving {window}-x window, normalized such that spei averages to 0 for calibration data. the window unit `x` is the minimal time period defined by the resampling frequency monthly.'
                            }
                            ind = ind.to_dataset()
                            ind = ind.rename({"month": f"spei{window}"})
                            ind.attrs = ds.attrs

                            for window in xs.CONFIG["spei"]["windows"][1:]:
                                good_months = [m for m in xs.CONFIG["spei"]["good_months"] if m - window >= -1]
                                with xr.set_options(keep_attrs=True):
                                    da_wl = xr.where(ds_wl["time.month"].isin(good_months), ds_wl["wb"].rolling({"time": window}).mean(), np.NaN)
                                ind[f"spei{window}"] = da_wl.groupby("time.month").map(lambda x: wrapped_spi(x, cal_period=cal_period))
                                ind[f"spei{window}"].attrs = {
                                    'units': '',
                                    'calibration_period': cal_period if "ERA5" in key.split(".")[0] else [str(ind.time.dt.year[0].values), str(ind.time.dt.year[-1].values)],
                                    'cell_methods': 'pr: time: mean evspsblpot: tasmin: time:  minimum tasmax: time:  maximum time: mean within days',
                                    'history': f"{ds['wb'].attrs['history']}\n[2023-01-xx] spei: SPEI(wb=month, wb_cal=month, freq='MS', window={window}, dist='fisk', method='ML') with options check_missing=skip - xclim version: 0.40.0",
                                    'standard_name': 'spei',
                                    'long_name': 'Standardized precipitation evapotranspiration index (spei)',
                                    'description': f'Water budget (precipitation minus evapotranspiration) over a moving {window}-x window, normalized such that spei averages to 0 for calibration data. the window unit `x` is the minimal time period defined by the resampling frequency monthly.'
                                }

                            ind.attrs.pop("cat:variable", None)
                            ind.attrs["cat:xrfreq"] = "MS"
                            ind.attrs["cat:frequency"] = "mon"
                            ind.attrs["cat:processing_level"] = "indicators" if "ERA5" in key.split(".")[0] else f"indicators-{ds_wl.attrs['cat:processing_level']}"

                            # Save
                            filename = f"{xs.CONFIG['io']['extract']}{ind.attrs['cat:id']}_{ind.attrs['cat:domain']}_{ind.attrs['cat:processing_level']}_{ind.attrs['cat:frequency']}.zarr"
                            print("Saving")
                            xs.save_to_zarr(ind, filename)
                            pcat.update_from_ds(ind, filename)


if __name__ == '__main__':
    main()
