from distributed import Client
import xscen as xs
import xclim
import xclim.indicators as xci

xs.load_config("project.yml", "paths.yml", "cfg.yml")


def main():

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=True, project=xs.CONFIG["project"], overwrite=False)

    if xs.CONFIG["tasks"]["extract"]:
        # Search
        cat_dict = xs.search_data_catalogs()
        for key, cat in cat_dict.items():
            if not pcat.exists_in_cat(id=key, processing_level="extracted"):

                with Client(**xs.CONFIG["dask"]) as c:

                    # Extract tas and pr
                    ds_dict = xs.extract_dataset(catalog=cat, region=xs.CONFIG["region"])
                    ds = ds_dict["D"]

                    # Compute evspsblpot and wb (separated steps, because wb requires evspsbltot)
                    module = xs.indicators.load_xclim_module(xs.CONFIG["indicatorsWB"])

                    tmp = xs.compute_indicators(ds, [module.evspsblpot])
                    ds["evspsblpot"] = tmp["D"]["evspsblpot"]
                    tmp = xs.compute_indicators(ds, [module.water_budget])
                    ds["wb"] = tmp["D"]["wb"]

                    # Drop tas
                    ds = ds.drop_vars("tas")

                    # Ajust attributes
                    ds.attrs["cat:variable"] = tuple(v for v in ds.data_vars)

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
        # Search
        ds_dict = pcat.search(processing_level="extracted", variable="wb").to_dataset_dict()
        for key, ds in ds_dict.items():
            if not pcat.exists_in_cat(id=key, processing_level="indicators"):

                with Client(**xs.CONFIG["dask"]) as c:
                    # Get reference period
                    cal_period = xs.CONFIG["spei"]["cal_period"]
                    da = ds["wb"].sel(time=slice(cal_period[0], cal_period[1]))

                    # Compute SPEI (currently not compatible with xscen)
                    with xclim.set_options(check_missing="skip"):
                        ind = xci.atmos.standardized_precipitation_evapotranspiration_index(wb=ds["wb"], wb_cal=da, freq="MS", window=1, method="ML", dist="fisk")
                    ind = ind.to_dataset()
                    ind = ind.rename({"spei": "spei1"})
                    ind.attrs = ds.attrs

                    for window in [3, 6, 9, 12]:
                        with xclim.set_options(check_missing="skip"):
                            ind[f"spei{window}"] = xci.atmos.standardized_precipitation_evapotranspiration_index(wb=ds["wb"], wb_cal=da, freq="MS", window=window, method="ML", dist="fisk")

                    ind.attrs.pop("cat:variable", None)
                    ind.attrs["cat:xrfreq"] = "MS"
                    ind.attrs["cat:frequency"] = "mon"
                    ind.attrs["cat:processing_level"] = "indicators"

                    # Save
                    filename = f"{xs.CONFIG['io']['extract']}{ind.attrs['cat:id']}_{ind.attrs['cat:domain']}_{ind.attrs['cat:processing_level']}_{ind.attrs['cat:frequency']}.zarr"
                    print("Saving")
                    xs.save_to_zarr(ind, filename)
                    pcat.update_from_ds(ind, filename)


if __name__ == '__main__':
    main()
