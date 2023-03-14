from distributed import Client
import os
import shutil
import xarray as xr
import xscen as xs
import json

xs.load_config("paths.yml", "configs/cfg_zgiebv.yml")

def main():

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=True, project=xs.CONFIG["project"], overwrite=False)

    if xs.CONFIG["tasks"]["extract"]:
        region = dict(xs.CONFIG["region"])
        region["shape"]["shape"] = f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp"

        xs.CONFIG["region"]["shape"]  .set("shape", f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")
        # Search
        cat_dict = xs.search_data_catalogs(**xs.CONFIG["extract"]["search_data_catalogs_ref"])
        # cat_dict_sim = xs.search_data_catalogs(**xs.CONFIG["extract"]["search_data_catalogs_sim"])
        # cat_dict |= cat_dict_sim  # Merge the two search results

        for key, cat in cat_dict.items():
            if not (pcat.exists_in_cat(source=cat.unique("source")[0], member=cat.unique("member")[0], processing_level="extracted", domain=region["name"]) if "ClimEx" in key else \
                    pcat.exists_in_cat(id=key, processing_level="extracted", domain=region["name"])):

                with Client(**xs.CONFIG["dask"]) as c:

                    # Extract tas and pr
                    if "ClimEx" in key:
                        ds_dict = xs.extract_dataset(catalog=cat, region=region)
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
                        ds_dict = xs.extract_dataset(catalog=cat, region=region)
                        ds = ds_dict["D"]

                    # Spatial average
                    ds = ds.chunk({"lon": -1, "lat": -1})
                    ds_mean = xs.spatial_mean(ds, method="xesmf", region=region, simplify_tolerance=0.01)

                    # Indicators
                    if xs.CONFIG["tasks"]["extract"]:
                        ind_dict = xs.compute_indicators(ds_mean, indicators="configs/indicators_zgiebv.yml")
                    else:
                        ind_dict = {"D": ds_mean}

                    for out in ind_dict.values():
                        if xs.CONFIG["tasks"]["deltas"]:
                            ref = xs.climatological_mean(out)
                            deltas = xs.compute_deltas(ds=out, reference_horizon=ref, kind="+")

                        # Cleanup
                        out = xs.clean_up(out, variables_and_units={"tasmax": "degC"}, change_attr_prefix="dataset:", attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})

                        # Prepare CSV
                        filename = f"{xs.CONFIG['io']['livraison']}{out.attrs['dataset:id']}_{out.attrs['dataset:domain']}"

                        # Write some metadata
                        metadata_geom = out.geom.to_dataframe()
                        metadata_geom.to_csv(f"{xs.CONFIG['io']['livraison']}zgiebv.csv")
                        with open(f"{filename}_metadata.json", 'w') as fp:
                            json.dump(out.attrs, fp)

                        metadata_geom.to_csv(f"{xs.CONFIG['io']['livraison']}zgiebv.csv")
                        for v in out.data_vars:
                            df = out[v].swap_dims({"geom": "SIGLE"}).to_pandas()
                            df.to_csv(f"{filename}_{v}.csv")

                            with open(f"{filename}_{v}_metadata.json", 'w') as fp:
                                json.dump(out[v].attrs, fp)


if __name__ == '__main__':
    main()
