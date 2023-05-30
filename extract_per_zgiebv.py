from distributed import Client
import os
import shutil
import pandas as pd
import xclim.ensembles as xce
import xarray as xr
import xscen as xs
import json

from utils import sort_analogs

xs.load_config("paths.yml", "configs/cfg_zgiebv.yml")

def main():

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=True, project=xs.CONFIG["project"], overwrite=False)

    if xs.CONFIG["tasks"]["extract"]:
        region = dict(xs.CONFIG["region"])
        region["shape"]["shape"] = f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp"

        for dataset in xs.CONFIG["datasets"]:
            # Search
            if "ref" in dataset:
                cat_dict = xs.search_data_catalogs(**xs.CONFIG["extract"]["search_data_catalogs_ref"])
            elif "analog" in dataset:
                dparts = dataset.split("-")
                perf = sort_analogs(pcat.search(processing_level=f"{dparts[2]}-performance-vs-{dparts[1]}").to_dask().rmse)
                members = []
                analog_years = []
                for i in range(5):
                    members.extend([str(perf.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]])
                    analog_years.extend([int(perf.isel(stacked=i).time.dt.year.values)])
                search_crits = xs.CONFIG["extract"]["search_data_catalogs_sim"]
                a = search_crits["other_search_criteria"]
                a["member"] = members
                search_crits["other_search_criteria"] = a
                cat_dict = xs.search_data_catalogs(**search_crits)
            else:
                raise ValueError

            for key, cat in cat_dict.items():
                with Client(**xs.CONFIG["dask"]) as c:

                    ds_dict = xs.extract_dataset(catalog=cat, region=region)
                    ds = ds_dict["D"]
                    if "ClimEx" in key:
                        ds.attrs["cat:mip_era"] = "CMIP5"
                        ds = xs.subset_warming_level(ds, wl=float(dparts[2]), window=30).squeeze()
                        [ds[c].load() for c in ds.coords]

                        # Regrid to ERA5-Land. Very important to do this here, otherwise evspsblpot makes dask explode in RAM
                        ds_grid = pcat.search(source="ERA5-Land", processing_level="extracted", xrfreq="MS").to_dask()
                        chunks = xs.io.estimate_chunks(ds, ["time"], target_mb=125)
                        ds = ds.chunk(chunks)
                        ds = xs.regrid_dataset(ds, ds_grid, weights_location=f"{xs.CONFIG['io']['extract']}weights/", to_level="extracted")
                        ds["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}

                        if os.path.isdir(f"{xs.CONFIG['io']['extract']}tmp.zarr"):
                            shutil.rmtree(f"{xs.CONFIG['io']['extract']}tmp.zarr")
                        xs.save_to_zarr(ds, f"{xs.CONFIG['io']['extract']}tmp.zarr")
                        ds = xr.open_zarr(f"{xs.CONFIG['io']['extract']}tmp.zarr")
                        [ds[c].load() for c in ds.coords]

                    # Separated because we don't want to use ERA5-Land's evap
                    if xs.CONFIG["tasks"]["evap"]:
                        # Compute evspsblpot and wb (separated steps, because wb requires evspsbltot)
                        module = xs.indicators.load_xclim_module("configs/conversion.yml")
                        ds["evspsblpot"] = xs.compute_indicators(ds, [module.evspsblpot])["D"]["evspsblpot"]
                        # ds = ds.drop_vars(["tas", "pr"])

                    # Spatial average
                    ds = ds.chunk({"lon": -1, "lat": -1})
                    ds_mean = xs.spatial_mean(ds, method="xesmf", region=region, simplify_tolerance=0.01)

                    # Indicators
                    if xs.CONFIG["tasks"]["indicators"]:
                        ind_dict = xs.compute_indicators(ds_mean, indicators="configs/indicators_zgiebv.yml")
                    else:
                        ind_dict = {"D": ds_mean}

                    for freq, out in ind_dict.items():
                        # Cleanup
                        if dataset == "ref":
                            out = out.sel(time=slice(f"{xs.CONFIG['storylines']['out_period'][0]}-01-01", f"{xs.CONFIG['storylines']['out_period'][1]}-12-31"))
                        else:
                            out.attrs["cat:processing_level"] = f"{out.attrs['cat:processing_level']}-{dparts[2]}"

                        # Save
                        filename = f"{xs.CONFIG['io']['extract_clim']}{out.attrs['cat:id']}_{out.attrs['cat:processing_level']}_{out.attrs['cat:xrfreq']}.zarr"
                        xs.save_to_zarr(out, filename, mode="a", rechunk={"time": -1})
                        pcat.update_from_ds(out, filename)



                        # out = xs.clean_up(out, variables_and_units=xs.CONFIG["variables_and_units"], change_attr_prefix="dataset:", attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})
                        #
                        # # Prepare CSV
                        # filename = f"{xs.CONFIG['io']['livraison']}{out.attrs['dataset:id']}_{out.attrs['dataset:domain']}"
                        #
                        # # Write some metadata
                        # os.makedirs(xs.CONFIG['io']['livraison'], exist_ok=True)
                        # metadata_geom = out.geom.to_dataframe()
                        # metadata_geom.to_csv(f"{xs.CONFIG['io']['livraison']}zgiebv.csv")
                        # with open(f"{filename}_metadata.json", 'w') as fp:
                        #     json.dump(out.attrs, fp)
                        #
                        # for v in out.data_vars:
                        #     df = out[v].swap_dims({"geom": "SIGLE"}).to_pandas().T
                        #     df.to_csv(f"{filename}_{v}.csv")
                        #
                        #     with open(f"{filename}_{v}_metadata.json", 'w') as fp:
                        #         json.dump(out[v].attrs, fp)

    if xs.CONFIG["tasks"]["chirps"]:
        with Client(**xs.CONFIG["dask"]) as c:
            region = dict(xs.CONFIG["region"])
            region["shape"]["shape"] = f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp"

            ds = xr.open_dataset(xs.CONFIG["chirps"], chunks={"longitude": 100, "latitude": 100})
            ds = ds.rename({"longitude": "lon", "latitude": "lat", "precip": "pr"})

            ds = xs.extract.clisops_subset(ds, region)

            # Spatial average
            ds = ds.chunk({"lon": -1, "lat": -1})
            ds_mean = xs.spatial_mean(ds, method="xesmf", region=region, simplify_tolerance=0.01)

            # Indicators
            ind_dict = {"MS": ds_mean}
            ind_dict["MS"] = ind_dict["MS"].rename({"pr": "pr_mon"})
            if xs.CONFIG["tasks"]["indicators"]:
                ind_dict["AS-DEC"] = ds_mean.resample({"time": "AS-DEC"}).sum(keep_attrs=True)
                ind_dict["AS-DEC"] = ind_dict["AS-DEC"].where((ind_dict["AS-DEC"].time.dt.year >= 1981) & (ind_dict["AS-DEC"].time.dt.year <= 2021))
                ind_dict["AS-DEC"] = ind_dict["AS-DEC"].rename({"pr": "pr_yr"})

            for freq, out in ind_dict.items():
                if xs.CONFIG["tasks"]["deltas"]:
                    ref = xs.climatological_mean(out, periods=[xs.CONFIG['storylines']['ref_period']], min_periods=20)
                    deltas = xs.compute_deltas(ds=out, reference_horizon=ref, kind="+")
                    for v in deltas.data_vars:
                        out[v] = deltas[v]

                # Cleanup
                out = out.sel(time=slice(f"{xs.CONFIG['storylines']['out_period'][0]}-01-01", f"{xs.CONFIG['storylines']['out_period'][1]}-12-31"))
                # Cut regions that aren't fully covered by CHIRPS
                out = out.where(~out.ZGIE.isin(['Abitibi-JamÃ©sie', 'Manicouagan', 'Duplessis', 'Haute-CÃ´te-Nord', 'Lac-Saint-Jean']))

                out.attrs["cat:frequency"] = 'mon' if freq == "MS" else 'yr'
                out.attrs["cat:xrfreq"] = freq
                out.attrs["cat:institution"] = "UCSB"
                out.attrs["cat:source"] = "CHIRPS2.0"
                out.attrs["cat:processing_level"] = "indicators"
                out.attrs["cat:domain"] = "ZGIEBV"
                out.attrs["cat:id"] = xs.catalog.generate_id(out)[0]

                # Save
                filename = f"{xs.CONFIG['io']['extract_clim']}{out.attrs['cat:id']}_{out.attrs['cat:processing_level']}_{out.attrs['cat:xrfreq']}.zarr"
                xs.save_to_zarr(out, filename, mode="a", rechunk={"time": -1})
                pcat.update_from_ds(out, filename)

    if xs.CONFIG["tasks"]["csv"]:
        for dataset in xs.CONFIG["datasets"]:
            if dataset == "ref":
                ds_dict = pcat.search(id=".*ERA5-Land.*", domain="ZGIEBV", processing_level=["extracted", "indicators"]).to_dataset_dict()
                pr_ref = pcat.search(id=".*CHIRPS.*", domain="ZGIEBV", processing_level="indicators").to_dataset_dict()

                ds_dict["ECMWF_ERA5-Land_NAM.ZGIEBV.indicators.AS-DEC"]["precip_accumulation_yr"] = pr_ref['UCSB_CHIRPS2.0_ZGIEBV.ZGIEBV.indicators.AS-DEC']["pr_yr"]
                ds_dict["ECMWF_ERA5-Land_NAM.ZGIEBV.indicators.MS"]["precip_accumulation_mon"] = pr_ref['UCSB_CHIRPS2.0_ZGIEBV.ZGIEBV.indicators.MS']["pr_mon"]
            else:
                ds_dict = pcat.search(processing_level=f"{dataset}.*", domain="ZGIEBV").to_dataset_dict()

            for key, ds in ds_dict.items():
                if "delta" not in key:
                    out = xs.clean_up(ds, variables_and_units=xs.CONFIG["variables_and_units"], change_attr_prefix="dataset:",
                                      attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})
                else:
                    out = xs.clean_up(ds, change_attr_prefix="dataset:",
                                      attrs_to_remove={"global": ["cat:_data_format_", "intake_esm_dataset_key"]})

                # Prepare CSV
                if dataset == "ref":
                    filename = f"{out.attrs['dataset:source']}_{out.attrs['dataset:domain']}"
                elif "analog" in out.attrs['dataset:processing_level']:
                    filename = f"{out.attrs['dataset:source']}_{out.attrs['dataset:domain']}_{out.attrs['dataset:processing_level']}degC"
                else:
                    filename = f"{out.attrs['dataset:activity']}_{out.attrs['dataset:member']}_{out.attrs['dataset:domain']}_{out.attrs['dataset:processing_level']}degC"

                # Write some metadata
                os.makedirs(xs.CONFIG['io']['livraison'], exist_ok=True)
                metadata_geom = out.geom.to_dataframe()
                metadata_geom.to_csv(f"{xs.CONFIG['io']['livraison']}dataset-metadata_ZGIEBV.csv")
                with open(f"{xs.CONFIG['io']['livraison']}dataset-metadata_{filename}.json", 'w') as fp:
                    json.dump(out.attrs, fp)

                for v in out.data_vars:
                    if out.attrs['dataset:frequency'] not in v:
                        out = out.rename({v: f"{v}_{out.attrs['dataset:frequency']}"})
                        v = f"{v}_{out.attrs['dataset:frequency']}"

                    df = out[v].swap_dims({"geom": "SIGLE"}).to_pandas()
                    if isinstance(df, pd.Series):
                        df = df.to_frame(name=v)
                    if df.index.name != "SIGLE":
                        df = df.T
                    df.to_csv(f"{xs.CONFIG['io']['livraison']}{v}_{filename}.csv")

                    with open(f"{xs.CONFIG['io']['livraison']}variable-metadata_{v}_{filename}.json", 'w') as fp:
                        json.dump(out[v].attrs, fp)


if __name__ == '__main__':
    main()
