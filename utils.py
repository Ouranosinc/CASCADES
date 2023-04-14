import geopandas as gpd
import xscen as xs
import pandas as pd
from distributed import Client
import glob
import xarray as xr
import shutil

def get_target_region(target_year: int):
    shp = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")
    target_region = shp.loc[shp["SIGLE"].isin(xs.CONFIG["analogs"]["targets"][target_year]["region"])]

    return target_region


def get_stations_within_target_region(data, target_region):
    from shapely.geometry import Point
    _pnts = [Point(data.lon[i], data.lat[i]) for i in range(len(data.lon))]
    pnts = gpd.GeoDataFrame(geometry=_pnts, index=data.station_id)
    pnts = pnts.assign(**{f"a{key}": pnts.within(geom) for key, geom in target_region.geometry.items()})
    pnts = pnts.drop("geometry", axis=1)

    return list(pnts.index[pnts.any(axis=1)])


def sort_analogs(da):
    if "time" in da.dims:
        da = da.stack({"stacked": ["time", "realization"]})
    else:
        da = da.stack({"stacked": ["year", "realization"]})
    da = da.sortby(da)

    return da


def atlas_radeau_common():
    stations_radeau = gpd.read_file(f"{xs.CONFIG['gis']}RADEAU/CONSOM_SURF_BV_CF1_WGS84.shp")["BV_ID"]
    # Fix IDs
    stations_radeau = stations_radeau.str[0:3].str.cat(stations_radeau.str[4:])
    stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
    stations = stations_atlas.loc[stations_atlas["ATLAS2018"].isin(stations_radeau)]
    # When an old reach is now covered by multiple, keep only the largest one
    stations = stations.sort_values('SUPERFICIE_TRONCON_num_km2')
    stations = stations.drop_duplicates(subset=["ATLAS2018"], keep="last")

    return stations


def fix_infocrue(cat):
    def _fix_infocrue_file(ds):
        # drainage_area as coordinate
        ds = ds.assign_coords({"drainage_area": ds["drainage_area"]})
        # load coordinates
        [ds[c].load() for c in ds.coords]
        # fix station_id
        if "nchar_station_id" in ds.dims:
            ds["station_id"] = ds.station_id.astype(str).str.join(dim="nchar_station_id")
        else:
            ds["station_id"] = ds.station_id.astype(str)
        # add coordinate to station
        ds = ds.assign_coords({"station": ds["station"]})
        # rename Dis
        ds = ds.rename({"Dis": "discharge"})
        ds["discharge"] = ds["discharge"].astype("float32")
        return ds

    ds_all = cat.to_dask(xarray_open_kwargs={"chunks": {"station": 500, "time": 365}})
    if "percentile" in ds_all:
        ds_all = ds_all.chunk({"percentile": 1})
        rechunk = {"station": 500, "time": 365, "percentile": 1}
    else:
        rechunk = {"station": 500, "time": 365}
    ds_all = _fix_infocrue_file(ds_all)

    # loop to prevent dask from dying
    for i in range(0, len(ds_all.station), 1500):
        with Client(**xs.CONFIG["dask"]) as c:
            ds = ds_all.isel(station=slice(i, i + 1500))
            xs.save_to_zarr(ds, filename=f"{xs.CONFIG['tmp_rechunk']}{ds.attrs['cat:id']}_{i}.zarr",
                            rechunk=rechunk)

    with Client(**xs.CONFIG["dask"]) as c:
        files = glob.glob(f"{xs.CONFIG['tmp_rechunk']}{cat.unique('id')[0]}_*.zarr")
        ds = xr.open_mfdataset(files, engine="zarr")
        [ds[c].load() for c in ds.coords]
        xs.save_to_zarr(ds, filename=f"{xs.CONFIG['tmp_rechunk']}{ds.attrs['cat:id']}.zarr")

    for f in files:
        shutil.rmtree(f)
