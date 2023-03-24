import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.feature as cpf


def cartopy_map(ax, ds, lon_bnds=None, lat_bnds=None, highlight=None, hide_labels=False, **kwargs):
    ax.coastlines(zorder=2)
    ax.add_feature(cpf.OCEAN, zorder=1)
    ax.add_feature(cpf.LAKES, zorder=1, edgecolor="black")
    ax.add_feature(cpf.STATES, edgecolor="black", linestyle="dotted")
    ds.plot.pcolormesh(ax=ax, zorder=0, **kwargs)

    if highlight is not None:
        highlight.plot.contour(levels=[0.99], colors='r')

    if lon_bnds is not None:
        plt.xlim([lon_bnds[0], lon_bnds[1]])
    if lat_bnds is not None:
        plt.ylim([lat_bnds[0], lat_bnds[1]])


def map_hydro(ax, data, shp, lon_bnds=None, lat_bnds=None, highlight=None, background=True, **kwargs):
    """

    Parameters
    ----------
    ax: cartopy.mpl.geoaxes.GeoAxes | cartopy.mpl.geoaxes.GeoAxesSubplot
        Created by initialising a matplotlib plot with the 'projection' argument.
    data: xr.DataArray
        Data to be plotted. Requires a 'station_id' coordinate.
    shp: geopandas.geodataframe.GeoDataFrame
        Spatial information data. Indexes should match 'station_id'
    cmap:
        Colormap to use.
    bnds: list
        Boundaries to use for matplotlib.colors.BoundaryNorm.

    """
    # Add data to the shapefile
    da2 = data.to_dataframe().set_index("station_id")
    i = da2.index.intersection(shp.index)
    shp["val"] = da2.loc[i][data.name]

    # Change CRS of the shapefile, for cartopy/geopandas compatibility
    crs_proj4 = ax.projection.proj4_init
    df_ae = shp.to_crs(crs_proj4)

    # add the background
    if background:
        ax.add_feature(cpf.LAND, color="#f0f0f0", zorder=0)
        ax.add_feature(cpf.OCEAN, zorder=0)
        ax.add_feature(cpf.STATES, edgecolor="black", linestyle="dotted", zorder=10)

    ax.add_feature(cpf.RIVERS, color="#cfd3d4")
    ax.add_feature(cpf.LAKES, color="#cfd3d4")

    # Plot data
    df_ae.plot(column="val", ax=ax, **kwargs)

    if highlight is not None:
        highlight.plot.contour(levels=[0.99], colors='r')

    if lon_bnds is not None:
        plt.xlim([lon_bnds[0], lon_bnds[1]])
    if lat_bnds is not None:
        plt.ylim([lat_bnds[0], lat_bnds[1]])


# def timeseries_hydro(ax, da, target_year):
#
#
#     da = convert_calendar(da, 'noleap')
#     da = da.chunk({"time": -1})
#     qt = da.groupby("time.dayofyear").quantile(q=[0.10, 0.25, 0.50, 0.75, 0.90])
#
#     # Plot
#     plt.fill_between(qt.dayofyear, da.groupby("time.dayofyear").min(dim="time"),
#                      da.groupby("time.dayofyear").max(dim="time"), facecolor="#c6dfe7")
#     plt.fill_between(qt.dayofyear, qt.sel(quantile=0.10), qt.sel(quantile=0.90),
#                      facecolor="#80b0c8")
#     plt.fill_between(qt.dayofyear, qt.sel(quantile=0.25), qt.sel(quantile=0.75),
#                      facecolor="#003366")
#     plt.plot(qt.dayofyear, qt.sel(quantile=0.50), c="k")
#     plt.plot(da.sel(time=slice(str(target_year), str(target_year))), c='r')
#
#     plt.ylim([0, portrait_station.Dis.max()])
#
#     # Stats
#     portrait_station_7q2 = xclim.land.freq_analysis(portrait_station.Dis, mode="min", window=7, t=2, dist="lognorm",
#                                                     **{"month": [5, 6, 7, 8, 9, 10, 11]})
#     plt.hlines(portrait_station_7q2, 1, 365, linestyle="--")
#
#     portrait_under_thresh = portrait_station.sel(time=slice(f"{target_year}-04", f"{target_year}-11")).Dis < portrait_station_7q2
#     portrait_under_thresh.sum()
#     consecutive = portrait_under_thresh.cumsum(dim='time') - portrait_under_thresh.cumsum(dim='time').where(portrait_under_thresh.values == 0).ffill(
#         dim='time').fillna(0)
#     consecutive.max()
#     consecutive.where(consecutive > 7, drop=True).time.min()
#     consecutive.where(consecutive > 7, drop=True).time.max()
#
#     q_spring = portrait_station.sel(time=slice(f"{target_year}-02", f"{target_year}-07")).Dis
#     qmax = q_spring.where(q_spring >= 0.5 * q_spring.max(), drop=True)
#     qmax.idxmax()
#     qsub = q_spring.sel(time=slice(str(qmax.idxmax().values.astype(str)), f"{target_year}-07")).where(q_spring < 0.1 * qmax.max().values)
#     qsub.dropna(dim="time").time.min()
#
#     q_spring_qt = qt.sel(quantile=.50).rolling({"dayofyear": 7}, center=True).mean()
#     qmax_qt = q_spring_qt.where(q_spring_qt >= 0.8 * q_spring_qt.max(), drop=True)
#     tmp = xr.DataArray(pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31"),
#                        coords={"time": pd.date_range(f"{target_year}-01-01", f"{target_year}-12-31")})
#     qmax_qt = tmp.where(tmp.dt.dayofyear.isin(qmax_qt.dayofyear), drop=True)
#     qmax_qt.idxmin()
#     qmax_qt.idxmax()
#     qmax_qt.idxmax().dt.dayofyear - qmax.idxmax().dt.dayofyear
#
#     # TODO: volume minimal vs 7q2
