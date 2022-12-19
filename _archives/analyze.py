import xscen as xs
import numpy as np
import geopandas as gpd
import os
import cartopy
import matplotlib.colors
import matplotlib.pyplot as plt

from glob import glob

xs.load_config("project.yml", "paths.yml", "cfg.yml")


def main(hide: float = 0):

    pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

    # Search
    ds_dict = pcat.search(processing_level="indicators").to_dataset_dict()

    levels = np.arange(-2, 2.5, 0.5)
    levels = np.arange(-2.5, 3, 0.5)
    cmap = make_cmap("BrWhGr", 25)

    for key, ds in ds_dict.items():
        for y in np.unique(ds.time.dt.year):

            # Prepare the figure
            fig, _ = plt.subplots(5, 6, figsize=(30, 15))
            plt.suptitle(f"{ds.attrs['cat:id']}: {y}")

            speis = [1, 3, 6, 9, 12]
            months = ["06", "07", "08", "09", "10", "11"]
            month_lbl = ["Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre"]
            s = 0
            for spei in range(len(speis)):
                for month in range(len(months)):
                    da = ds[f"spei{speis[spei]}"].sel(time=f"{y}-{months[month]}-01")
                    da = da.where(np.abs(da) > hide)
                    if s <= len(month_lbl) - 1:
                        title = month_lbl[s]
                    else:
                        title = None
                    if s in [0, 6, 12, 18, 24]:
                        ylabel = f"SPEI-{speis[spei]}"
                    else:
                        ylabel = None

                    ax = plt.subplot(5, 6, s + 1, projection=cartopy.crs.PlateCarree())
                    make_spatial_distribution_plot(ax, da, levels=levels, cmap=cmap, title=title, ylabel=ylabel)

                    s = s + 1

            plt.tight_layout()
            os.makedirs(xs.CONFIG['io']['figures'], exist_ok=True)
            plt.savefig(f"{xs.CONFIG['io']['figures']}{ds.attrs['cat:id']}_{y}.png")
            plt.close()


def make_spatial_distribution_plot(ax, ds, levels, cmap, title=None, ylabel=None, add_cbar=False):

    """
    Produces maps (spatial distributions) of data.

    Parameters
    ----------
    ds : xarray.DataArray
        Data to be plotted.
    levels : numpy.ndarray
        Levels used for the color map.
    cmap : matplotlib.colors.LinearSegmentedColormap
        Colors used for the color map.
    lowflow_epicenter_lat : float
        Latitude of the epicenter of the lowflow.
    lowflow_epicenter_lon : float
        Longitude of the epicenter of the lowflow.
    cmap_label : str
        Label for the color map.
    lowflow_epicenter_label : str
        Label for the lowflow epicenter.
    title : str
        Title of map.
    filename : str
        File name and path for the plot.

    Note
    ----
    This function was used to produce the climatology, anomaly and SPEI figures.
    """

    regions = glob(f"{xs.CONFIG['gis']}*.shp")

    min_lat = ds.coords['lat'].values.min()
    max_lat = ds.coords['lat'].values.max()

    min_lon = ds.coords['lon'].values.min()
    max_lon = ds.coords['lon'].values.max()

    # ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.coastlines(zorder=2)
    ax.add_feature(cartopy.feature.OCEAN, zorder=1)
    # ax.add_feature(cartopy.feature.STATES, edgecolor="black", linestyle="dotted")
    # ax.add_feature(cartopy.feature.RIVERS, color="#404040")
    ax.add_feature(cartopy.feature.LAKES, edgecolor="black")

    ds.plot.contourf(ax=ax, levels=levels, colors=cmap, extend='both', zorder=0, add_colorbar=add_cbar)

    for shp in regions :
        sf = gpd.read_file(shp)
        sf.plot(facecolor='none', edgecolor='k', linewidth=0.6, ax=ax, zorder=2)

    plt.xlim([min_lon, max_lon])
    plt.ylim([min_lat, max_lat])

    # Cartopy hijacks ax labels
    ax.text(-0.07, 0.55, ylabel, va='bottom', ha='center',
            rotation='vertical', rotation_mode='anchor',
            transform=ax.transAxes)
    plt.title(title)

    # return ax

    # plt.margins(x=0.1, y=0.005)
    # plt.tight_layout()

    # plt.savefig(filename)
    # plt.show()


def make_cmap(name, n_cluster):
    # Custom color maps (not in matplotlib). The order assumes a vertical color bar.
    hex_wh  = "#ffffff"  # White.
    hex_gy  = "#808080"  # Grey.
    hex_gr  = "#008000"  # Green.
    hex_yl  = "#ffffcc"  # Yellow.
    hex_or  = "#f97306"  # Orange.
    hex_br  = "#662506"  # Brown.
    hex_rd  = "#ff0000"  # Red.
    hex_pi  = "#ffc0cb"  # Pink.
    hex_pu  = "#800080"  # Purple.
    hex_bu  = "#0000ff"  # Blue.
    hex_lbu = "#7bc8f6"  # Light blue.
    hex_lbr = "#d2b48c"  # Light brown.
    hex_sa  = "#a52a2a"  # Salmon.
    hex_tu  = "#008080"  # Turquoise.

    code_hex_l = {
        "Pinks": [hex_wh, hex_pi],
        "PiPu": [hex_pi, hex_wh, hex_pu],
        "Browns": [hex_wh, hex_br],
        "Browns_r": [hex_br, hex_wh],
        "YlBr": [hex_yl, hex_br],
        "BrYl": [hex_br, hex_yl],
        "BrYlGr": [hex_br, hex_yl, hex_gr],
        "GrYlBr": [hex_gr, hex_yl, hex_br],
        "YlGr": [hex_yl, hex_gr],
        "GrYl": [hex_gr, hex_yl],
        "BrWhGr": [hex_br, hex_wh, hex_gr],
        "GrWhBr": [hex_gr, hex_wh, hex_br],
        "TuYlSa": [hex_tu, hex_yl, hex_sa],
        "YlTu": [hex_yl, hex_tu],
        "YlSa": [hex_yl, hex_sa],
        "LBuWhLBr": [hex_lbu, hex_wh, hex_lbr],
        "LBlues": [hex_wh, hex_lbu],
        "BuYlRd": [hex_bu, hex_yl, hex_rd],
        "LBrowns": [hex_wh, hex_lbr],
        "LBuYlLBr": [hex_lbu, hex_yl, hex_lbr],
        "YlLBu": [hex_yl, hex_lbu],
        "YlLBr": [hex_yl, hex_lbr],
        "YlBu": [hex_yl, hex_bu],
        "Turquoises": [hex_wh, hex_tu],
        "Turquoises_r": [hex_tu, hex_wh],
        "PuYlOr": [hex_pu, hex_yl, hex_or],
        "YlOrRd": [hex_yl, hex_or, hex_rd],
        "YlOr": [hex_yl, hex_or],
        "YlPu": [hex_yl, hex_pu],
        "PuYl": [hex_pu, hex_yl],
        "GyYlRd": [hex_gy, hex_yl, hex_rd],
        "RdYlGy": [hex_rd, hex_yl, hex_gy],
        "YlGy": [hex_yl, hex_gy],
        "GyYl": [hex_gy, hex_yl],
        "YlRd": [hex_yl, hex_rd],
        "RdYl": [hex_rd, hex_yl],
        "GyWhRd": [hex_gy, hex_wh, hex_rd]}

    hex_l = None
    if name in list(code_hex_l.keys()):
        hex_l = code_hex_l[name]

    # List of positions.
    if len(hex_l) == 2:
        pos_l = [0.0, 1.0]
    else:
        pos_l = [0.0, 0.5, 1.0]

    # Reverse hex list.
    if "_r" in name:
        hex_l.reverse()

    # Build colour map.
    rgb_l = [rgb_to_dec(hex_to_rgb(i)) for i in hex_l]
    if pos_l:
        pass
    else:
        pos_l = list(np.linspace(0, 1, len(rgb_l)))
    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_l = [[pos_l[i], rgb_l[i][num], rgb_l[i][num]] for i in range(len(pos_l))]
        cdict[col] = col_l

    return matplotlib.colors.LinearSegmentedColormap("custom_cmap", segmentdata=cdict, N=n_cluster)


def hex_to_rgb(
    value: str
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Converts hex to RGB colors
    Parameters
    ----------
    value: str
        String of 6 characters representing a hex color.
    Returns
    -------
        list of 3 RGB values
    --------------------------------------------------------------------------------------------------------------------
    """

    value = value.strip("#")
    lv = len(value)

    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(
    value: [int]
):

    """
    --------------------------------------------------------------------------------------------------------------------
    Converts RGB to decimal colors (i.e. divides each value by 256)
    Parameters
    ----------
    value: [int]
        List of 3 RGB values.
    Returns
    -------
        List of 3 decimal values.
    --------------------------------------------------------------------------------------------------------------------
    """

    return [v/256 for v in value]

if __name__ == '__main__':
    main(hide=1.5)