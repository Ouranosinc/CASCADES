import streamlit as st
import numpy as np
import xarray as xr
import cartopy
from matplotlib import pyplot as plt
import matplotlib
import glob
import pandas as pd
import geopandas as gpd


@st.cache(hash_funcs={xr.core.dataset.Dataset: id},ttl=60)
def load_zarr(path):
    return xr.open_zarr(path,decode_timedelta= False)

@st.cache(hash_funcs={xr.core.dataset.Dataset: id},ttl=60)
def load_nc(path):
    return xr.open_dataset(path,decode_timedelta= False)


def make_spatial_distribution_plot(ax, ds, levels, cmap, lon_bnds, lat_bnds, title=None, ylabel=None, add_cbar=False, shp=None):

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

    ax.coastlines(zorder=2)
    ax.add_feature(cartopy.feature.OCEAN, zorder=1)
    ax.add_feature(cartopy.feature.LAKES, edgecolor="black")

    ds.plot.imshow(ax=ax, levels=levels, colors=cmap, extend='both', zorder=0, add_colorbar=add_cbar)

    if shp is not None:
        for s in shp:
            s.plot(facecolor='none', edgecolor='k', linewidth=0.6, ax=ax, zorder=2)

    plt.xlim([lon_bnds[0], lon_bnds[1]])
    plt.ylim([lat_bnds[0], lat_bnds[1]])
    plt.title(title, fontsize=18)


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
    value = value.strip("#")
    lv = len(value)

    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(
    value: [int]
):
    return [v/256 for v in value]

st.set_page_config(layout="wide")

st.title('CASCADES')
st.header("Cons??quences Attendues Survenant en Contexte d???Aggravation des D??ficits d???Eau S??v??res au Qu??bec")

useCat = st.checkbox("Use Local Data")

# Tabs
tab1, tab2 = st.tabs(["Historique", "TBD"])

with tab1:

    st.header("ERA5-Land")

    useSom = st.checkbox("Sommaire: SPEI minimal entre juin et octobre", True)
    useHydro = st.checkbox("Afficher les rivi??res", False)
    if useHydro:
        cols = st.columns(2)
        option_rivtype = cols[0].selectbox('Afficher les rivi??res:', ["Tous les tron??ons", "Tron??ons diffus??s seulement"], index=1)
        option_hidehydro = cols[1].selectbox("Niveau d\'intensit?? minimal (D??bits)", ["Tout afficher", "??? 1 (Mod??r??)", "??? 1.5 (S??v??re)", "??? 2 (Extr??me)", "??? 2.5 (Tr??s extr??me)"], index=2)

    cols = st.columns(2)
    option_y = cols[0].selectbox('Ann??e', np.arange(1970, 2022), index=51)
    option_hide = cols[1].selectbox("Niveau d\'intensit?? minimal (SPEI)", ["Tout afficher", "??? 1 (Mod??r??)", "??? 1.5 (S??v??re)", "??? 2 (Extr??me)", "??? 2.5 (Tr??s extr??me)"], index=2)
    hide = 0 if option_hide == "Tout afficher" else eval(option_hide.split("??? ")[1].split(" (")[0])

    speis = [1, 3, 6, 9, 12]
    levels = np.arange(-2.5, 3, 0.5)
    cmap = make_cmap("BrWhGr", 25)
    months = ["Janvier", "F??vrier", "Mars", "Avril", "Mai", "Juin", "Juillet", "Ao??t", "Septembre", "Octobre", "Novembre", "D??cembre"]

    # load data
    if useCat:
        import xscen as xs
        xs.load_config("project.yml", "paths.yml", "cfg.yml", reset=True)
        pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

        ds = pcat.search(processing_level="indicators", source="ERA5-Land").to_dask()

        # Shapefiles
        regions = glob.glob(f"{xs.CONFIG['gis']}ZGIEBV/*.shp")
        zgiebv = [gpd.read_file(shp) for shp in regions]
        atlas = gpd.read_file(glob.glob(f"{xs.CONFIG['gis']}atlashydroclimatique_2022/*.shp")[0])
        atlas_meta = pd.read_csv(f"{xs.CONFIG['dpphc']}Metadata_Portrait.csv", encoding='latin-1')

    else:
        raise NotImplementedError()

    # Selections
    cols = st.columns(3)
    lon_bnds = cols[0].slider('Longitude', -83., -55.5, (-83., -55.5))
    cols = st.columns(3)
    lat_bnds = cols[0].slider('Latitude', 43., 54., (43., 54.))

    cols = st.columns(5)
    cols[0].markdown("<h3 style='text-align: center; color: black;'>SPEI-1</h3>", unsafe_allow_html=True)
    cols[1].markdown("<h3 style='text-align: center; color: black;'>SPEI-3</h3>", unsafe_allow_html=True)
    cols[2].markdown("<h3 style='text-align: center; color: black;'>SPEI-6</h3>", unsafe_allow_html=True)
    cols[3].markdown("<h3 style='text-align: center; color: black;'>SPEI-9</h3>", unsafe_allow_html=True)
    cols[4].markdown("<h3 style='text-align: center; color: black;'>SPEI-12</h3>", unsafe_allow_html=True)

    if useSom is True:
        # Plot the minimum of all SPEI between June and October
        cols = st.columns(len(speis))
        titles = ["min(SPEI-1)", "min(SPEI-3)", "min(SPEI-6)", "min(SPEI-9)", "min(SPEI-12)"]

        for s in range(len(speis)):
            fig, _ = plt.subplots(1, 1)
            da = ds[f"spei{speis[s]}"].sel(time=slice(f"{option_y}-06-01", f"{option_y}-10-01")).min(dim="time")
            da = da.where(np.abs(da) > hide)
            title = titles[s]

            ax = plt.subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
            make_spatial_distribution_plot(ax, da, levels=levels, cmap=cmap, title=title, shp=zgiebv, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
            if useHydro == True:
                atlas.where(atlas_meta["MASQUE"] >= (0 if option_rivtype == "Tous les tron??ons" else 2)).plot(ax=ax)
            plt.tight_layout()

            cols[s].pyplot(fig)

    else:
        # Plot every month from June to October
        for month in [6, 7, 8, 9, 10]:
            cols = st.columns(len(speis))

            for s in range(len(speis)):

                titles = [months[month - 1],
                          f"{months[(month - 2 - 1)%12]} ?? {months[month - 1]}",
                          f"{months[(month - 5 - 1)%12]} ?? {months[month - 1]}",
                          f"{months[(month - 8 - 1)%12]}{' (' + str(option_y - 1) + ')' if month - 8 - 1 < 0 else ''} ?? {months[month - 1]}",
                          f"{months[(month - 11 - 1)%12]}{' (' + str(option_y - 1) + ')' if month - 11 - 1 < 0 else ''} ?? {months[month - 1]}"]

                fig, _ = plt.subplots(1, 1)
                da = ds[f"spei{speis[s]}"].sel(time=f"{option_y}-{month}-01")
                da = da.where(np.abs(da) > hide)
                title = titles[s]

                ax = plt.subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
                make_spatial_distribution_plot(ax, da, levels=levels, cmap=cmap, title=title, shp=zgiebv, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
                if useHydro == True:
                    atlas.where(atlas_meta["MASQUE"] >= (0 if option_rivtype == "Tous les tron??ons" else 2)).plot(ax=ax)
                plt.tight_layout()

                cols[s].pyplot(fig)
