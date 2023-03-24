import streamlit as st
import numpy as np
import cartopy
import matplotlib
from matplotlib import pyplot as plt
import glob
import geopandas as gpd

import figures


st.set_page_config(layout="wide")

st.title('CASCADES')
st.header("Conséquences Attendues Survenant en Contexte d’Aggravation des Déficits d’Eau Sévères au Québec")

# useCat = st.checkbox("Use Local Data", True)
useCat = True

cols = st.columns(4)
lon_bnds = cols[0].slider('Longitude', -83., -55.5, (-80., -64.))
# cols = st.columns(3)
lat_bnds = cols[1].slider('Latitude', 43., 54., (44.75, 54.))

# Tabs
tab1, tab2 = st.tabs(["SPEI (hist)", "Débits (hist)"])
with tab1:

    st.header("ERA5-Land")

    # Selections
    cols = st.columns(2)
    option_y1 = cols[0].selectbox('Année', np.arange(1970, 2022), index=51)
    option_hide = cols[1].selectbox("Niveau d\'intensité minimal (SPEI)",
                                    ["Tout afficher", "≥ 1 (Modéré)", "≥ 1.5 (Sévère)", "≥ 2 (Extrême)", "≥ 2.5 (Très extrême)"], index=2)
    hide = 0 if option_hide == "Tout afficher" else eval(option_hide.split("≥ ")[1].split(" (")[0])
    show_som = st.checkbox("Sommaire: SPEI minimal sur la période estivale", True)

    speis = [3, 6, 9]
    levels = np.arange(-3, 3.5, 0.5)
    cmap = figures.utils.make_cmap("BrWhGr", 25)
    months = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

    # load data
    if useCat:
        import xscen as xs
        xs.load_config("paths.yml", reset=True)
        pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

        ds = pcat.search(processing_level="indicators", source="ERA5-Land").to_dask()

        # Open the ZGIEBV
        shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")

    else:
        raise NotImplementedError()

    cols = st.columns(3)
    cols[0].markdown("<h3 style='text-align: center; color: black;'>SPEI-3</h3>", unsafe_allow_html=True)
    cols[1].markdown("<h3 style='text-align: center; color: black;'>SPEI-6</h3>", unsafe_allow_html=True)
    cols[2].markdown("<h3 style='text-align: center; color: black;'>SPEI-9</h3>", unsafe_allow_html=True)

    proj = cartopy.crs.PlateCarree()
    if show_som is True:
        # Plot the minimum of all SPEI between June and October
        cols = st.columns(len(speis))
        titles = ["min(SPEI-3)", "min(SPEI-6)", "min(SPEI-9)"]

        for s in range(len(speis)):
            fig, _ = plt.subplots(1, 1)
            da = ds[f"spei{speis[s]}"].sel(time=slice(f"{option_y1}-06-01", f"{option_y1}-10-01")).clip(min=-3.09, max=3.09).min(dim="time")
            da = da.where(np.abs(da) > hide)
            title = titles[s]

            ax = plt.subplot(1, 1, 1, projection=proj)
            figures.templates.cartopy_map(ax, da, hide_labels=False, lon_bnds=lon_bnds, lat_bnds=lat_bnds, levels=levels, cmap=cmap, add_colorbar=False)
            shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
            plt.title(title)
            plt.tight_layout()

            cols[s].pyplot(fig)

    else:
        # Plot every month from June to October
        for month in [5, 6, 7, 8, 9, 10, 11]:
            cols = st.columns(len(speis))

            for s in range(len(speis)):

                titles = [f"{months[(month - 2 - 1)%12]} à {months[month - 1]}",
                          f"{months[(month - 5 - 1)%12]}{' (' + str(option_y1 - 1) + ')' if month - 5 - 1 < 0 else ''}  à {months[month - 1]}",
                          f"{months[(month - 8 - 1)%12]}{' (' + str(option_y1 - 1) + ')' if month - 8 - 1 < 0 else ''} à {months[month - 1]}"]

                da = ds[f"spei{speis[s]}"].sel(time=f"{option_y1}-{month}-01").clip(min=-3.09, max=3.09)
                da = da.where(np.abs(da) > hide)

                if not np.isnan(da).all():
                    fig, _ = plt.subplots(1, 1)
                    title = titles[s]

                    ax = plt.subplot(1, 1, 1, projection=proj)
                    figures.templates.cartopy_map(ax, da, hide_labels=False, lon_bnds=lon_bnds, lat_bnds=lat_bnds, levels=levels, cmap=cmap, add_colorbar=False)
                    shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                    plt.title(title)
                    plt.tight_layout()

                    cols[s].pyplot(fig)

with tab2:

    st.header("Portrait")

    # load data
    if useCat:
        import xscen as xs
        xs.load_config("paths.yml", "configs/cfg_figures.yml", reset=True)
        pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

        ds = pcat.search(processing_level="indicators", type="reconstruction-hydro", xrfreq="AS-DEC").to_dask()
        ds["season_length"] = ds["season_end"] - ds["season_start"]
        ds = ds.compute()
        ds_fx = pcat.search(processing_level="indicators", type="reconstruction-hydro", xrfreq="fx").to_dask()
        ds_fx = ds_fx.compute()

        # ZGIEBV shapefiles
        regions = glob.glob(f"{xs.CONFIG['gis']}ZGIEBV/*.shp")
        zgiebv = [gpd.read_file(shp) for shp in regions]

        # RADEAU shapefiles
        stations = ds_fx.where(ds_fx.atlas2018 != '', drop=True)
        cv = dict(zip(stations["atlas2018"].values, stations["station_id"].values))
        shp = gpd.read_file(f"{xs.CONFIG['gis']}RADEAU/CONSOM_SURF_BV_CF1_WGS84.shp")
        shp["BV_ID"] = shp["BV_ID"].map(cv)
        shp = shp.dropna(subset=["BV_ID"])
        shp = shp.set_index("BV_ID")
        shp = shp.sort_values("Shape_Area", ascending=False)

        # Portrait shapefiles
        shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
        shp_portrait = shp_portrait.set_index("TRONCON")

    else:
        raise NotImplementedError()

    # Selections
    cols = st.columns(2)
    option_y2 = cols[0].selectbox('Année', np.arange(1992, 2022), index=29)
    cols = st.columns(3)
    show_clim = cols[0].checkbox("Climatologie (année sera ignorée)", True)
    show_spec = cols[1].checkbox("Débits spécifiques", True)

    proj = cartopy.crs.PlateCarree()
    if show_clim is True:
        v = ['doy_14qmax', '14qmax', 'season_start', 'season_end', 'season_length', '7q2', '7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']
        q = [.10, .50, .90]
        for vv in v:
            cols = st.columns(3)

            bounds = np.linspace(**xs.CONFIG["figures"][vv]["clim"]["bnds"])
            norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            cmap = xs.CONFIG["figures"][vv]["clim"]["cmap"]
            if vv != "7q2":
                for s in range(3):
                    fig, _ = plt.subplots(1, 1)
                    if vv not in ['days_under_7q2', 'max_consecutive_days_under_7q2']:
                        da = ds[vv].sel(time=slice("1992-01-01", "2021-01-01")).quantile(q[s], dim="time")
                    else:
                        da = ds[vv].where(ds[vv] > 0).sel(time=slice("1992-01-01", "2021-01-01")).quantile(q[s], dim="time")
                    if show_spec and (vv in ['14qmax', '7q2', '7qmin']):
                        da = da / ds["drainage_area"] * 1000
                        da.name = vv

                    ax = plt.subplot(1, 1, 1, projection=proj)
                    # figures.templates.map_hydro(ax, da, shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                    # figures.templates.map_hydro(ax, da, shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, background=False, linewidth=1, norm=norm, cmap=cmap)
                    figures.templates.map_hydro(ax, da, shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=1, norm=norm, cmap=cmap)
                    plt.title(f"{vv}: {q[s]}")
                    plt.tight_layout()
                    cols[s].pyplot(fig)
            else:
                fig, _ = plt.subplots(1, 1)
                da = ds_fx[vv]
                if show_spec and (vv in ['14qmax', '7q2', '7qmin']):
                    da = da / ds["drainage_area"] * 1000
                    da.name = vv

                ax = plt.subplot(1, 1, 1, projection=proj)
                figures.templates.map_hydro(ax, da, shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=1, norm=norm, cmap=cmap)
                plt.title(f"{vv}")
                plt.tight_layout()

                cols[1].pyplot(fig)

    else:
        v = ['doy_14qmax', '14qmax', 'season_start', 'season_end', 'season_length', '7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']

        for vv in v:
            cols = st.columns(3)

            for s in range(3):
                fig, _ = plt.subplots(1, 1)
                if s == 0:
                    # Climatology
                    bounds = np.linspace(**xs.CONFIG["figures"][vv]["clim"]["bnds"])
                    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                    cmap = xs.CONFIG["figures"][vv]["clim"]["cmap"]
                    if vv not in ['days_under_7q2', 'max_consecutive_days_under_7q2']:
                        da = ds[vv].sel(time=slice("1992-01-01", "2021-01-01")).quantile(0.5, dim="time")
                    else:
                        da = ds[vv].where(ds[vv] > 0).sel(time=slice("1992-01-01", "2021-01-01")).quantile(0.5, dim="time")
                    if show_spec and (vv in ['14qmax', '7qmin']):
                        da = da / ds["drainage_area"] * 1000
                        da.name = vv
                    clim = da
                    title = f"{vv} - Climatologie"
                elif s == 1:
                    # Year
                    bounds = np.linspace(**xs.CONFIG["figures"][vv]["clim"]["bnds"])
                    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                    cmap = xs.CONFIG["figures"][vv]["clim"]["cmap"]
                    da = ds[vv].sel(time=slice(f"{option_y2}-01-01", f"{option_y2}-01-01"))
                    if show_spec and (vv in ['14qmax', '7qmin']):
                        da = da / ds["drainage_area"] * 1000
                        da.name = vv
                    title = f"{vv} - {option_y2}"
                else:
                    # Delta
                    bounds = np.linspace(**xs.CONFIG["figures"][vv]["deltas"]["bnds"])
                    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                    cmap = xs.CONFIG["figures"][vv]["deltas"]["cmap"]
                    da = ds[f"{vv}"].sel(time=slice(f"{option_y2}-01-01", f"{option_y2}-01-01"))
                    if show_spec and (vv in ['14qmax', '7qmin']):
                        da = da / ds["drainage_area"] * 1000
                        da.name = vv
                    if vv in ["14qmax", "7qmin"]:
                        da = (da - clim) / clim * 100
                    else:
                        da = da - clim
                    title = f"{vv} - Delta"

                if show_spec and (vv in ['14qmax', '7q2', '7qmin']):
                    da = da / ds["drainage_area"] * 1000
                    da.name = vv

                ax = plt.subplot(1, 1, 1, projection=proj)
                # figures.templates.map_hydro(ax, da, shp=shp, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, linestyle=":", edgecolor="k", norm=norm, cmap=cmap)
                # figures.templates.map_hydro(ax, da, shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, background=False, linewidth=1, norm=norm, cmap=cmap)
                figures.templates.map_hydro(ax, da, shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=1, norm=norm,
                                            cmap=cmap)
                plt.title(title)
                plt.tight_layout()
                cols[s].pyplot(fig)
