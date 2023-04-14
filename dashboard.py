import streamlit as st
import numpy as np
import cartopy
import matplotlib
from matplotlib import pyplot as plt
import geopandas as gpd
from xclim.sdba.utils import ecdf

import figures


st.set_page_config(layout="wide")

st.title('CASCADES')
st.header("Conséquences Attendues Survenant en Contexte d’Aggravation des Déficits d’Eau Sévères au Québec")

# useCat = st.checkbox("Use Local Data", True)
useCat = True
cols = st.columns(18)
run_spei = cols[0].checkbox("SPEI", False)
run_streamflow = cols[1].checkbox("Débits", False)
# Selections
cols = st.columns(2)
option_y = cols[0].selectbox('Année', np.arange(2021, 1971, -1), index=0)
option_hide = cols[1].selectbox("Niveau d\'intensité minimal",
                                ["Tout afficher", "≥ 1 (Modéré)", "≥ 1.5 (Sévère)", "≥ 2 (Extrême)", "≥ 2.5 (Très extrême)"], index=2)
hide = 0 if option_hide == "Tout afficher" else eval(option_hide.split("≥ ")[1].split(" (")[0])
show_som = st.checkbox("Sommaire seulement", True)

cols = st.columns(4)
lon_bnds = cols[0].slider('Longitude', -83., -55.5, (-80., -64.))
# cols = st.columns(3)
lat_bnds = cols[1].slider('Latitude', 43., 54., (44.75, 54.))

# Tabs
tab1, tab2 = st.tabs(["SPEI (hist)", "Débits (hist)"])
with tab1:
    st.header("ERA5-Land")

    speis = [3, 6, 9]
    levels = np.arange(-3, 3.5, 0.5)
    cmap = figures.utils.make_cmap("BrWhGr", 25)
    months = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

    if run_spei:

        # load data
        if useCat:
            import xscen as xs
            xs.load_config("paths.yml", reset=True)
            pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

            ds = pcat.search(processing_level="indicators", source="ERA5-Land").to_dask()
            ds = ds.where(ds.drainage_area > 25, drop=True)

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
            titles = ["mean(SPEI-3)", "mean(SPEI-6)", "mean(SPEI-9)"]

            for s in range(len(speis)):
                fig, _ = plt.subplots(1, 1)
                # da = ds[f"spei{speis[s]}"].sel(time=slice(f"{option_y}-06-01", f"{option_y}-10-01")).clip(min=-3.09, max=3.09).min(dim="time")
                da = ds[f"spei{speis[s]}"].sel(time=slice(f"{option_y}-06-01", f"{option_y}-10-01")).clip(min=-3.09, max=3.09).mean(dim="time")
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
                              f"{months[(month - 5 - 1)%12]}{' (' + str(option_y - 1) + ')' if month - 5 - 1 < 0 else ''}  à {months[month - 1]}",
                              f"{months[(month - 8 - 1)%12]}{' (' + str(option_y - 1) + ')' if month - 8 - 1 < 0 else ''} à {months[month - 1]}"]

                    da = ds[f"spei{speis[s]}"].sel(time=f"{option_y}-{month}-01").clip(min=-3.09, max=3.09)
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

    # Additional selections
    if show_som is False:
        cols = st.columns(3)
        show_clim = cols[0].checkbox("Climatologie (année sera ignorée)", False)
        show_spec = cols[1].checkbox("Débits spécifiques", False)
    else:
        show_clim = False

    hide_conv = {1: 0.15866, 1.5: 0.066807, 2: 0.034, 2.5: 0.006}

    if run_streamflow:
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

            # Open the ZGIEBV
            shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")

            # # RADEAU shapefiles
            # stations = ds_fx.where(ds_fx.atlas2018 != '', drop=True)
            # cv = dict(zip(stations["atlas2018"].values, stations["station_id"].values))
            # shp = gpd.read_file(f"{xs.CONFIG['gis']}RADEAU/CONSOM_SURF_BV_CF1_WGS84.shp")
            # shp["BV_ID"] = shp["BV_ID"].map(cv)
            # shp = shp.dropna(subset=["BV_ID"])
            # shp = shp.set_index("BV_ID")
            # shp = shp.sort_values("Shape_Area", ascending=False)

            # Portrait shapefiles
            shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
            shp_portrait = shp_portrait.set_index("TRONCON")

        else:
            raise NotImplementedError()

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
                        figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                        shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
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
                    figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, norm=norm, cmap=cmap)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                    shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                    plt.title(f"{vv}")
                    plt.tight_layout()

                    cols[1].pyplot(fig)

        elif show_som:
            v = [['doy_14qmax', '14qmax', ''], ['season_start', 'season_end', 'season_length'],
                 ['discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son'], ['7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']]

            for vv in v:
                cols = st.columns(3)

                for s in range(3):
                    if vv[s] != "":
                        # Empirical CDF
                        da = ds[vv[s]].sel(time=slice("1992-01-01", "2021-01-01"))
                        if vv[s] in ['days_under_7q2', 'max_consecutive_days_under_7q2']:
                            da = da.where(da > 0)
                            da.name = "days_under"
                        da = ecdf(da, da.sel(time=slice(f"{option_y}-01-01", f"{option_y}-01-01")).squeeze())
                        da = da.where(da > 0)
                        if hide != 0:
                            da = da.where((da <= hide_conv[hide]) | (da >= 1 - hide_conv[hide]))

                        # bounds = np.linspace(-3, 3, 13)
                        # bounds = np.linspace(0, 1, 13)
                        bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
                        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                        cmap = 'RdBu' if vv[s] not in ['season_end', 'season_length', 'days_under_7q2', 'max_consecutive_days_under_7q2'] else 'RdBu_r'

                        fig, _ = plt.subplots(1, 1)
                        title = f"{vv[s]}"

                        ax = plt.subplot(1, 1, 1, projection=proj)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, linewidth=0.25, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                        shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                        plt.title(title)
                        plt.tight_layout()
                        cols[s].pyplot(fig)

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
                        da = ds[vv].sel(time=slice(f"{option_y}-01-01", f"{option_y}-01-01"))
                        if show_spec and (vv in ['14qmax', '7qmin']):
                            da = da / ds["drainage_area"] * 1000
                            da.name = vv
                        title = f"{vv} - {option_y}"
                    else:
                        # Delta
                        bounds = np.linspace(**xs.CONFIG["figures"][vv]["deltas"]["bnds"])
                        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                        cmap = xs.CONFIG["figures"][vv]["deltas"]["cmap"]
                        da = ds[f"{vv}"].sel(time=slice(f"{option_y}-01-01", f"{option_y}-01-01"))
                        if show_spec and (vv in ['14qmax', '7qmin']):
                            da = da / ds["drainage_area"] * 1000
                            da.name = vv
                        if vv in ["14qmax", "7qmin"]:
                            da = (da - clim) / clim * 100
                        else:
                            da = da - clim
                        title = f"{vv} - Delta"

                    ax = plt.subplot(1, 1, 1, projection=proj)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area<=150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, norm=norm, cmap=cmap)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area>150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area>1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                    shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                    plt.title(title)
                    plt.tight_layout()
                    cols[s].pyplot(fig)
