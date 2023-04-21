import streamlit as st
import numpy as np
import cartopy
import matplotlib
from matplotlib import pyplot as plt
import xarray as xr
import geopandas as gpd
import pandas as pd

import figures


st.set_page_config(layout="wide")

st.title('CASCADES')
st.header("Conséquences Attendues Survenant en Contexte d’Aggravation des Déficits d’Eau Sévères au Québec")

# useCat = st.checkbox("Use Local Data", True)
useCat = True
cols = st.columns(10)
run_spei = cols[0].checkbox("SPEI", False)
run_streamflow = cols[1].checkbox("Débits historiques", False)
run_deltas = cols[2].checkbox("Deltas futurs", False)
run_analogs = cols[3].checkbox("Analogues", False)
# Selections
cols = st.columns(2)
option_y = cols[0].selectbox('Année', np.arange(2021, 1971, -1), index=0)
option_hide = cols[1].selectbox("Niveau d\'intensité minimal",
                                ["Tout afficher", "≥ 1 (Modéré)", "≥ 1.5 (Sévère)", "≥ 2 (Extrême)", "≥ 2.5 (Très extrême)"], index=2)
hide = 0 if option_hide == "Tout afficher" else eval(option_hide.split("≥ ")[1].split(" (")[0])
show_som = st.checkbox("Sommaire seulement", True)
show_spec = st.checkbox("Débits spécifiques", True)

cols = st.columns(4)
lon_bnds = cols[0].slider('Longitude', -83., -55.5, (-80., -64.))
# cols = st.columns(3)
lat_bnds = cols[1].slider('Latitude', 43., 54., (44.75, 54.))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["SPEI (hist)", "Débits (hist)", "Deltas futurs", "Analogues futurs"])
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

            ds = pcat.search(processing_level="indicators", source="ERA5-Land", variable=["spei3", "spei6", "spei9"]).to_dask()

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
    hide_conv = {1: 0.15866, 1.5: 0.066807, 2: 0.034, 2.5: 0.006}

    if run_streamflow:
        # load data
        if useCat:
            import xscen as xs

            xs.load_config("paths.yml", "configs/cfg_figures.yml", reset=True)
            pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

            ds = pcat.search(processing_level="indicators", type="reconstruction-hydro", xrfreq="AS-JAN").to_dask()
            ds["season_length"] = ds["season_end"] - ds["season_start"]
            ds["season_length"].attrs["units"] = "d"
            ecdf = pcat.search(processing_level=f"deltas-{option_y}", type="reconstruction-hydro", xrfreq="fx").to_dask()
            clim = pcat.search(processing_level="climatology", type="reconstruction-hydro", xrfreq="fx").to_dask()

            # Open the ZGIEBV
            shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")

            # Portrait shapefiles
            shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
            shp_portrait = shp_portrait.set_index("TRONCON")

            # Cleanup
            stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
            stations_atlas = stations_atlas.loc[stations_atlas["MASQUE"] == 0]  # Remove invalid/fake stations
            ds = ds.where(~ds.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ecdf = ecdf.where(~ecdf.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            clim = clim.where(~clim.station_id.isin(list(stations_atlas["TRONCON_ID"])))

        else:
            raise NotImplementedError()

        proj = cartopy.crs.PlateCarree()
        if show_som:
            v = [['doy_14qmax', '14qmax', ''], ['season_start', 'season_end', 'season_length'],
                 ['discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son'], ['7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']]

            for vv in v:
                cols = st.columns(3)

                for s in range(3):
                    if vv[s] != "":
                        da = ecdf[f"{vv[s]}_ecdf"]
                        if hide != 0:
                            da = da.where((da <= hide_conv[hide]) | (da >= 1 - hide_conv[hide]))

                        bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
                        norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                        cmap = 'RdBu' if vv[s] not in ['season_end', 'season_length', 'days_under_7q2', 'max_consecutive_days_under_7q2'] else 'RdBu_r'

                        fig, _ = plt.subplots(1, 1)
                        title = f"{vv[s]}"

                        ax = plt.subplot(1, 1, 1, projection=proj)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, linewidth=0.5, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=1, norm=norm, cmap=cmap)
                        figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                        shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                        plt.title(title)
                        plt.tight_layout()
                        cols[s].pyplot(fig)

        else:
            v = ['doy_14qmax', '14qmax', 'season_start', 'season_end', 'season_length',
                 'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son',
                 '7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']
            cmaps = {'doy_14qmax': "nipy_spectral", '14qmax': figures.utils.make_cmap("BrWhBu", 256), 'season_start': "nipy_spectral", 'season_end': "nipy_spectral", 'season_length': "hot_r",
                     'discharge_mean_mam': figures.utils.make_cmap("BrWhBu", 256), 'discharge_mean_jja': figures.utils.make_cmap("BrWhBu", 256), 'discharge_mean_son': figures.utils.make_cmap("BrWhBu", 256),
                     '7qmin': figures.utils.make_cmap("BrWhBu", 256), 'days_under_7q2': "hot_r", 'max_consecutive_days_under_7q2': "hot_r"}

            for vv in v:
                cols = st.columns(4)

                for s in range(4):
                    if s == 0:
                        da = clim[vv]
                        da4vm = ds[vv]
                        da.attrs['units'] = da4vm.attrs['units']
                        if show_spec and (vv in ["14qmax", 'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son', '7qmin']):
                            units = da.attrs["units"]
                            name = da.name
                            with xr.set_options(keep_attrs=True):
                                da = da / da.drainage_area
                                da4vm = da4vm / da4vm.drainage_area
                            da.attrs["units"] = f"{units} km-2"
                            da.name = name
                        title = f"{vv} - Climatologie ({da.attrs['units']})"
                        vmin = np.abs(da4vm).compute().quantile(0.01)
                        vmax = np.abs(da4vm).compute().quantile(0.99)
                        bounds = np.arange(vmin, vmax + (vmax - vmin) / 10, (vmax - vmin) / 10)
                        cmap = cmaps[vv]

                    elif s == 1:
                        da = ds[vv].sel(time=slice(str(option_y), str(option_y)))
                        if show_spec and (vv in ["14qmax", 'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son', '7qmin']):
                            name = da.name
                            units = da.attrs["units"]
                            with xr.set_options(keep_attrs=True):
                                da = da / da.drainage_area
                            da.name = name
                            da.attrs["units"] = f"{units} km-2"
                        title = f"{vv} - {option_y} ({da.attrs['units']})"

                    elif s == 2:
                        da = ecdf[f"{vv}_delta-abs"]
                        if show_spec and (vv in ["14qmax", 'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son', '7qmin']):
                            name = da.name
                            units = da.attrs["units"]
                            with xr.set_options(keep_attrs=True):
                                da = da / da.drainage_area
                            da.attrs["units"] = f"{units} km-2"
                            da.name = name
                        title = f"{vv} - Delta ({ecdf[f'{vv}_delta-abs'].attrs['units']})"
                        vm = np.abs(da).compute().quantile(0.99).values
                        bounds = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]) * vm
                        cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2',
                                                    'max_consecutive_days_under_7q2'] else 'RdBu_r'

                    else:
                        da = ecdf[f"{vv}_ecdf"]
                        title = f"{vv} - ECDF"
                        bounds = [0, 0.034, 0.066807, 0.15866, 0.30854, 0.5, 0.69146, 0.84134, 0.93319, 0.966, 1]
                        cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2',
                                                    'max_consecutive_days_under_7q2'] else 'RdBu_r'
                        if hide != 0:
                            da = da.where((da <= hide_conv[hide]) | (da >= 1 - hide_conv[hide]))

                    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                    fig, _ = plt.subplots(1, 1)
                    ax = plt.subplot(1, 1, 1, projection=proj)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, norm=norm, cmap=cmap)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                    figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                    shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                    plt.title(title)
                    plt.tight_layout()
                    cols[s].pyplot(fig)

with tab3:
    st.header("Deltas")

    if run_deltas:
        # load data
        if useCat:
            import xscen as xs

            xs.load_config("paths.yml", "configs/cfg_figures.yml", reset=True)
            pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

            ds1 = pcat.search(processing_level=f"deltas-{option_y}-1.5", activity="ClimEx.*", xrfreq="AS-JAN").to_dask()
            ds2 = pcat.search(processing_level=f"deltas-{option_y}-2", activity="ClimEx.*", xrfreq="AS-JAN").to_dask()
            ds3 = pcat.search(processing_level=f"deltas-{option_y}-3", activity="ClimEx.*", xrfreq="AS-JAN").to_dask()
            ds4 = pcat.search(processing_level=f"deltas-{option_y}-4", activity="ClimEx.*", xrfreq="AS-JAN").to_dask()

            # Open the ZGIEBV
            shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")

            # Portrait shapefiles
            shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
            shp_portrait = shp_portrait.set_index("TRONCON")

            # Cleanup
            stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
            stations_atlas = stations_atlas.loc[stations_atlas["MASQUE"] == 0]  # Remove invalid/fake stations
            ds1 = ds1.where(~ds1.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ds2 = ds2.where(~ds2.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ds3 = ds3.where(~ds3.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ds4 = ds4.where(~ds4.station_id.isin(list(stations_atlas["TRONCON_ID"])))

        else:
            raise NotImplementedError()

        proj = cartopy.crs.PlateCarree()

        v = ['doy_14qmax', '14qmax', 'season_start', 'season_end', 'season_length',
             'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son',
             '7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']
        wl = ["+1.5°C", "+2.0°C", "+3.0°C", "+4.0°C"]

        for vv in v:
            cols = st.columns(4)
            dav = [ds1[vv], ds2[vv], ds3[vv], ds4[vv]]
            if dav[0].attrs['delta_kind'] == "percentage":
                vm = 100
            else:
                vm = np.abs(xr.concat(dav, dim="warming_level")).compute().quantile(0.99).values
            bounds = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]) * vm

            for s in range(4):
                da = dav[s]

                title = f"{vv} - Deltas {wl[s]} ({da.attrs['units']})"
                cmap = 'RdBu' if vv not in ['season_end', 'season_length', 'days_under_7q2', 'max_consecutive_days_under_7q2',
                                            'season_histend', 'season_histlength', 'days_under_hist7q2',
                                            'max_consecutive_days_under_hist7q2'] else 'RdBu_r'

                norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                fig, _ = plt.subplots(1, 1)
                ax = plt.subplot(1, 1, 1, projection=proj)
                figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, norm=norm, cmap=cmap)
                figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                plt.title(title)
                plt.tight_layout()
                cols[s].pyplot(fig)

with tab4:
    st.header("Analogues")

    if run_analogs:
        # load data
        if useCat:
            import xscen as xs

            xs.load_config("paths.yml", "configs/cfg_figures.yml", reset=True)
            pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

            ds = pcat.search(processing_level="indicators", type="reconstruction-hydro", xrfreq="AS-JAN").to_dask()
            ds["season_length"] = ds["season_end"] - ds["season_start"]
            ds["season_length"].attrs["units"] = "d"
            ds = ds.assign_coords({"warminglevel": "hist"})
            ds1 = pcat.search(processing_level=f"analog-{option_y}-1.5", type="reconstruction-hydro", xrfreq="AS-JAN").to_dask()
            ds2 = pcat.search(processing_level=f"analog-{option_y}-2", type="reconstruction-hydro", xrfreq="AS-JAN").to_dask()
            ds3 = pcat.search(processing_level=f"analog-{option_y}-3", type="reconstruction-hydro", xrfreq="AS-JAN").to_dask()
            ds4 = pcat.search(processing_level=f"analog-{option_y}-4", type="reconstruction-hydro", xrfreq="AS-JAN").to_dask()

            # Open the ZGIEBV
            shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")

            # Portrait shapefiles
            shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
            shp_portrait = shp_portrait.set_index("TRONCON")

            # Cleanup
            stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
            stations_atlas = stations_atlas.loc[stations_atlas["MASQUE"] == 0]  # Remove invalid/fake stations
            ds = ds.where(~ds.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ds1 = ds1.where(~ds1.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ds2 = ds2.where(~ds2.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ds3 = ds3.where(~ds3.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            ds4 = ds4.where(~ds4.station_id.isin(list(stations_atlas["TRONCON_ID"])))

        else:
            raise NotImplementedError()

        proj = cartopy.crs.PlateCarree()

        v = ['doy_14qmax', '14qmax', 'season_start', 'season_end', 'season_length',
             'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son',
             '7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']
        cmaps = {'doy_14qmax': "nipy_spectral", '14qmax': figures.utils.make_cmap("BrWhBu", 256), 'season_start': "nipy_spectral",
                 'season_end': "nipy_spectral", 'season_length': "hot_r",
                 'discharge_mean_mam': figures.utils.make_cmap("BrWhBu", 256), 'discharge_mean_jja': figures.utils.make_cmap("BrWhBu", 256),
                 'discharge_mean_son': figures.utils.make_cmap("BrWhBu", 256),
                 '7qmin': figures.utils.make_cmap("BrWhBu", 256), 'days_under_7q2': "hot_r", 'max_consecutive_days_under_7q2': "hot_r"}
        wl = ["+1.5°C", "+2.0°C", "+3.0°C", "+4.0°C"]

        for vv in v:
            cols = st.columns(5)
            dav = [ds[vv].sel(time=slice(f"{option_y}", f"{option_y}")), ds1[vv], ds2[vv], ds3[vv], ds4[vv]]
            if show_spec and (vv in ["14qmax", 'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son', '7qmin']):
                name = vv
                units = dav[0].attrs["units"]
                with xr.set_options(keep_attrs=True):
                    for i in range(len(dav)):
                        dav[i] = dav[i] / dav[i].drainage_area
                        dav[i].name = name
                        dav[i].attrs["units"] = f"{units} km-2"

            da4vm = xr.concat(dav, dim="warminglevel").compute()
            vmin = da4vm.quantile(0.01)
            vmax = da4vm.quantile(0.99)
            bounds = np.arange(vmin, vmax + (vmax - vmin) / 10, (vmax - vmin) / 10)

            for s in range(5):
                da = dav[s]

                if s == 0:
                    title = f"{vv} - {option_y} ({da.attrs['units']})"
                else:
                    title = f"{vv} - Analogue {wl[s-1]} ({da.attrs['units']})"
                cmap = cmaps[vv]

                norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
                fig, _ = plt.subplots(1, 1)
                ax = plt.subplot(1, 1, 1, projection=proj)
                figures.templates.map_hydro(ax, da.where(da.drainage_area <= 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=True, linewidth=0.25, norm=norm, cmap=cmap)
                figures.templates.map_hydro(ax, da.where(da.drainage_area > 150), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=0.5, norm=norm, cmap=cmap)
                figures.templates.map_hydro(ax, da.where(da.drainage_area > 1000), shp=shp_portrait, lon_bnds=lon_bnds, lat_bnds=lat_bnds, legend=False, background=False, linewidth=2, norm=norm, cmap=cmap)
                shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k")
                plt.title(title)
                plt.tight_layout()
                cols[s].pyplot(fig)