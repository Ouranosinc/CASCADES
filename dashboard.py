import streamlit as st
import numpy as np
import cartopy
from matplotlib import pyplot as plt
import xarray as xr
import geopandas as gpd
import pandas as pd
import xscen as xs
from xclim.sdba.utils import ecdf
import spirograph.matplotlib as sp
from copy import deepcopy


# xscen setup
xs.load_config("paths.yml", reset=True)
pcat = xs.ProjectCatalog(xs.CONFIG["project"]["path"], create=False)

# Open the ZGIEBV
shp_zg = gpd.read_file(f"{xs.CONFIG['gis']}ZGIEBV/ZGIEBV_WGS84.shp")
shp_zg = shp_zg.set_index("SIGLE")
# Portrait shapefiles
shp_portrait = gpd.read_file(f"{xs.CONFIG['gis']}atlas2022/AtlasHydroclimatique_2022.shp")
shp_portrait = shp_portrait.set_index("TRONCON")
stations_atlas = pd.read_csv(f"{xs.CONFIG['dpphc']['portrait']}Metadata_Portrait.csv", encoding="ISO-8859-1")
stations_atlas = stations_atlas.loc[stations_atlas["MASQUE"] == 0]  # Remove invalid/fake stations

proj = cartopy.crs.PlateCarree()
shp_zg = shp_zg.to_crs(proj)
shp_portrait = shp_portrait.to_crs(proj)
features = {'coastline': {}, 'borders': {}, 'lakes': {"edgecolor": "black"}, 'ocean': {}, 'rivers': {}, 'states': {"edgecolor": "black", "linestyle": "dotted"}}

st.set_page_config(layout="wide")

st.title('CASCADES')
st.header("Conséquences Attendues Survenant en Contexte d’Aggravation des Déficits d’Eau Sévères au Québec")

cols = st.columns(4)
run = cols[0].selectbox('Afficher', ["Rien", "SPEI", "Climat", "Débits"], index=0)
if run != "Rien":
    if run in ["Climat", "Débits"]:
        what = cols[1].selectbox('Afficher', ["Valeurs absolues", "Deltas vs Année", "Deltas vs Moy."], index=0)

    # Bounds
    cols = st.columns(4)
    lon_bnds = cols[0].slider('Longitude', -83., -55.5, (-80., -64.))
    # cols = st.columns(3)
    lat_bnds = cols[1].slider('Latitude', 43., 54., (44.75, 54.))

    # Selections
    cols = st.columns(2)
    option_y = cols[0].selectbox('Année', np.arange(2021, 1971, -1), index=0)
    option_hide = cols[1].selectbox("Niveau d\'intensité minimal",
                                    ["Tout afficher", "≥ 1 (Modéré)", "≥ 1.5 (Sévère)", "≥ 2 (Extrême)", "≥ 2.5 (Très extrême)"], index=2)
    hide = 0 if option_hide == "Tout afficher" else eval(option_hide.split("≥ ")[1].split(" (")[0])
    hide_conv = {1: 0.15866, 1.5: 0.066807, 2: 0.034, 2.5: 0.006}

    # Plot the mean of all SPEI between June and October
    cols = st.columns(5)
    cols[0].markdown("<h3 style='text-align: center; color: black;'>Référence</h3>", unsafe_allow_html=True)
    cols[1].markdown("<h3 style='text-align: center; color: black;'>+1.5°C</h3>", unsafe_allow_html=True)
    cols[2].markdown("<h3 style='text-align: center; color: black;'>+2.0°C</h3>", unsafe_allow_html=True)
    cols[3].markdown("<h3 style='text-align: center; color: black;'>+3.0°C</h3>", unsafe_allow_html=True)
    cols[4].markdown("<h3 style='text-align: center; color: black;'>+4.0°C</h3>", unsafe_allow_html=True)
    a = ["ref", "1.5", "2", "3", "4"]

    if run == "SPEI":
        v = ['spei3-5', 'spei3-6', 'spei3-7', 'spei3-8', 'spei3-9', 'spei3-10', 'spei6-5', 'spei6-10']

        ref = pcat.search(processing_level="indicators", source="ERA5-Land", variable=["spei3", "spei6"]).to_dask()

        analogs = {}
        for wl in a[1:]:
            tmp = pcat.search(processing_level=f"{wl}-performance-vs-{option_y}").to_dask().rmse.stack({"stacked": ["time", "realization"]})
            tmp = tmp.sortby(tmp)
            tmp2 = []
            for i in range(5):
                member = str(tmp.isel(stacked=i).realization.values).split(".")[0].split("_")[-1]
                analog_year = int(tmp.isel(stacked=i).time.dt.year.values)
                tmp2.extend([pcat.search(activity="ClimEx", processing_level=f"indicators-warminglevel-{wl}vs.*", member=member, variable=["spei3", "spei6"]).to_dask().sel(time=slice(str(analog_year), str(analog_year))).compute().squeeze()])

            analogs[f"{wl}decC"] = xr.concat(tmp2, dim="realization").groupby("time.month").mean(dim=["realization", "time"], keep_attrs=True)

        for vv in v:
            spei = vv.split("-")[0]
            month = int(vv.split("-")[1])

            cols = st.columns(5)
            for s in range(5):
                if (s == 0) or (len(analogs) > 0):
                    if s == 0:
                        da = ref[spei].sel(time=slice(f"{option_y}-{month:02}", f"{option_y}-{month:02}")).squeeze()
                    else:
                        da = analogs[f"{a[s]}decC"][spei].sel(month=month)

                    da = da.where(np.abs(da) > hide)

                    fig, _ = plt.subplots(1, 1)
                    ax = plt.subplot(1, 1, 1, projection=proj)
                    ax = sp.gridmap(da, ax=ax, cmap='prec_div', contourf=True, features=features, divergent=0, frame=True,
                                    plot_kw={"vmin": -3, "vmax": 3, "levels": 13, 'add_colorbar': False})
                    ax.set_title(f"{spei}-{month}")
                    shp_zg.plot(ax=ax, facecolor="None", edgecolor="k", linewidth=0.5)
                    ax.set_xlim(lon_bnds[0], lon_bnds[1])
                    ax.set_ylim(lat_bnds[0], lat_bnds[1])

                    cols[s].pyplot(fig)
        #
        #
        #
        #
        #
        #
        #
        #
        #
        # v = ['spei3-5', 'spei3-6']
        # speis = [3, 6, 9]
        # months = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
        #
        #
        #
        #     # Plot the mean of all SPEI between June and October
        #     cols = st.columns(3)
        #     cols[0].markdown("<h3 style='text-align: center; color: black;'>Mean SPEI-3</h3>", unsafe_allow_html=True)
        #     cols[1].markdown("<h3 style='text-align: center; color: black;'>Mean SPEI-6</h3>", unsafe_allow_html=True)
        #     cols[2].markdown("<h3 style='text-align: center; color: black;'>Mean SPEI-9</h3>", unsafe_allow_html=True)
        #     cols = st.columns(len(speis))
        #     titles = ["mean(SPEI-3)", "mean(SPEI-6)", "mean(SPEI-9)"]
        #
        #     for s in range(len(speis)):
        #         da = ds[f"spei{speis[s]}"].sel(time=slice(f"{option_y}-06-01", f"{option_y}-10-01")).clip(min=-3.09, max=3.09).mean(dim="time")
        #         da = da.sel(lat=slice(lat_bnds[0], lat_bnds[1]), lon=slice(lon_bnds[0], lon_bnds[1]))
        #         da = da.where(np.abs(da) > hide)
        #         title = titles[s]
        #
        #         fig, _ = plt.subplots(1, 1)
        #         ax = plt.subplot(1, 1, 1, projection=proj)
        #         sp.gridmap(da, ax=ax, cmap='prec_div', contourf=True, features=features, divergent=0, frame=True, plot_kw={"vmin": -3, "vmax": 3, "levels": 13})
        #         shp_zg.plot(ax=ax, facecolor="None", edgecolor="k", linewidth=0.5)
        #         ax.set_xlim(lon_bnds[0], lon_bnds[1])
        #         ax.set_ylim(lat_bnds[0], lat_bnds[1])
        #
        #         cols[s].pyplot(fig)
        #
        # else:
        #     # Plot every month from June to October
        #     cols = st.columns(3)
        #     cols[0].markdown("<h3 style='text-align: center; color: black;'>SPEI-3</h3>", unsafe_allow_html=True)
        #     cols[1].markdown("<h3 style='text-align: center; color: black;'>SPEI-6</h3>", unsafe_allow_html=True)
        #     cols[2].markdown("<h3 style='text-align: center; color: black;'>SPEI-9</h3>", unsafe_allow_html=True)
        #     for month in [5, 6, 7, 8, 9, 10, 11]:
        #         cols = st.columns(len(speis))
        #         for s in range(len(speis)):
        #             titles = [f"{months[(month - 2 - 1) % 12]} à {months[month - 1]}",
        #                       f"{months[(month - 5 - 1) % 12]}{' (' + str(option_y - 1) + ')' if month - 5 - 1 < 0 else ''}  à {months[month - 1]}",
        #                       f"{months[(month - 8 - 1) % 12]}{' (' + str(option_y - 1) + ')' if month - 8 - 1 < 0 else ''} à {months[month - 1]}"]
        #             da = ds[f"spei{speis[s]}"].sel(time=f"{option_y}-{month}-01").clip(min=-3.09, max=3.09)
        #             da = da.sel(lat=slice(lat_bnds[0], lat_bnds[1]), lon=slice(lon_bnds[0], lon_bnds[1]))
        #             da = da.where(np.abs(da) > hide)
        #
        #             if not np.isnan(da).all():
        #                 title = titles[s]
        #
        #                 fig, _ = plt.subplots(1, 1)
        #                 ax = plt.subplot(1, 1, 1, projection=proj)
        #                 sp.gridmap(da, ax=ax, cmap='prec_div', contourf=True, features=features, divergent=0, frame=True,
        #                            plot_kw={"vmin": -3, "vmax": 3, "levels": 13, "add_colorbar": False})
        #                 shp_zg.to_crs(proj).plot(ax=ax, facecolor="None", edgecolor="k", linewidth=0.5)
        #                 ax.set_xlim(lon_bnds[0], lon_bnds[1])
        #                 ax.set_ylim(lat_bnds[0], lat_bnds[1])
        #                 plt.title(title)
        #
        #                 cols[s].pyplot(fig)

    else:
        v = ['precip_accumulation_yr', 'evspsblpot_accumulation_yr', 'tg_mean_yr'] if run == "Climat" else \
            ['doy_14qmax', '14qmax', 'season_start', 'season_end', 'season_length',
             'discharge_mean_mam', 'discharge_mean_jja', 'discharge_mean_son',
             '7qmin', 'days_under_7q2', 'max_consecutive_days_under_7q2']
        if what == "Valeurs absolues":
            cmaps = {'precip_accumulation_yr': 'prec_seq', 'evspsblpot_accumulation_yr': 'prec_seq', 'tg_mean_yr': 'temp_seq',
                     'doy_14qmax': "nipy_spectral", 'season_start': "nipy_spectral", 'season_end': "nipy_spectral",
                     '14qmax': "Blues", 'discharge_mean_mam': "Blues", 'discharge_mean_jja': "Blues", 'discharge_mean_son': "Blues", '7qmin': "Blues",
                     'season_length': "hot_r", 'days_under_7q2': "hot_r", 'max_consecutive_days_under_7q2': "hot_r"}
        else:
            cmaps = {'precip_accumulation_yr': 'prec_div', 'evspsblpot_accumulation_yr': 'prec_div', 'tg_mean_yr': 'temp_div',
                     'doy_14qmax': "PuOr_r", 'season_start': "PuOr_r", 'season_end': "PuOr_r",
                     '14qmax': "RdBu", 'discharge_mean_mam': "RdBu", 'discharge_mean_jja': "RdBu", 'discharge_mean_son': "RdBu", '7qmin': "RdBu",
                     'season_length': "misc_div", 'days_under_7q2': "misc_div", 'max_consecutive_days_under_7q2': "misc_div"}

        if run == "Climat":
            ref = pcat.search(processing_level="indicators", source="ERA5.*", xrfreq="AS-DEC").to_dask().sel(time=slice("1992", "2021")).compute()
            analogs = pcat.search(processing_level=f"analog-{option_y}-.*", source="ERA5.*", xrfreq="AS-DEC").to_dataset_dict()
            if len(analogs) > 0:
                for k in analogs:
                    analogs[k] = analogs[k].compute()
        else:
            ref = pcat.search(processing_level="indicators", type="reconstruction-hydro", xrfreq="AS-JAN").to_dask().compute()
            ref["season_length"] = ref["season_end"] - ref["season_start"]
            ref["season_length"].attrs["units"] = "d"
            ref = ref.where(~ref.station_id.isin(list(stations_atlas["TRONCON_ID"])))
            analogs = pcat.search(processing_level=f"analog-{option_y}-.*", type="reconstruction-hydro", xrfreq="AS-JAN").to_dataset_dict()
            if len(analogs) > 0:
                for k in analogs:
                    analogs[k] = analogs[k].compute()
                    analogs[k] = analogs[k].where(~analogs[k].station_id.isin(list(stations_atlas["TRONCON_ID"])))

        for vv in v:
            # Specific discharge
            if (vv in ["14qmax", "7qmin", "discharge_mean_mam", "discharge_mean_jja", "discharge_mean_son"]) and ("km-2" not in ref[vv].attrs["units"]):
                attrs = ref[vv].attrs
                ref[vv] = ref[vv] / ref.drainage_area * 1000
                ref[vv].attrs = attrs
                ref[vv].attrs["units"] = ref[vv].attrs["units"].replace("m3", "L").replace("m^3", "L") + ' km-2'
            if len(analogs) > 0:
                for k in analogs:
                    if (vv in ["14qmax", "7qmin", "discharge_mean_mam", "discharge_mean_jja", "discharge_mean_son"]) and ("km-2" not in analogs[k][vv].attrs["units"]):
                        attrs = analogs[k][vv].attrs
                        analogs[k][vv] = analogs[k][vv] / ref.drainage_area * 1000
                        analogs[k][vv].attrs = attrs
                        analogs[k][vv].attrs["units"] = analogs[k][vv].attrs["units"].replace("m3", "L").replace("m^3", "L") + ' km-2'

            if len(analogs) > 0:
                vmin = xr.concat([ref[vv].assign_coords({"warminglevel": 0.91})] + [analogs[k][vv] for k in analogs], dim="warminglevel").quantile(0.01).values
                vmax = xr.concat([ref[vv].assign_coords({"warminglevel": 0.91})] + [analogs[k][vv] for k in analogs], dim="warminglevel").quantile(0.99).values
            else:
                vmin = ref[vv].quantile(0.01)
                vmax = ref[vv].quantile(0.99)

            cols = st.columns(5)
            for s in range(5):
                if (s == 0) or (len(analogs) > 0):
                    if s == 0:
                        da = ref[vv].sel(time=slice(f"{option_y if ref.attrs['cat:xrfreq'] == 'AS-JAN' else option_y - 1}", f"{option_y if ref.attrs['cat:xrfreq'] == 'AS-JAN' else option_y - 1}"))
                    else:
                        k = [k for k in analogs if f"-{a[s]}." in k][0]
                        da = analogs[k][vv]

                    if what == "Deltas vs Année":
                        da = da - ref[vv].sel(time=slice(f"{option_y if ref.attrs['cat:xrfreq'] == 'AS-JAN' else option_y - 1}", f"{option_y if ref.attrs['cat:xrfreq'] == 'AS-JAN' else option_y - 1}"))
                        stdev = ref[vv].std().values
                        vmin = -(3 * stdev)
                        vmax = 3 * stdev
                    elif what == "Deltas vs Moy.":
                        da = da - ref[vv].mean(dim="time")
                        stdev = ref[vv].std().values
                        vmin = -(5 * stdev)
                        vmax = 5 * stdev

                    if hide != 0 and what == "Valeurs absolues":
                        da_ecdf = ecdf(ref[vv].squeeze(), da.squeeze())
                        da = da.where((da_ecdf <= hide_conv[hide]) | (da_ecdf >= 1 - hide_conv[hide]), drop=True).squeeze()

                    levels = list(np.arange(vmin, vmax + (vmax - vmin) / 12 - 0.1, (vmax - vmin) / 12))
                    if len(da.dims) > 0:
                        # Add data to the shapefile
                        if run == "Climat":
                            shp = deepcopy(shp_zg)
                            da2 = da.to_dataframe().set_index("SIGLE")
                            cmap = cmaps[da.name]
                            plot_kw = None
                        else:
                            shp = deepcopy(shp_portrait)
                            da2 = da.to_dataframe().set_index("station_id")
                            cmap = cmaps[da.name]
                            plot_kw = {"linewidth": 1}
                        i = da2.index.intersection(shp.index)
                        shp[da.name] = da2.loc[i][da.name]

                        fig, _ = plt.subplots(1, 1)
                        ax = plt.subplot(1, 1, 1, projection=proj)
                        sp.gdfmap(shp, da.name, ax=ax, features=features, cmap=cmap, frame=True, levels=levels, plot_kw=plot_kw)
                        shp_zg.plot(ax=ax, facecolor="None", edgecolor="k", linewidth=0.5)
                        ax.set_xlim(lon_bnds[0], lon_bnds[1])
                        ax.set_ylim(lat_bnds[0], lat_bnds[1])

                        cols[s].pyplot(fig)
