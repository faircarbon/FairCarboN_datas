import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import folium
from folium.features import CustomIcon
from streamlit_folium import st_folium
from streamlit.components.v1 import html
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
import json
from shapely.geometry import shape, Polygon, MultiPolygon
import shapely
from folium.features import GeoJson
import time
import os
import geopandas as gpd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
pio.templates.default = "plotly"

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON DATA",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

######################################################################################################################
########### FONCTIONS SUPPORTS #######################################################################################
######################################################################################################################

@st.cache_data
def read_data(path):
    # Chemin vers le fichier Excel
    #fichier_excel = "Data\FairCarboN_Datas_V2.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=0,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")

    return df

def save_map_as_png(html_path, png_path):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--window-size=3000,2000")


    driver = webdriver.Chrome(options=options)
    driver.get("file://" + os.path.abspath(html_path))

    time.sleep(2)  # laisse la carte se charger

    driver.save_screenshot(png_path)
    driver.quit()

def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_countries_geojson():
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    r = requests.get(url)
    geojson = r.json()

    # 🔥 Sauvegarde locale du dictionnaire des géométries
    with open("countries.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    return geojson


def get_country_shape(geojson, country_name):
    for feature in geojson["features"]:
        if feature["properties"]["name"].lower() == country_name.lower():
            return shape(feature["geometry"])
    return None

def create_world_mask(country_shape):
    # Polygone géant couvrant la planète
    world = Polygon([
        (-180, -90), (-180, 90),
        (180, 90), (180, -90)
    ])

    # On retire le pays → trou dans le masque
    mask = world.difference(country_shape)
    return mask

######################################################################################################################
########### PARAMETRES ###############################################################################################
######################################################################################################################


df_Labo_Site3 = read_data("Data\FairCarboN_Datas_Labo3")


cartounitparprojet = st.checkbox("Carto par projet")
cartositesparprojet = st.checkbox("Carto sites par projet")
tests_tous = st.checkbox("toutes les unités")
tests_tous2 = st.checkbox("tous les sites")

projects2 = sorted(df_Labo_Site3['projet'].unique())

#choix de visualisation2
col1, col2, col3 =st.columns([0.4,0.3,0.3])
with col1:
    st.subheader(f":grey[Choix de visualisation]")
    st.markdown("Choix obligatoire")
with col2:
    Unites2 = st.checkbox('Unités2')
with col3:
    Sites2 = st.checkbox('Sites2')

Selection_projets2 = st.multiselect('',options=projects2)

if len(Selection_projets2)==0: #aucun choix
    df_selected2 = df_Labo_Site3 #le dataframe ne change pas, c'est l'original
else:
    df_selected2 = df_Labo_Site3[df_Labo_Site3['projet'].isin(Selection_projets2)]

# Regrouper par projet
grouped2 = df_selected2.groupby(['Sigle structure','Type_Data','Latitude', 'Longitude','Pays','Logos'])['projet'].apply(list).reset_index()

if Unites2:
    grouped_2 = grouped2[grouped2['Type_Data']=='Labo']

elif Sites2:
    grouped_2 = grouped2[grouped2['Type_Data']=='Site']
else:
    grouped_2 = pd.DataFrame()

############################################################
############## PARAMETRES ##################################

contour_unites = "#5F8114"
ligne_unites = "#EC4715"
markers_unites = "#EC4715"
inter_unites = 1
marge_unites = 3
marge_unites_v = 0

contour_sites = "#5F8114"
ligne_sites = "#F1AD18"
markers_sites = "#F3BF31"
inter_sites = 1.5
marge_sites = 3
marge_sites_v = -2

#pin_icon_url = "https://cdn-icons-png.flaticon.com/512/2776/2776067.png"
pin_icon_url = "Data/pin_test.png"
pin_icon_url2 = "Data/pin_orange2.png"
############################################################



if cartounitparprojet:
    st.dataframe(grouped_2)

    grouped_2 = grouped_2.sort_values(by="Latitude", ascending=True)

    countries = grouped_2["Pays"].unique()

    # Charger GeoJSON mondial
    geojson_world = load_countries_geojson()

    for country in countries:
        st.subheader(f"📍 {country}")

        df_country = grouped_2[grouped_2["Pays"] == country]

        # Récupérer tous les pays présents dans les données
        countries_in_data = df_country["Pays"].unique()

        # Créer une MultiPolygon avec tous les pays concernés
        all_countries_shapes = []
        for country in countries_in_data:
            country_shape = get_country_shape(geojson_world, country)
            if country_shape is not None:
                all_countries_shapes.append(country_shape)

        # Combiner tous les pays en une seule forme
        from shapely.ops import unary_union
        combined_shape = unary_union(all_countries_shapes)

        # Création du masque mondial
        mask_shape = create_world_mask(combined_shape)

        # Carte centrée sur le monde
        m = folium.Map(
            location=[df_country["Latitude"].mean(), df_country["Longitude"].mean()],
            zoom_start=2
        )

        # Ajouter le masque blanc
        GeoJson(
            data=json.loads(json.dumps(mask_shape.__geo_interface__)),
            style_function=lambda x: {
                "fillColor": "white",
                "color": "white",
                "weight": 1,
                "fillOpacity": 1
            }
        ).add_to(m)

        # Ajouter TOUS les pays du monde avec contours gris
        GeoJson(
            data=geojson_world,
            style_function=lambda x: {
                "fillColor": "#ffffff00",
                "color": "#cccccc",  # Gris clair pour tous les pays
                "weight": 1,
                "fillOpacity": 0
            }
        ).add_to(m)

        # Ajouter les pays avec données en rouge par-dessus
        GeoJson(
            data=json.loads(json.dumps(combined_shape.__geo_interface__)),
            style_function=lambda x: {
                "fillColor": "#ffffff00",
                "color": contour_unites,
                "weight": 2,
                "fillOpacity": 0
            }
        ).add_to(m)

        # Position de la colonne des labels (à droite de la carte)
        label_lon = df_country["Longitude"].max() + marge_unites

        # Espacement vertical entre les labels
        lat_min = df_country["Latitude"].min()
        lat_max = df_country["Latitude"].max()
        step = (lat_max - lat_min) * inter_unites / (len(df_country) + 1)

        # Index pour placer les labels
        i = 1

        # Ajouter les marqueurs avec lignes et labels
        for _, row in df_country.iterrows():
            img_b64 = image_to_base64(row["Logos"])
            # Marqueur circulaire (point vert)
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                icon=folium.CustomIcon(
                icon_image=pin_icon_url,
                icon_size=(24, 24))
            ).add_to(m)
            
            # Position du label dans la colonne
            label_lat = lat_min + step * i + marge_unites_v
            i += 1
            
            # Ligne entre le point et le label
            folium.PolyLine(
                locations=[[row["Latitude"], row["Longitude"]], [label_lat, label_lon]],
                color=ligne_unites,
                weight=1.5,
                opacity=0.9
            ).add_to(m)
            
            # Label avec marqueur circulaire + nom de la structure
            html_label = f"""
            <div style="
                display: flex;
                flex-direction: row;
                align-items: center;
            ">
                <div style="
                    width: 64px;
                    height: 64px;
                    border-radius: 50%;
                    overflow: hidden;
                    border: 3px solid {markers_unites};
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <img src="data:image/png;base64,{img_b64}"
                        style="width: 100%; height: 100%; object-fit: contain;">
                </div>
                <span style="color:black; font-size:16px; font-weight:600;">
                    {row['Sigle structure']}
                </span>
            </div>
            """
            
            folium.Marker(
                location=[label_lat, label_lon],
                icon=folium.DivIcon(
                    html=html_label,
                    icon_size=(250, 30),
                    icon_anchor=(0, 15)
                )
            ).add_to(m)

        # Affichage Streamlit
        map_html = m._repr_html_()
        html(map_html, height=1200)

        # Sauvegarde et téléchargement
        html_file = f"carte_{country}.html"
        png_file = f"carte_{country}.png"

if cartositesparprojet:
    st.dataframe(grouped_2)

    grouped_2 = grouped_2.sort_values(by="Latitude", ascending=True)

    countries = grouped_2["Pays"].unique()

    # Charger GeoJSON mondial
    geojson_world = load_countries_geojson()

    for country in countries:
        st.subheader(f"📍 {country}")

        df_country = grouped_2[grouped_2["Pays"] == country]

        # Récupérer tous les pays présents dans les données
        countries_in_data = df_country["Pays"].unique()

        # Créer une MultiPolygon avec tous les pays concernés
        all_countries_shapes = []
        for country in countries_in_data:
            country_shape = get_country_shape(geojson_world, country)
            if country_shape is not None:
                all_countries_shapes.append(country_shape)

        # Combiner tous les pays en une seule forme
        from shapely.ops import unary_union
        combined_shape = unary_union(all_countries_shapes)

        # Création du masque mondial
        mask_shape = create_world_mask(combined_shape)

        # Carte centrée sur le monde
        m = folium.Map(
            location=[df_country["Latitude"].mean(), df_country["Longitude"].mean()],
            zoom_start=2
        )

        # Ajouter le masque blanc
        GeoJson(
            data=json.loads(json.dumps(mask_shape.__geo_interface__)),
            style_function=lambda x: {
                "fillColor": "white",
                "color": "white",
                "weight": 1,
                "fillOpacity": 1
            }
        ).add_to(m)

        # Ajouter TOUS les pays du monde avec contours gris
        GeoJson(
            data=geojson_world,
            style_function=lambda x: {
                "fillColor": "#ffffff00",
                "color": "#cccccc",  # Gris clair pour tous les pays
                "weight": 1,
                "fillOpacity": 0
            }
        ).add_to(m)

        # Ajouter les pays avec données en rouge par-dessus
        GeoJson(
            data=json.loads(json.dumps(combined_shape.__geo_interface__)),
            style_function=lambda x: {
                "fillColor": "#ffffff00",
                "color": contour_sites,
                "weight": 2,
                "fillOpacity": 0
            }
        ).add_to(m)

        # Position de la colonne des labels (à droite de la carte)
        label_lon = df_country["Longitude"].max() + marge_sites

        # Espacement vertical entre les labels
        lat_min = df_country["Latitude"].min()
        lat_max = df_country["Latitude"].max()
        step = (lat_max - lat_min) * inter_sites / (len(df_country) + 1)

        # Index pour placer les labels
        i = 1

        # Ajouter les marqueurs avec lignes et labels
        for _, row in df_country.iterrows():
            # Marqueur circulaire (point vert)
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                icon=folium.CustomIcon(
                icon_image=pin_icon_url2,
                icon_size=(24, 24))
            ).add_to(m)
            
            # Position du label dans la colonne
            label_lat = lat_min + step * i + marge_sites_v
            i += 1
            
            # Ligne entre le point et le label
            folium.PolyLine(
                locations=[[row["Latitude"], row["Longitude"]], [label_lat, label_lon]],
                color=ligne_sites,
                weight=2.5,
                opacity=0.9
            ).add_to(m)
            
            # Label avec marqueur circulaire + nom de la structure
            html_label = f"""
            <div style="
                display: flex;
                flex-direction: row;
                align-items: center;
            ">
                <div style="
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    background-color: {markers_sites};
                    border: 2px solid {markers_sites};
                ">
                </div>
                <span style="color:black; font-size:16px; font-weight:600;">
                    {row['Sigle structure']}
                </span>
            </div>
            """
            
            folium.Marker(
                location=[label_lat, label_lon],
                icon=folium.DivIcon(
                    html=html_label,
                    icon_size=(250, 30),
                    icon_anchor=(0, 15)
                )
            ).add_to(m)

        # Affichage Streamlit
        map_html = m._repr_html_()
        html(map_html, height=1200)

        # Sauvegarde et téléchargement
        html_file = f"carte_{country}.html"
        png_file = f"carte_{country}.png"

        m.save(html_file)
        save_map_as_png(html_file, png_file)
        with open(png_file, "rb") as f:
            st.download_button(
                label="📥 Télécharger la carte en PNG",
                data=f,
                file_name=png_file,
                mime="image/png"
            )

if tests_tous:
    grouped_2 = grouped_2.sort_values(by="Sigle structure", ascending=True, key=lambda col: col.str.lower())

    st.dataframe(grouped_2)

    countries = grouped_2["Pays"].unique()

    # Charger GeoJSON mondial
    geojson_world = load_countries_geojson()

    for country in countries:
        st.subheader(f"📍 {country}")

        df_country = grouped_2[grouped_2["Pays"] == country]

        # Récupérer tous les pays présents dans les données
        countries_in_data = df_country["Pays"].unique()

        # Créer une MultiPolygon avec tous les pays concernés
        all_countries_shapes = []
        for c in countries_in_data:
            country_shape = get_country_shape(geojson_world, c)
            if country_shape is not None:
                all_countries_shapes.append(country_shape)

        # Combiner tous les pays en une seule forme
        from shapely.ops import unary_union
        combined_shape = unary_union(all_countries_shapes)

        # Création du masque mondial
        mask_shape = create_world_mask(combined_shape)

        # Carte centrée sur le monde
        m = folium.Map(
            location=[df_country["Latitude"].mean(), df_country["Longitude"].mean()],
            zoom_start=2
        )

        # Ajouter le masque blanc
        GeoJson(
            data=json.loads(json.dumps(mask_shape.__geo_interface__)),
            style_function=lambda x: {
                "fillColor": "white",
                "color": "white",
                "weight": 1,
                "fillOpacity": 1
            }
        ).add_to(m)

        # Ajouter TOUS les pays du monde avec contours gris
        GeoJson(
            data=geojson_world,
            style_function=lambda x: {
                "fillColor": "#ffffff00",
                "color": "#cccccc",
                "weight": 1,
                "fillOpacity": 0
            }
        ).add_to(m)

        # Ajouter les pays avec données par-dessus
        GeoJson(
            data=json.loads(json.dumps(combined_shape.__geo_interface__)),
            style_function=lambda x: {
                "fillColor": "#ffffff00",
                "color": contour_unites,
                "weight": 2,
                "fillOpacity": 0
            }
        ).add_to(m)

        # Ajouter les marqueurs circulaires sur la carte
        for _, row in df_country.iterrows():
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                #radius=3,
                #color=markers_unites,
                #fill=True,
                #fill_color=markers_unites
                icon=folium.CustomIcon(
                icon_image=pin_icon_url,
                icon_size=(24, 24))
            ).add_to(m)

        # --- Construction de la grille HTML des logos/noms ---
        max_rows = 13
        items = list(df_country.iterrows())
        cols_needed = max(1, -(-len(items) // max_rows))

        # Construire les colonnes HTML
        columns_html = ""
        for col_idx in range(cols_needed):
            col_items = items[col_idx * max_rows : (col_idx + 1) * max_rows]
            col_html = '<div style="display:flex; flex-direction:column; gap:4px;">'
            for _, row in col_items:
                img_b64 = image_to_base64(row["Logos"])
                col_html += f"""
                <div style="display:flex; align-items:center; gap:5px;">
                    <div style="
                        width:60px; height:60px;
                        border-radius:50%; overflow:hidden;
                        border:2px solid {markers_unites};
                        flex-shrink:0;
                    ">
                        <img src="data:image/png;base64,{img_b64}"
                            style="width:100%; height:100%; object-fit:contain;">
                    </div>
                    <span style="font-size:14px; font-weight:600; white-space:nowrap; color:black;">
                        {row['Sigle structure']}
                    </span>
                </div>
                """
            col_html += '</div>'
            columns_html += col_html

        # Wrapper global de la grille
        grid_html = f"""
        <div style="
            display:flex;
            flex-direction:row;
            gap:12px;
            background:rgba(255,255,255,0.85);
            padding:8px;
            border-radius:8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        ">
            {columns_html}
        </div>
        """

        # Placement de la grille en haut à droite de la carte
        label_lon = df_country["Longitude"].max() + marge_unites
        label_lat = df_country["Latitude"].max() + marge_unites_v

        folium.Marker(
            location=[label_lat, label_lon],
            icon=folium.DivIcon(
                html=grid_html,
                icon_size=(cols_needed * 180, min(len(items), max_rows) * 36),
                icon_anchor=(0, 0)
            )
        ).add_to(m)

        # Affichage Streamlit
        map_html = m._repr_html_()
        html(map_html, height=1000)

        # Sauvegarde et téléchargement
        html_file = f"carte_{country}.html"
        png_file = f"carte_{country}.png"

        m.save(html_file)
        save_map_as_png(html_file, png_file)
        with open(png_file, "rb") as f:
            st.download_button(
                label="📥 Télécharger la carte en PNG",
                data=f,
                file_name=png_file,
                mime="image/png"
            )

if tests_tous2:
    # Tri
    grouped_2 = grouped_2.sort_values(
        by="Sigle structure",
        ascending=True,
        key=lambda col: col.str.lower()
    )

    st.dataframe(grouped_2)

    # Charger GeoJSON mondial
    geojson_world = load_countries_geojson()

    # Récupérer tous les pays présents dans les données
    countries_in_data = grouped_2["Pays"].unique()

    # Construire la MultiPolygon de tous les pays concernés
    all_countries_shapes = []
    for c in countries_in_data:
        country_shape = get_country_shape(geojson_world, c)
        if country_shape is not None:
            all_countries_shapes.append(country_shape)

    from shapely.ops import unary_union
    combined_shape = unary_union(all_countries_shapes)

    # Masque mondial
    mask_shape = create_world_mask(combined_shape)

    # Carte centrée sur la moyenne des points
    m = folium.Map(
        location=[grouped_2["Latitude"].mean(), grouped_2["Longitude"].mean()],
        zoom_start=2
    )

    # Masque blanc
    GeoJson(
        data=json.loads(json.dumps(mask_shape.__geo_interface__)),
        style_function=lambda x: {
            "fillColor": "white",
            "color": "white",
            "weight": 1,
            "fillOpacity": 1
        }
    ).add_to(m)

    # Tous les pays du monde (contours gris)
    GeoJson(
        data=geojson_world,
        style_function=lambda x: {
            "fillColor": "#ffffff00",
            "color": "#cccccc",
            "weight": 1,
            "fillOpacity": 0
        }
    ).add_to(m)

    # Pays concernés (contours colorés)
    GeoJson(
        data=json.loads(json.dumps(combined_shape.__geo_interface__)),
        style_function=lambda x: {
            "fillColor": "#ffffff00",
            "color": contour_unites,
            "weight": 2,
            "fillOpacity": 0
        }
    ).add_to(m)

    # Marqueurs
    for _, row in grouped_2.iterrows():
        folium.Marker(
            location=[row["Latitude"], row["Longitude"]],
            icon=folium.CustomIcon(
                icon_image=pin_icon_url2,
                icon_size=(24, 24)
            )
        ).add_to(m)

    # Affichage Streamlit
    map_html = m._repr_html_()
    html(map_html, height=1000)

    # Sauvegarde et téléchargement
    html_file = "carte_monde.html"
    png_file = "carte_monde.png"

    m.save(html_file)
    save_map_as_png(html_file, png_file)

    with open(png_file, "rb") as f:
        st.download_button(
            label="📥 Télécharger la carte en PNG",
            data=f,
            file_name=png_file,
            mime="image/png"
        )