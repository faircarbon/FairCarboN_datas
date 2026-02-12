import streamlit as st
import pandas as pd
import folium
import base64
import requests
import json
from shapely.geometry import shape, Polygon, MultiPolygon
from folium.features import GeoJson
from streamlit.components.v1 import html

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON SLAMB",
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
    """
    lecture d'un fichier excel, retourné dans le script en format csv prêt à l'emploi

    Paramètres: 
        un chemin vers un fichier excel
    retour: 
        un tableau CSV
    """
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=1,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")
    return df



import streamlit as st
import pandas as pd
import folium
import base64
import requests
import json
from shapely.geometry import shape, Polygon, MultiPolygon
from folium.features import GeoJson
from streamlit.components.v1 import html

# ---------------------------------------------------
# Utilitaires
# ---------------------------------------------------
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

# ---------------------------------------------------
# Données
# ---------------------------------------------------
df = pd.DataFrame([
    ["Entreprise Alpha", 48.8566, 2.3522, "France", "Data/logos/alpha.png"],
    ["Entreprise Beta", 45.7640, 4.8357, "France", "Data/logos/beta.png"],
    ["Entreprise Gamma", 43.2965, 5.3698, "France", "Data/logos/gamma.png"],
    ["Entreprise Delta", 51.5074, -0.1278, "United Kingdom", "Data/logos/delta.png"],
    ["Entreprise Omega", 40.4168, -3.7038, "Spain", "Data/logos/omega.png"],
], columns=["name", "lat", "lon", "country", "logo"])

st.title("Cartes des entreprises par pays (avec masque)")

countries = df["country"].unique()

# Charger GeoJSON mondial
geojson_world = load_countries_geojson()

# ---------------------------------------------------
# Génération d'une carte par pays
# ---------------------------------------------------
for country in countries:
    st.subheader(f"📍 {country}")

    df_country = df[df["country"] == country]

    # Récupérer la forme du pays
    country_shape = get_country_shape(geojson_world, country)

    if country_shape is None:
        st.error(f"Impossible de trouver le pays : {country}")
        continue

    # Création du masque
    mask_shape = create_world_mask(country_shape)

    # Carte centrée sur le pays
    m = folium.Map(
        location=[df_country["lat"].mean(), df_country["lon"].mean()],
        zoom_start=5
    )

    # Ajouter le masque gris
    GeoJson(
        data=json.loads(json.dumps(mask_shape.__geo_interface__)),
        style_function=lambda x: {
            "fillColor": "white",
            "color": "white",
            "weight": 1,
            "fillOpacity": 1
        }
    ).add_to(m)

    # Ajouter le pays en clair
    GeoJson(
        data=json.loads(json.dumps(country_shape.__geo_interface__)),
        style_function=lambda x: {
            "fillColor": "#ffffff00",
            "color": "red",
            "weight": 2,
            "fillOpacity": 0
        }
    ).add_to(m)

    # Ajouter les entreprises
    for _, row in df_country.iterrows():
        img_b64 = image_to_base64(row["logo"])

        html_icon = f"""
        <div style="
            width: 60px;
            height: 60px;
            border-radius: 50%;
            overflow: hidden;
            border: 3px solid #333;
            box-shadow: 0 0 5px rgba(0,0,0,0.4);
        ">
            <img src="data:image/png;base64,{img_b64}"
                 style="width: 100%; height: 100%; object-fit: contain;">
        </div>
        """

        icon = folium.DivIcon(html=html_icon)

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=row["name"],
            icon=icon
        ).add_to(m)

    # Affichage Streamlit
    map_html = m._repr_html_()
    html(map_html, height=1000)
