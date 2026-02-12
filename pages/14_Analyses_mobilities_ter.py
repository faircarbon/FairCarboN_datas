import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.components.v1 import html
import base64

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

def image_to_base64(path): 
    with open(path, "rb") as f: data = f.read() 
    return base64.b64encode(data).decode("utf-8")

df = pd.DataFrame([ ["Entreprise Alpha", 48.8566, 2.3522, "Data/logos/alpha.png"], ["Entreprise Beta", 45.7640, 4.8357, "Data/logos/beta.png"], ["Entreprise Gamma", 43.2965, 5.3698, "Data/logos/gamma.png"], ], columns=["name", "lat", "lon", "logo"])

import folium 
# Création de la carte centrée sur Grenoble 
m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=6) 
# Ajout des marqueurs 
for _, row in df.iterrows():
    img_b64 = image_to_base64(row["logo"])
    html_icon = f""" <div style=" width: 60px; height: 60px; border-radius: 50%; overflow: hidden; border: 3px solid #333; box-shadow: 0 0 5px rgba(0,0,0,0.4); "> <img src="data:image/png;base64,{img_b64}" style="width: 100%; height: 100%; object-fit: contain;"> </div> """ 
    icon = folium.DivIcon(html=html_icon) 
    folium.Marker( location=[row["lat"], row["lon"]], popup=row["name"], icon=icon ).add_to(m)
# Sauvegarde 
m.save("carte_entreprises.html")

map_html = m._repr_html_()

html(map_html, height=600)