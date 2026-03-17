import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
import datetime
import requests
import random
from geopy.geocoders import Nominatim 
import time
import numpy as np


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



######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
#file1 = "Data/Mobilites_internationales_FairCarboN_JD"

#data = read_data(file1)

#geolocator = Nominatim(user_agent="geoapi")

#def get_coords(city):
#    try:
#        loc = geolocator.geocode(city)
#        time.sleep(1)  # éviter surcharge API
#        if loc:
#            return pd.Series([loc.latitude, loc.longitude])
#        else:
#            return pd.Series([None, None])
#    except:
#        return pd.Series([None, None])
    
# Coordonnées origine
#data[['origine_lat', 'origine_lon']] = data['Origine'].apply(get_coords)

# Coordonnées destination
#data[['destination_lat', 'destination_lon']] = data['Destination'].apply(get_coords)

#st.dataframe(data)
#data.to_csv("Data/Mobilites_internationales_FairCarboN_JD2")

import numpy as np

def great_circle(lon1, lat1, lon2, lat2, n_points=50):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    t = np.linspace(0, 1, n_points)

    # interpolation sphérique
    delta = np.arccos(
        np.sin(lat1)*np.sin(lat2) +
        np.cos(lat1)*np.cos(lat2)*np.cos(lon2 - lon1)
    )

    A = np.sin((1 - t) * delta) / np.sin(delta)
    B = np.sin(t * delta) / np.sin(delta)

    x = A * np.cos(lat1) * np.cos(lon1) + B * np.cos(lat2) * np.cos(lon2)
    y = A * np.cos(lat1) * np.sin(lon1) + B * np.cos(lat2) * np.sin(lon2)
    z = A * np.sin(lat1) + B * np.sin(lat2)

    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon = np.degrees(np.arctan2(y, x))

    return lat, lon

def categoriser_duree(x):
    if x <= 1:
        return "<=1 mois"
    elif x <= 3:
        return "1-3 mois"
    else:
        return ">3 mois"


data2 = pd.read_csv("Data/Mobilites_internationales_FairCarboN_JD2")

data2 = data2[data2['Id']=='AAP#3']

st.dataframe(data2)

# Normalisation du type
data2["Type"] = data2["Type"].str.upper()

# Regroupement
grouped = data2.groupby(["Origine", "Destination", "Type"]).agg(
    origine_lat=("origine_lat", "first"),
    origine_lon=("origine_lon", "first"),
    destination_lat=("destination_lat", "first"),
    destination_lon=("destination_lon", "first"),
    duree_moy=("Duree", "mean"),
    n=("Type", "count")
).reset_index()

grouped["categorie_duree"] = grouped["duree_moy"].apply(categoriser_duree)

import numpy as np
import plotly.graph_objects as go

# --- Palette Type + Catégorie ---
palette = {
    ("ENTRANTE", "<=1 mois"): "#9ecae1",
    ("ENTRANTE", "1-3 mois"): "#4292c6",
    ("ENTRANTE", ">3 mois"):  "#08519c",

    ("SORTANTE", "<=1 mois"): "#fdd0a2",
    ("SORTANTE", "1-3 mois"): "#fd8d3c",
    ("SORTANTE", ">3 mois"):  "#a63603",
}

fig = go.Figure()

# --- Tracer les lignes courbes ---
for _, row in grouped.iterrows():

    lat_curve, lon_curve = great_circle(
        row["origine_lon"], row["origine_lat"],
        row["destination_lon"], row["destination_lat"],
        n_points=60
    )

    couleur = palette[(row["Type"], row["categorie_duree"])]

    fig.add_trace(go.Scattergeo(
        lat=lat_curve,
        lon=lon_curve,
        mode="lines",
        line=dict(
            width=2 + 0.3 * row["duree_moy"],
            color=couleur
        ),
        showlegend=False
    ))

    # --- Afficher le nombre de trajets au niveau du point origine ---
    if row["n"] > 1:
        fig.add_trace(go.Scattergeo(
            lat=[row["origine_lat"]],
            lon=[row["origine_lon"]],
            mode="text",
            text=[f"<b>{row['n']}</b>"],
            textfont=dict(color=couleur, size=16),
            textposition="top center",
            hoverinfo="skip",
            showlegend=False
        ))


# --- Points origine ---
fig.add_trace(go.Scattergeo(
    lat=grouped["origine_lat"],
    lon=grouped["origine_lon"],
    mode="markers",
    marker=dict(
        size=10,
        color=[palette[(t, c)] for t, c in zip(grouped["Type"], grouped["categorie_duree"])],
    ),
    hoverinfo="skip",
    showlegend=False
))

# --- Points destination ---
fig.add_trace(go.Scattergeo(
    lat=grouped["destination_lat"],
    lon=grouped["destination_lon"],
    mode="markers",
    marker=dict(
        size=10,
        color=[palette[(t, c)] for t, c in zip(grouped["Type"], grouped["categorie_duree"])],
    ),
    hoverinfo="skip",
    showlegend=False
))


# --- Légende : 6 catégories ---
for (typ, cat), col in palette.items():
    fig.add_trace(go.Scattergeo(
        lat=[None], lon=[None],
        mode="markers",
        marker=dict(size=12, color=col),
        name=f"{typ} — {cat}"
    ))


# --- Mise en forme ---
fig.update_layout(
    title="Mobilités internationales FairCarboN AAP#3",
    geo=dict(
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(230,230,230)"
    ),
    legend=dict(
        orientation="v",
        x=1.05,
        y=0.95
    )
)

fig.show()

