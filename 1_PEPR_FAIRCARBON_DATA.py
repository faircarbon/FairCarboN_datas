import streamlit as st
from PIL import Image
import pandas as pd
import folium
from folium.features import CustomIcon
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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

###############################################################################################
########### TRANSFORMATION FICHIER XLS ########################################################
###############################################################################################
@st.cache_data
def read_data():
    # Chemin vers le fichier Excel
    fichier_excel = "Data\FairCarboN_RNSR_copie.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(fichier_excel, sheet_name=1,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv("Data\FairCarboN_RNSR_copie.csv", index=False, encoding="utf-8")

    ######## NETTOYAGES EVENTUELS ######################

    # filtrer les lignes incomplètes
    df_filtré = df.dropna(subset=["Latitude", "Longitude","Acronyme projet","Acronyme unité"])
    # Renommer les colonnes
    df_filtré_renommé = df_filtré.rename(columns={
        "Acronyme projet": "projet",
        "Acronyme unité": "laboratoire"
    })
    df_filtré_renommé.to_csv("Data\FairCarboN_RNSR_copie_filtré_renommé.csv", index=False)

    return df_filtré_renommé

# Charger les données
df = read_data()

# Couleurs associées à chaque projet
projects = sorted(df['projet'].unique())
laboratoires = sorted(df['laboratoire'].unique())
colors = plt.cm.tab20.colors  # Palette de couleurs
project_color_map = {project: colors[i % len(colors)] for i, project in enumerate(projects)}

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################
st.sidebar.title('Filtrage')
Selection_projets = st.sidebar.multiselect('Projets',options=projects)
if len(Selection_projets)==0:
    df_selected = df
else:
    df_selected = df[df['projet'].isin(Selection_projets)]

Selection_laboratoires = st.sidebar.multiselect('Unités',options=laboratoires)
if len(Selection_laboratoires)==0:
    df_selected_ = df_selected
else:
    df_selected_ = df[df['laboratoire'].isin(Selection_laboratoires)]

# Regrouper par laboratoire
grouped = df_selected_.groupby(['laboratoire','Type_Data','Latitude', 'Longitude'])['projet'].apply(list).reset_index()
st.sidebar.write(len(grouped))

# Créer la carte
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB positron',  # Ou 'Stamen Toner Lite'
    control_scale=True)  # Centrée sur la France

# Générer des marqueurs en camembert
for _, row in grouped.iterrows():
    projets = row['projet']
    latitude = row['Latitude']
    longitude = row['Longitude']
    type_data = row['Type_Data']

    if type_data == "Analyses":
        # Créer un graphique en camembert
        fig, ax = plt.subplots(figsize=(1, 1))
        projet_counts = [1] * len(projets)  # égale pondération
        colors_used = [project_color_map[proj] for proj in projets]
        #ax.pie(projet_counts, colors=colors_used) version sans bordure
        wedges, _ = ax.pie(
            projet_counts,
            colors=colors_used,
            wedgeprops={'edgecolor': 'black', 'linewidth': 5}  # Bordure noire épaisse
        )
        plt.axis('off')

        # Sauvegarder en mémoire
        img_data = BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        img_data.seek(0)
        encoded = base64.b64encode(img_data.read()).decode()

        icon_url = f"data:image/png;base64,{encoded}"
        icon = folium.CustomIcon(icon_image=icon_url, icon_size=(20, 20))
    
    elif type_data == "Modelisation":
        # Créer un graphique en camembert
        fig, ax = plt.subplots(figsize=(1, 1))
        projet_counts = [1] * len(projets)  # égale pondération
        colors_used = [project_color_map[proj] for proj in projets]
        #ax.pie(projet_counts, colors=colors_used) version sans bordure
        wedges, _ = ax.pie(
            projet_counts,
            colors=colors_used,
            wedgeprops={'edgecolor': 'red', 'linewidth': 5}  # Bordure rouge épaisse
        )
        plt.axis('off')

        # Sauvegarder en mémoire
        img_data = BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        img_data.seek(0)
        encoded = base64.b64encode(img_data.read()).decode()

        icon_url = f"data:image/png;base64,{encoded}"
        icon = folium.CustomIcon(icon_image=icon_url, icon_size=(30, 30))

    # Ajouter le marqueur
    popup = folium.Popup("<br>".join(projets), max_width=200)
    tooltip = row['laboratoire']
    folium.Marker(location=[latitude, longitude], popup=popup, tooltip=tooltip, icon=icon).add_to(m)


st.title("Carte interactive FAIRCARBON")
st_folium(m, width=800)


st.dataframe(df_selected_)