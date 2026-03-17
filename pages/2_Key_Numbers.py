import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

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

######################################################################################################################
########### PARAMETRES ###############################################################################################
######################################################################################################################



######################################################################################################################
########### DONNEES ##################################################################################################
######################################################################################################################
data = read_data("Data/FairCarboN_Datas_Contacts")

######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################
st.title("Chiffres clés | Key Numbers")

col1 , col2, col3 = st.columns(3)
with col1:
    st.write("Budget (Millions d'Euros)")
    st.metric(value=40, label="", label_visibility="hidden")
with col2:
    st.write("Projets Ciblés | Target projects")
    st.metric(value=5, label="", label_visibility="hidden")
with col3:
    st.write("Projets sélectionnés | Selected Projects")
    st.metric(value=11, label="", label_visibility="hidden")

col1 , col2, col3 = st.columns(3)
with col1:
    st.write("Labos impliqués | Research units involved")
    st.metric(value=114, label="", label_visibility="hidden")
with col2:
    st.write("Communauté fairCarboN | FairCarboN community")
    st.metric(value=498, label="", label_visibility="hidden")
with col3:
    st.write("Sites étudiés/ expérimentaux | Sites localisations")
    st.metric(value=150, label="", label_visibility="hidden")


st.title("Chiffres clés - Key Numbers || Par projet - By Project")


import pandas as pd
import plotly.express as px

# --- Création d'un DataFrame exemple ---
data = {
    "Continent": ["FC", "FC", "FC","FC", "FC", "FC","FC", "FC", "FC","FC", "FC", "FC","FC", "FC"],
    "Pays": ["P1", "P1", "P1", "P1", "P1", "P1","P1", "P2", "P2", "P2", "P2", "P2", "P2", "P2" ],
    "Ville": ["CR", "DIR", "DOC", "POSTDOC", "INGE", "TECH","Autres", "CR", "DIR", "DOC", "POSTDOC", "INGE", "TECH","Autres"],
    "Population": [15, 10 , 3, 6 , 5, 12, 0, 18, 20 , 13, 0 , 5, 2, 4]
}

df = pd.DataFrame(data)

# --- Création du graphique Sunburst ---
fig = px.sunburst(
    df,
    path=["Continent", "Pays", "Ville"],  # hiérarchie
    values="Population",                 # taille des segments
    color="Continent",                   # coloration par continent
    title="Répartition de population par continent, pays et ville"
)

st.plotly_chart(fig, use_container_width=True)
