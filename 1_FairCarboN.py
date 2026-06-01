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
# Variables Python
couleur_h1 = "#748114"
taille_h1 = "60px"
police_h1 = "Cascadia Code"

couleur_h2 = "#FF6347"
taille_h2 = "32px"
police_h2 = "Cascadia Code"

couleur_h3 = "#1B657C"
taille_h3 = "25px"
police_h3 = "Cascadia Code"

taille_metrique = "70px"
couleur_metrique = "#081E25"

# Injection CSS
st.markdown(f"""
<style>
h1 {{
    color: {couleur_h1}!important;
    font-size: {taille_h1}!important;
    font-family: {police_h1}!important;
    text-align: center;
}}

h2 {{
    color: {couleur_h2} !important;
    font-size: {taille_h2} !important;
    font-family: {police_h2} !important;
    text-align: center;
}}

h3 {{
    color: {couleur_h3} !important;
    font-size: {taille_h3} !important;
    font-family: {police_h3} !important;
    font-weight: bold;
    text-align: center;
}}

[data-testid="stMetricValue"] {{
    font-size: {taille_metrique} !important;
    color: {couleur_metrique} !important;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)


######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################
with st.container(height=300):
    col1, col2, col3 = st.columns([0.4,0.2,0.4])
    with col1:
        st.image("Data/logos/France2030.png", width=500)
    with col2:
        st.image("Data/logos/logoFC.png",  width=300)
    with col3:
        st.image("Data/logos/logosTutelles.png", width=700)

col1, col2 = st.columns([0.1,0.9])

with col2:
    st.title("Bienvenue - Welcome")
    st.title("sur l'application FairCarboN ! - on FairCarboN app !")