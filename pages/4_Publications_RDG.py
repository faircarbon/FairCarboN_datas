import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import re
from datetime import datetime
import requests


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
couleurs = {"ALAMOD":"#fa0404",
              "SLAM-B":"#e7b204",
              "RIFT":"#05fc6c",
              "CrosyeN":"#b9fc01",
              "CarboNium":"#ec8129",
              "CABESTAN":"#03fabc",
              "CANETE":"#067ff0",
              "DEEP-C":"#6d03f8",
              "Drought for C":"#cb05fc",
              "PEACE":"#f0047a",
              "TROPECOS":"#090080",
              "CLIM-FAS":"#ca7ff8",
              "CO2_CMPhi":"#793305",
              "GREENSCALE":"#035718",
              "PREFALIM":"#B83E0E",
              "RhizoSeqC":"#636303",
              "Gouvernance":"#0A0A0A",
              "ART": "#020242",
              "UNDEFINED": "#0B9BEE",
              "COM": "#16DA6E",
              "POSTER":"#8de42a",
              "MEM":"#eedf14",
              "REPORT" : "#eba21b",
              "OTHER": "#f5760f",
              "LECTURE": "#ff0909"}


# Variables Python
couleur_h1 = "#748114"
taille_h1 = "60px"
police_h1 = "Cascadia Code"

couleur_h2 = "#FF6347"
taille_h2 = "32px"
police_h2 = "Cascadia Code"

couleur_h3 = "#1B657C"
taille_h3 = "35px"
police_h3 = "Cascadia Code"

taille_metrique = "50px"
couleur_metrique = "#0C495C"

couleur_graphes = "#0C495C"

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
########### DONNEES ##################################################################################################
######################################################################################################################

dossier = "Data/RechercheDataGouv/"
pattern = dossier + "all_datasets_rdg_*.csv"
fichiers = glob.glob(pattern)

# Expression régulière pour extraire la date
regex = re.compile(r"all_datasets_rdg_(\d{4}-\d{2}-\d{2}).csv")

def extraire_date(f):
    match = regex.search(f)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    return datetime.min

# Trier par date décroissante
fichiers_tries = sorted(fichiers, key=extraire_date, reverse=True)
fichiers_sans_prefixe = [f[22:] for f in fichiers_tries]

derniere_date = fichiers_sans_prefixe[0] if fichiers_tries else None
dernier_fichier = dossier + derniere_date

data = pd.read_csv(dernier_fichier)
data_sans_doublon = data.drop_duplicates(subset='PersistentUrl')
filtered = data_sans_doublon[data_sans_doublon["Date de publication"]>2024].reset_index()
#data_sans_doublon["Date de publication"]=pd.to_datetime(data_sans_doublon["Date de publication"],format='%Y')

df = read_data("Data/FairCarboN_Datas_Contacts")

######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################

st.title(f"FairCarboN sur RechercheDataGouv - {derniere_date[18:-4]}")

#st.write(len(data_sans_doublon))
#st.dataframe(filtered)

#base = "https://entrepot.recherche.data.gouv.fr"
#response_init = requests.get(base + '/api/v1/search?q=faircarbon')
#data_init = response_init.json().get("data", {})
