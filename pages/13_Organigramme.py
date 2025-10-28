import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np
from plotly.subplots import make_subplots

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON ORGANIGRAMME",
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
    df = pd.read_excel(f"{path}.xlsx", sheet_name=0,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")
    return df


    
######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df = read_data("Data\FairCarboN_Datas_Contacts")

st.title(":grey[Organigramme de FairCarboN]")

fig_organigramme = px.treemap(
    df,
    path=["Acronyme PEPR", "projet", "Sigle structure", "Fonction", "Contact"],
    title="Organigramme hiérarchique (Treemap)"
)
fig_organigramme.write_html("organigramme_treemap.html")

fig_organigramme2 = px.sunburst(
    df,
    path=["Acronyme PEPR", "projet", "Sigle structure", "Fonction", "Contact"],
    title="Organigramme hiérarchique des projets",
    width=800,
    height=600
)

fig_organigramme2.write_html("organigramme_sunburst.html")

fig_organigramme3 = px.icicle(
    df,
    path=["Acronyme PEPR", "projet", "Sigle structure", "Fonction", "Contact"],
    title="Organigramme en arbre hiérarchique",
    width=800,
    height=600
)

fig_organigramme3.write_html("organigramme_arbre.html")

st.plotly_chart(fig_organigramme, use_container_width=True)

st.plotly_chart(fig_organigramme2, use_container_width=True)

st.plotly_chart(fig_organigramme3, use_container_width=True)