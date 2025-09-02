import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON CATALOGUE",
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

st.title(f":grey[Etude des data papers publiés (via scraping)]")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label='Data In Brief', value=int(df['DataInBrief'].sum()))
    st.metric(label='Earth System Science Data', value=int(df['EarthSystemScienceData'].sum()))
with col2:
    st.metric(label='Nombre de contacts', value=len(df['Contact'][df['DataInBrief']>0].drop_duplicates()))
    st.metric(label='Nombre de contacts', value=len(df['Contact'][df['EarthSystemScienceData']>0].drop_duplicates()))
with col3:
    st.metric(label='Maximum pour un contact', value=int(df['DataInBrief'].max()))
    st.metric(label='Maximum pour un contact', value=int(df['EarthSystemScienceData'].max()))