import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go

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
df = read_data("Data\FairCarboN_Datas_Recensement")

df_reduit = df[['PROJET','NOM','INTITULE_DONNEES','NATURE','TYPE','SOURCE','SOURCE_URL','AUTRE_REFERENCE',
                'MOYENS_PRODUCTION','MODELE','MODELE_STATUT','AUTRE_MOYENS','DOCS_ASSOCIES','AUTRE_DOCS', 'ATTRIBUTS','METADATA_EMBARQUEES','METADATA_ENRICHIES']]

#st.dataframe(df_reduit)

df_reduit['Value']= 1
df_reduit['NOM'] = df_reduit['NOM'].str.upper()

fig = px.sunburst(df_reduit, path=['NATURE','TYPE','NOM'], values='Value', color='NATURE',  # La couleur sera attribuée au 1er niveau
    color_discrete_map={
        'PRODUITES': 'lightgreen',
        'PRE_EXISTANTES': 'pink'})

fig.update_layout(
                width=800,
                height=800)

st.title(":grey[Données en cours - Recensements]")
st.plotly_chart(fig, use_container_width=True)