import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np
from plotly.subplots import make_subplots
import datetime

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
d = datetime.date.today()
df = read_data("Data\FairCarboN_Datas_Contacts")

df_hal = st.session_state['df_hal']

st.title(":grey[Suivi Collection HAL]")

df_grouped = df.groupby("Contact", as_index=False).agg({
    "projet": lambda x: list(x)
})

df_merged = pd.merge(
    df_hal,
    df_grouped,
    left_on="Auteur_recherché",
    right_on="Contact",
    how="left"   # "left" → garde tous les auteurs recherchés
)

df_merged['projet'] = df_merged['projet'].apply(lambda row: row[0])


projects = sorted(df_merged['projet'].unique())
Selection_projets = st.multiselect("Choix d'un ou plusieurs projets à visualiser (par défaut TOUS)",options=projects)

if len(Selection_projets)==0: #aucun choix
    df_selected = df_merged
else:
    df_selected = df_merged[df_merged['projet'].isin(Selection_projets)]

df_hal_ = df_selected[['Titre_unique', 'Type de document', 'Date complete depot','In_FairCarboN','projet']].drop_duplicates(subset='Titre_unique')
df_hal__ = df_hal_[df_hal_['In_FairCarboN']==True]
df_hal__.reset_index(inplace=True)
df_hal__.drop(columns='index', inplace=True)


# Convertir les dates
df_hal__['Date complete depot'] = pd.to_datetime(df_hal__['Date complete depot'])

# Compter les documents par jour et par type
counts = df_hal__.groupby(['Date complete depot', 'Type de document']).size().reset_index(name="nb_docs")
counts2 = df_hal__.groupby(['Date complete depot','projet']).size().reset_index(name="nb_docs_projet")

# Calcul du cumul par type
counts["cumul"] = counts.groupby('Type de document')["nb_docs"].cumsum()
counts2["cumul"] = counts2.groupby('projet')["nb_docs_projet"].cumsum()


# Tracé avec plotly
fig = px.line(
    counts,
    x='Date complete depot',
    y="cumul",
    color='Type de document',
    markers=True,
    title="Évolution/ Cumul des dépôts dans notre collection FairCarboN par type de document"
)

# Tracé avec plotly
fig2 = px.line(
    counts2,
    x='Date complete depot',
    y="cumul",
    color='projet',
    markers=True,
    title="Évolution/ Cumul des dépôts dans notre collection FairCarboN par projet"
)

st.plotly_chart(fig, use_container_width=True)

st.plotly_chart(fig2, use_container_width=True)

