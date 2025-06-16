import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
from Publications import afficher_publications_hal
import datetime
import requests


###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON HAL DATA MINING",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

######################################################################################################################
########### CHOIX VISUELS ############################################################################################
######################################################################################################################
# taille et couleurs des sous-titres
couleur_subtitles = (250,150,150)
taille_subtitles = "25px"
couleur_subsubtitles = (60,150,160)
taille_subsubtitles = "25px"

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

st.dataframe(df[df['Type_Data']=='Contact'])

#st.dataframe(df[df['Type_Data']=="Contact"])

###############################################################################################
########### REQUETES HAL ######################################################################
###############################################################################################

st.title(":grey[Extraction des publications sur HAL]")

start_year=2020
end_year=2025

Liste_chercheurs = df['laboratoire'][df['Type_Data']=='Contact']

with st.spinner("Recherche en cours"):
    liste_columns_hal = ['Store','Auteur_recherché','Ids','Titre et auteurs','Uri','Type','Type de document', 'Date de production','Collection','Collection_code','Auteur_organisme','Auteur','Labo_all','Labo_auteur','Titre']
    df_global_hal = pd.DataFrame(columns=liste_columns_hal)
    for i, s in enumerate(Liste_chercheurs):
        url_type = f'http://api.archives-ouvertes.fr/search/?q=text:"{s.lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s,labStructName_s'
        df = afficher_publications_hal(url_type, s)
        dfi = pd.concat([df_global_hal,df], axis=0)
        dfi.reset_index(inplace=True)
        dfi.drop(columns='index', inplace=True)
        df_global_hal = dfi
    df_global_hal.sort_values(by='Ids', inplace=True, ascending=False)
    df_global_hal.reset_index(inplace=True)
    df_global_hal.drop(columns='index', inplace=True)

    df_global_hal['Labo_filter1'] = 0
    df_global_hal['Labo_filter2'] = 0
    df_global_hal['Auteur_Labo'] = 0

    for i in range(len(df_global_hal)):
        df_global_hal['Labo_filter1'].loc[i] = [item for item in df_global_hal['Labo_all'].loc[i] if df_global_hal['Auteur_recherché'].loc[i] in item]
        df_global_hal['Labo_filter2'].loc[i] = [item.split('_')[-1] for item in df_global_hal['Labo_filter1'].loc[i]]
        df_global_hal['Auteur_Labo'].loc[i] = [item for item in df_global_hal['Labo_filter2'].loc[i] if item in df_global_hal['Labo_'].loc[i]]


filtered_df = df_global_hal[df_global_hal['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]

st.metric(label="Nombre de publications globales", value=len(df_global_hal))

st.metric(label="Nombre de publications dans la collection FairCarboN", value=len(filtered_df))

#st.dataframe(filtered_df[['Auteur_recherché','Type de document','Date de production','Titre','Auteur_Labo']])

df_global_hal['In_FairCarboN'] = df_global_hal['Titre'].isin(filtered_df['Titre'])

st.dataframe(df_global_hal[['Auteur_recherché','Type de document','Date de production','Titre','Auteur_Labo','In_FairCarboN']])