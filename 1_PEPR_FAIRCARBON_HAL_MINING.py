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

###############################################################################################
########### REQUETES HAL ######################################################################
###############################################################################################

st.title(":grey[Etude des publications sur HAL]")

start_year=2023
end_year=2025

df_research = df[['projet','laboratoire']][df['Type_Data']=='Contact']
df_research.reset_index(inplace=True)
df_research.drop(columns='index', inplace=True)

liste_chercheurs = df_research['laboratoire']
liste_projet = df_research['projet']
#Liste_chercheurs = ['Claire Chenu']
#requete_api_hal = f'http://api.archives-ouvertes.fr/search/?q=text:"{Liste_chercheurs[0].lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s'
#reponse = requests.get(requete_api_hal, timeout=5)
#print(reponse.json()['response']['docs'][-1])

@st.cache_data
def acquisition_data(start_year,end_year,liste_chercheurs, liste_projet):
    liste_columns_hal = ['Store','Auteur_recherché','Projet','Ids','Titre et auteurs','Uri','Type','Type de document', 'Date de production','Collection','Collection_code','Auteur_organisme','Auteur','Labo_all','Labo_','Titre','Langue','Mots_Clés']
    df_global_hal = pd.DataFrame(columns=liste_columns_hal)
    for i, s in enumerate(liste_chercheurs):
        url_type = f'http://api.archives-ouvertes.fr/search/?q=text:"{s.lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s,labStructName_s,language_s,keyword_s'
        df = afficher_publications_hal(url_type, s, liste_projet.iloc[i])
        dfi = pd.concat([df_global_hal,df], axis=0)
        dfi.reset_index(inplace=True)
        dfi.drop(columns='index', inplace=True)
        df_global_hal = dfi
    df_global_hal.sort_values(by='Ids', inplace=True, ascending=False)
    df_global_hal.reset_index(inplace=True)
    df_global_hal.drop(columns='index', inplace=True)

    df_global_hal['Labo_filter1'] = df_global_hal['Labo_all']
    df_global_hal['Labo_filter2'] = df_global_hal['Labo_all']

    for i in range(len(df_global_hal)):
        try:
            df_global_hal['Labo_filter1'].loc[i] = [item for item in df_global_hal['Labo_all'].loc[i] if df_global_hal['Auteur_recherché'].loc[i] in item]
        except:
            pass
        try:
            df_global_hal['Labo_filter2'].loc[i] = [item.split('_')[-1] for item in df_global_hal['Labo_filter1'].loc[i]]
        except:
            pass

    df_global_hal['Auteur_Labo'] = df_global_hal.apply(intersect_lists, axis=1)
    df_global_hal['Titre_bis'] = df_global_hal['Titre'].apply(lambda row: row[0])

    return df_global_hal

def intersect_lists(row):
    return list(set(row['Labo_filter2']) & set(row['Labo_']))

with st.spinner("Recherche en cours"):
    df_global_hal = acquisition_data(start_year=start_year,end_year=end_year,liste_chercheurs=liste_chercheurs, liste_projet=liste_projet)

filtered_df = df_global_hal[df_global_hal['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]

col1,col2 = st.columns(2)

with col1:
    st.metric(label="Nombre de contacts étudiés", value=len(set(liste_chercheurs)))
    st.metric(label="Nombre de dépôts HAL global", value=len(set(df_global_hal['Titre_bis'].values)))
    st.metric(label="Nombre de dépôts HAL dans la collection FairCarboN", value=len(set(filtered_df['Titre_bis'].values)))

with col2:
    st.metric(label="Nombre de contacts trouvés dans HAL", value=len(set(df_global_hal['Auteur_recherché'])))
    st.metric(label="Nombre d'articles global", value=len(set(df_global_hal['Titre_bis'][df_global_hal['Type de document']=="ART"].values)))
    st.metric(label="Nombre d'artciles dans la collection FairCarboN", value=len(set(filtered_df['Titre_bis'][filtered_df['Type de document']=="ART"].values)))

df_global_hal['In_FairCarboN'] = df_global_hal['Titre'].isin(filtered_df['Titre'])

st.dataframe(df_global_hal[['Auteur_recherché','Projet','Type de document','Date de production','Titre','Langue','In_FairCarboN','Auteur_Labo','Mots_Clés']])

st.write(liste_chercheurs)