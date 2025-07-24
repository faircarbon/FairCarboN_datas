import streamlit as st
import pandas as pd
from pyDataverse.models import Dataset
from pyDataverse.utils import read_file
from pyDataverse.api import NativeApi
import datetime
import numpy as np
import re
import plotly.express as px
import requests
import os
import json

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON RDG DATA MINING",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

###############################################################################################
########### RECUPERATION DES DATAFRAMES #######################################################
###############################################################################################

df_hal = st.session_state['df_hal']
df_rdg = st.session_state['df_rdg']
df_zenodo = st.session_state['df_zenodo']

mots_cles_recherches = ['pepr faircarbon','faircarbon','alamod','slam-b','rift','crosyen','greenscale','canete','carbonium','deep-c','climfas','rhizoseqc','cabestan','tropecos','peace','prefalim','co2cmphi']

df_hal_reduit = df_hal[['Nom_archive','Auteur_recherché','Uri','Titre_unique','Date de publication','Mots_clés','Type de document','ANR project acronyme','DOI sources','In_FairCarboN']]
df_rdg_reduit = df_rdg[['Nom_archive','Auteur_recherché','Titre_unique','Mots_clés','DOI sources','Date de publication']]
df_zenodo_reduit = df_zenodo[['Nom_archive','Auteur_recherché','Titre_unique','Date de publication']]

df_concat = pd.concat([df_hal_reduit,df_rdg_reduit,df_zenodo_reduit], axis=0)
df_concat['In_FairCarboN'] = df_concat['In_FairCarboN'].fillna(False)
df_concat['Type de document'] = df_concat['Type de document'].fillna('DATASET')
df_concat['Mots_clés'] = df_concat['Mots_clés'].apply(lambda x: [] if not isinstance(x, list) else x)
df_concat['Mots_clés'] = df_concat['Mots_clés'].apply(lambda lst: [mot.lower() for mot in lst])
df_concat['ANR project acronyme'] = df_concat['ANR project acronyme'].apply(lambda x: [] if not isinstance(x, list) else x)
df_concat['ANR project acronyme'] = df_concat['ANR project acronyme'].apply(lambda lst: [mot.lower() for mot in lst])
df_concat['Référencement par mots clés'] = df_concat['Mots_clés'].apply(
    lambda liste: any(mot in liste for mot in mots_cles_recherches))
df_concat['Projet ANR dans FairCarboN'] = df_concat['ANR project acronyme'].apply(
    lambda liste: any(mot in liste for mot in mots_cles_recherches))
df_concat["Contient Zenodo"] = df_concat["DOI sources"].apply(
    lambda lst: any("zenodo" in s for s in lst) if isinstance(lst, list) else False
)
df_concat["Contient hal"] = df_concat["DOI sources"].apply(
    lambda lst: any("hal" in s for s in lst) if isinstance(lst, list) else False
)

def is_empty_list(x):
    return isinstance(x, list) and len(x) == 0
df_concat["projet ANR à vérifier"] = df_concat["ANR project acronyme"].apply(is_empty_list)  & (df_concat["In_FairCarboN"]==False)

st.dataframe(df_concat)

df_referencé = df_concat[df_concat['Référencement par mots clés']==True]

p = set(df_concat['Auteur_recherché'].values)

Selection_p = st.selectbox(label='Selection', options=p)

df_selected = df_concat[df_concat['Auteur_recherché']==Selection_p]
df_selected.reset_index(inplace=True)
df_selected.drop(columns='index', inplace=True)

st.dataframe(df_selected)

tous_doi_sources_hal = sum(df_hal_reduit['DOI sources'], [])
tous_doi_sources_rdg = sum(df_rdg_reduit['DOI sources'], [])

# Enlever les doublons en gardant l'ordre
liste_doi_sources_hal = list(dict.fromkeys(tous_doi_sources_hal))
liste_doi_sources_rdg = list(dict.fromkeys(tous_doi_sources_rdg))


liste3 = [element for element in liste_doi_sources_hal if element in liste_doi_sources_rdg]
st.write(liste3)