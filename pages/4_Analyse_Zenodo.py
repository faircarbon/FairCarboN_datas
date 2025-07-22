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
########### FONCTIONS SUPPORT #################################################################
###############################################################################################
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
    df = pd.read_excel(f"{path}.xlsx", sheet_name=1,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")
    return df


def recuperation_zenodo(url_zenodo, params_zenodo, headers_zenodo):
    """
    Connexion à l'API Zenodo et récupération des résultats de recherche.

    Paramètres :
        url_zenodo (str) : URL de l'API Zenodo
        params_zenodo (dict) : paramètres de la requête
        headers_zenodo (dict) : en-têtes HTTP

    Retour :
        list : liste des éléments trouvés (dictionnaires)
    """
    try:
        response = requests.get(url_zenodo, params=params_zenodo, headers=headers_zenodo)
        response.raise_for_status()
        data = response.json()
        return data.get('hits', {}).get('hits', [])
    except requests.RequestException as e:
        print(f"[Erreur API Zenodo] {e}")
        return []

def extraire_valeur(dico, cle, default=""):
    """Extraction sécurisée d'une valeur dans un dictionnaire"""
    return dico.get(cle, default) if dico else default

def Recup_contenu_zenodo(url_zenodo, params_zenodo, headers_zenodo, auteur_recherche, projet):
    """
    Extraction des informations bibliographiques à partir de l'API Zenodo.

    Paramètres :
        url_zenodo (str) : URL de l'API Zenodo
        params_zenodo (dict) : paramètres de la requête
        headers_zenodo (dict) : en-têtes HTTP
        auteur_recherche (str) : nom de l'auteur à rechercher
        projet (str) : nom du projet associé

    Retour :
        pd.DataFrame : tableau des résultats formaté
    """
    contenu = recuperation_zenodo(url_zenodo, params_zenodo, headers_zenodo)

    donnees = {
        'Store': [],
        'Auteur_recherché': [],
        'Projet': [],
        'ID': [],
        'Titre': [],
        'Auteur': [],
        'Résumé': [],
        'Date de publication': [],
        'Publication Url': []
    }

    for item in contenu:
        metadata = item.get('metadata', {})
        creators = metadata.get('creators', [{}])

        donnees['Store'].append('Zenodo')
        donnees['Auteur_recherché'].append(auteur_recherche)
        donnees['Projet'].append(projet)
        donnees['ID'].append(item.get('id', ''))
        donnees['Titre'].append(item.get('title', ''))
        donnees['Auteur'].append(creators[0].get('name', '') if creators else '')
        donnees['Résumé'].append(metadata.get('description', ''))
        donnees['Date de publication'].append(metadata.get('publication_date', ''))
        donnees['Publication Url'].append(metadata.get('doi', ''))

    return pd.DataFrame(donnees)

@st.cache_data
def acquisition_data_zenodo(liste_chercheurs, liste_projet):
    liste_columns = ['Store','Auteur_recherché','Projet','ID','Titre','Auteur',"Résumé","Date de publication","Publication Url"]
    df_global_zenodo = pd.DataFrame(columns=liste_columns)
    for i, s in enumerate(liste_chercheurs):
        params_zenodo = {'q': f'"{s.lower()}"',
                            'access_token': zenodo_token}
                    
        df = Recup_contenu_zenodo(url_zenodo,params_zenodo, headers_zenodo, s, liste_projet[i])
        dfi = pd.concat([df_global_zenodo,df], axis=0)
        dfi.reset_index(inplace=True)
        dfi.drop(columns='index', inplace=True)
        df_global_zenodo = dfi
    df_global_zenodo.sort_values(by='ID', inplace=True, ascending=False)
    df_global_zenodo.reset_index(inplace=True)
    df_global_zenodo.drop(columns='index', inplace=True)
    return df_global_zenodo

######################################################################################################################
######################## ZENODO ######################################################################################
url_zenodo = 'https://zenodo.org/api/records/'
zenodo_token = "OMMGEVUcApEKSt4JEkSK7OzpqZQPMvGKAlB2yP2MXG6APstRn2hWpiHfpjaA"
headers_zenodo = {"Content-Type": "application/json"}


st.title(":grey[Analyse des dépôts dans Zenodo]")

# Charger les données
df = read_data("Data\FairCarboN_Datas_Contacts")
liste_chercheurs = df['Contact']
liste_projet = df['projet']

df_global_zenodo = acquisition_data_zenodo(liste_chercheurs, liste_projet)

st.dataframe(df_global_zenodo)