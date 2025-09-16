import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import time, re, sys
import datetime
import unicodedata
import requests


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

def normalize_name(name: str) -> str:
    # minuscule
    name = name.lower()
    # suppression des accents
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name

def rechercher_orcid(nom, prenom):
    url = "https://pub.orcid.org/v3.0/search/"
    headers = {"Accept": "application/json"}
    params = {
        "q": f'"{prenom} {nom}"'
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        #st.write(data)
        if data.get("result", []):
            for item in data["result"]:
                orcid = item.get("orcid-identifier", {}).get("path")
                return orcid
    return None

def get_publication_count(orcid_id):
    """Interroge l'API ORCID et retourne le nombre de publications pour un ORCID donné."""
    url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    headers = {"Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return len(data.get("group", []))
    except Exception as e:
        print(f"Erreur pour ORCID {orcid_id} : {e}")
        return None  # ou 0 si tu préfères

def ajouter_nombre_publications(df, colonne_orcid):
    """Ajoute une colonne 'Nombre_publis' au DataFrame en comptant les publications via ORCID."""
    df["Nombre_publis"] = df[colonne_orcid].apply(get_publication_count)
    return df

######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
d = datetime.date.today()
data = read_data("Data/FairCarboN_Datas_Contacts")  
#df = data.copy()
#df["Contact_norm"]=df["Contact"].apply(normalize_name)


#df['Nom']=df['Contact_norm'].apply(lambda x: x.split(" ")[-1])
#df['Prenom']=df['Contact_norm'].apply(lambda x: x.split(" ")[0])
#df['ORCID']=None
#df['Nombre']=None

#for i in range(len(df)):
#    print(i)
#    df.loc[i,'ORCID'] = rechercher_orcid(df.loc[i,'Nom'], df.loc[i,'Prenom'])

#df = pd.read_csv("resultats_extraction_nombre_publis.csv")

#df2 = df[['Contact','ORCID']]
#df2['Nombre_publis']=None

# Ajout de la colonne Nombre_publis
#df2 = ajouter_nombre_publications(df2, "ORCID")

#df2.to_csv("resultats_extraction_nombre_publis2.csv", index=False, encoding="utf-8-sig")