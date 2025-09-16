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

YEARS = list(range(2018, 2026))  # De 2018 à 2025 inclus

def get_publications_by_year(orcid_id, years=YEARS):
    """Retourne un dictionnaire {année: nombre de publications} pour un ORCID donné."""
    url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    headers = {"Accept": "application/json"}
    counts = {str(year): 0 for year in years}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for group in data.get("group", []):
            for summary in group.get("work-summary", []):
                pub_year = summary.get("publication-date", {}).get("year", {}).get("value")
                if pub_year and str(pub_year) in counts:
                    counts[str(pub_year)] += 1
                    
    except Exception as e:
        print(f"Erreur pour ORCID {orcid_id} : {e}")
    
    return counts

def ajouter_publications_par_annee(df, colonne_orcid):
    """Ajoute une colonne par année avec le nombre de publications pour chaque ORCID."""
    for year in YEARS:
        df[str(year)] = 0  # Initialisation des colonnes
    
    for idx, orcid in df[colonne_orcid].items():
        counts = get_publications_by_year(orcid)
        for year, count in counts.items():
            df.at[idx, year] = count
            
    return df

def get_titles_and_types_by_year(orcid_id, years=YEARS):
    """Retourne un dict {année: {'titres': [...], 'types': [...]}} pour un ORCID donné."""
    url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    headers = {"Accept": "application/json"}
    
    result = {str(year): {"titres": [], "types": []} for year in years}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        for group in data.get("group", []):
            for summary in group.get("work-summary", []):
                title = summary.get("title", {}).get("title", {}).get("value", "")
                pub_type = summary.get("type", "")
                pub_year = summary.get("publication-date", {}).get("year", {}).get("value", "")
                
                if pub_year and str(pub_year) in result:
                    result[str(pub_year)]["titres"].append(title)
                    result[str(pub_year)]["types"].append(pub_type)
                    
    except Exception as e:
        print(f"Erreur pour ORCID {orcid_id} : {e}")
    
    return result

def enrichir_dataframe_publications(df, colonne_orcid):
    """Ajoute les colonnes de titres et types par année au DataFrame."""
    for year in YEARS:
        df[f"Titres_{year}"] = [[] for _ in range(len(df))]
        df[f"Types_{year}"] = [[] for _ in range(len(df))]
    
    for idx, orcid in df[colonne_orcid].items():
        print(idx)
        data = get_titles_and_types_by_year(orcid)
        for year in YEARS:
            df.at[idx, f"Titres_{year}"] = data[str(year)]["titres"]
            df.at[idx, f"Types_{year}"] = data[str(year)]["types"]
    
    return df

######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
d = datetime.date.today()
data = read_data("Data/FairCarboN_Datas_Contacts")  
df = data.copy()

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

df = ajouter_publications_par_annee(df, "ORCID")
# Enrichissement du DataFrame
df = enrichir_dataframe_publications(df, "ORCID")

st.dataframe(df)

df_final = df[['Contact','2018','2019','2020','2021','2022','2023','2024','2025','Titres_2018',
               'Titres_2019','Titres_2020','Titres_2021','Titres_2022','Titres_2023','Titres_2024','Titres_2025',
               'Types_2018','Types_2019','Types_2020','Types_2021','Types_2022','Types_2023','Types_2024','Types_2025']]
df_final.to_csv("test.csv", index=False, encoding="utf-8-sig")