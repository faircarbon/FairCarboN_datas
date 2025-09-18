import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import plotly.graph_objects as go
from collections import Counter
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

@st.cache_data
def ajouter_nombre_publications(df, colonne_orcid):
    """Ajoute une colonne 'Nombre_publis' au DataFrame en comptant les publications via ORCID."""
    df["Nombre_publis"] = df[colonne_orcid].apply(get_publication_count)
    return df

def get_publications_by_year(orcid_id, years):
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

@st.cache_data
def ajouter_publications_par_annee(df, colonne_orcid):
    """Ajoute une colonne par année avec le nombre de publications pour chaque ORCID."""
    for year in YEARS:
        df[str(year)] = 0  # Initialisation des colonnes
    
    for idx, orcid in df[colonne_orcid].items():
        counts = get_publications_by_year(orcid,YEARS)
        for year, count in counts.items():
            df.at[idx, year] = count
            
    return df


def get_publications_flat(orcid_id):
    """Retourne une liste de dicts avec les infos de chaque publication pour un ORCID donné."""
    url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    headers = {"Accept": "application/json"}
    publications = []

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        for group in data.get("group", []):
            for summary in group.get("work-summary", []):
                title = summary.get("title", {}).get("title", {}).get("value", "")
                pub_type = summary.get("type", "")
                pub_year = summary.get("publication-date", {}).get("year", {}).get("value", "")
                source = summary.get("url", {}).get("value", "")  # parfois vide

                if pub_year:  # On ignore les publications sans date
                    publications.append({
                        "Orcid": orcid_id,
                        "Année": int(pub_year),
                        "Titre": title,
                        "Type": pub_type,
                        "Source": source
                    })

    except Exception as e:
        print(f"Erreur pour ORCID {orcid_id} : {e}")

    return publications

@st.cache_data
def construire_dataframe_publications(df_contacts):
    """Construit un DataFrame aplati avec les publications et les contacts associés."""
    all_publications = []

    for _, row in df_contacts.iterrows():
        orcid = row["ORCID"]
        contact = row["Contact"]
        pubs = get_publications_flat(orcid)
        for pub in pubs:
            pub["contact"] = contact
        all_publications.extend(pubs)

    return pd.DataFrame(all_publications)


######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
d = datetime.date.today()
YEARS = list(range(2018, 2026))  # De 2018 à 2025 inclus
data = read_data("Data/FairCarboN_Datas_Contacts")  
df = data[['Contact','ORCID']]

st.success("Connexion établie avec ORCID")
st.title(f":grey[Etude des travaux de la communauté FairCarboN]")

#Ajout de la colonne Nombre_publis
#df = ajouter_nombre_publications(df, "ORCID")

# Enrichissement du DataFrame
#df = ajouter_publications_par_annee(df, "ORCID")

df.to_csv(f"Data/ORCID/all_publications_ORCID_{d}.csv", index=False, encoding="utf-8-sig")


df_publications = construire_dataframe_publications(df)

st.dataframe(df_publications)

df_publications.to_csv(f"Data/ORCID/all_publications_ORCID_{d}_bis.csv", index=False, encoding="utf-8-sig")

df_hal = st.session_state['df_hal']

st.dataframe(df_hal)

start_year = 2018
end_year = 2025
# Assurer que la colonne 'Date de publication' est bien en datetime
df_hal["Date complete depot"] = pd.to_datetime(df_hal["Date complete depot"], errors="coerce")
df_hal["Année"] = df_hal["Date complete depot"].dt.year

# Filtrer les années souhaitées
df_filtered = df_hal[df_hal["Année"].between(start_year, end_year)]

# Compter les types par année
grouped = df_filtered.groupby(["Année", "Type de document"]).size().reset_index(name="Nombre")

df_pivot = grouped.pivot_table(index="Année", columns="Type de document", values="Nombre", fill_value=0)

# Création du graphique
fig = go.Figure()
for pub_type in df_pivot.columns:
    fig.add_bar(
            x=df_pivot.index,
            y=df_pivot[pub_type],
            name=pub_type
        )

fig.update_layout(
        barmode="stack",
        title="Publications par type et par année HAL",
        xaxis_title="Année",
        yaxis_title="Nombre de publications",
        legend_title="Type de document",
        template="plotly_white"
    )

st.plotly_chart(fig, use_container_width=True)


# Filtrer les années souhaitées
df_filtered3 = df_publications[df_publications["Année"].between(start_year, end_year)]

# Compter les types par année
grouped3 = df_filtered3.groupby(["Année", "Type"]).size().reset_index(name="Nombre")

df_pivot3 = grouped3.pivot_table(index="Année", columns="Type", values="Nombre", fill_value=0)

# Création du graphique
fig3 = go.Figure()
for pub_type in df_pivot3.columns:
    fig3.add_bar(
            x=df_pivot3.index,
            y=df_pivot3[pub_type],
            name=pub_type
        )

fig3.update_layout(
        barmode="stack",
        title="Publications par type et par année ORCID",
        xaxis_title="Année",
        yaxis_title="Nombre de publications",
        legend_title="Type de document",
        template="plotly_white"
    )

st.plotly_chart(fig3, use_container_width=True)