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

def get_titles_and_types_by_year(orcid_id, years):
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

@st.cache_data
def enrichir_dataframe_publications(df, colonne_orcid):
    """Ajoute les colonnes de titres et types par année au DataFrame."""
    for year in YEARS:
        df[f"Titres_{year}"] = [[] for _ in range(len(df))]
        df[f"Types_{year}"] = [[] for _ in range(len(df))]
    
    for idx, orcid in df[colonne_orcid].items():
        print(idx)
        data = get_titles_and_types_by_year(orcid, YEARS)
        for year in YEARS:
            df.at[idx, f"Titres_{year}"] = data[str(year)]["titres"]
            df.at[idx, f"Types_{year}"] = data[str(year)]["types"]
    
    return df

def prepare_data_for_plot(df, years):
    """Transforme les colonnes Types_YYYY en un DataFrame de comptage par type et année."""
    data = []

    for year in years:
        col = f"Types_{year}"
        if col not in df.columns:
            continue
        
        # Aplatir toutes les listes de types pour cette année
        all_types = [typ for sublist in df[col] for typ in sublist if isinstance(sublist, list)]
        counts = Counter(all_types)
        
        for pub_type, count in counts.items():
            data.append({
                "Année": year,
                "Type": pub_type,
                "Nombre": count
            })
    
    return pd.DataFrame(data)

def prepare_counts(df, start_year=2018, end_year=2025):
    """Prépare les données agrégées par année et type de publication."""
    # Assurer que la colonne 'Date de publication' est bien en datetime
    df["Date complete"] = pd.to_datetime(df["Date complete"], errors="coerce")
    df["Année"] = df["Date complete"].dt.year

    # Filtrer les années souhaitées
    df_filtered = df[df["Année"].between(start_year, end_year)]

    # Compter les types par année
    grouped = df_filtered.groupby(["Année", "Type de document"]).size().reset_index(name="Nombre")

    return grouped

def plot_stacked_bar2(df_counts):
    """Crée un graphique à barres empilées avec Plotly."""
    # Pivot pour avoir les types en colonnes
    df_pivot = df_counts.pivot_table(index="Année", columns="Type de document", values="Nombre", fill_value=0)

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
        title="Publications par type et par année",
        xaxis_title="Année",
        yaxis_title="Nombre de publications",
        legend_title="Type de document",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_stacked_bar(df_counts):
    """Crée un bar plot empilé avec Plotly à partir du DataFrame de comptage."""
    # Pivot pour avoir les types en colonnes
    df_pivot = df_counts.pivot_table(index="Année", columns="Type", values="Nombre", fill_value=0)
    
    # Création des barres empilées
    fig = go.Figure()
    for pub_type in df_pivot.columns:
        fig.add_bar(
            x=df_pivot.index,
            y=df_pivot[pub_type],
            name=pub_type
        )
    
    fig.update_layout(
        barmode="stack",
        title="Nombre de publications par type et par année",
        xaxis_title="Année",
        yaxis_title="Nombre de publications",
        legend_title="Type de publication",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)



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
df = ajouter_nombre_publications(df, "ORCID")

# Enrichissement du DataFrame
df = ajouter_publications_par_annee(df, "ORCID")
df = enrichir_dataframe_publications(df, "ORCID")

st.dataframe(df)

df.to_csv(f"Data/ORCID/all_publications_ORCID_{d}.csv", index=False, encoding="utf-8-sig")

# 🧪 Exemple d’utilisation
df_counts = prepare_data_for_plot(df, YEARS)
plot_stacked_bar(df_counts)

st.dataframe(df_counts)

df_hal = st.session_state['df_hal']

df_counts2 = prepare_counts(df_hal, start_year=2018, end_year=2025)
plot_stacked_bar2(df_counts2)

# Filtrer les types pertinents
df1_filtered = df_counts[df_counts['Type'] == 'journal-article']
df2_filtered = df_counts2[df_counts2['Type de document'] == 'ART']

# S'assurer que les années sont bien triées
df1_filtered = df1_filtered.sort_values(by='Année')
df2_filtered = df2_filtered.sort_values(by='Année')

# Créer le graphique à barres
fig = go.Figure()

fig.add_trace(go.Bar(
    x=df1_filtered['Année'],
    y=df1_filtered['Nombre'],
    name='journal-article',
    marker_color='blue'
))

fig.add_trace(go.Bar(
    x=df2_filtered['Année'],
    y=df2_filtered['Nombre'],
    name='ART',
    marker_color='orange'
))

# Mise en forme du graphique
fig.update_layout(
    title='Comparaison des publications par année',
    xaxis_title='Année',
    yaxis_title='Nombre de publications',
    barmode='group',  # Affiche les barres côte à côte
    template='plotly_white',
    legend_title='Type de publication'
)

st.plotly_chart(fig, use_container_width=True)