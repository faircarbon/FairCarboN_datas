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
import ast


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

def get_publications_flat2(orcid_id):
    """Retourne une liste de dicts avec les infos de chaque publication pour un ORCID donné, incluant les auteurs."""
    base_url = f"https://pub.orcid.org/v3.0/{orcid_id}"
    headers = {"Accept": "application/json"}
    publications = []

    try:
        response = requests.get(f"{base_url}/works", headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        for group in data.get("group", []):
            for summary in group.get("work-summary", []):
                try:
                    put_code = summary.get("put-code")
                except:
                    put_code = None
                try:
                    title = summary.get("title", {}).get("title", {}).get("value", "")
                except:
                    title = None
                try:
                    pub_type = summary.get("type", "")
                except:
                    pub_type = None
                try:
                    pub_year = summary.get("publication-date", {}).get("year", {}).get("value", "")
                except:
                    pub_year= None
                try:
                    source = summary.get("url", {}).get("value", "")
                except:
                    source = None

                # Requête supplémentaire pour récupérer les auteurs
                authors = []
                try:
                    detail_url = f"{base_url}/work/{put_code}"
                    detail_resp = requests.get(detail_url, headers=headers, timeout=10)
                    detail_resp.raise_for_status()
                    detail_data = detail_resp.json()

                    for contributor in detail_data.get("contributors", {}).get("contributor", []):
                        name = contributor.get("credit-name", {}).get("value", "")
                        if name:
                            authors.append(name)

                except Exception as e:
                    authors = ["Erreur récupération auteurs"]

                if pub_year:
                    publications.append({
                        "Orcid": orcid_id,
                        "Année": int(pub_year),
                        "Titre": title,
                        "Type": pub_type,
                        "Source": source,
                        "Auteurs": authors
                    })

    except Exception as e:
        print(f"Erreur pour ORCID {orcid_id} : {e}")

    return publications

@st.cache_data
def construire_dataframe_publications(df_contacts):
    """Construit un DataFrame aplati avec les publications et les contacts associés."""
    all_publications = []

    i = 0
    for _, row in df_contacts.iterrows():
        print(i)
        orcid = row["ORCID"]
        contact = row["Contact"]
        pubs = get_publications_flat2(orcid)
        for pub in pubs:
            pub["contact"] = contact
        all_publications.extend(pubs)
        i += 1

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

lancement_recherche = st.checkbox(label='Lancer recherche sur ORCID')

if lancement_recherche:

    df_publications = construire_dataframe_publications(df)
    df_publications.to_csv(f"Data/ORCID/all_publications_ORCID_{d}.csv", index=False, encoding="utf-8-sig")

else:
    df_publications = pd.read_csv("Data/ORCID/all_publications_ORCID_2025-09-19.csv")
    df_publications['Auteurs']=df_publications['Auteurs'].apply(ast.literal_eval)


df_publications['Premier_auteur']=df_publications['Auteurs'].apply(lambda row: row[0] if (len(row)>0) else None)

st.session_state['df_publications'] = df_publications

df_hal = st.session_state['df_hal']

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

# Comparaison des titres non dupliqués
liste_article = ['preprint','journal-issue','journal-article']
df_hal_non_dupliqués = df_filtered[['Année','Premier_auteur','Auteur_recherché','Titre_unique','Type de document']][df_filtered['Type de document']=='ART'].drop_duplicates()
df_hal_non_dupliqués['from']='HAL'
df_publications_non_dupliqués = df_filtered3[['Année','Premier_auteur','contact','Titre','Type']][df_filtered3['Type'].isin(liste_article)].drop_duplicates()


df_publications_non_dupliqués['from']='ORCID'
df_publications_non_dupliqués['Auteur_recherché']=df_publications_non_dupliqués['contact']
df_publications_non_dupliqués['Titre_unique']=df_publications_non_dupliqués['Titre']

df_to_be_compared = pd.concat([df_hal_non_dupliqués[['Année','Auteur_recherché','Premier_auteur','from','Titre_unique']],df_publications_non_dupliqués[['Année','Auteur_recherché','Premier_auteur','from','Titre_unique']]], axis=0)
df_to_be_compared.reset_index(inplace=True)
df_to_be_compared.drop(columns='index', inplace=True)

# Créer un ensemble des titres présents dans les lignes 'HAL'
titres_hal = set(df_to_be_compared[df_to_be_compared['from'] == 'HAL']['Titre_unique'])

# Appliquer une fonction pour vérifier si chaque ligne ORCID est dans titres_hal
df_to_be_compared['Present_in_HAL'] = df_to_be_compared.apply(
    lambda row: row['Titre_unique'] in titres_hal if row['from'] == 'ORCID' else None,
    axis=1
)

st.dataframe(df_to_be_compared[df_to_be_compared['from'] == 'ORCID'], hide_index=True)

df_to_be_compared_non_dupliqués = df_to_be_compared[['Année','Auteur_recherché','from','Titre_unique','Present_in_HAL']].drop_duplicates()

st.session_state['df_publications_orcid_compared'] = df_to_be_compared[df_to_be_compared['from'] == 'ORCID']

# Filtrer les lignes avec des valeurs True ou False
df_filt = df_to_be_compared_non_dupliqués[df_to_be_compared_non_dupliqués['Present_in_HAL'].isin([True, False])]

# Compter les occurrences
#counts = df_filt['Present_in_HAL'].value_counts().reset_index()
#counts.columns = ['Present_in_HAL', 'count']
#counts['Present_in_HAL'] = counts['Present_in_HAL'].map({True: 'Présent dans HAL', False: 'Absent de HAL'})

# Remapper les valeurs pour plus de lisibilité
df_filt['Présence'] = df_filt['Present_in_HAL'].map({True: 'Présent dans HAL', False: 'Absent de HAL'})

# Agréger les données
counts = df_filt.groupby(['Année', 'Present_in_HAL']).size().unstack(fill_value=0)

# Calculer les pourcentages pour 'Présent dans HAL'
percentages = (counts[True] / (counts[True] + counts[False]) * 100).round(1)


# Créer le graphe en barres empilées avec go.Bar
fig4 = go.Figure()

fig4.add_bar(
    x=counts.index,
    y=counts[True],
    name='Présent dans HAL',
    marker_color='mediumseagreen',
    text=[f"{p}%" for p in percentages],
    textposition='inside'
)

fig4.add_bar(
    x=counts.index,
    y=counts[False],
    name='Absent de HAL',
    marker_color='salmon'
)

fig4.update_layout(
    barmode='stack',
    title='Présence des titres ORCID dans HAL par année',
    xaxis_title='Année',
    yaxis_title='Nombre de titres',
    template='plotly_white'
)

st.plotly_chart(fig4,use_container_width=True)