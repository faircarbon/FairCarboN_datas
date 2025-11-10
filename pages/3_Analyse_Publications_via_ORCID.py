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
        response = requests.get(f"{base_url}/works", headers=headers, timeout=20)
        time.sleep(1) 
        response.raise_for_status()
        data = response.json()
        
        for group in data.get("group", []):
            for summary in group.get("work-summary", []):
                put_code = summary.get("put-code")
                title = summary.get("title", {}).get("title", {}).get("value", "")
                pub_type = summary.get("type", "")
                pub_year = summary.get("publication-date", {}).get("year", {}).get("value", "")
                source = summary.get("url", {}).get("value", "")

                authors = []
                try:
                    detail_url = f"{base_url}/work/{put_code}"
                    detail_resp = requests.get(detail_url, headers=headers, timeout=10)
                    detail_resp.raise_for_status()
                    detail_data = detail_resp.json()

                    contributors = detail_data.get("contributors", {}).get("contributor", [])
                    if contributors:
                        for contributor in contributors:
                            name = contributor.get("credit-name", {}).get("value", "")
                            if name:
                                authors.append(name)
                            else:
                                authors = ["Auteurs non renseignés"]

                except Exception as e:
                    authors = ["Erreur récupération auteurs"]

                # Requête supplémentaire pour récupérer les auteurs
                #authors = []
                #try:
                #    detail_url = f"{base_url}/work/{put_code}"
                #    detail_resp = requests.get(detail_url, headers=headers, timeout=20)
                #    time.sleep(1)
                #    detail_resp.raise_for_status()
                #    detail_data = detail_resp.json()

                #    for contributor in detail_data.get("contributors", {}).get("contributor", []):
                #        name = contributor.get("credit-name", {}).get("value", "")
                #        if name:
                #            authors.append(name)

                #except Exception as e:
                #    authors = ["Erreur récupération auteurs"]

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

def get_publications_flat3(orcid_id):
    """Retourne une liste de dicts avec les infos de chaque publication pour un ORCID donné, incluant les auteurs."""
    base_url = f"https://pub.orcid.org/v3.0/{orcid_id}"
    headers = {"Accept": "application/json"}
    publications = []

    try:
        response = requests.get(f"{base_url}/works", headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            print(f"[ERREUR] ORCID {orcid_id} → réponse JSON invalide : {data}")
            return []

        groups = data.get("group")
        if not isinstance(groups, list):
            print(f"[ERREUR] ORCID {orcid_id} → champ 'group' absent ou mal formé : {json.dumps(data, indent=2)}")
            return []

        for group in groups:
            for summary in group.get("work-summary", []):
                put_code = summary.get("put-code")

                # Titre
                title = ""
                title_data = summary.get("title")
                if isinstance(title_data, dict):
                    title_title = title_data.get("title")
                    if isinstance(title_title, dict):
                        title = title_title.get("value", "")

                # Type
                pub_type = summary.get("type", "")

                # Année
                pub_year = ""
                pub_date = summary.get("publication-date")
                if isinstance(pub_date, dict):
                    year_data = pub_date.get("year")
                    if isinstance(year_data, dict):
                        pub_year = year_data.get("value", "")

                # Source (URL)
                source = ""
                url_data = summary.get("url")
                if isinstance(url_data, dict):
                    source = url_data.get("value", "")

                # Auteurs
                authors = []
                try:
                    detail_url = f"{base_url}/work/{put_code}"
                    detail_resp = requests.get(detail_url, headers=headers, timeout=10)
                    detail_resp.raise_for_status()
                    detail_data = detail_resp.json()

                    if not isinstance(detail_data, dict):
                        print(f"[ERREUR] ORCID {orcid_id}, put-code {put_code} → JSON détail invalide : {detail_data}")
                        authors = ["Erreur JSON détail"]
                    else:
                        contributors = detail_data.get("contributors", {}).get("contributor", [])
                        if isinstance(contributors, list) and contributors:
                            for contributor in contributors:
                                name = contributor.get("credit-name", {}).get("value", "")
                                if name:
                                    authors.append(name)
                        else:
                            authors = ["Auteurs non renseignés"]

                except Exception as e:
                    print(f"[ERREUR] ORCID {orcid_id}, put-code {put_code} → erreur récupération auteurs : {e}")
                    authors = ["Erreur récupération auteurs"]

                publications.append({
                    "Orcid": orcid_id,
                    "Année": int(pub_year) if pub_year else None,
                    "Titre": title,
                    "Type": pub_type,
                    "Source": source,
                    "Auteurs": authors
                })

    except Exception as e:
        print(f"[ERREUR] ORCID {orcid_id} → erreur requête principale : {e}")
        return []

    print(f"[INFO] ORCID {orcid_id} → {len(publications)} publications récupérées")
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
        pubs = get_publications_flat3(orcid)
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
    df_publications = pd.read_csv("Data/ORCID/all_publications_ORCID_2025-09-20.csv")
    df_publications['Auteurs']=df_publications['Auteurs'].apply(ast.literal_eval)


df_publications['Premier_auteur']=df_publications['Auteurs'].apply(lambda row: row[0] if (len(row)>0) else None)
df_publications['Auteur_recherché'] = df_publications['contact'] 
df_publications['Titre'] = df_publications['Titre'].str.lower()
df_publications = df_publications[['Orcid','Année','Titre','Type','Auteur_recherché']].drop_duplicates(subset='Titre')
df_publications.reset_index(inplace=True)
df_publications.drop(columns='index', inplace=True)

df_hal = st.session_state['df_hal']


start_year = 2021
end_year = 2025

################# AFFICHAGE HAL ##############################################################

df_hal = df_hal.drop_duplicates(subset=['Ids'])

# Assurer que la colonne 'Date de publication' est bien en datetime
df_hal["Date complete depot"] = pd.to_datetime(df_hal["Date complete depot"], errors="coerce")
df_hal["Année"] = df_hal["Date complete depot"].dt.year

# Filtrer les années souhaitées
df_filtered = df_hal[df_hal["Année"].between(start_year, end_year)]

# Compter les types par année
grouped = df_filtered.groupby(["Année", "Type de document"]).size().reset_index(name="Nombre")

df_pivot = grouped.pivot_table(index="Année", columns="Type de document", values="Nombre", fill_value=0)

# Clés communes (types génériques)
category_map = {
    "ART": "Article",
    "VIDEO": "Video",
    "OTHER": "Autre",
    "other":"Autre",
    "TRAD": "Traduction",
    "SON": "Audio",
    "SOFTWARE": "Logiciel",
    "software": "Logiciel",
    "PATENT": "Brevet",
    "REPORT": "Rapport",
    "report": "Rapport",
    "NOTICE" :"Notice",
    "manual":"Notice",
    "BLOG": "Ressource online",
    "online-resource": "Ressource online",
    "PROCEEDINGS" : "Procédure",
    "UNDEFINED": "Indéfini",
    "MEM": "Mémoire",
    "LECTURE": "Cours",
    "HDR": "HDR",
    "CREPORT":"Chapitre de rapport",
    "IMG":"Image",
    "ISSUE": "Issue",
    "journal-article": "Article",
    "journal-issue": "Issue",
    "COMM": "Communication",
    "conference-abstract": "Communication (abstract)",
    "POSTER": "Poster",
    "conference-poster": "Poster",
    "THESE": "Thèse",
    "dissertation-thesis": "Thèse",
    "COUV": "Chapitre d'ouvrage",
    "OUV": "Ouvrage",
    "book-chapter": "Chapitre d'ouvrage",
    "book": "Ouvrage",
    "edited-book":"Ouvrage édité",
    "working-paper": "Papier en cours",
    "review": "Article de review",
    "newspaper-article":"Article de presse",
    "data-set":"Données",
    "conference-paper": "Communication",
    "dictionary-entry":"Dictionnaire de données",
    "encyclopedia-entry":"Entrée d'encyclopédie"
}

shared_palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#9c9ede", "#6b6ecf", "#b5cf6b", "#cedb9c",
    "#e7ba52", "#e7969c", "#a55194", "#bd9e39"
]

# Associer chaque type générique à une couleur
generic_types = sorted(set(category_map.values()))
color_by_type = {typ: shared_palette[i % len(shared_palette)] for i, typ in enumerate(generic_types)}



# Création du graphique
fig = go.Figure()
for pub_type in df_pivot.columns:
    generic_type = category_map.get(pub_type, pub_type)  # fallback si non mappé
    fig.add_bar(
        x=df_pivot.index,
        y=df_pivot[pub_type],
        name=generic_type,
        marker_color=color_by_type.get(generic_type, "#7f7f7f")  # gris si inconnu
    )

fig.update_layout(
        barmode="stack",
        title="Publications par type et par année HAL",
        xaxis_title="Année",
        yaxis_title="Nombre de publications",
        legend_title="Type de document",
        template="plotly_white",
        height=600
    )


fig.update_layout(
    legend=dict(
        orientation="h",
        y=-0.2,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=10)
    ),
    margin=dict(b=200)  # marge basse élargie
)

st.plotly_chart(fig, use_container_width=True)

################# AFFICHAGE ORCID ##############################################################

# Filtrer les années souhaitées
df_filtered3 = df_publications[df_publications["Année"].between(start_year, end_year)]

# Compter les types par année
grouped3 = df_filtered3.groupby(["Année", "Type"]).size().reset_index(name="Nombre")

df_pivot3 = grouped3.pivot_table(index="Année", columns="Type", values="Nombre", fill_value=0)

# Création du graphique
fig3 = go.Figure()
for pub_type in df_pivot3.columns:
    # Utiliser le type générique comme nom de légende
    generic_type = category_map.get(pub_type, pub_type)  # fallback si non mappé
    fig3.add_bar(
        x=df_pivot3.index,
        y=df_pivot3[pub_type],
        name=generic_type,  # légende simplifiée
        marker_color=color_by_type.get(generic_type, "#7f7f7f")
    )

fig3.update_layout(
        barmode="stack",
        title="Publications par type et par année ORCID",
        xaxis_title="Année",
        yaxis_title="Nombre de publications",
        legend_title="Type de document",
        template="plotly_white",
        height=600
    )

fig3.update_layout(
    legend=dict(
        orientation="h",
        y=-0.2,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=10)
    ),
    margin=dict(b=200)  # marge basse élargie
)

st.plotly_chart(fig3, use_container_width=True)

###############################################################################################################

# Comparaison des titres non dupliqués
liste_article = ['preprint','journal-issue','journal-article']

#LES ARTICLES SUR HAL UNIQUEMENT
df_hal_non_dupliqués = df_filtered[['Année','Premier_auteur','Ids','Auteur_recherché','Titre_unique','Type de document']][df_filtered['Type de document']=='ART'].drop_duplicates(subset='Ids')
df_hal_non_dupliqués['from']='HAL'


df_filtered3['from']='ORCID'
df_filtered3['Titre_unique']=df_filtered3['Titre']
#LES ARTICLES SUR ORCID UNIQUEMENT
df_publications_non_dupliqués = df_filtered3[['Année','Auteur_recherché','Titre','Type','from','Titre_unique']][df_filtered3['Type'].isin(liste_article)].drop_duplicates(subset='Titre')


st.metric(label="Nombre de contacts recherchés", value=len(set(df['Contact'])))

df_avec_ORCID = df.dropna()

st.metric(label="Nombre de contacts ayant un numéro ORCID", value=len(set(df_avec_ORCID['Contact'])))

st.metric(label="Nombre de contacts ORCID", value=len(set(df_publications_non_dupliqués['Auteur_recherché'])))



# CONCATENATION DES DEUX DATAFRAME
df_to_be_compared = pd.concat([df_hal_non_dupliqués[['Année','Auteur_recherché','from','Titre_unique']],df_publications_non_dupliqués[['Année','Auteur_recherché','from','Titre_unique']]], axis=0)
df_to_be_compared.reset_index(inplace=True)
df_to_be_compared.drop(columns='index', inplace=True)


import numpy as np
import unicodedata

# Fonction de normalisation : retire les accents, met en minuscules, supprime les espaces superflus
def normalize(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()

# Appliquer la normalisation à la colonne 'Titre_unique'
df_to_be_compared['Titre_normalisé'] = df_to_be_compared['Titre_unique'].apply(normalize)

# Créer l'ensemble des titres HAL normalisés
titres_hal = set(df_to_be_compared[df_to_be_compared['from'] == 'HAL']['Titre_normalisé'])

# Comparer pour les lignes ORCID
df_to_be_compared['Present_in_HAL'] = np.where(
    (df_to_be_compared['from'] == 'ORCID') & (df_to_be_compared['Titre_normalisé'].isin(titres_hal)),
    True,
    False
)

st.session_state['df_publications'] = df_to_be_compared

# Étape 1 : filtrer les lignes
df_orcid = df_to_be_compared[df_to_be_compared['from'] == 'ORCID']
df_hal = df_to_be_compared[df_to_be_compared['from'] == 'HAL']

# Étape 2 : titres ORCID présents ou non dans HAL
df_orcid['in_HAL'] = df_orcid['Present_in_HAL'] == True
#df_orcid['Année'] = df_orcid['Année'].astype(str)  # pour l'affichage

# Étape 3 : comptage par année
orcid_total = df_orcid.groupby('Année')['Titre_unique'].nunique().reset_index(name='ORCID_total')
orcid_in_hal = df_orcid[df_orcid['in_HAL']].groupby('Année')['Titre_unique'].nunique().reset_index(name='HAL_count')
orcid_not_in_hal = df_orcid[~df_orcid['in_HAL']].groupby('Année')['Titre_unique'].nunique().reset_index(name='ORCID_only')

# Étape 4 : fusion des données
df_yearly = pd.merge(orcid_total, orcid_in_hal, on='Année', how='left')
df_yearly = pd.merge(df_yearly, orcid_not_in_hal, on='Année', how='left')
df_yearly.fillna(0, inplace=True)

# Étape 5 : calcul des pourcentages
df_yearly['HAL_pct'] = df_yearly['HAL_count'] / df_yearly['ORCID_total'] * 100
df_yearly['ORCID_only_pct'] = df_yearly['ORCID_only'] / df_yearly['ORCID_total'] * 100

# Étape 6 : création du graphique
fig5 = go.Figure()

# Barres HAL
fig5.add_trace(go.Bar(
    x=df_yearly['Année'],
    y=df_yearly['HAL_count'],
    name='Titres dans HAL',
    marker_color='green',
    text=df_yearly['HAL_pct'].round(1).astype(str) + '%',
    textposition='inside'
))

# Barres ORCID uniquement
fig5.add_trace(go.Bar(
    x=df_yearly['Année'],
    y=df_yearly['ORCID_only'],
    name='Titres ORCID non trouvés dans HAL',
    marker_color='orange',
    text=df_yearly['ORCID_only_pct'].round(1).astype(str) + '%',
    textposition='inside'
))

# Mise en page
fig5.update_layout(
    barmode='stack',
    title='Comparaison des titres ORCID vs HAL par année',
    xaxis_title='Année',
    yaxis_title='Nombre de titres',
    legend_title='Source',
    template='plotly_white'
)


st.plotly_chart(fig5,use_container_width=True)

#st.session_state['df_publications_orcid_compared'] = df_orcid[df_orcid['in_HAL']==False]


labels = [
    "Compte ORCID (partiel)",
    "Compte ORCID non opé",
    "Aucun compte ORCID"
]

values = [
    252,              # "Compte ORCID permettant de recenser (partiellement ou totalement) les productions scientifiques"
    383 - 252,        # "Compte ORCID mais non opérationnel"
    474 - 383         # "Aucun compte ORCID"
    ]

# Couleurs personnalisées (tu peux les adapter)
custom_colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # vert, orange, rouge

# Créer le pie chart
fig_usageORCID = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    textinfo='label+percent',
    hoverinfo='label+value',
    marker=dict(colors=custom_colors, line=dict(color='#000000', width=1))
)])

fig_usageORCID.update_layout(
    title="Statistiques d'usage ORCID",
    template="plotly_white"
)

st.plotly_chart(fig_usageORCID, use_container_width=True)