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

######################################################################################################################
########### CHOIX VISUELS ############################################################################################
######################################################################################################################
# taille et couleurs des sous-titres
couleur_subtitles = (250,150,150)
taille_subtitles = "25px"
couleur_subsubtitles = (60,150,160)
taille_subsubtitles = "25px"

######################################################################################################################
######################## RDG #########################################################################################
BASE_URL_RDG="https://entrepot.recherche.data.gouv.fr/"
API_TOKEN_RDG="13b493ed-e02b-4e65-95de-d97d6896916a"

###################### CREATION CONNEXION #############################
def connect_to_dataverse(BASE_URL, API_TOKEN):
    try:
        # Create d'une connexion à l'api
        api = NativeApi(BASE_URL, API_TOKEN)
        resp = api.get_info_version()
        response = resp.json()
            
        # vérification de la connexion
        if response['status']=='OK':
            st.session_state['rdg_api'] = api
            st.success("Connexion établie avec Recherche Data Gouv")
        else:
            st.error("Connexion échouée!")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return api

##################################################################################################################
######### RECUPERATION CONTENU DATAVERSE #########################################################################
def Recup_contenu_dataverse(api,s):
    """récupération du contenu du dataverse
    Paramètre = la connexion api réalisée, l'identifiant du dataverse"""
    datav = api.get_dataverse_contents(s)
    datav_contenu = datav.json()
    return datav_contenu

##################################################################################################################
######### RECUPERATION CONTENU DATASET ###########################################################################
def Recup_contenu_dataset(api,persistenteUrl):
    """récupération du contenu du dataset
    Paramètre = la connexion api réalisée,identifiant du dataset"""
    dataset = api.get_dataset(persistenteUrl)
    dataset_contenu = dataset.json()
    return dataset_contenu

##################################################################################################################
######### RECUPERATION DES ENTREPOTS RDG #########################################################################

def get_all_subdataverses(api, dataverse_id, parent_path="root"):
    """
    Recursively fetch all sub-dataverses under a given dataverse.
    
    Parameters:
    - api: connection object with .get_dataverse_contents()
    - dataverse_id: ID or alias of the dataverse to query
    - parent_path: String path showing hierarchy for clarity
    
    Returns:
    - List of dictionaries with each dataverse and its metadata
    """
    results = []
    try:
        response = api.get_dataverse_contents(dataverse_id)
        content = response.json().get("data", [])
    except Exception as e:
        st.write(f"Error retrieving dataverse {dataverse_id}: {e}")
        return results

    for item in content:
        if item.get("type") == "dataverse":
            entry = {
                "name": item.get("title"),
                "id": item.get("id"),
                "parent": dataverse_id,
                "path": parent_path + "/" + item.get("title")
            }
            results.append(entry)
            # Recursive call
            sub_results = get_all_subdataverses(api, item.get("id"), parent_path=entry["path"])
            results.extend(sub_results)
    return results


def recup_dataverses_rdg_recursive(api, output_filename="all_dataverses_rdg.csv"):
    """
    Recursively retrieves all dataverses starting from 'root' in Recherche Data Gouv.
    
    Parameters:
    - api: dataverse API connection
    - output_filename: name of the output CSV file
    """
    all_data = get_all_subdataverses(api, "root", parent_path="Recherche Data Gouv")
    df = pd.DataFrame(all_data)
    
    # Optional: add root level manually if needed
    root_entry = {
        "name": "Recherche Data Gouv",
        "id": "root",
        "parent": None,
        "path": "Recherche Data Gouv"
    }
    df = pd.concat([pd.DataFrame([root_entry]), df], ignore_index=True)

    # Output path
    os.makedirs("Data/RechercheDataGouv", exist_ok=True)
    output_path = os.path.join("Data", "RechercheDataGouv", output_filename)
    
    df.to_csv(output_path, index=False)
    st.write(f"Saved dataverse hierarchy to: {output_path}")
    return df

def Recup_contenu_dataverse(api,s):
    """récupération du contenu du dataverse
    Paramètre = la connexion api réalisée, l'identifiant du dataverse"""
    datav = api.get_dataverse_contents(s)
    datav_contenu = datav.json()
    return datav_contenu

def Recup_contenu_dataset(api,persistenteUrl):
    """récupération du contenu du dataset
    Paramètre = la connexion api réalisée,identifiant du dataset"""
    dataset = api.get_dataset(persistenteUrl)
    dataset_contenu = dataset.json()
    return dataset_contenu

#création du connecteur
api_rdg = connect_to_dataverse(BASE_URL_RDG,  API_TOKEN_RDG)


#récupération des dataverses présents dans RDG
d = datetime.date.today()
fichier = rf'tableau_dataverses_rdg-{d}.csv'

# Code à décommenter pour faire la récupération
#with st.spinner('Recupération des dataverses disponibles et leurs identifiants'):
#    data = recup_dataverses_rdg_recursive(api_rdg)


# Load the previously saved dataverses
df = pd.read_csv("Data/RechercheDataGouv/all_dataverses_rdg.csv")




# Split path into hierarchical levels
df[['level_0','level_1','level_2','level_3','level_4','level_5']] = df['path'].str.split('/', expand=True, n=5)
df['val']=1
df.fillna('', inplace=True)
liste_entrepots_rdg = df['name'].values

liste_entrepots_rdg_visu0 = set(df['level_0'].values)
liste_entrepots_rdg_visu1 = set(df['level_1'].values)
liste_entrepots_rdg_visu2 = set(df['level_2'].values)
liste_entrepots_rdg_visu3 = set(df['level_3'].values)
liste_entrepots_rdg_visu4 = set(df['level_4'].values)
liste_entrepots_rdg_visu5 = set(df['level_5'].values)

l0 = len(liste_entrepots_rdg_visu0)
l1 = len(liste_entrepots_rdg_visu1)
l2 = len(liste_entrepots_rdg_visu2)
l3 = len(liste_entrepots_rdg_visu3)
l4 = len(liste_entrepots_rdg_visu4)
l5 = len(liste_entrepots_rdg_visu5)

cola,colb =st.columns([0.8,0.2])
with cola:
    st.title('Etude du contenu de Recherche Data Gouv')
with colb:
    st.metric(label='Nombre de collections total', value=len(liste_entrepots_rdg))

col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    st.metric(label="NB au niveau 1", value=l1)
with col2:
    st.metric(label="NB au niveau 2", value=l2)
with col3:
    st.metric(label="NB au niveau 3", value=l3)
with col4:
    st.metric(label="NB au niveau 4", value=l4)
with col5:
    st.metric(label="NB au niveau 5", value=l5)


st.write("Total",l0+l1+l2+l3+l4+l5)

df_drop = df.dropna(axis=0)

fig = px.sunburst(df_drop, path=['level_0','level_1','level_2'], values='val')
fig.update_layout(
                width=1000,
                height=1000)

st.subheader("Visualisation de la struturation des entrepôts (2 premiers niveaux)")
st.plotly_chart(fig, use_container_width=True)

stest = "84494"
test = Recup_contenu_dataverse(api_rdg,stest)


#testurl = "https://doi.org/10.57745/IVWMIR"
#testtest = Recup_contenu_dataset(api_rdg,testurl)
#st.write(testtest)



# code pour faire la récupération de l'ensemble des datasets
@st.cache_data
def Recup_datasets_metadata():
    base = "https://entrepot.recherche.data.gouv.fr"
    rows = 10
    start = 0
    page = 1
    condition = True # emulate do-while


    response_init = requests.get(base + '/api/v1/search?q=*&type=dataset')
    response_init.raise_for_status()  # Sécurité : stoppe si erreur
    data_init = response_init.json().get("data", {})
    total_count = data_init.get("total_count", 0)

    all_items = []

    while (condition):
        url = base + '/api/v1/search?q=*&type=dataset' + "&start=" + str(start)
        
        response = requests.get(url)
        response.raise_for_status()  # Sécurité : stoppe si erreur

        data = response.json().get("data", {})
        items = data.get("items", [])

        if not items:
            break

        all_items.extend(items)
        start = start + rows
        page += 1
        print(page)
        condition = start < total_count


    # 🔍 Filtrer uniquement les datasets
    dataset_items = [item for item in all_items if item.get("type") == "dataset"]

    # 🎯 Extraction des champs souhaités
    filtered_data = [
            {"name": item.get("name"), 
            "global_id": item.get("global_id"), 
            'entrepot':item.get('publisher'), 
            'parent':item.get('storageIdentifier'), 
            "Date_Création":item.get('createdAt'),
            "Date_Update":item.get('updatedAt'),
            "Mots_clés":item.get('keywords'),
            "Sujet":item.get('subjects'), 
            "Auteurs":item.get('authors')}
            for item in dataset_items
    ]

    # 📊 DataFrame
    df2 = pd.DataFrame(filtered_data)

    df2['PersistentUrl'] = df2['global_id'].str.replace(r'^doi:', 'https://doi.org/', regex=True)

    # 💾 Sauvegarde en CSV
    df2.to_csv("Data/RechercheDataGouv/all_datasets_rdg.csv", index=False)
    return df2

df2 = Recup_datasets_metadata()

def recup_license(df2):
    df2['status']=""
    df2['license']=""
    for i, item in enumerate(df2["PersistentUrl"]):
        ex = Recup_contenu_dataset(api_rdg, item)
        df2['status'].loc[i] = ex['status']
        try:
            df2['license'].loc[i] =ex['data']['latestVersion']['license']['name']
        except:
            df2['license'].loc[i] ='License inconnue'
    return df2

def transform_name(name):
    name = name.strip()
    if ',' in name:
        # Format: "Lastname, Firstname"
        parts = [part.strip().title() for part in name.split(',', 1)]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    else:
        # Format: "Lastname Firstname"
        parts = name.split()
        if len(parts) >= 2:
            return f"{' '.join(parts[1:]).title()} {parts[0].title()}"
    return name.title()  # fallback

# Append transformed names to original list
df2['Auteurs'] = df2['Auteurs'].apply(
    lambda author_list: author_list + [transform_name(name) for name in author_list]
    if isinstance(author_list, list) else author_list
)

df3 =pd.read_csv("Data\FairCarboN_Datas_Contacts.csv")
liste_contacts = df3['Contact'].values

#df2_filtré = df2[df2["Auteurs"].apply(lambda auteurs: any(nom in liste_contacts for nom in auteurs))]

df2["Contacts_trouvés"] = df2["Auteurs"].apply(
    lambda auteurs: [nom for nom in auteurs if nom in liste_contacts]
)

# Filtrer ensuite les lignes où au moins un contact a été trouvé
df2_filtré = df2[df2["Contacts_trouvés"].apply(lambda x: len(x) > 0)]

liste_contacts_trouves = list(set(nom for sous_liste in df2["Contacts_trouvés"] for nom in sous_liste))

df2_filtré['Date_Update'] = pd.to_datetime(df2_filtré['Date_Update'])
df2_filtré['Value']=1

df2_filtré['Year'] = df2_filtré['Date_Update'].dt.year

df2_filtré_recent = df2_filtré[df2_filtré['Year']>=2023]


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label='Nombre de datasets récupérés', value=len(df2))
with col2:
    st.metric(label='Nombre de datasets rattachés à nos contacts', value=len(df2_filtré))
with col3:
    st.metric(label='Nombre de datasets entre 2023 et 2025', value=len(df2_filtré_recent))
with col4:
    st.metric(label='Nombre de contacts', value=len(liste_contacts_trouves))
st.dataframe(df2_filtré)



# Aggregate (e.g., sum) values by year
df_yearly = df2_filtré.groupby('Year')['Value'].sum().reset_index()

# Plot aggregated data
fig_test = px.bar(df_yearly, x='Year', y='Value', title='Dépôts rattachés aux contacts FaircarboN')
st.plotly_chart(fig_test, use_container_width=True)