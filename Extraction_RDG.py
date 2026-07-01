import streamlit as st
import pandas as pd
from pyDataverse.models import Dataset
from pyDataverse.utils import read_file
from pyDataverse.api import NativeApi
import datetime
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

######################################################################################################################
######################## RDG #########################################################################################
BASE_URL_RDG="https://entrepot.recherche.data.gouv.fr/"
API_TOKEN_RDG="d6ee4496-c075-4ba9-a280-d752513b6af4"

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

########## création du connecteur ###################################################################
api_rdg = connect_to_dataverse(BASE_URL_RDG,  API_TOKEN_RDG)
#####################################################################################################

def extract_funding_info_from_url(url):

    try:
        data = Recup_contenu_dataset(api_rdg,url)

        project_info = data['data']['latestVersion']['metadataBlocks']['citation']['fields']

        # Initialiser les valeurs
        grant_number = None
        acronym = None

        for field in project_info:
            if field['typeName'] == 'grantNumber':
                grant_number = field['value'][0]['grantNumberValue']['value']
            if field['typeName'] == 'project':
                acronym = field['value'][0]['projectAcronym']['value']

        return pd.Series([grant_number, acronym])

    except Exception as e:
        print(f"Erreur pour l'URL {url}: {e}")
        return pd.Series([None, None])

def extraire_urls(source):
    if isinstance(source, list):
        return [item['url'] for item in source if isinstance(item, dict) and 'url' in item]
    return []

def get_suffix_after_third_slash(source_list):
    if not source_list:
        return ''
    
    suffixes = []
    for url in source_list:
        parts = url.split("/", 3)  # split into at most 4 parts
        if len(parts) > 3:
            suffixes.append(parts[3])  # the part after the third "/"
    return suffixes

def forcer_en_liste(val):
    if isinstance(val, list):
        return val
    elif pd.isna(val):
        return []
    else:
        return [val]
    
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
            {"Nom_archive":"Recherche Data Gouv",
            "Titre_unique": item.get("name"), 
            "global_id": item.get("global_id"), 
            'entrepot':item.get('publisher'), 
            'parent':item.get('storageIdentifier'), 
            "Date_Création":item.get('createdAt'),
            "Date_Update":item.get('updatedAt'),
            "Mots_clés":item.get('keywords'),
            "Sujet":item.get('subjects'), 
            "Auteurs":item.get('authors'),
            "Sources":item.get('publications'),
            "Type de document":"dataset-rdg"}
            for item in dataset_items
    ]

    # 📊 DataFrame
    df2 = pd.DataFrame(filtered_data)

    df2['PersistentUrl'] = df2['global_id'].str.replace(r'^doi:', 'https://doi.org/', regex=True)

    #df2[['grant_number', 'project_acronym']] = df2['PersistentUrl'].apply(extract_funding_info_from_url)

    # Application de la fonction
    df2["Sources"] = df2["Sources"].apply(extraire_urls)
    df2["DOI sources"] = df2["Sources"].apply(get_suffix_after_third_slash)

    # Append transformed names to original list
    df2['Auteurs'] = df2['Auteurs'].apply(
        lambda author_list: author_list + [transform_name(name) for name in author_list]
        if isinstance(author_list, list) else author_list
    )

    df2["Contacts_trouvés"] = df2["Auteurs"].apply(
        lambda auteurs: [nom for nom in auteurs if nom in liste_contacts]
    )
    df2['Auteur_recherché'] = df2["Contacts_trouvés"]
    df2 = df2.explode('Auteur_recherché').reset_index(drop=True)

    df2['DOI sources'] = df2['DOI sources'].apply(forcer_en_liste)

    # Filtrer ensuite les lignes où au moins un contact a été trouvé
    df2_filtré = df2[df2['Auteur_recherché'].notna() & (df2['Auteur_recherché'] != '')]
    df2_merged = df2_filtré.merge(df_contacts_grouped, on='Auteur_recherché', how='left')
    df2_merged.reset_index(drop=True)

    df2_merged['Date_Update'] = pd.to_datetime(df2_merged['Date_Update'])
    df2_merged['Value']=1

    df2_merged['Date de publication'] = df2_merged['Date_Update'].dt.year

    df2_merged['projet'] = df2_merged['projet'].str.split(',').apply(lambda x: [p.strip() for p in x if p.strip()])
    df2_merged = df2_merged.explode('projet').reset_index(drop=True)

    # 💾 Sauvegarde en CSV
    df2_merged.to_csv(f"Data/RechercheDataGouv/all_datasets_rdg_{d}.csv", index=False)
    return df2_merged


def recup_license_publication(df2):
    df2['status'] = ""
    df2['license'] = ""
    df2['publication_DOI'] = ""

    for i, item in enumerate(df2["PersistentUrl"]):
        try:
            ex = Recup_contenu_dataset(api_rdg, item)
        except Exception as e:
            # If API call fails entirely
            df2.loc[i, 'status'] = f"Erreur API: {type(e).__name__}"
            df2.loc[i, 'license'] = 'Erreur API'
            df2.loc[i, 'publication_DOI'] = 'Erreur API'
            continue

        status = ex.get('status', 'inconnu')
        df2.loc[i, 'status'] = status

        # 🚫 Skip if status is not 'OK'
        if status.lower() != 'ok':
            df2.loc[i, 'license'] = 'Non récupéré'
            df2.loc[i, 'publication_DOI'] = 'Non récupérée'
            continue

        # ✅ License
        try:
            df2.loc[i, 'license'] = ex['data']['latestVersion']['license']['name']
        except (KeyError, TypeError):
            df2.loc[i, 'license'] = 'License inconnue'

        # ✅ Publication DOI
        try:
            fields = ex['data']['latestVersion']['metadataBlocks']['citation']['fields']
            publication_found = False

            for field in fields:
                if field.get('typeName') == "publication":
                    values = field.get('value', [])
                    publication_dois = [pub.get('publicationIDNumber', {}).get('value') for pub in values]
                    df2.loc[i, 'publication_DOI'] = "; ".join(publication_dois)
                    publication_found = True
                    break

            if not publication_found:
                df2.loc[i, 'publication_DOI'] = 'pas de publication trouvée'
        except (KeyError, TypeError):
            df2.loc[i, 'publication_DOI'] = 'pas de publication trouvée'

    return df2

######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################

#récupération des dataverses présents dans RDG
d = datetime.date.today()
start_year=2024
end_year=d.year

######################################################################################################################
# Code à décommenter pour faire la récupération des dataverses
#with st.spinner('Recupération des dataverses disponibles et leurs identifiants'):
    #data = recup_dataverses_rdg_recursive(api_rdg)
######################################################################################################################

# Load the previously saved dataverses
df = pd.read_csv("Data/RechercheDataGouv/all_dataverses_rdg.csv")
df_contacts =read_data("Data\FairCarboN_Datas_Contacts2")
df_contacts['Auteur_recherché']=df_contacts['Contact']
df_contacts_grouped = df_contacts.groupby('Auteur_recherché')['projet'].apply(lambda x: ', '.join(sorted(set(x)))).reset_index()

liste_contacts = df_contacts['Contact'].values

#df2 = Recup_datasets_metadata()
#print("Récupération RDG réalisée avec succès!")