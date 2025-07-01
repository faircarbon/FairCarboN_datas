import streamlit as st
import pandas as pd
from pyDataverse.models import Dataset
from pyDataverse.utils import read_file
from pyDataverse.api import NativeApi
import datetime
import numpy as np
import re
import plotly.express as px

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
def Recup_dataverses_rdg(api, fichier):
    """récupération de différents sous-dataverses existants dans RDG
    Paramètre = la connexion api réalisée , le nom du fichier où les enregistrer"""
    RDG = api.get_dataverse_contents("root")
    RDG_json = RDG.json()
    liste_dataverses_1= []
    liste_ids = []
    for d in range(len(RDG_json['data'])):
        if RDG_json['data'][d]['type']=="dataverse":
            liste_dataverses_1.append(RDG_json['data'][d]['title'])
            liste_ids.append(RDG_json['data'][d]['id'])

    df_liste_dataverses_1=pd.DataFrame(data=[liste_dataverses_1,liste_ids], index=['Dataverses_niv1','Ids'])
    df_liste_dataverses_1=df_liste_dataverses_1.T
    
    liste = []
    ids = []
    for i in range(len(df_liste_dataverses_1)):
        datav = api.get_dataverse_contents(df_liste_dataverses_1.loc[i,'Ids'])
        datav_dv = datav.json()
        liste_dataverses_2 = []
        ids_niv2 = []
        for d in range(len(datav_dv['data'])):
            try:
                if datav_dv['data'][d]['type']=="dataverse":
                    liste_dataverses_2.append(datav_dv['data'][d]['title'])
                    ids_niv2.append(datav_dv['data'][d]['id'])
            except:
                    liste_dataverses_2.append()
                    ids_niv2.append()
        liste.append(liste_dataverses_2)
        ids.append(ids_niv2)
            
    df_liste_dataverses_1['Dataverses_niv2']=liste
    df_liste_dataverses_1['Ids_niv2']=ids
    df_liste_dataverses_1.to_csv(f"Data\RechercheDataGouv\liste_dataverses_rdg.csv")
            
    df_liste_dataverses_2=pd.DataFrame(data=[liste,ids], index=['Dataverses_niv2','Ids_niv2'])
    df_liste_dataverses_2=df_liste_dataverses_2.T
    df_liste_dataverses_2.to_csv(f"Data\RechercheDataGouv\liste_dataverses_rdg2.csv")

    data = pd.read_csv(f"Data\RechercheDataGouv\liste_dataverses_rdg.csv")
    data.drop(columns=['Unnamed: 0'], inplace=True)
    for i in range(len(data)):
            data.loc[i,'val']=int(len(re.split(',',data.loc[i,'Dataverses_niv2'].replace('[','').replace(']','').replace("'",'').strip())))

    som = sum(data['val'].values)
    new_data = pd.DataFrame(index=np.arange(0,som), columns=['niv1','niv2'])
    i=0
    for j in range(len(data)):
        for k in range(int(data.loc[j,'val'])):
            new_data.loc[i,'niv1']=data.loc[j,'Dataverses_niv1']
            new_data.loc[i,'ids_niv1']=data.loc[j,'Ids']
            new_data.loc[i,'niv2']=re.split(',',data.loc[j,'Dataverses_niv2'].replace('[','').replace(']','').strip())[k]
            new_data.loc[i,'niv2']=new_data.loc[i,'niv2'].replace("'","")
            try:
                new_data.loc[i,'ids_niv2']=re.split(',',data.loc[j,'Ids_niv2'].replace('[','').replace(']','').replace('"','').strip())[k]
            except:
                pass
            i+=1
            print(i)
    new_data['val']=1
    new_data['niv0']="Recherche Data Gouv"
    new_data.to_csv(f"Data\RechercheDataGouv\{fichier}")
    return new_data


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

#création du connecteur
api_rdg = connect_to_dataverse(BASE_URL_RDG,  API_TOKEN_RDG)


#récupération des dataverses présents dans RDG
d = datetime.date.today()
fichier = rf'tableau_dataverses_rdg-{d}.csv'
#with st.spinner('Recupération des dataverses disponibles et leurs identifiants'):
#    data = recup_dataverses_rdg_recursive(api_rdg)


# Load the previously saved dataverses
df = pd.read_csv("Data/RechercheDataGouv/all_dataverses_rdg.csv")
# Split path into hierarchical levels
df[['level_0','level_1','level_2','level_3','level_4','level_5']] = df['path'].str.split('/', expand=True, n=5)
df['val']=1
df.fillna('', inplace=True)
st.dataframe(df)
liste_entrepots_rdg = df['name'].values
st.write(len(liste_entrepots_rdg))

liste_entrepots_rdg_visu0 = set(df['level_0'].values)
liste_entrepots_rdg_visu1 = set(df['level_1'].values)
liste_entrepots_rdg_visu2 = set(df['level_2'].values)
liste_entrepots_rdg_visu3 = set(df['level_3'].values)
liste_entrepots_rdg_visu4 = set(df['level_4'].values)
liste_entrepots_rdg_visu5 = set(df['level_5'].values)
#liste_entrepots_rdg_visu6 = set(df['level_6'].values)
l0 = len(liste_entrepots_rdg_visu0)
l1 = len(liste_entrepots_rdg_visu1)
l2 = len(liste_entrepots_rdg_visu2)
l3 = len(liste_entrepots_rdg_visu3)
l4 = len(liste_entrepots_rdg_visu4)
l5 = len(liste_entrepots_rdg_visu5)
#l6 = len(liste_entrepots_rdg_visu6)
st.write(l0)
st.write(l1)
st.write(l2)
st.write(l3)
st.write(l4)
st.write(l5)
#st.write(l6)
st.write("Total",l0+l1+l2+l3+l4+l5)

df_drop = df.dropna(axis=0)

fig = px.sunburst(df_drop, path=['level_0','level_1','level_2'], values='val')
fig.update_layout(
                title=f'Visuel des différents Dataverses',
                width=1000,
                height=1000)

st.plotly_chart(fig, use_container_width=True)