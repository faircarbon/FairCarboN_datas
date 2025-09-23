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

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON RDG DATA",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

######################################################################################################################
######################## RDG #########################################################################################
BASE_URL_RDG="https://entrepot.recherche.data.gouv.fr/"
API_TOKEN_RDG="13b493ed-e02b-4e65-95de-d97d6896916a"

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
    df = pd.read_excel(f"{path}.xlsx", sheet_name=1,header=0, engine='openpyxl')
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

# Code à décommenter pour faire la récupération des dataverses
#with st.spinner('Recupération des dataverses disponibles et leurs identifiants'):
#    data = recup_dataverses_rdg_recursive(api_rdg)

# Load the previously saved dataverses
df = pd.read_csv("Data/RechercheDataGouv/all_dataverses_rdg.csv")
df_contacts =pd.read_csv("Data\FairCarboN_Datas_Contacts.csv")
df_contacts['Auteur_recherché']=df_contacts['Contact']
df_contacts_grouped = df_contacts.groupby('Auteur_recherché')['projet'].apply(lambda x: ', '.join(sorted(set(x)))).reset_index()

liste_contacts = df_contacts['Contact'].values
df2 = Recup_datasets_metadata()

st.session_state['df_rdg'] = df2

######################################################################################################################
########### Visualisation contenu dataverses RDG #####################################################################
######################################################################################################################

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


#st.write("Total",l0+l1+l2+l3+l4+l5)

df_drop = df.dropna(axis=0)

fig = px.sunburst(df_drop, path=['level_0','level_1','level_2'], values='val')
fig.update_layout(
                width=1000,
                height=1000)

st.subheader("Visualisation de la struturation des entrepôts (2 premiers niveaux)")
st.plotly_chart(fig, use_container_width=True)

# Aggregate (e.g., sum) values by year
df_yearly = df2.groupby('Date de publication')['Value'].sum().reset_index()

# Plot aggregated data
fig_dates = px.bar(df_yearly, x='Date de publication', y='Value', title='Dépôts rattachés aux contacts FaircarboN')
st.plotly_chart(fig_dates, use_container_width=True)

#stest = "84494"
#test = Recup_contenu_dataverse(api_rdg,stest)

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################
projets = list(set(df2['projet']))
auteurs = list(set(df2['Auteur_recherché']))
col1,col2 = st.columns(2)
with col1:
    st.subheader(f":grey[Choix du/des projet(s) visualisé(s)]")
    choix_projet = st.multiselect(label='', options=projets )
    if len(choix_projet)==0:
        choix_p = projets
    else:
        choix_p = choix_projet
with col2:
    st.subheader(f":grey[Choix de(s) l'auteur(e(s)) visualisé(e(s))]")
    choix_auteur = st.multiselect(label='', options=list(set(df2['Auteur_recherché'][df2['projet'].isin(choix_p)])))
    if len(choix_auteur)==0:
        choix_a = df2['Auteur_recherché'][df2['projet'].isin(choix_p)]
    else:
        choix_a = choix_auteur


######################################################################################################################
########### Visualisation contenu RDG ################################################################################
######################################################################################################################

df_rdg_proj =df2[df2['projet'].isin(choix_p)][df2['Auteur_recherché'].isin(choix_a)][df2['Date de publication']>=start_year]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label='Nombre de datasets rattachés à nos contacts', value=len(df2))
with col2:
    st.metric(label=f'Nombre de datasets entre {start_year} et {end_year}', value=len(df_rdg_proj))
with col3:
    st.metric(label='Nombre de contacts', value=len(set(df_rdg_proj['Auteur_recherché'].values)))


unique_projet_titles = df_rdg_proj[['projet','Titre_unique']].drop_duplicates()
projects_count = unique_projet_titles['projet'].value_counts().reset_index()
projects_count.columns = ['Projet', 'compte']

unique_person_titles = df_rdg_proj[['Auteur_recherché','Titre_unique']].drop_duplicates()
row_counts = unique_person_titles['Auteur_recherché'].value_counts().reset_index()
row_counts.columns = ['Auteur', 'compte']

###################################################################################################################################
fig = px.pie(
    projects_count,
    names='Projet',
    values='compte',
    title='Répartition des publications parmi les membres des projets',
    color_discrete_sequence=px.colors.qualitative.Set3,
    hole=0.3  
)

fig1 = px.pie(
    projects_count,
    names='Projet',
    values='compte',
    title='Participation aux projets',
    color_discrete_sequence=px.colors.qualitative.Set3,
    hole=0.3
)
fig1.update_traces(textinfo='label')
fig1.update_layout(showlegend=False)

# Box plot using Plotly
fig2 = px.box(row_counts, y='compte', points="all",hover_data=['Auteur'], title="Distribution du nombre de publications parmi ces membres")
fig2.update_traces(marker_color='tomato', line_color='tomato')

# Affichage
col1,col2 = st.columns(2)
with col1:
    if len(choix_auteur)==0:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig1, use_container_width=True)
    
with col2:
    st.plotly_chart(fig2, use_container_width=True)



#url_test = "https://entrepot.recherche.data.gouv.fr" + '/api/v1/search?q="Laurent Augusto"&type=dataset'        
#response_t = requests.get(url_test)
#response_t.raise_for_status()  # Sécurité : stoppe si erreur
#data_t = response_t.json().get("data", {})
#items_t = data_t.get("items", [])
#st.write(items_t)

#testurl = "https://doi.org/10.57745/NEBK4J"
#testtest = Recup_contenu_dataset(api_rdg,testurl)
#st.write(testtest)

#df2_test = df2_filtré[df2_filtré['Auteur_recherché']=="Laurent Augusto"]

#df2_test[['grant_number', 'project_acronym']] = df2_test['PersistentUrl'].apply(extract_funding_info_from_url)

#st.dataframe(df2_test)