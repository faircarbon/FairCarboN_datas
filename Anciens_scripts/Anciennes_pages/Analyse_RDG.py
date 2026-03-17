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
#    data = recup_dataverses_rdg_recursive(api_rdg)
######################################################################################################################

# Load the previously saved dataverses
df = pd.read_csv("Data/RechercheDataGouv/all_dataverses_rdg.csv")
df_contacts =pd.read_csv("Data\FairCarboN_Datas_Contacts.csv")
df_contacts['Auteur_recherché']=df_contacts['Contact']
df_contacts_grouped = df_contacts.groupby('Auteur_recherché')['projet'].apply(lambda x: ', '.join(sorted(set(x)))).reset_index()

liste_contacts = df_contacts['Contact'].values

df2 = pd.read_csv("Data/RechercheDataGouv/all_datasets_rdg_2025-11-10.csv")

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
fig_dates = px.bar(df_yearly, x='Date de publication', y='Value', title='Dépôts rattachés aux contacts FaircarboN', color_discrete_sequence=['green'])
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


st.write(len(df2['Auteur_recherché'].unique()))

labels = [
    "Dépôts RDG depuis 2024",
    "Dépôts RDG",
    "Aucun dépôt RDG identifié"
]

values = [
    114,              # Dépôts depuis 2024
    126 - 114,        # Dépôts uniquement avant 2024
    474 - 114         # Aucun dépôt
]

# Couleurs personnalisées (tu peux les adapter)
custom_colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # vert, orange, rouge

# Créer le pie chart
fig_usagerdg = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    textinfo='label+percent',
    hoverinfo='label+value',
    marker=dict(colors=custom_colors, line=dict(color='#000000', width=1))
)])

fig_usagerdg.update_layout(
    title="Statistiques de dépôt RDG",
    template="plotly_white"
)

st.plotly_chart(fig_usagerdg, use_container_width=True)