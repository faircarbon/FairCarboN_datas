import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
from Publications import afficher_publications_hal
import datetime
import requests
import seaborn as sns
from deep_translator import GoogleTranslator
from stqdm import stqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON HAL DATA MINING",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

# Download necessary NLTK data files (run once)
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')

######################################################################################################################
########### FONCTIONS SUPPORTS #######################################################################################
######################################################################################################################
def intersect_lists(row):
    return list(set(row['Labo_filter2']) & set(row['Labo_']))


def filtre_labo1(row):
    try:
        return [item for item in row['Labo_all'] if row['Auteur_recherché'] in item]
    except:
        return []

# Fonction pour extraire le suffixe après le dernier '_'
def filtre_labo2(liste):
    try:
        return [item.split('_')[-1] for item in liste]
    except:
        return []
    
def extraire_doi(cellule):
    morceaux = cellule.split(';')
    for m in morceaux:
        if m.strip().startswith('10.'):
            return m.strip()
    return ""  # ou "" si tu préfères une chaîne vide

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

@st.cache_data
def acquisition_data(start_year,end_year,liste_chercheurs, liste_projet):
    liste_columns_hal = ['Nom_archive',
                         'Auteur_recherché',
                         'Projet',
                         'Ids',
                         'Titre et auteurs',
                         'Uri',
                         'Type',
                         'Type de document', 
                         'Date de publication',
                         'Collection',
                         'Collection_code',
                         'Organisme',
                         'Auteur',
                         'Labo_all',
                         'Labo_',
                         'Titre',
                         'Langue',
                         'Mots_clés',
                         'Publication_source',
                         'ANR project acronyme',
                         'ANR project titre',
                         'EU project acronyme',
                         'EU project titre',
                         'Financement']
    df_global_hal = pd.DataFrame(columns=liste_columns_hal)
    #progress = stqdm(total=len(liste_chercheurs))
    for i, s in enumerate(liste_chercheurs):
        #url_type = f'http://api.archives-ouvertes.fr/search/?q=text:"{s.lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s,labStructName_s,language_s,keyword_s,anrProjectAcronym_s,anrProjectTitle_s,europeanProjectAcronym_s,europeanProjectTitle_s,funding_s'
        url_type = f'http://api.archives-ouvertes.fr/search/?q=text:"{s.lower().strip()}"&rows=1500&wt=json&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s,labStructName_s,language_s,keyword_s,anrProjectAcronym_s,anrProjectTitle_s,europeanProjectAcronym_s,europeanProjectTitle_s,funding_s'
        df = afficher_publications_hal(url_type, s, liste_projet.iloc[i])
        dfi = pd.concat([df_global_hal,df], axis=0)
        dfi.reset_index(inplace=True)
        dfi.drop(columns='index', inplace=True)
        df_global_hal = dfi
        #progress.update(i/len(liste_chercheurs))
    df_global_hal.sort_values(by='Ids', inplace=True, ascending=False)
    df_global_hal.reset_index(inplace=True)
    df_global_hal.drop(columns='index', inplace=True)

    
    df_global_hal['Labo_filter1'] = df_global_hal.apply(filtre_labo1, axis=1)
    df_global_hal['Labo_filter2'] = df_global_hal['Labo_filter1'].apply(filtre_labo2)


    # Colonne Auteur Labo qui est la résultante
    df_global_hal['Auteur_Labo'] = df_global_hal.apply(intersect_lists, axis=1)

    # On ne garde qu'un titre
    df_global_hal['Titre_unique'] = df_global_hal['Titre'].apply(lambda row: row[0])
    # On ne garde qu'une langue
    df_global_hal['Langue_unique'] = df_global_hal['Langue'].apply(lambda row: row[0])
    df_global_hal['Labo_unique'] = df_global_hal['Auteur_Labo'].apply(lambda row: row[0] if (len(row)>0) else None)
    #df_global_hal['Mots_Clés'] = df_global_hal['Mots_Clés'].apply(lambda x: ' '.join(x))
    #df_global_hal['combined'] = df_global_hal['Titre_bis'] + ' ' + df_global_hal['Mots_Clés']
    df_global_hal['DOI sources'] = df_global_hal['Publication_source'].apply(extraire_doi)
    df_global_hal['Mots_clés_'] = df_global_hal['Mots_clés'].apply(
    lambda x: '/'.join(x) if isinstance(x, list) else '')
    df_global_hal['ANR project acronyme_'] = df_global_hal['ANR project acronyme'].apply(
    lambda x: '/'.join(x) if isinstance(x, list) else '')
    return df_global_hal

######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df = read_data("Data\FairCarboN_Datas_Contacts")
d = datetime.date.today()
start_year=2024
end_year=d.year
#st.slider(label='Choix plage de dates',min_value=2020, max_value=2025)
liste_chercheurs = df['Contact']
liste_projet = df['projet']

###############################################################################################
########### ACQUISITION DONNEES DE HAL ########################################################
###############################################################################################
st.success("Connexion établie avec HAL")
df_global_hal = acquisition_data(start_year=start_year,end_year=end_year,liste_chercheurs=liste_chercheurs, liste_projet=liste_projet)

# Tableau de l'existant dans la collection FAIRCARBON
filtered_df = df_global_hal[df_global_hal['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]

# Ajout colonne In_FairCarboN
df_global_hal['In_FairCarboN'] = df_global_hal['Titre'].isin(filtered_df['Titre'])

###########################################################################################################################################
df_inter = df_global_hal[['Nom_archive','Auteur_recherché','Ids','Uri','Titre_unique','Labo_unique','Langue_unique','DOI sources','Type de document','Date de publication','Mots_clés_','ANR project acronyme_','In_FairCarboN']].drop_duplicates()
df_inter['Mots_clés'] = df_inter['Mots_clés_'].apply(
    lambda x: x.split('/') if isinstance(x, str) and x else []
)
df_inter['ANR project acronyme'] = df_inter['ANR project acronyme_'].apply(
    lambda x: x.split('/') if isinstance(x, str) and x else []
)

df_inter['DOI sources'] = df_inter['DOI sources'].apply(lambda x: [x])
df_inter['Value']=1

df_inter.to_csv(f"Data/HAL/all_publications_hal_{d}.csv",index=False, encoding="utf-8")

st.session_state['df_hal'] = df_inter

###############################################################################################
########### VISUALISATION GENERALE ############################################################
###############################################################################################
st.title(f":grey[Etude des publications sur HAL]")
col1,col2 = st.columns(2)

with col1:
    st.metric(label="Nombre de contacts étudiés", value=len(set(liste_chercheurs)))
    st.metric(label="Nombre de dépôts HAL global", value=len(set(df_global_hal['Ids'].values)))
    st.metric(label="Nombre de dépôts HAL dans la collection FairCarboN", value=len(set(filtered_df['Ids'].values)))

with col2:
    st.metric(label="Nombre de contacts trouvés dans HAL", value=len(set(df_global_hal['Auteur_recherché'])))
    st.metric(label="Nombre d'articles global", value=len(set(df_global_hal['Ids'][df_global_hal['Type de document']=="ART"].values)))
    st.metric(label="Nombre d'articles dans la collection FairCarboN", value=len(set(filtered_df['Ids'][filtered_df['Type de document']=="ART"].values)))

# Aggregate (e.g., sum) values by year
df_yearly = df_inter.groupby('Date de publication')['Value'].sum().reset_index()

# Plot aggregated data
fig_dates = px.bar(df_yearly, x='Date de publication', y='Value', title='Dépôts rattachés aux contacts FaircarboN')
st.plotly_chart(fig_dates, use_container_width=True)

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################

projets = list(set(df_global_hal['Projet']))
auteurs = list(set(df_global_hal['Auteur_recherché']))
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
    choix_auteur = st.multiselect(label='', options=list(set(df_global_hal['Auteur_recherché'][df_global_hal['Projet'].isin(choix_p)])))
    if len(choix_auteur)==0:
        choix_a = df_global_hal['Auteur_recherché'][df_global_hal['Projet'].isin(choix_p)]
    else:
        choix_a = choix_auteur


df_global_hal_proj =df_global_hal[df_global_hal['Projet'].isin(choix_p)][df_global_hal['Auteur_recherché'].isin(choix_a)][df_global_hal['Date de publication']>=start_year]
ifc = df_global_hal_proj['Ids'][df_global_hal_proj['In_FairCarboN']==True].drop_duplicates()
In_FC = len(ifc)

col1,col2,col3 = st.columns([0.25,0.25,0.5])
with col1:
    st.metric(label=f'Nombre de dépôts dans HAL depuis {start_year}',value=len(list(set(df_global_hal_proj['Ids'][df_global_hal_proj['Date de publication']>=start_year]))))
with col2:
    st.metric(label="dans la collection FairCarboN", value=In_FC)
with col3:
    st.metric(label=f"Nombre d'auteur(e)s ayant publié depuis {start_year}", value=len(list(set(df_global_hal_proj['Auteur_recherché'][df_global_hal_proj['Date de publication']>=start_year]))))

unique_projet_titles = df_global_hal_proj[['Projet','Titre_unique']].drop_duplicates()
projects_count = unique_projet_titles['Projet'].value_counts().reset_index()
projects_count.columns = ['Projet', 'compte']

unique_person_titles = df_global_hal_proj[['Auteur_recherché','Titre_unique']].drop_duplicates()
row_counts = unique_person_titles['Auteur_recherché'].value_counts().reset_index()
row_counts.columns = ['Auteur', 'compte']

unique_labo_titles = df_global_hal_proj[['Labo_unique','Titre_unique']].drop_duplicates()
labo_count = unique_labo_titles['Labo_unique'].value_counts().reset_index()
labo_count.columns = ['Labo', 'compte']
top10_labo_count = labo_count.sort_values(by='compte', ascending=False).head(20)

df_pareto = labo_count.sort_values(by='compte', ascending=False).reset_index(drop=True)
df_pareto['cum_percentage'] = df_pareto['compte'].cumsum() / df_pareto['compte'].sum() * 100

# 3. Limit to top N labs
top_n = 20
df_pareto_plot = df_pareto.head(top_n)

unique_auteurs_titles = df_global_hal_proj[['Labo_unique','Auteur_recherché']].drop_duplicates()
labo_count2 = unique_auteurs_titles['Labo_unique'].value_counts().reset_index()
labo_count2.columns = ['Labo', 'compte']
top10_labo_count2 = labo_count2.sort_values(by='compte', ascending=False).head(20)

# Sort by count descending
df_pareto2 = labo_count2.sort_values(by='compte', ascending=False).reset_index(drop=True)
df_pareto2['cum_percentage'] = df_pareto2['compte'].cumsum() / df_pareto2['compte'].sum() * 100

# Limit to top N labs for readability
df_pareto_plot2 = df_pareto2.head(top_n)


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

fig_pareto_pub = go.Figure()

# Bar: publication count
fig_pareto_pub.add_trace(go.Bar(
    x=df_pareto_plot['Labo'],
    y=df_pareto_plot['compte'],
    name='Nombre de publications',
    marker_color="mediumturquoise",
    yaxis='y1'
))

# Line: cumulative percentage
fig_pareto_pub.add_trace(go.Scatter(
    x=df_pareto_plot['Labo'],
    y=df_pareto_plot['cum_percentage'],
    name='Pourcentage cumulé',
    yaxis='y2',
    mode='lines+markers',
    marker=dict(color='tomato'),
    line=dict(width=2)
))

# 5. Layout with dual axes
fig_pareto_pub.update_layout(
    title='Pareto des publications par labo (Top 20)',
    xaxis=dict(title='Labo'),
    yaxis=dict(title='Nombre de publications', side='left'),
    yaxis2=dict(
        title='Pourcentage cumulé (%)',
        overlaying='y',
        side='right',
        range=[0, 110]
    ),
    legend=dict(x=1.1, y=0.85),
    margin=dict(l=40, r=40, t=60, b=80),
    height=500
)

for i, row in df_pareto_plot.iterrows():
    fig_pareto_pub.add_annotation(
        x=row['Labo'],
        y=row['cum_percentage'],
        yref='y2',
        text=f"{row['cum_percentage']:.1f}%",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-20,
        font=dict(size=10, color='tomato'),
        arrowcolor='tomato',
        align='center'
    )

# Create bar + line plot (Pareto)
fig_pareto = go.Figure()

# Bar for publication counts
fig_pareto.add_trace(go.Bar(
    x=df_pareto_plot2['Labo'],
    y=df_pareto_plot2['compte'],
    name='Nombre d\'auteurs',
    marker_color='slateblue',
    yaxis='y1'
))

# Line for cumulative %
fig_pareto.add_trace(go.Scatter(
    x=df_pareto_plot2['Labo'],
    y=df_pareto_plot2['cum_percentage'],
    name='Pourcentage cumulé',
    yaxis='y2',
    mode='lines+markers',
    marker=dict(color='tomato'),
    line=dict(width=2)
))

# Layout with dual axes
fig_pareto.update_layout(
    title="Pareto des publications avec nombre de contacts par labo (Top 20)",
    xaxis=dict(title='Labo'),
    yaxis=dict(title='Nombre d\'auteurs', side='left'),
    yaxis2=dict(
        title='Pourcentage cumulé (%)',
        overlaying='y',
        side='right',
        range=[0, 110]
    ),
    legend=dict(x=1.1, y=0.85),
    margin=dict(l=40, r=40, t=60, b=80),
    height=500
)

for i, row in df_pareto_plot2.iterrows():
    fig_pareto.add_annotation(
        x=row['Labo'],
        y=row['cum_percentage'],
        yref='y2',
        text=f"{row['cum_percentage']:.1f}%",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-20,
        font=dict(size=10, color='tomato'),
        arrowcolor='tomato',
        align='center'
    )


# Affichage
col1,col2 = st.columns(2)
with col1:
    if len(choix_auteur)==0:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig1, use_container_width=True)
    
with col2:
    st.plotly_chart(fig2, use_container_width=True)

st.plotly_chart(fig_pareto_pub, use_container_width=True)
st.plotly_chart(fig_pareto, use_container_width=True)

# Exemple de requête
#Liste_chercheurs = ['Olivier Bornet']
#requete_api_hal = f'http://api.archives-ouvertes.fr/search/?q=text:"{Liste_chercheurs[0].lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s'
#reponse = requests.get(requete_api_hal, timeout=5)
#test_liste_coll=[]
#st.write(reponse.json()['response']['docs'][4])
#test_liste_coll.append(reponse.json()['response']['docs'][0]['collCode_s'])
#st.write(test_liste_coll)