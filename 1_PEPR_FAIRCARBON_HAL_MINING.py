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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import numpy as np


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

######################################################################################################################
########### CHOIX VISUELS ############################################################################################
######################################################################################################################
# taille et couleurs des sous-titres
couleur_subtitles = (250,150,150)
taille_subtitles = "25px"
couleur_subsubtitles = (60,150,160)
taille_subsubtitles = "25px"

###############################################################################################
########### TRANSFORMATION FICHIER XLS ########################################################
###############################################################################################
@st.cache_data
def read_data():
    # Chemin vers le fichier Excel
    fichier_excel = "Data\FairCarboN_RNSR_copie.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(fichier_excel, sheet_name=1,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv("Data\FairCarboN_RNSR_copie.csv", index=False, encoding="utf-8")

    ######## NETTOYAGES EVENTUELS ######################

    # filtrer les lignes incomplètes
    df_filtré = df.dropna(subset=["Latitude", "Longitude","Acronyme projet","Acronyme unité"])
    # Renommer les colonnes
    df_filtré_renommé = df_filtré.rename(columns={
        "Acronyme projet": "projet",
        "Acronyme unité": "laboratoire"
    })
    df_filtré_renommé.to_csv("Data\FairCarboN_RNSR_copie_filtré_renommé.csv", index=False)

    return df_filtré_renommé

# Charger les données
df = read_data()

###############################################################################################
########### REQUETES HAL ######################################################################
###############################################################################################
start_year=2023
end_year=2025
st.title(f":grey[Etude des publications sur HAL de {start_year} à {end_year}]")

df_research = df[['projet','laboratoire']][df['Type_Data']=='Contact']
df_research.reset_index(inplace=True)
df_research.drop(columns='index', inplace=True)

liste_chercheurs = df_research['laboratoire']
liste_projet = df_research['projet']
#Liste_chercheurs = ['Claire Chenu']
#requete_api_hal = f'http://api.archives-ouvertes.fr/search/?q=text:"{Liste_chercheurs[0].lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s'
#reponse = requests.get(requete_api_hal, timeout=5)
#print(reponse.json()['response']['docs'][-1])

@st.cache_data
def acquisition_data(start_year,end_year,liste_chercheurs, liste_projet):
    liste_columns_hal = ['Store','Auteur_recherché','Projet','Ids','Titre et auteurs','Uri','Type','Type de document', 'Date de production','Collection','Collection_code','Auteur_organisme','Auteur','Labo_all','Labo_','Titre','Langue','Mots_Clés']
    df_global_hal = pd.DataFrame(columns=liste_columns_hal)
    #progress = stqdm(total=len(liste_chercheurs))
    for i, s in enumerate(liste_chercheurs):
        url_type = f'http://api.archives-ouvertes.fr/search/?q=text:"{s.lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s,labStructName_s,language_s,keyword_s'
        df = afficher_publications_hal(url_type, s, liste_projet.iloc[i])
        dfi = pd.concat([df_global_hal,df], axis=0)
        dfi.reset_index(inplace=True)
        dfi.drop(columns='index', inplace=True)
        df_global_hal = dfi
        #progress.update(i/len(liste_chercheurs))
    df_global_hal.sort_values(by='Ids', inplace=True, ascending=False)
    df_global_hal.reset_index(inplace=True)
    df_global_hal.drop(columns='index', inplace=True)

    df_global_hal['Labo_filter1'] = df_global_hal['Labo_all']
    df_global_hal['Labo_filter2'] = df_global_hal['Labo_all']

    for i in range(len(df_global_hal)):
        try:
            df_global_hal['Labo_filter1'].loc[i] = [item for item in df_global_hal['Labo_all'].loc[i] if df_global_hal['Auteur_recherché'].loc[i] in item]
        except:
            pass
        try:
            df_global_hal['Labo_filter2'].loc[i] = [item.split('_')[-1] for item in df_global_hal['Labo_filter1'].loc[i]]
        except:
            pass

    df_global_hal['Auteur_Labo'] = df_global_hal.apply(intersect_lists, axis=1)
    df_global_hal['Titre_bis'] = df_global_hal['Titre'].apply(lambda row: row[0])
    df_global_hal['Langue_bis'] = df_global_hal['Langue'].apply(lambda row: row[0])

    return df_global_hal

def intersect_lists(row):
    return list(set(row['Labo_filter2']) & set(row['Labo_']))

# Translate French titles to English
def translate_list(titles, languages):
    translated = []
    for title, lang in zip(titles, languages):
        if lang == 'fr':
            translated.append(GoogleTranslator(source='fr', target='en').translate(title))
        else:
            translated.append(title)
    return translated


df_global_hal = acquisition_data(start_year=start_year,end_year=end_year,liste_chercheurs=liste_chercheurs, liste_projet=liste_projet)

filtered_df = df_global_hal[df_global_hal['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]

col1,col2 = st.columns(2)

with col1:
    st.metric(label="Nombre de contacts étudiés", value=len(set(liste_chercheurs)))
    st.metric(label="Nombre de dépôts HAL global", value=len(set(df_global_hal['Titre_bis'].values)))
    st.metric(label="Nombre de dépôts HAL dans la collection FairCarboN", value=len(set(filtered_df['Titre_bis'].values)))

with col2:
    st.metric(label="Nombre de contacts trouvés dans HAL", value=len(set(df_global_hal['Auteur_recherché'])))
    st.metric(label="Nombre d'articles global", value=len(set(df_global_hal['Titre_bis'][df_global_hal['Type de document']=="ART"].values)))
    st.metric(label="Nombre d'artciles dans la collection FairCarboN", value=len(set(filtered_df['Titre_bis'][filtered_df['Type de document']=="ART"].values)))

df_global_hal['In_FairCarboN'] = df_global_hal['Titre'].isin(filtered_df['Titre'])

#st.dataframe(df_global_hal[['Auteur_recherché','Projet','Type de document','Date de production','Titre','Langue','In_FairCarboN','Auteur_Labo','Mots_Clés']])

projets = list(set(df_global_hal['Projet']))
auteurs = list(set(df_global_hal['Auteur_recherché']))
col1,col2 = st.columns(2)
with col1:
    choix_projet = st.multiselect(label='Projets', options=projets )
    if len(choix_projet)==0:
        choix_p = projets
    else:
        choix_p = choix_projet
with col2:
    choix_auteur = st.multiselect(label='Auteur(e)s', options=list(set(df_global_hal['Auteur_recherché'][df_global_hal['Projet'].isin(choix_p)])))
    if len(choix_auteur)==0:
        choix_a = df_global_hal['Auteur_recherché'][df_global_hal['Projet'].isin(choix_p)]
    else:
        choix_a = choix_auteur

df_global_hal_proj =df_global_hal[df_global_hal['Projet'].isin(choix_p)][df_global_hal['Auteur_recherché'].isin(choix_a)]

col1,col2 = st.columns(2)
with col1:
    st.metric(label='Nombre de dépôts dans HAL',value=len(list(set(df_global_hal_proj['Titre_bis']))))
with col2:
    st.metric(label="Nombre d'auteur(e)s", value=len(list(set(df_global_hal_proj['Auteur_recherché']))))

# Nombre de ligne par auteur
unique_person_titles = df_global_hal_proj[['Auteur_recherché','Titre_bis']].drop_duplicates()
row_counts = unique_person_titles['Auteur_recherché'].value_counts().reset_index()
row_counts.columns = ['Titre', 'compte']

# Box plot using Plotly
fig2 = px.box(row_counts, y='compte', points="all",hover_data=['Titre'], title="Distribution du nombre de publications")

unique_projet_titles = df_global_hal_proj[['Projet','Titre_bis']].drop_duplicates()
projects_count = unique_projet_titles['Projet'].value_counts().reset_index()
projects_count.columns = ['Projet', 'compte']

fig = px.pie(
    projects_count,
    names='Projet',
    values='compte',
    title='Répartition des publications par projet',
    hole=0.3  
)

fig1 = px.pie(
    projects_count,
    names='Projet',
    values='compte',
    title='Participation aux projets',
    hole=0.3
)
fig1.update_traces(textinfo='label')
fig1.update_layout(showlegend=False)

# Affichage
col1,col2 = st.columns(2)
with col1:
    if len(choix_auteur)==0:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.plotly_chart(fig2, use_container_width=True)


df_final = df_global_hal_proj[['Auteur_recherché','Projet','Type de document','Date de production','Titre_bis','Langue_bis','In_FairCarboN']].drop_duplicates()
df_final_english = df_final[df_final['Langue_bis']=='en']

df_test = df_final_english.copy()
#df_test = df_final_english[df_final_english['Projet']=='SLAM-B']

#st.dataframe(df_test)


st.subheader('Prototype Clustering')

# Step 1: Vectorize titles using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_test['Titre_bis'])

# Step 2: Apply KMeans clustering
num_clusters = st.slider(label='Nombre de clusters', value=10, max_value=20)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_test['cluster'] = kmeans.fit_predict(X)

# Optional: Visualize using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(X.toarray())
df_test['pca_x'] = reduced[:, 0]
df_test['pca_y'] = reduced[:, 1]

score = silhouette_score(X, df_test['cluster'])

# Get the cluster centers and feature names
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

fig_clustering = px.scatter(
                                df_test,
                                x='pca_x',
                                y='pca_y',
                                color=df_test['cluster'].astype(str),
                                hover_data=['Titre_bis'],
                                title=f"Book Title Clusters (TF-IDF + KMeans) / Silhouette Score: {score:.3f}",
                                labels={'color': 'Cluster'}
                            )
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_clustering, se_container_width=True)
with col2:
    for i in range(num_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :5]]
        st.write(f"\nCluster {i}: ", ", ".join(top_terms))


st.subheader('Finding K')
# Range of cluster numbers to try
K_range = range(1, 30)
inertias = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

fig_k = go.Figure()
fig_k.add_trace(go.Scatter(
        x=list(K_range),
        y=inertias,
        mode='lines+markers',
        marker=dict(size=10),
        name='Inertia'
    ))

fig_k.update_layout(
        title="Elbow Method for Optimal Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
    )

st.plotly_chart(fig_k, se_container_width=True)