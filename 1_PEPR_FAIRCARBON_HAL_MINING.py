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

def intersect_lists(row):
    return list(set(row['Labo_filter2']) & set(row['Labo_']))

# Translate French titles to English
def translate_list(titles, languages):
    translated = []
    for title, lang in zip(titles, languages):
        if lang == 'fr':
            try:
                translated.append(GoogleTranslator(source='fr', target='en').translate(title))
            except:
                translated.append(title)
        else:
            translated.append(title)
    return translated

# Download necessary NLTK data files (run once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

@st.cache_data
def acquisition_data(start_year,end_year,liste_chercheurs, liste_projet, stop_words):
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
    #df_global_hal['Mots_Clés'] = df_global_hal['Mots_Clés'].apply(lambda x: ' '.join(x))
    #df_global_hal['combined'] = df_global_hal['Titre_bis'] + ' ' + df_global_hal['Mots_Clés']
    translated = translate_list(df_global_hal['Titre_bis'].values, df_global_hal['Langue_bis'].values)
    df_global_hal['translated']=translated
    filtered_titles = []
    for title in df_global_hal['translated']:
        words = word_tokenize(title)
        filtered = [word for word in words if word.lower() not in stop_words]
        filtered_titles.append(" ".join(filtered))
    df_global_hal['filtered']=filtered_titles
    
    return df_global_hal


df_global_hal = acquisition_data(start_year=start_year,end_year=end_year,liste_chercheurs=liste_chercheurs, liste_projet=liste_projet, stop_words= stop_words)

df_global_hal.to_csv("test_csv.csv",index=False, encoding="utf-8")

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


df_inter = df_global_hal_proj[['Auteur_recherché','Projet','Type de document','Date de production','filtered','In_FairCarboN']].drop_duplicates()

df_final = df_inter[['Projet','Type de document','Date de production','filtered','In_FairCarboN']].drop_duplicates()
df_test = df_final.copy()
df_test.reset_index(inplace=True)
df_test.drop(columns='index', inplace=True)
#df_test = df_final_english[df_final_english['Projet']=='SLAM-B']

st.dataframe(df_test['filtered'])

clustering1 = st.checkbox(label='clustering_v1')

clustering2 = st.checkbox(label='clustering_v2')

if clustering1:
    st.subheader('Clustering TF-IDF + KMEANS')

    # Vectorize
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1,2))
    X = vectorizer.fit_transform(df_test['filtered'])

# Range of cluster numbers to try
    K_range = range(1, 15)
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
            name='Inertie'
        ))

    fig_k.update_layout(
            title="Méthode du coude pour trouver le meilleur K",
            xaxis_title="Nombre de Clusters (k)",
            yaxis_title="Inertie (Within-Cluster Sum of Squares)",
        )

    st.plotly_chart(fig_k, se_container_width=True)

    # Try different values of k
    sil_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)

    # Find the best k
    best_k = K_range[sil_scores.index(max(sil_scores))]
    st.write(f"Estimation du meilleur nombre de clusters (k): {best_k}")

    # Final model
    final_model = KMeans(n_clusters=best_k, random_state=42)
    df_test['cluster'] = final_model.fit_predict(X)

    # Get feature names from TF-IDF
    terms = vectorizer.get_feature_names_out()

    # Get centroids of clusters from final KMeans model
    order_centroids = final_model.cluster_centers_.argsort()[:, ::-1]

    # Extract top N keywords per cluster
    top_n = 1
    cluster_keywords = {}

    for i in range(best_k):
        top_terms = [terms[ind] for ind in order_centroids[i, :top_n]]
        cluster_keywords[i] = ", ".join(top_terms)

    # Reduce to 2D
    pca = PCA(n_components=2)
    X_2D = pca.fit_transform(X.toarray())

    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PCA1': X_2D[:, 0],
        'PCA2': X_2D[:, 1],
        'cluster': df_test['cluster'],
        'Projet': df_test['Projet'],
        'clean_title': df_test['filtered']
    })
    plot_df['cluster_label'] = plot_df['cluster'].apply(
        lambda x: f"{cluster_keywords.get(x, '')}"
    )

    final_score = silhouette_score(X, df_test['cluster'])

    # Plot with Plotly
    fig_clustering = px.scatter(
        plot_df,
        x='PCA1', y='PCA2',
        color='cluster_label',
        hover_data=['clean_title'],
        title=f"Clusters (Silhouette Score = {final_score:.2f})",
        labels={'cluster_label': 'Cluster'}
    )
    # Plot with Plotly
    fig_clustering_proj = px.scatter(
        plot_df,
        x='PCA1', y='PCA2',
        color='Projet',
        hover_data=['clean_title'],
        title=f"Clusters",
        labels={'cluster_label': 'Cluster'}
    )


    col1, col2 = st.columns([0.6,0.4])
    with col1:
        st.plotly_chart(fig_clustering, use_container_width=True)
    with col2:
            st.plotly_chart(fig_clustering_proj, use_container_width=True)


elif clustering2:
    st.subheader('Clusters avec embeddings')

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small & fast model
    embeddings = model.encode(df_test['filtered'], show_progress_bar=False)
    embeddings = normalize(embeddings)

    # --- 3. Elbow Method ---
    inertias = []
    K_range = range(1, 20)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)

    fig_k2 = go.Figure()
    fig_k2.add_trace(go.Scatter(
        x=list(K_range),
        y=inertias,
        mode='lines+markers',
        marker=dict(size=10),
        name='Inertia'
    ))
    fig_k2.update_layout(
        title="Elbow Method with Sentence Embeddings",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia"
    )
    st.plotly_chart(fig_k2, se_container_width=True)

    # --- 4. Choose k and Cluster ---
    num_clusters2 = st.slider(label='Nombre de clusters', value=10, max_value=20)
    kmeans2 = KMeans(n_clusters=num_clusters2, random_state=42)
    df_test['cluster'] = kmeans2.fit_predict(embeddings)

    # --- 5. 2D Plot with PCA or UMAP ---
    reduced = PCA(n_components=2).fit_transform(embeddings)

    #svd = TruncatedSVD(n_components=3)
    #reduced = svd.fit_transform(embeddings)
    df_test['pca_x'] = reduced[:, 0]
    df_test['pca_y'] = reduced[:, 1]
    #df_test['pca_z'] = reduced[:, 2]

    score2 = silhouette_score(embeddings, df_test['cluster'])

    fig_clustering2 = px.scatter(
                                    df_test,
                                    x='pca_x',
                                    y='pca_y',
                                    #z='pca_z',
                                    color='cluster',
                                    hover_data=['filtered'],
                                    title=f"Clusters (embeddings) / Silhouette Score: {score2:.3f}",
                                    labels={'color': 'Cluster'},
                                    color_discrete_sequence=px.colors.qualitative.Dark2
                                )
    
    fig_clustering_proj2 = px.scatter(
                                    df_test,
                                    x='pca_x',
                                    y='pca_y',
                                    #z='pca_z',
                                    color='Projet',
                                    hover_data=['filtered'],
                                    title=f"Clusters (embeddings)",
                                    labels={'color': 'Cluster'},
                                    color_discrete_sequence=px.colors.qualitative.Dark2
                                )
    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_clustering2, use_container_width=True)
    with col2:
        st.plotly_chart(fig_clustering_proj2, use_container_width=True)

else:
    st.write("")

