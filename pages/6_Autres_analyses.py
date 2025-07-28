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
    page_title="FAIRCARBON AUTRES ANALYSES",
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

def translate_clean(df_global_hal):
    translated = translate_list(df_global_hal['Titre_bis'].values, df_global_hal['Langue_bis'].values)
    df_global_hal['translated']=translated
    filtered_titles = []
    for title in df_global_hal['translated']:
        title = re.sub(r'[^\w\s]', '', title)
        words = word_tokenize(title)
        filtered = [word for word in words if word.lower() not in stop_words]
        filtered_ = [word for word in filtered if word.lower() not in stop_words_fr]
        filtered_titles.append(" ".join(filtered_))
    df_global_hal['filtered']=filtered_titles
    return df_global_hal

#################### DONNEES RECUPEREES #######################################################
df_hal = st.session_state['df_hal']

###############################################################################################
########### ESSAIS DE CLUSTERING ##############################################################
###############################################################################################

st.title(f":grey[Analyse par clustering]")

# Initialize tools
stop_words = set(stopwords.words('english'))
stop_words_fr = set(stopwords.words('french'))
lemmatizer = WordNetLemmatizer()

col1, col2 = st.columns(2)
with col1:
    clustering1 = st.checkbox(label='clustering_v1')
with col2:
    clustering2 = st.checkbox(label='clustering_v2')

if clustering1:
    st.subheader('Clustering TF-IDF + KMEANS')

    df_test = translate_clean(df_hal)

    # Vectorize
    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1,2))
    X = vectorizer.fit_transform(df_test['filtered'])

# Range of cluster numbers to try
    if len(df_test)<50:
        K_range = range(1, int(len(df_test)/2))
    else:
        K_range = range(1, 50)
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

    # Try different values of k
    sil_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)


    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_k, se_container_width=True)
    with col2:
        # Find the best k
        best_k = K_range[sil_scores.index(max(sil_scores))]
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write(f"Estimation du meilleur nombre de clusters (k): {best_k}")

        # Final model
        choix_k = st.number_input('choix de K', value=best_k)
    final_model = KMeans(n_clusters=choix_k, random_state=42)
    df_test['cluster'] = final_model.fit_predict(X)

    # Get feature names from TF-IDF
    terms = vectorizer.get_feature_names_out()

    # Get centroids of clusters from final KMeans model
    order_centroids = final_model.cluster_centers_.argsort()[:, ::-1]

    # Extract top N keywords per cluster
    top_n = 2
    cluster_keywords = {}

    for i in range(choix_k):
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

    df_test = translate_clean(df_final)

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
        title="Méthode du coude pour trouver le meilleur K",
        xaxis_title="Nombre de Clusters (k)",
        yaxis_title="Inertie"
    )

    # Try different values of k
    sil_scores = []
    K_range = range(2, 10)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        sil_scores.append(score)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_k2, se_container_width=True)
    with col2:
        best_k = K_range[sil_scores.index(max(sil_scores))]
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write(f"Estimation du meilleur nombre de clusters (k): {best_k}")

        # Final model
        choix_k = st.number_input('choix de K', value=best_k)

    # --- 4. Choose k and Cluster ---
    kmeans2 = KMeans(n_clusters=choix_k, random_state=42)
    df_test['cluster'] = kmeans2.fit_predict(embeddings)

    # --- 5. 2D Plot with PCA or UMAP ---
    reduced = PCA(n_components=2).fit_transform(embeddings)

    #svd = TruncatedSVD(n_components=3)
    #reduced = svd.fit_transform(embeddings)
    df_test['pca_x'] = reduced[:, 0]
    df_test['pca_y'] = reduced[:, 1]
    #df_test['pca_z'] = reduced[:, 2]

    try:
        score2 = silhouette_score(embeddings, df_test['cluster'])
    except:
        score2 = 0

    fig_clustering2 = px.scatter(
                                    df_test,
                                    x='pca_x',
                                    y='pca_y',
                                    #z='pca_z',
                                    color=df_test['cluster'].astype(str),
                                    hover_data=['filtered'],
                                    title=f"Clusters (embeddings) / Silhouette Score: {score2:.3f}",
                                    labels={'color': 'Cluster'},
                                    #color_discrete_sequence=px.colors.qualitative.Dark2
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