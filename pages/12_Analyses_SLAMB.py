import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
import datetime
import requests
import random
from deep_translator import GoogleTranslator
#from stqdm import stqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
#from sklearn.decomposition import TruncatedSVD
#from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics import silhouette_score
#from sentence_transformers import SentenceTransformer
import numpy as np
#from sklearn.preprocessing import normalize
#from sklearn.cluster import DBSCAN
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON SLAMB",
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

@st.cache_data
def translate_clean(df_global_hal):
    translated = translate_list(df_global_hal['Titre_unique'].values, df_global_hal['Langue_unique'].values)
    df_global_hal['translated']=translated
    filtered_titles = []
    i = 0
    for title in df_global_hal['translated']:
        title = re.sub(r'[^\w\s]', '', title)
        words = word_tokenize(title)
        filtered = [word for word in words if word.lower() not in stop_words]
        filtered_ = [word for word in filtered if word.lower() not in stop_words_fr]
        filtered_titles.append(" ".join(filtered_))
        i += 1
        print(i)
    df_global_hal['filtered']=filtered_titles
    return df_global_hal


######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
file1 = "Data\Questionnaires\Questionnaires_SLAM_B"

df = read_data(file1)
df["NOM_PRENOM"] = df["NOM"]+ "_" +df["PRENOM"]
# Colonne pour les valeurs (poids des liens)
value_column = "VOLUMETRIE_MAX"
if value_column not in df.columns:
    df[value_column] = 0.1  # valeur par défaut


df_hal = st.session_state['df_hal']

col1, col2 = st.columns(2)
with col1:
    diagram = st.checkbox("Diagramme")
with col2:
    analyse_semantique = st.checkbox("Analyse sémantique")

if diagram:
    # -------------------------------
    # Choix du nombre de noeuds
    # -------------------------------
    nombre_nodes = st.number_input("Nombre de noeuds", min_value=2)

    colonnes_disponibles = list(df.columns)
    node_columns = st.multiselect(
        "Choisis les colonnes pour les noeuds (dans l'ordre)",
        options=colonnes_disponibles,
        #default=colonnes_disponibles[:nombre_nodes]
    )
    if len(node_columns) == nombre_nodes:
        # -------------------------------
        # Construction des labels
        # -------------------------------
        all_labels = list(pd.concat([df[col] for col in node_columns]).unique())
        label_to_index = {label: i for i, label in enumerate(all_labels)}

        sources, targets, values, link_colors = [], [], [], []

        # -------------------------------
        # Couleurs par contributeur initial
        # -------------------------------
        unique_contributors = df[node_columns[0]].unique()

        def generate_colors(n=30):
            colors = []
            for _ in range(n):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                colors.append(f'#{r:02x}{g:02x}{b:02x}')
            return colors

        colors = generate_colors(len(unique_contributors))
        contributor_to_color = {contrib: colors[i] for i, contrib in enumerate(unique_contributors)}

        # -------------------------------
        # Création des liens entre colonnes successives
        # -------------------------------
        for i in range(len(node_columns) - 1):
            src_col = node_columns[i]
            tgt_col = node_columns[i + 1]
            for src, tgt, val, contrib in zip(df[src_col], df[tgt_col], df["VOLUMETRIE_MAX"], df[node_columns[0]]):
                sources.append(label_to_index[src])
                targets.append(label_to_index[tgt])
                values.append(val)
                link_colors.append(contributor_to_color[contrib])  # couleur du chemin

        # -------------------------------
        # Couleurs des noeuds par colonne
        # -------------------------------
        palette_nodes = ["#FFD700", "#87CEEB", "#90EE90", "#FFA07A", "#FF69B4", "#D3D3D3"]
        colors_for_nodes = []
        for i, col in enumerate(node_columns):
            colors_for_nodes.extend([palette_nodes[i % len(palette_nodes)]] * len(df[col].unique()))

        # -------------------------------
        # Sankey diagram
        # -------------------------------
        fig = go.Figure(go.Sankey(
            arrangement='freeform',
            node=dict(
                pad=30,
                thickness=20,
                line=dict(color="grey", width=1),
                label=all_labels,
                color=colors_for_nodes
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors   # chaque chemin garde sa couleur
            )
        ))

        # -------------------------------
        # Annotations automatiques
        # -------------------------------
        for i, col in enumerate(node_columns):
            fig.add_annotation(
                dict(
                    font=dict(color="black", size=16),
                    x=i / (len(node_columns) - 1),
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    text=f"<b>{col}</b>"
                )
            )

        fig.update_layout(
            height=1500,  # hauteur augmentée
            hovermode='x',
            #title=dict(text="<b>Diagramme Sankey dynamique</b>", font=dict(size=18), x=0.5),
            font=dict(size=18, color='white'),
            plot_bgcolor='black',
            paper_bgcolor='black'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Le nombre de colonnes sélectionnées doit correspondre au nombre de noeuds choisi.")
elif analyse_semantique:
    df_hal_ = df_hal[df_hal["Projet"]=="SLAM-B"][df_hal["In_FairCarboN"]].drop_duplicates(subset=['Titre_unique'])
    df_hal_.reset_index(inplace=True)
    df_hal_.drop(columns='index', inplace=True)
    
    # Initialize tools
    stop_words = set(stopwords.words('english'))
    stop_words_fr = set(stopwords.words('french'))
    lemmatizer = WordNetLemmatizer()

    df_test = translate_clean(df_hal_)
    df_test.reset_index(inplace=True)
    df_test.drop(columns='index', inplace=True)
    st.dataframe(df_test)


    # 1. Transformer les textes en vecteurs TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")  # stopwords français
    X = vectorizer.fit_transform(df_test["filtered"])

    # 2. Clustering avec KMeans
    n_clusters = 3  # nombre de clusters à définir
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_test["cluster"] = kmeans.fit_predict(X)

    # 3. Réduction de dimension avec t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    X_embedded = tsne.fit_transform(X.toarray())

    df_test["x"] = X_embedded[:,0]
    df_test["y"] = X_embedded[:,1]

    # 4. Visualisation interactive avec Plotly
    fig = px.scatter(
        df_test,
        x="x",
        y="y",
        color="cluster",
        text="filtered",  # afficher le titre en survol
        title="Visualisation 2D des clusters de titres avec t-SNE"
    )

    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Appuyer sur le bouton de votre choix")