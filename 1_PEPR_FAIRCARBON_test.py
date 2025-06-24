import streamlit as st
import pandas as pd
import pandas as pd
import re
import nltk
nltk.data.path.append(r'C:\Users\dutroncj\Documents\Outils\FairCarboN_datas\nltk_data')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.express as px

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="TEST",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

# Download necessary NLTK data files (run once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Sample DataFrame
data = pd.read_csv("test_csv.csv")


def preprocess_title(title):
    # Lowercase
    title = title.lower()
    # Remove punctuation and special characters
    title = re.sub(r'[^a-z\s]', '', title)
    # Tokenize
    tokens = word_tokenize(title)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    # Return cleaned title as a string (for vectorization)
    return ' '.join(tokens)

# Apply preprocessing
data['clean_title'] = data['translated'].apply(preprocess_title)

st.dataframe(data[['translated','clean_title']])


# Vectorize
vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1,2))
X = vectorizer.fit_transform(data['clean_title'])

# Try different values of k
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    sil_scores.append(score)

# Find the best k
best_k = K_range[sil_scores.index(max(sil_scores))]
st.write(f"Best number of clusters (k): {best_k}")

# Final model
final_model = KMeans(n_clusters=best_k, random_state=42)
data['cluster'] = final_model.fit_predict(X)

# Get feature names from TF-IDF
terms = vectorizer.get_feature_names_out()

# Get centroids of clusters from final KMeans model
order_centroids = final_model.cluster_centers_.argsort()[:, ::-1]

# Extract top N keywords per cluster
top_n = 5
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
    'cluster': data['cluster'],
    'clean_title': data['clean_title']
})
plot_df['cluster_label'] = plot_df['cluster'].apply(
    lambda x: f"Cluster {x}: {cluster_keywords.get(x, '')}"
)

final_score = silhouette_score(X, data['cluster'])

# Plot with Plotly
fig = px.scatter(
    plot_df,
    x='PCA1', y='PCA2',
    color='cluster_label',
    hover_data=['clean_title'],
    title=f"2D PCA of Clusters (Silhouette Score = {final_score:.2f})",
    labels={'cluster_label': 'Cluster'}
)
st.plotly_chart(fig, use_container_width=True)