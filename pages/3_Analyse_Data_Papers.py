import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import networkx as nx

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON CATALOGUE",
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

def parse_col(value):
    if not value or pd.isna(value):  # si vide ou NaN
        return []
    return [(part.split("-")[0], part.split("-")[1]) for part in value.split("/") if "-" in part]
    
######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df = read_data("Data\FairCarboN_Datas_Contacts")

df["DIB_titres_tuples"] = df["DIB_titres"].apply(
    lambda x: x.split("/") if isinstance(x, str) and x else []
)
df["ESSD_titres_tuples"] = df["ESSD_titres"].apply(
    lambda x: x.split("/") if isinstance(x, str) and x else []
)

liste_DIB_titres = df["DIB_titres_tuples"].drop_duplicates()
liste_ESSD_titres = df["ESSD_titres_tuples"].drop_duplicates()


st.title(f":grey[Etude des data papers publiés (via scraping)]")
col1, col2= st.columns(2)
with col1:
    st.metric(label='NB Articles - Data In Brief', value=len(liste_DIB_titres)) #int(df['DataInBrief'].sum())
    st.metric(label='NB Articles - Earth System Science Data', value=len(liste_ESSD_titres)) #int(df['EarthSystemScienceData'].sum())
with col2:
    st.metric(label='Nombre de contacts', value=len(df['Contact'][df['DataInBrief']>0].drop_duplicates()))
    st.metric(label='Nombre de contacts', value=len(df['Contact'][df['EarthSystemScienceData']>0].drop_duplicates()))




st.title(f":grey[Analyse des liens pour les data papers]")

# 1) On "explose" la colonne titre
df_long_DIB = df.explode("DIB_titres_tuples").dropna(subset=["DIB_titres_tuples"])
df_long_ESSD = df.explode("ESSD_titres_tuples").dropna(subset=["ESSD_titres_tuples"])
# Creation du graphe
G = nx.Graph()
G2 = nx.Graph()

# Ajout de noeuds et lignes
for _, row in df_long_DIB.iterrows():
    nom = row['Contact']
    titres = row["DIB_titres_tuples"]
    G.add_node(nom, type='contact')
    G.add_node(titres, type='titres')
    G.add_edge(titres, nom)

for _, row in df_long_ESSD.iterrows():
    nom = row['Contact']
    titres = row["ESSD_titres_tuples"]
    G2.add_node(nom, type='contact')
    G2.add_node(titres, type='titres')
    G2.add_edge(titres, nom)

# Création de la couche du graphe
pos_DIB = nx.spring_layout(G, seed=1, iterations=100)
pos_ESSD = nx.spring_layout(G2, seed=1, iterations=100)

project_x, project_y, project_text = [], [], []
unit_x, unit_y, unit_text = [], [], []
project_x2, project_y2, project_text2 = [], [], []
unit_x2, unit_y2, unit_text2 = [], [], []

for node in G.nodes():
    x, y = pos_DIB[node]
    if G.nodes[node]['type'] == 'contact':
        project_x.append(x)
        project_y.append(y)
        project_text.append(f"<b>{node}</b>")
    else:
        unit_x.append(x)
        unit_y.append(y)
        unit_text.append(node)

for node in G2.nodes():
    x, y = pos_ESSD[node]
    if G2.nodes[node]['type'] == 'contact':
        project_x2.append(x)
        project_y2.append(y)
        project_text2.append(f"<b>{node}</b>")
    else:
        unit_x2.append(x)
        unit_y2.append(y)
        unit_text2.append(node)

# Création des lignes
edge_x = []
edge_y = []
edge_x2 = []
edge_y2 = []

for edge in G.edges():
    x0, y0 = pos_DIB[edge[0]]
    x1, y1 = pos_DIB[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
for edge in G2.edges():
    x0, y0 = pos_ESSD[edge[0]]
    x1, y1 = pos_ESSD[edge[1]]
    edge_x2 += [x0, x1, None]
    edge_y2 += [y0, y1, None]


edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)
edge_trace2 = go.Scatter(
    x=edge_x2, y=edge_y2,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Préparation des Noeuds
unit_trace = go.Scatter(
    x=unit_x, y=unit_y,
    mode='markers+text',
    text=unit_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        color='gold',
        size=10,
        line_width=2
    ),
    textfont=dict(
        size=12,
        color='black'
    )
)

unit_trace2 = go.Scatter(
    x=unit_x2, y=unit_y2,
    mode='markers+text',
    text=unit_text2,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        color='gold',
        size=10,
        line_width=2
    ),
    textfont=dict(
        size=12,
        color='black'
    )
)

project_trace = go.Scatter(
    x=project_x, y=project_y,
    mode='markers+text',
    text=project_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        color='green',
        size=25,
        line_width=2
    ),
    textfont=dict(
        size=16,
        color='darkgreen'
    )
)
project_trace2 = go.Scatter(
    x=project_x2, y=project_y2,
    mode='markers+text',
    text=project_text2,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        color='green',
        size=25,
        line_width=2
    ),
    textfont=dict(
        size=16,
        color='darkgreen'
    )
)

# Préparation de la figure
fig = go.Figure(
    data=[edge_trace, unit_trace, project_trace],
    layout=go.Layout(
        width=600,
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)
fig2 = go.Figure(
    data=[edge_trace2, unit_trace2, project_trace2],
    layout=go.Layout(
        width=600,
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)

# Affichage
st.subheader(f":grey[Pour Data In Brief]")
st.plotly_chart(fig, use_container_width=True)

st.subheader(f":grey[Pour ESSD]")
st.plotly_chart(fig2, use_container_width=True)