import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np
from plotly.subplots import make_subplots
import random

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
        plot_bgcolor='snow',
        paper_bgcolor='snow'
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("⚠️ Le nombre de colonnes sélectionnées doit correspondre au nombre de noeuds choisi.")