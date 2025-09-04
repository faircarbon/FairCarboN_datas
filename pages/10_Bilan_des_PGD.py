import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np
from plotly.subplots import make_subplots

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

    
######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df = read_data("Data\PGD\PGD_structuration")

projects = sorted(df['projet'].unique())
Selection_projets = st.multiselect('',options=projects)

renseignements_projet = ['DMP Opidor', 'Modèle PGD', 'Financeur', 'Titre_complet', 'projet',
       'Résumé', 'Sources de financement', 'Date de début', 'Date de fin',
       'Partenaires', 'Coordinateur']
renseignements_plan =['Titre du plan', 'Numéro de livrable',
       'Version', 'Objet', 'Domaine OCDE', 'Langue_PGD', 'Responsable du plan',
       'Identifiant_archive', 'Licence', 'autres PGD associés']

all = ['Nom PR abrégé', 'Nom PR', 'Type', 'Données_personnelles?',
       'Description succinte PR', 'WP_tache', 'Mots clés contrôlés',
       'mots_clés', 'Langue_PR', 'personne contact', 'Date de publication',
       'Identifiant', 'Type_identifiant', 'Ethique', 'Existantes?',
       'justification', 'Description des données réutilisées',
       'Couts_réutilisation', 'Nom méthode production',
       'Description méthode production', 'Nature des données',
       'Equipements_production', 'Protocoles', 'Contact_production',
       'Couts_production', 'Description documentation', 'Documents',
       'Standards métadonnées', 'langue_métadonnées', 'logiciel',
       'Contact_documentation', 'Couts_documentation', 'Procédure qualité',
       'docs_qualité', 'Contact_qualité', 'Aspects juridiques généraux',
       'docs_ juridiques', 'Contact_ juridiques',
       'Description méthode éthique', 'docs_éthiques', 'Contact_éthiques',
       'Description Traitement des données', 'Références_traitements',
       'Equipements traitements', 'Contact_traitements', 'Couts_traitements',
       'Description besoin stockage', 'volume', 'volume_unités',
       'Equipements_stockage', 'Docs_stockage', 'Description sécurité',
       'Couts_stockage', 'Description partage', 'Potentiel réutilisation',
       'Entrepot', 'Description caractéristiques du PR partagé',
       'Contact_partage', 'Justification conservation long terme',
       'Volume_conservation', 'Unités', 'Date début conservation',
       'Date fin conservation', 'Archive', 'Dispositions finales',
       'Contact_conservation', 'Couts_conservation']

st.dataframe(df)

df["projet_index"] = df["projet"].astype(str) + "-" + df.index.astype(str)  
df = df.set_index("projet_index", drop=False)

# Définir manuellement les blocs
block1 = renseignements_projet
block2 = renseignements_plan
block3 = all

blocks = [
    block1,
    block2,
    block3
]

df1 = df[block1]
df2 = df[block2]
df3 = df[block3]

# Nombre de lignes par bloc
rows_per_bloc = [len(cols) for cols in blocks]

# Calculer des proportions de hauteur (hauteur proportionnelle au nombre de lignes)
total_rows = sum(rows_per_bloc)
row_heights = [r / total_rows for r in rows_per_bloc]


# Créer une figure avec 2 "rows" pour juxtaposer verticalement
fig2 = make_subplots(
    rows=len(blocks), cols=1,
    shared_xaxes=False,  # axe X commun
    vertical_spacing=0.02,
    row_heights=row_heights,
)


# Bloc1 
mask1 = df1.notna().astype(int).T
fig2.add_trace(
    go.Heatmap(
        z=mask1.values,
        x=df1.index,
        y=df1.columns,
        colorscale=[[0,'lightgray'],[1,'red']],
        showscale=False,
        text=df1.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=1, col=1
)

# Bloc2 
mask2 = df2.notna().astype(int).T
fig2.add_trace(
    go.Heatmap(
        z=mask2.values,
        x=df2.index,
        y=df2.columns,
        colorscale=[[0,'lightgray'],[1,'green']],
        showscale=False,
        text=df2.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=2, col=1
)

# Bloc3
mask3 = df3.notna().astype(int).T
fig2.add_trace(
    go.Heatmap(
        z=mask3.values,
        x=df3.index,
        y=df3.columns,
        colorscale=[[0,'lightgray'],[1,'blue']],
        showscale=False,
        text=df3.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=3, col=1
)

fig2.update_layout(height= 1200)
fig2.update_yaxes(autorange='reversed',dtick=1,row=1, col=1)
fig2.update_yaxes(autorange='reversed',dtick=1,row=2, col=1)
fig2.update_yaxes(autorange='reversed',dtick=1,row=3, col=1)
# Modifier uniquement le X du subplot 1
fig2.update_xaxes(side='top',tickangle=90,dtick=1, row=1, col=1)

# Modifier uniquement le X du subplot 2
fig2.update_xaxes(showticklabels=False, row=2, col=1)
fig2.update_xaxes(showticklabels=False, row=3, col=1)

st.plotly_chart(fig2, use_container_width=True)