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

def make_mask(df: pd.DataFrame):
    # Tout est NaN au départ
    mask = pd.DataFrame(0.0, index=df.index, columns=df.columns)

    # Mettre 1 si non vide
    mask[df.notna()] = 1


    regex = r"^/.*/$"
    regex2 = r"^&.*&$"
    regex2b = r"^/&.*&/$"
    regex3 = r"^µ.*µ$"
    regex4 = "done"
    #regex5 = r"^£.*£$"
    for col in df.columns:
        # accompagné
        mask.loc[df[col].astype(str).str.match(regex, na=False), col] = 0.5
        # info partielle
        mask.loc[df[col].astype(str).str.match(regex2, na=False), col] = 0.7
        # accompagné info partielle
        mask.loc[df[col].astype(str).str.match(regex2b, na=False), col] = 0.8
        # non concerné
        mask.loc[df[col].astype(str).str.match(regex3, na=False), col] = 0.2
        # analysé
        mask.loc[df[col].astype(str).str.match(regex4, na=False), col] = 0.9
        # benoit Marie
        #mask.loc[df[col].astype(str).str.match(regex5, na=False), col] = 0.6

    return mask.T  # transpose pour affichage Plotly
    
######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df = read_data("Data\PGD\PGD_structuration")
df_Contacts = read_data("Data\FairCarboN_Datas_Contacts")

st.title(":grey[Bilan quantitatif des PGD de FairCarboN]")
projects = sorted(df['projet'].unique())
Selection_projets = st.multiselect("Choix d'un ou plusieurs projets à visualiser (par défaut TOUS)",options=projects)

if len(Selection_projets)==0: #aucun choix
    df_selected = df 
else:
    df_selected = df[df['projet'].isin(Selection_projets)]


renseignements_projet = ['DMP Opidor', 'Modèle PGD', 'Financeur', 'Titre_complet', 'projet',
       'Résumé', 'Sources de financement', 'Date de début', 'Date de fin',
       'Noms_Partenaires','Acronymes_Partenaires','ID_Partenaires','Coordinateur','Affiliation_Coordinateur','ID_Coordinateur']
renseignements_plan =['Titre du plan','Date_création_plan','Date_dernière_modif', 'Numéro de livrable',
       'Version', 'Objet', 'Domaine OCDE', 'Langue_PGD', 'Responsable du plan',
       'ID_PGD', 'Licence_PGD', 'Autres_docs_associés','ID_Autres_docs_associés']

Description_PR = ['Nom PR abrégé', 'Nom PR', 'Type', 'Données_personnelles?',
       'Description succinte PR', 'WP_tache', 'Mots clés contrôlés',
       'Mots_clés', 'Langue_PR', 'personne contact_PR', 'Ethique']

Donnees_existantes = ['Justification_Existantes','Variables_Existantes','Nature_Existantes','Format_Existantes','Volume_Existantes', 'ID_Source_ données_réutilisées','Version_données réutilisées',
       'Licence_données réutilisées','Couts_euros_réutilisation']

Donnees_produites = ['Variables_Produites','Description méthode production', 'Nature_Produites','Format_Produites','Volume_Produites',
       'Equipements_production', 'Protocoles_production', 'Contact_production',
       'Couts_production']

Documentations = ['Description documentation', 'Documents',
       'Standards métadonnées', 'Liste_métadonnées', 'logiciel',
       'Contact_documentation', 'Couts_documentation', 'Procédure qualité',
       'docs_qualité','Contact_qualité', 'Aspects juridiques généraux',
       'Description méthode éthique', 'docs_éthiques']

Traitement_donnees = ['Description Traitement des données', 'Références_traitements',
       'Equipements traitements', 'Contact_traitements', 'Couts_traitements']
       
Stockage = ['Description besoin stockage', 'Volume_Stockage',
       'Equipements_stockage','Mode_accès', 'Docs_stockage', 'Description sécurité',
       'Couts_stockage']

Partage = ['Description partage', 'Potentiel réutilisation',
       'Entrepot','Date de diffusion','Format','URL_acces','Licence_Partage', 'Contact_partage'] 
       
Conservation = ['Justification conservation long terme',
       'Volume_conservation', 'Date début conservation',
       'Date fin conservation', 'Nom_Archive', 'Dispositions finales',
       'Contact_conservation', 'Couts_conservation']


df_selected["projet_index"] = df_selected["projet"].astype(str) + " // " + df_selected['Nom PR abrégé'].astype(str)
df_selected = df_selected.set_index("projet_index", drop=False)

# Définir manuellement les blocs
block1 = renseignements_projet
block2 = renseignements_plan
block3 = Description_PR
block4 = Donnees_existantes
block5 = Donnees_produites
block6 = Documentations
block7 = Traitement_donnees
block8 = Stockage
block9 = Partage
block10 = Conservation

blocks = [
    block1,block2,block3,block4,block5,block6,block7,block8,block9,block10
]

df1 = df_selected[block1]
df2 = df_selected[block2]
df3 = df_selected[block3]
df4 = df_selected[block4]
df5 = df_selected[block5]
df6 = df_selected[block6]
df7 = df_selected[block7]
df8 = df_selected[block8]
df9 = df_selected[block9]
df10 = df_selected[block10]


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
    row_heights=row_heights
)


# Bloc1 
mask1 = make_mask(df1)
fig2.add_trace(
    go.Heatmap(
        z=mask1.values,
        x=df1.index,
        y=df1.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df1.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=1, col=1
)

# Bloc2 
mask2 = make_mask(df2)
fig2.add_trace(
    go.Heatmap(
        z=mask2.values,
        x=df2.index,
        y=df2.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df2.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=2, col=1
)

# Bloc3
mask3 = make_mask(df3)
fig2.add_trace(
    go.Heatmap(
        z=mask3.values,
        x=df3.index,
        y=df3.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df3.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=3, col=1
)

# Bloc4
mask4 = make_mask(df4)
fig2.add_trace(
    go.Heatmap(
        z=mask4.values,
        x=df4.index,
        y=df4.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df4.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=4, col=1
)

# Bloc5
mask5 = make_mask(df5)
fig2.add_trace(
    go.Heatmap(
        z=mask5.values,
        x=df5.index,
        y=df5.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df5.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=5, col=1
)

# Bloc6
mask6 = make_mask(df6)
fig2.add_trace(
    go.Heatmap(
        z=mask6.values,
        x=df6.index,
        y=df6.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df6.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=6, col=1
)

# Bloc7
mask7 = make_mask(df7)
fig2.add_trace(
    go.Heatmap(
        z=mask7.values,
        x=df7.index,
        y=df7.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df7.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=7, col=1
)

# Bloc8
mask8 = make_mask(df8)
fig2.add_trace(
    go.Heatmap(
        z=mask8.values,
        x=df8.index,
        y=df8.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df8.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=8, col=1
)

# Bloc9
mask9 = make_mask(df9)
fig2.add_trace(
    go.Heatmap(
        z=mask9.values,
        x=df9.index,
        y=df9.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df9.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=9, col=1
)

# Bloc10
mask10 = make_mask(df10)
fig2.add_trace(
    go.Heatmap(
        z=mask10.values,
        x=df10.index,
        y=df10.columns,
        colorscale=[[0,'white'],[0.2,'gray'],[0.5,'sandybrown'],[0.7,'skyblue'],[0.8,'darkorange'],[0.9,'red'],[1,'royalblue']],
        showscale=False,
        text=df10.columns.T.fillna(''),
        hoverinfo='text'
    ),
    row=10, col=1
)


fig2.update_layout(height= 2000)
fig2.update_yaxes(autorange='reversed',dtick=1,row=1, col=1,title_text="Renseignement du projet",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=2, col=1,title_text="Renseignement du plan",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=3, col=1,title_text="Description du produit de R",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=4, col=1,title_text="Existantes",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=5, col=1,title_text="Produites",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=6, col=1,title_text="Documentations",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=7, col=1,title_text="Traitement des données",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=8, col=1,title_text="Stockage",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=9, col=1,title_text="Partage",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
fig2.update_yaxes(autorange='reversed',dtick=1,row=10, col=1,title_text="Conservation",title_font=dict(size=14, family="Arial", color="black"),tickfont=dict(size=12, family="Arial", color="red"))
# Modifier uniquement le X du subplot 1
fig2.update_xaxes(side='top',tickangle=90,dtick=1, row=1, col=1, tickfont=dict(size=12, family="Arial", color="brown"))

# Modifier uniquement le X du subplot 2
fig2.update_xaxes(showticklabels=False, row=2, col=1)
fig2.update_xaxes(showticklabels=False, row=3, col=1)
fig2.update_xaxes(showticklabels=False, row=4, col=1)
fig2.update_xaxes(showticklabels=False, row=5, col=1)
fig2.update_xaxes(showticklabels=False, row=6, col=1)
fig2.update_xaxes(showticklabels=False, row=7, col=1)
fig2.update_xaxes(showticklabels=False, row=8, col=1)
fig2.update_xaxes(showticklabels=False, row=9, col=1)
fig2.update_xaxes(showticklabels=False, row=10, col=1)

st.plotly_chart(fig2, use_container_width=True)