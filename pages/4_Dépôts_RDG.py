import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import re
from datetime import datetime
import requests


pio.templates.default = "plotly"

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON DATA",
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
    # Chemin vers le fichier Excel
    #fichier_excel = "Data\FairCarboN_Datas_V2.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=0,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")
    return df

def extraire_date(f):
    match = regex.search(f)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    return datetime.min

def extraire_nom(contact):
    if pd.isna(contact):
        return None

    noms = []

    for personne in contact.split('|'):
        personne = personne.strip()

        # Si format "Nom, Prénom"
        if ',' in personne:
            premier_terme = personne.split(',')[0].strip().lower()
        else:
            # Sinon on prend le premier mot
            premier_terme = personne.split()[0].strip().lower()

        noms.append(premier_terme)

    return " | ".join(noms)

def trouver_projet(noms):
    if pd.isna(noms):
        return "projet inconnu"

    # Séparer les noms (déjà en minuscules normalement)
    liste_noms = [n.strip().lower() for n in noms.split('|')]

    # Tester chaque nom dans l'ordre
    for nom in liste_noms:
        if nom in dico_projets:
            return dico_projets[nom]

    return "projet inconnu"

######################################################################################################################
########### PARAMETRES ###############################################################################################
######################################################################################################################
couleurs = {"ALAMOD":"#fa0404",
              "SLAM-B":"#e7b204",
              "RIFT":"#05fc6c",
              "CrosyeN":"#b9fc01",
              "CarboNium":"#ec8129",
              "CABESTAN":"#03fabc",
              "CANETE":"#067ff0",
              "DEEP-C":"#6d03f8",
              "Drought for C":"#cb05fc",
              "PEACE":"#f0047a",
              "TROPECOS":"#090080",
              "CLIM-FAS":"#ca7ff8",
              "CO2_CMPhi":"#793305",
              "GREENSCALE":"#035718",
              "PREFALIM":"#B83E0E",
              "RhizoSeqC":"#636303",
              "Gouvernance":"#0A0A0A",
              "FairCarboN":"#0A0A0A",
              "ART": "#020242",
              "UNDEFINED": "#0B9BEE",
              "COM": "#16DA6E",
              "POSTER":"#8de42a",
              "MEM":"#eedf14",
              "REPORT" : "#eba21b",
              "OTHER": "#f5760f",
              "LECTURE": "#ff0909"}


# Variables Python
couleur_h1 = "#748114"
taille_h1 = "60px"
police_h1 = "Cascadia Code"

couleur_h2 = "#FF6347"
taille_h2 = "32px"
police_h2 = "Cascadia Code"

couleur_h3 = "#1B657C"
taille_h3 = "35px"
police_h3 = "Cascadia Code"

taille_metrique = "50px"
couleur_metrique = "#0C495C"

couleur_graphes = "#0C495C"

# Injection CSS
st.markdown(f"""
<style>
h1 {{
    color: {couleur_h1}!important;
    font-size: {taille_h1}!important;
    font-family: {police_h1}!important;
    text-align: center;
}}

h2 {{
    color: {couleur_h2} !important;
    font-size: {taille_h2} !important;
    font-family: {police_h2} !important;
    text-align: center;
}}

h3 {{
    color: {couleur_h3} !important;
    font-size: {taille_h3} !important;
    font-family: {police_h3} !important;
    font-weight: bold;
    text-align: center;
}}

[data-testid="stMetricValue"] {{
    font-size: {taille_metrique} !important;
    color: {couleur_metrique} !important;
    text-align: center;
}}
</style>
""", unsafe_allow_html=True)


######################################################################################################################
########### DONNEES ##################################################################################################
######################################################################################################################
dossier = "Data/RechercheDataGouv/"
pattern = dossier + "all_datasets_rdg_multi_*.csv"
fichiers = glob.glob(pattern)

# Expression régulière pour extraire la date
regex = re.compile(r"all_datasets_rdg_multi_(\d{4}-\d{2}-\d{2}).csv")

# Trier par date décroissante
fichiers_tries = sorted(fichiers, key=extraire_date, reverse=True)
fichiers_sans_prefixe = [f[22:] for f in fichiers_tries]

derniere_date = fichiers_sans_prefixe[0] if fichiers_tries else None
dernier_fichier = dossier + derniere_date

data = pd.read_csv(dernier_fichier)
data_sans_doublon = data.drop_duplicates(subset='url')
data_sans_doublon["published_at"] = pd.to_datetime(data_sans_doublon["published_at"])
data_sans_doublon["Annee"] = data_sans_doublon["published_at"].dt.year
filtered = data_sans_doublon[data_sans_doublon["Annee"]>2024].reset_index()
#data_sans_doublon["Date de publication"]=pd.to_datetime(data_sans_doublon["Date de publication"],format='%Y')

dico_projets = {"alamod":"ALAMOD",
                "slam-b":"SLAM-B",
                "crosyen":"CrosyeN",
                'rift':"RIFT",
                'carbonium':"CarboNium",
                'canete': "CANETE",
                'peace':"PEACE",
                'clim-fas':"CLIM-FAS",
                'prefalim':"PREFALIM",
                'rhizoseqc':"RhizoSeqC",
                'greenscale':"GREENSCALE",
                'co2_cmphi':"CO2_CMPhi",
                'drought_forc':"Drought for C",
                'tropecos':"TROPECOS",
                'deep-c':"DEEP-C",
                'cabestan':"CABESTAN",
                'faircarbon':"FairCarboN"}

filtered["Projet"]=filtered["Projet"].replace(dico_projets)

###############################################################################################################
############# AFFICHAGE #################################################################################
###############################################################################################################

st.title(f"FairCarboN => RechercheDataGouv {derniere_date[24:-4]}")

col1 , col2 = st.columns(2)
with col1:
    st.subheader("All deposits")
    st.metric(value=len(filtered), label="", label_visibility="hidden", border=True)
with col2:
    st.subheader("Nb of files uploaded")
    st.metric(value=sum(filtered["nb_fichiers"]), label="", label_visibility="hidden", border=True)

projects = sorted(filtered['Projet'].unique())
col1, col2, col3 = st.columns(3)
with col1:
    st.header("Choisir le projet | choose project")
with col2:
    Selection_projets = st.multiselect("",options=projects)
with col3:
    st.markdown("")
    st.markdown("")
    st.markdown(":red[par défaut tous | all]")

if len(Selection_projets)==0: #aucun choix
    df_selected = filtered
else:
    df_selected = filtered[filtered['Projet'].isin(Selection_projets)]

###############################################################################################################
###############################################################################################################
###############################################################################################################

#df = read_data("Data/FairCarboN_Datas_Contacts2")
# Ajout de la colonne "Nom"
#df["Nom"] = df["Contact"].str.split().str[-1]
#dico_projets = dict(zip(df["Nom"].str.lower(), df["projet"]))

# Application au dataframe
#filtered["Nom"] = filtered["authors"].apply(extraire_nom)
#filtered["Projet"] = filtered["Nom"].apply(trouver_projet)

###############################################################################################################

df_selected = df_selected.sort_values("published_at")


# Compter les documents par jour et par type
counts = df_selected.groupby(['published_at','Projet','title']).size().reset_index(name="nb_docs")

# Calcul du cumul par type
counts["Cumul"] = counts.groupby('Projet')["nb_docs"].cumsum()
counts["index_plus_1"] = counts.index + 1

fig = px.scatter(
    counts,
    x='published_at',
    y="Cumul",
    color='Projet',
    color_discrete_map=couleurs,
    text="index_plus_1"
)

fig.update_traces(
    marker=dict(size=10),
    textposition="top center",   # <<< position du texte
    textfont=dict(size=14, color="black")
)  # taille des points

fig.update_layout(
    legend=dict(
        title=dict(
            text="Projets",
            font=dict(size=22, color=couleur_h3)),
        orientation="h",
        y=-0.25,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=20)
    ),
    margin=dict(b=200),
    height=600,
    xaxis_title=dict(
        text="Date",
        font=dict(size=22, color=couleur_graphes)
    ),
    yaxis_title=dict(
        text="Nombre cumulé | Cumulative number",
        font=dict(size=22, color=couleur_graphes)
    )
)

fig.update_xaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    tickfont=dict(size=16, color=couleur_graphes),
    gridcolor="lightgray"
)

fig.update_yaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    tickfont=dict(size=16, color=couleur_graphes),
    gridcolor="lightgray"
)

import plotly.graph_objects as go

projets_visibles = df_selected["Projet"].unique()
# Colonne index +1
indices = [i+1 for i in range(len(df_selected))]

# Couleurs associées à chaque ligne
cell_colors = [[couleurs[p] for p in df_selected["Projet"]],  # colonne index
               ["white"] * len(df_selected)]                 # colonne titre

# Création de la table
fig_table = go.Figure(data=[go.Table(
    columnwidth=[20, 400],
    header=dict(
        values=["", ""],
        fill_color="white",
        align="center",
        font=dict(color="black", size=25)
    ),
    cells=dict(
        values=[indices, df_selected["title"]],
        fill_color=cell_colors,
        align="left",
        font=dict(color="black", size=16)
    )
)])

# Ajout de la légende sous la table
annotations = []
y_start = -0.15
step = 0.07

for i, (projet, color) in enumerate(couleurs.items()):
    annotations.append(dict(
        x=0.02,
        y=y_start - i * step,
        xanchor="left",
        yanchor="middle",
        text=f"<b>{projet}</b>",
        font=dict(color=color, size=14),
        showarrow=False
    ))

# --- LÉGENDE (uniquement projets visibles) ---
for projet in projets_visibles:
    fig_table.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=10, color=couleurs[projet]),
        name=projet,
        showlegend=True
    ))

# --- STYLE DE LA LÉGENDE ---
fig_table.update_layout(
    legend=dict(
        title=dict(
            text="Projets",
            font=dict(size=22, color=couleur_h3)
        ),
        orientation="h",
        y=-0.25,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=20)
    ),
    margin=dict(b=200),
    height=600
)

fig_table.update_xaxes(visible=False)
fig_table.update_yaxes(visible=False)


######################################################################################################################
########### AFFICHAGE SUITE ################################################################################################
######################################################################################################################

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.subheader("Titles of the datasets by project")
        st.plotly_chart(fig_table, use_container_width=True)
with col2:
    with st.container(border=True):
        st.subheader("Cumulative Deposits by project")
        st.plotly_chart(fig, use_container_width=True)
