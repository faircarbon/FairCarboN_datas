import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import re
from datetime import datetime
from pathlib import Path

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

def extraire_nom(contact):
    if pd.isna(contact):
        return None

    noms = []

    for personne in contact.split('|'):
        personne = personne.strip()

        # Si format "Nom, Prénom"
        if ',' in personne:
            premier_terme = personne.split(',')[1].strip().lower()
        else:
            # Sinon on prend le premier mot
            premier_terme = personne.split()[1].strip().lower()

        noms.append(premier_terme)

    return " | ".join(noms)

def trouver_projets(cell):
    if pd.isna(cell):
        return None
    acronymes = [a.strip() for a in str(cell).split("|")]
    trouvés = [a for a in acronymes if a in projets]
    return trouvés[0] if trouvés else None

def trouver_projet_par_nom(row):
    if pd.notna(row["Projet"]):
        return row["Projet"]
    if pd.isna(row["Nom"]):
        return "INCONNU"
    noms = [n.strip().lower() for n in str(row["Nom"]).split("|")]
    for nom in noms:
        if nom in nom_to_projet:
            return nom_to_projet[nom]
    return "INCONNU"


######################################################################################################################
########### PARAMETRES ###############################################################################################
######################################################################################################################
projets = ["ALAMOD","SLAM-B","RIFT","CrosyeN","CarboNium","CABESTAN","CANETE","DEEP-C","DroughtForC","PEACE","TROPECOS","CLIM-FAS","CO2_CMPhi","GREENSCALE","PREFALIM","RhizoSeqC"]
couleurs = {"ALAMOD":"#fa0404",
              "SLAM-B":"#e7b204",
              "RIFT":"#05fc6c",
              "CrosyeN":"#b9fc01",
              "CarboNium":"#ec8129",
              "CABESTAN":"#03fabc",
              "CANETE":"#067ff0",
              "DEEP-C":"#6d03f8",
              "DroughtForC":"#cb05fc",
              "PEACE":"#f0047a",
              "TROPECOS":"#090080",
              "CLIM-FAS":"#ca7ff8",
              "CO2_CMPhi":"#793305",
              "GREENSCALE":"#035718",
              "PREFALIM":"#B83E0E",
              "RhizoSeqC":"#636303",
              "Gouvernance":"#0A0A0A",
              "ART": "#020242",
              "UNDEFINED": "#0B9BEE",
              "COM": "#16DA6E",
              "POSTER":"#8de42a",
              "MEM":"#eedf14",
              "REPORT" : "#eba21b",
              "OTHER": "#f5760f",
              "LECTURE": "#ff0909"}

couleurs2 = {"ALAMOD":"#fa0404",
              "SLAM-B":"#e7b204",
              "RIFT":"#05fc6c",
              "CrosyeN":"#b9fc01",
              "CarboNium":"#ec8129",
              "CABESTAN":"#03fabc",
              "CANETE":"#067ff0",
              "DEEP-C":"#6d03f8",
              "DroughtForC":"#cb05fc",
              "PEACE":"#f0047a",
              "TROPECOS":"#090080",
              "CLIM-FAS":"#ca7ff8",
              "CO2_CMPhi":"#793305",
              "GREENSCALE":"#035718",
              "PREFALIM":"#B83E0E",
              "RhizoSeqC":"#636303",
              "Gouvernance":"#0A0A0A"}


# Variables Python
couleur_h1 = "#748114"
taille_h1 = "60px"
police_h1 = "Cascadia Code"

couleur_h2 = "#FF6347"
taille_h2 = "32px"
police_h2 = "Cascadia Code"

couleur_h3 = "#1B657C"
taille_h3 = "25px"
police_h3 = "Cascadia Code"

taille_metrique = "40px"
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
# Récupérer le fichier HAL le plus récent
hal_dir = Path("Data/HAL")
files = sorted(hal_dir.glob("all_publications_hal_FC_*.csv"))
latest_file = files[-1]

# Lire le fichier
data = pd.read_csv(latest_file, sep=";")

#data = pd.read_csv("all_publications_hal_FC_2026-06-04.csv")
df = read_data("Data/FairCarboN_Datas_Contacts2")
# Ajout de la colonne "Nom"
df["Nom"] = df["Contact"].str.split().str[-1]
nom_to_projet = {k.lower(): v for k, v in zip(df["Nom"], df["projet"])}

# Application au dataframe
data["Nom"] = data["Auteurs"].apply(extraire_nom)
data["Projet"] = data["Acronyme projet ANR"].apply(trouver_projets)

data["Projet"] = data.apply(trouver_projet_par_nom, axis=1)

######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################
st.title(f"FairCarboN => HAL {latest_file.name[24:-4]}")

col1 , col2 = st.columns(2)
with col1:
    st.subheader("All deposits")
    st.metric(value=len(data), label="", label_visibility="hidden", border=True)
with col2:
    st.subheader("Published Articles")
    st.metric(value=len(data[data["Type de document"]=="ART"]), label="", label_visibility="hidden", border=True)

######################################################################################################################
######################################################################################################################

projects = sorted(data['Projet'].unique())
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
    df_selected = data
else:
    df_selected = data[data['Projet'].isin(Selection_projets)]

df_hal_ = df_selected[['Titre', 'Type de document', 'Date de dépôt','Projet']].drop_duplicates(subset='Titre')

#st.dataframe(df_hal__)

# Convertir les dates
df_hal_['Date de dépôt'] = pd.to_datetime(df_hal_['Date de dépôt'])
df_hal_['Date'] = df_hal_['Date de dépôt']
df_hal_['Année'] = df_hal_['Date de dépôt'].dt.year
#df_hal___ = df_hal__[df_hal__["Année"]>=2024]

# Compter les documents par jour et par type
counts = df_hal_.groupby(['Date', 'Type de document']).size().reset_index(name="nb_docs")
counts2 = df_hal_.groupby(['Date','Projet']).size().reset_index(name="nb_docs_projet")
min_counts = counts["Date"].min()

# Calcul du cumul par type
counts["Cumul"] = counts.groupby('Type de document')["nb_docs"].cumsum()
counts2["Cumul"] = counts2.groupby('Projet')["nb_docs_projet"].cumsum()

# Calculer le total final par projet
totaux = counts2.groupby("Projet")["Cumul"].max()

# Ajouter le total dans le nom du projet
counts2["Projet_label"] = counts2["Projet"].map(
    lambda p: f"{p} (n={totaux[p]})"
)

# Mettre à jour le color_discrete_map avec les nouveaux labels
#couleurs_label = {f"{p} (n={totaux[p]})": c for p, c in couleurs2.items()}
couleurs_label = {
    f"{p} (n={totaux[p]})": c 
    for p, c in couleurs2.items() 
    if p in totaux  # <-- on ne garde que les projets présents
}
######################################################################################################################
######################################################################################################################

# Tracé avec plotly
fig = px.line(
    counts,
    x='Date',
    y="Cumul",
    color="Type de document",
    color_discrete_map=couleurs
)

mapping = {
    "ART": "ARTICLE",
    "UNDEFINED": "INDEFINI | UNDEFINED",
    "COMM": "COMMUNICATION",
    "POSTER":"POSTER",
    "MEM":"MEMOIRE | PROFESSIONNAL THESIS",
    "REPORT" : "RAPPORT | REPORT",
    "OTHER": "AUTRE | OTHER",
    "LECTURE": "COURS | LECTURE"
}

fig.update_traces(line=dict(width=4))

fig.for_each_trace(
    lambda t: t.update(name=mapping[t.name]) if t.name in mapping else None
)


fig.update_layout(
    legend=dict(
        title=dict(
            text="Types de document",
            font=dict(size=22, color=couleur_h3)), 
        orientation="h",
        y=-0.3,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=20)
    ),
    margin=dict(b=200),  # marge basse élargie
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
    gridcolor="lightgray",
    range=["2024-01-01", datetime.today()]
)

fig.update_yaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    tickfont=dict(size=16, color=couleur_graphes),
    gridcolor="lightgray"
)

######################################################################################################################
######################################################################################################################

fig2 = px.line(
    counts2,
    x='Date',
    y="Cumul",
    color='Projet_label',
    color_discrete_map=couleurs_label
)

fig2.update_traces(line=dict(width=4))

fig2.update_layout(
    legend=dict(
        title=dict(
            text="Projets",
            font=dict(size=22, color=couleur_h3)),
        orientation="h",
        y=-0.3,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=20)
    ),
    margin=dict(b=200),  # marge basse élargie
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

fig2.update_xaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    tickfont=dict(size=16, color=couleur_graphes),
    gridcolor="lightgray",
    range=["2024-01-01", datetime.today()]
)

fig2.update_yaxes(
    showline=True,
    linewidth=2,
    linecolor="black",
    tickfont=dict(size=16, color=couleur_graphes),
    gridcolor="lightgray"
)

######################################################################################################################
######################################################################################################################
col1, col2 = st.columns([0.42,0.58])
with col1:
    with st.container(border=True):
        st.subheader("Cumulative Deposits types ")
        st.plotly_chart(fig, use_container_width=True)
with col2:
    with st.container(border=True):
        st.subheader("Cumulative Deposits by project")
        st.plotly_chart(fig2, use_container_width=True)