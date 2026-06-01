import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import glob
import re
from datetime import datetime


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
              "ART": "#020242",
              "UNDEFINED": "#0B9BEE",
              "COM": "#16DA6E",
              "POSTER":"#8de42a",
              "MEM":"#eedf14",
              "REPORT" : "#eba21b",
              "OTHER": "#f5760f",
              "LECTURE": "#ff0909"}


# Variables Python
couleur_h1 = "#1F8B09"
taille_h1 = "48px"
police_h1 = "Marianne"

couleur_h2 = "#FF6347"
taille_h2 = "32px"
police_h2 = "Marianne"

couleur_h3 = "#4C98AF"
taille_h3 = "24px"
police_h3 = "Marianne"

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

dossier = "Data/HAL/"
pattern = dossier + "all_publications_hal_*.csv"
fichiers = glob.glob(pattern)

# Expression régulière pour extraire la date
regex = re.compile(r"all_publications_hal_(\d{4}-\d{2}-\d{2}).csv")

def extraire_date(f):
    match = regex.search(f)
    if match:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    return datetime.min

# Trier par date décroissante
fichiers_tries = sorted(fichiers, key=extraire_date, reverse=True)
fichiers_sans_prefixe = [f[9:] for f in fichiers_tries]

derniere_date = fichiers_sans_prefixe[0] if fichiers_tries else None
dernier_fichier = dossier + derniere_date

data = pd.read_csv(dernier_fichier)
filtered_df = data[data['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]
filtered_df_uniq = filtered_df[['Titre_unique', 'Type de document', 'Date complete depot','In_FairCarboN','Projet']].drop_duplicates(subset='Titre_unique')
df = read_data("Data/FairCarboN_Datas_Contacts")

######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################
st.title(f"Suivi Collection HAL - {derniere_date[21:-4]}")

col1 , col2 = st.columns(2)
with col1:
    st.subheader("Nombre global de depôts | All deposits")
    st.metric(value=len(filtered_df_uniq), label="", label_visibility="hidden", border=True)
with col2:
    st.subheader("Nombre d'articles | Published Articles")
    st.metric(value=len(filtered_df_uniq[filtered_df_uniq["Type de document"]=="ART"]), label="", label_visibility="hidden", border=True)

######################################################################################################################
######################################################################################################################

projects = sorted(filtered_df_uniq['Projet'].unique())
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
    df_selected = filtered_df_uniq
else:
    df_selected = filtered_df_uniq[filtered_df_uniq['Projet'].isin(Selection_projets)]

df_hal_ = df_selected[['Titre_unique', 'Type de document', 'Date complete depot','In_FairCarboN','Projet']].drop_duplicates(subset='Titre_unique')
df_hal__ = df_hal_[df_hal_['In_FairCarboN']==True]
df_hal__.reset_index(inplace=True)
df_hal__.drop(columns='index', inplace=True)

#st.dataframe(df_hal__)

# Convertir les dates
df_hal__['Date complete depot'] = pd.to_datetime(df_hal__['Date complete depot'])
df_hal__['Date'] = df_hal__['Date complete depot']
df_hal__['Année'] = df_hal__['Date complete depot'].dt.year
#df_hal___ = df_hal__[df_hal__["Année"]>=2024]

# Compter les documents par jour et par type
counts = df_hal__.groupby(['Date', 'Type de document']).size().reset_index(name="nb_docs")
counts2 = df_hal__.groupby(['Date','Projet']).size().reset_index(name="nb_docs_projet")
min_counts = counts["Date"].min()

# Calcul du cumul par type
counts["Cumul"] = counts.groupby('Type de document')["nb_docs"].cumsum()
counts2["Cumul"] = counts2.groupby('Projet')["nb_docs_projet"].cumsum()

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
        y=-0.2,
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
    color='Projet',
    color_discrete_map=couleurs
)

fig2.update_traces(line=dict(width=4))

fig2.update_layout(
    legend=dict(
        title=dict(
            text="Projets",
            font=dict(size=22, color=couleur_h3)),
        orientation="h",
        y=-0.2,
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
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.subheader("Cumul des dépôts par type de document | Cumulative Deposits types ")
        st.plotly_chart(fig, use_container_width=True)
with col2:
    with st.container(border=True):
        st.subheader("Cumul des dépôts par projet | Cumulative Deposits by project")
        st.plotly_chart(fig2, use_container_width=True)




