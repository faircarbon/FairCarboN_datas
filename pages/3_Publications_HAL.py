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
couleurs = {"ALAMOD":"#1f77b4",
              "SLAM-B":"#ff7f0e",
              "RIFT":"#2ca02c",
              "CrosyeN":"#d62728",
              "CarboNium":"#9467bd",
              "CABESTAN":"#8c564b",
              "CANETE":"#e377c2",
              "DEEP-C":"#7f7f7f",
              "Drought for C":"#bcbd22",
              "PEACE":"#17becf",
              "TROPECOS":"#393b79",
              "CLIM-FAS":"#637939",
              "CO2_CMPhi":"#8c6d31",
              "GREENSCALE":"#843c39",
              "PREFALIM":"#7b4173",
              "RhizoSeqC":"#3182bd",
              "Gouvernance":"#CCCCFF",
              "Labo":"#020242",
              "Site":"#AC1B08",
              "DIR":"#313695",
              "CR":"#4575b4",
              "INGE":"#74add1",
              "DOC":"#a6cee3",
              "POSTDOC":"#fdae61",
              "PROFESSEUR":"#f46d43",
              "MAITRE_DE_CONF":"#d73027",
              "ASSIT_INGE":"#a50026"}

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
couleur_metrique = "#081E25"

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
df = read_data("Data/FairCarboN_Datas_Contacts")

######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################
st.title(f"Suivi Collection HAL - {derniere_date[21:-4]}")

df_grouped = df.groupby("Contact", as_index=False).agg({
    "projet": lambda x: list(x)
})

df_merged = pd.merge(
    data,
    df_grouped,
    left_on="Auteur_recherché",
    right_on="Contact",
    how="left"   # "left" → garde tous les auteurs recherchés
)

df_merged['projet'] = df_merged['projet'].apply(lambda row: row[0])

projects = sorted(df_merged['projet'].unique())
col1, col2, col3 = st.columns(3)
with col2:
    st.header("Choisir le projet | choose project")
    Selection_projets = st.multiselect("",options=projects)
with col3:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown(":red[par défaut tous | all]")

if len(Selection_projets)==0: #aucun choix
    df_selected = df_merged
else:
    df_selected = df_merged[df_merged['projet'].isin(Selection_projets)]

df_hal_ = df_selected[['Titre_unique', 'Type de document', 'Date complete depot','In_FairCarboN','projet']].drop_duplicates(subset='Titre_unique')
df_hal__ = df_hal_[df_hal_['In_FairCarboN']==True]
df_hal__.reset_index(inplace=True)
df_hal__.drop(columns='index', inplace=True)


# Convertir les dates
df_hal__['Date complete depot'] = pd.to_datetime(df_hal__['Date complete depot'])
df_hal__['Année'] = df_hal__['Date complete depot'].dt.year
df_hal___ = df_hal__[df_hal__["Année"]>2024]

# Compter les documents par jour et par type
counts = df_hal___.groupby(['Date complete depot', 'Type de document']).size().reset_index(name="nb_docs")
counts2 = df_hal___.groupby(['Date complete depot','projet']).size().reset_index(name="nb_docs_projet")

# Calcul du cumul par type
counts["cumul"] = counts.groupby('Type de document')["nb_docs"].cumsum()
counts2["cumul"] = counts2.groupby('projet')["nb_docs_projet"].cumsum()


# Tracé avec plotly
fig = px.line(
    counts,
    x='Date complete depot',
    y="cumul",
    color='Type de document',
    markers=True,
    title="Évolution/ Cumul des dépôts dans notre collection FairCarboN par type de document"
)

fig.update_layout(
    legend=dict(
        orientation="h",
        y=-0.2,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=20)
    ),
    margin=dict(b=200),  # marge basse élargie
    height=600,
    width=600
)

fig2 = px.line(
    counts2,
    x='Date complete depot',
    y="cumul",
    color='projet',
    markers=True,
    title="Évolution/ Cumul des dépôts dans notre collection FairCarboN par projet",
    color_discrete_map=couleurs
)

for trace in fig2.data:
    trace.marker.size = 15

fig2.update_layout(
    legend=dict(
        orientation="h",
        y=-0.2,
        x=0.5,
        xanchor="center",
        yanchor="top",
        font=dict(size=20)
    ),
    margin=dict(b=200),  # marge basse élargie
    height=600,
    width=600
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Cumul des dépôts par type de document")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Cumul des dépôts par projet")
    st.plotly_chart(fig2, use_container_width=True)




