import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from Publications import afficher_publications_hal
import datetime


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
color_map2 = {"ALAMOD":"#2DCCB2",
              "SLAM-B":"#FD5436",
              "RIFT":"#7262DA",
              "CrosyeN":"#FBA972",
              "CarboNium":"#3F97D6",
              "CABESTAN":"#FCDA45",
              "CANETE":"#A9DF4D",
              "DEEP-C":"#F78CC3",
              "Drought for C":"#993921",
              "PEACE":"#BB62BD",
              "TROPECOS":"#648BC5",
              "CLIM-FAS":"#FFB52C",
              "CO2_CMPhi":"#B37700",
              "GREENSCALE":"#B3FFBF",
              "PREFALIM":"#FFEECC",
              "RhizoSeqC":"#1FFCB2",
              "PEPR":"#848486"}

d = datetime.date.today()

######################################################################################################################
########### DONNEES ##################################################################################################
######################################################################################################################
data = pd.read_csv("Data/HAL/all_publications_hal_2026-01-19.csv")
filtered_df = data[data['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]
df = read_data("Data\FairCarboN_Datas_Contacts")

######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################
st.title(":grey[Suivi Collection HAL]")

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
Selection_projets = st.multiselect("Choix d'un ou plusieurs projets à visualiser (par défaut TOUS)",options=projects)

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

# Compter les documents par jour et par type
counts = df_hal__.groupby(['Date complete depot', 'Type de document']).size().reset_index(name="nb_docs")
counts2 = df_hal__.groupby(['Date complete depot','projet']).size().reset_index(name="nb_docs_projet")

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

fig2 = px.line(
    counts2,
    x='Date complete depot',
    y="cumul",
    color='projet',
    markers=True,
    title="Évolution/ Cumul des dépôts dans notre collection FairCarboN par projet",
    color_discrete_map=color_map2
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
    height=600
)

st.plotly_chart(fig, use_container_width=True)

#st.plotly_chart(fig2, use_container_width=True)




