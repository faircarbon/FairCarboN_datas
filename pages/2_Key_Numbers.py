import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

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

size_sunburst = 600

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
Contacts = read_data("Data/FairCarboN_Datas_Contacts")
Budgets = read_data("Data/Budgets_PEPR_FairCarboN")
Labo = read_data("Data/FairCarboN_Datas_Labo3")

######################################################################################################################
########### AFFICHAGE ################################################################################################
######################################################################################################################
st.title("Chiffres Clés | Key Numbers")

col1 , col2, col3 = st.columns(3)
with col1:
    st.subheader("Budget (M. Euros)")
    st.metric(value=40, label="", label_visibility="hidden", border=True)
with col2:
    st.subheader("Projets Ciblés | Target Projects")
    st.metric(value=5, label="", label_visibility="hidden", border=True)
with col3:
    st.subheader("Projets sélectionnés | Selected Projects")
    st.metric(value=11, label="", label_visibility="hidden", border=True)

col1 , col2, col3 = st.columns(3)
with col1:
    st.subheader("Labos impliqués | Research Units involved")
    st.metric(value=114, label="", label_visibility="hidden",border=True)
with col2:
    st.subheader("Communauté FairCarboN | FairCarboN Community")
    st.metric(value=498, label="", label_visibility="hidden",border=True)
with col3:
    st.subheader("Sites étudiés/ expérimentaux | Sites localisations")
    st.metric(value=150, label="", label_visibility="hidden",border=True)


st.title("Chiffres Clés - Key Numbers || Par projet - By project")

# Comptage par projet et statut
Contacts_selec = Contacts[["projet", "Fonction"]]
Contacts_ag = Contacts.groupby(["projet", "Fonction"]).size().unstack(fill_value=0)
Contacts_long = Contacts_ag.reset_index().melt(
    id_vars="projet",
    var_name="Fonction",
    value_name="compte"
)
# On enlève les lignes où count = 0 (sinon ça crée des branches vides)
Contacts_long = Contacts_long[Contacts_long["compte"] > 0]
Contacts_long["PEPR FairCarboN"]="  "


Labo_selec = Labo[["PEPR FairCarboN", "projet", "Type_Data"]]
Labo_ag = Labo.groupby(["projet", "Type_Data"]).size().unstack(fill_value=0)
Labo_long = Labo_ag.reset_index().melt(
    id_vars="projet",
    var_name="Type_Data",
    value_name="compte"
)
# On enlève les lignes où count = 0 (sinon ça crée des branches vides)
Labo_long["compte_affiche"] = Labo_long["compte"].replace(0, 1)
Labo_long = Labo_long[Labo_long["compte_affiche"] > 0]
Labo_long["PEPR FairCarboN"]="  "
Labo_long["label"] = Labo_long.apply(
    lambda r: f"0" if r["compte"] == 0 else f"{r['compte']}",
    axis=1
)

fig1 = px.sunburst(
    Budgets,
    path=["PEPR FairCarboN", "Projet", "Budget"],  
    values="Budget",                 
    color="Projet",
    color_discrete_map=couleurs
)

fig1.update_layout(
    width=size_sunburst,   
    height=size_sunburst    
)

fig1.update_traces(
    insidetextfont=dict(size=20, color="black")
)

fig2 = px.sunburst(
    Labo_long,
    path=["PEPR FairCarboN", "projet", "Type_Data","label"],  
    values="compte_affiche",                 
    color="Type_Data",
    color_discrete_map=couleurs
)

fig2.update_layout(
    width=size_sunburst,   
    height=size_sunburst    
)

fig2.update_traces(
    insidetextfont=dict(size=20, color="black")
)


fig3 = px.sunburst(
    Contacts_long,
    path=["PEPR FairCarboN", "projet", "Fonction","compte"],  
    values="compte",                 
    color="Fonction",
    color_discrete_map=couleurs
)

fig3.update_layout(
    width=size_sunburst,   
    height=size_sunburst    
)

fig3.update_traces(
    insidetextfont=dict(size=20, color="black")
)


col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Budgets (M. Euros)")
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.subheader("Unités impliquées/Sites | Units involved/ Experiments")
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    st.subheader("Communauté | Community")
    st.plotly_chart(fig3, use_container_width=True)