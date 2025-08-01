import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

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
df = read_data("Data\FairCarboN_Datas_Recensement")

df_reduit = df[['PROJET','NOM','INTITULE_DONNEES','NATURE','TYPE','SOURCE','SOURCE_URL','AUTRE_REFERENCE',
                'MOYENS_PRODUCTION','MODELE','MODELE_STATUT','AUTRE_MOYENS','DOCS_ASSOCIES','AUTRE_DOCS', 'ATTRIBUTS','METADATA_EMBARQUEES','METADATA_ENRICHIES']]

#st.dataframe(df_reduit)

df_reduit['Value']= 1
df_reduit['NOM'] = df_reduit['NOM'].str.upper()
df_reduit['MOYENS_PRODUCTION'].fillna('NON RENSEIGNE', inplace=True)
df_reduit['SOURCE'].fillna('NON RENSEIGNE', inplace=True)

fig = px.sunburst(df_reduit, path=['NATURE','TYPE','NOM'], values='Value', color='NATURE',  # La couleur sera attribuée au 1er niveau
    color_discrete_map={
        'PRODUITES': 'lightgreen',
        'PRE_EXISTANTES': 'pink'})

fig.update_layout(
                width=800,
                height=800)


st.title(":grey[Données en cours - Recensements]")

st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    row_counts_prod = df_reduit['MOYENS_PRODUCTION'].value_counts().reset_index()
    row_counts_prod.columns = ['Production des données', 'compte']

    # Calcul du total et du pourcentage
    total_prod = row_counts_prod['compte'].sum()

    row_counts_prod['pourcentage'] = (row_counts_prod['compte'] / total_prod) * 100

    # Génération des étiquettes conditionnelles
    labels_prod = row_counts_prod['Production des données']
    values_prod = row_counts_prod['compte']
    text_labels_prod = [
                f"{pct_prod:.1f}%" if pct_prod > 1 else "" 
                for label_prod, pct_prod in zip(labels_prod, row_counts_prod['pourcentage'])
            ]


    # Création du graphique avec go.Figure
    fig_prod = go.Figure(
                data=[go.Pie(
                    labels=labels_prod,
                    values=values_prod,
                    text=text_labels_prod,
                    textinfo='text',  # N'affiche que text, donc rien si vide
                    hoverinfo='percent+value',
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )]
            )

    fig_prod.update_layout(
                title='Répartition des moyens de production de données',
                showlegend=True
            )
    
    fig_prod.update_layout(
                width=500,
                height=500)
    st.plotly_chart(fig_prod,use_container_width=True)    

with col2:

    row_counts_preex = df_reduit['SOURCE'].value_counts().reset_index()
    row_counts_preex.columns = ['Sources des données', 'compte']

    # Calcul du total et du pourcentage
    total_preex = row_counts_preex['compte'].sum()

    row_counts_preex['pourcentage'] = (row_counts_preex['compte'] / total_preex) * 100

    # Génération des étiquettes conditionnelles
    labels_preex = row_counts_preex['Sources des données']
    values_preex = row_counts_preex['compte']
    text_labels_preex = [
                f"{pct_preex:.1f}%" if pct_preex > 1 else "" 
                for label, pct_preex in zip(labels_preex, row_counts_preex['pourcentage'])
            ]


    # Création du graphique avec go.Figure
    fig_preex = go.Figure(
                data=[go.Pie(
                    labels=labels_preex,
                    values=values_preex,
                    text=text_labels_preex,
                    textinfo='text',  # N'affiche que text, donc rien si vide
                    hoverinfo='percent+value',
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3)
                )]
            )

    fig_preex.update_layout(
                title='Répartition des sources de données',
                showlegend=True
            )
    fig_preex.update_layout(
                width=500,
                height=500)
    st.plotly_chart(fig_preex,use_container_width=True)