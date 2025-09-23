import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np
from plotly.subplots import make_subplots
import datetime


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
d = datetime.date.today()
df_Contacts = read_data("Data\FairCarboN_Datas_Contacts")
#df_Labo_Site = read_data("Data\FairCarboN_Datas_Labo")

df_hal = st.session_state['df_hal']
df_publications = st.session_state['df_publications']

st.write(set(df_hal['Type de document'].values))

st.title(":grey[Baromètre FairCarboN]")

start_year = 2021
end_year = 2023

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label='Nombre de contacts', value=len(df_Contacts['Contact'].unique()))
with col2:
    st.metric(label='Nombre de contacts avec compte ORCID', value=len(df_Contacts['ORCID'].unique()))
with col3:
    st.metric(label='Nombre de contacts avec compte ORCID non vide', value=len(df_publications['contact'].unique()))


###########################################################################################################################################################
############################ COMPTAGES ####################################################################################################################

#################### TOUT TYPES ###########################################

# Filtrer les années souhaitées
liste_type_ORCID = ['journal-article','preprint','book','book-chapter','conference-paper','conference-poster']
#liste_type_HAL = ['NOTICE', 'REPORT', 'THESE', 'CREPORT', 'TRAD', 'LECTURE', 'ISSUE', 'OTHER', 'BLOG', 'UNDEFINED', 'OUV', 'COUV', 'SON', 'IMG', 'VIDEO', 'ART', 'HDR', 'COMM', 'MEM', 'PATENT', 'POSTER', 'SOFTWARE', 'PROCEEDINGS']
liste_type_HAL = ['ART','POSTER', 'COMM']


#df_publications_filtered_touttypes = df_publications[df_publications['Type'].isin(liste_type_ORCID)]
df_publications_filtered_touttypes1 = df_publications[df_publications["Année"].between(start_year, end_year)][df_publications['Type'].isin(liste_type_ORCID)]
df_publications_non_dupliqués_touttypes1 = df_publications_filtered_touttypes1.drop_duplicates(subset='Titre')
df_publications_non_dupliqués_touttypes1['from']='ORCID'
df_publications_non_dupliqués_touttypes1['Titre_unique']=df_publications_non_dupliqués_touttypes1['Titre']

# Assurer que la colonne 'Date de publication' est bien en datetime
df_hal["Date complete depot"] = pd.to_datetime(df_hal["Date complete depot"], errors="coerce")
df_hal["Année"] = df_hal["Date complete depot"].dt.year
#df_HAL_filtered = df_hal[df_hal['Type de document'].isin(liste_type_HAL)]
df_HAL_filtered_touttypes1 = df_hal[df_hal["Année"].between(start_year, end_year)][df_hal['Type de document'].isin(liste_type_HAL)]
df_hal_non_dupliqués_touttypes1 = df_HAL_filtered_touttypes1 .drop_duplicates(subset='Titre_unique')
df_hal_non_dupliqués_touttypes1['from']='HAL'


# Alignement des publications HAL et ORCID (prêts à être comparés)
df_to_be_compared1 = pd.concat([df_hal_non_dupliqués_touttypes1[['Année','Date complete depot','Premier_auteur','from','Titre_unique','In_FairCarboN']],df_publications_non_dupliqués_touttypes1[['Année','Premier_auteur','from','Titre_unique']]], axis=0)
df_to_be_compared1.reset_index(inplace=True)
df_to_be_compared1.drop(columns='index', inplace=True)

st.metric(label='DF à comparer', value=len(df_to_be_compared1))

# Étape 1 : Pivot pour regrouper les sources
df_to_be_compared1['In_FairCarboN'] = df_to_be_compared1['In_FairCarboN'].fillna(False)
df_to_be_compared1['is_HAL'] = df_to_be_compared1['from'] == 'HAL'
df_to_be_compared1['is_ORCID'] = df_to_be_compared1['from'] == 'ORCID'

# Regrouper par titre
grouped1 = df_to_be_compared1.groupby(['Titre_unique','Année']).agg({
    'is_HAL': 'any',
    'is_ORCID': 'any',
    'In_FairCarboN': 'any'
}).reset_index()

# Étape 2 : Créer la colonne de statut
def classify(row):
    if row['is_HAL'] and row['is_ORCID'] and row['In_FairCarboN']:
        return 'HAL + ORCID + Collection'
    elif row['is_HAL'] and row['is_ORCID'] and not row['In_FairCarboN']:
        return 'HAL + ORCID sans Collection'
    elif row['is_HAL'] and not row['is_ORCID'] and row['In_FairCarboN']:
        return 'HAL avec Collection hors ORCID'
    elif row['is_HAL'] and not row['is_ORCID'] and not row['In_FairCarboN']:
        return 'HAL seul sans Collection'
    elif row['is_ORCID'] and not row['is_HAL'] and not row['In_FairCarboN']:
        return 'ORCID seul'
    else:
        return 'Autre'


grouped1['categorie'] = grouped1.apply(classify, axis=1)


st.metric(label='grouped', value=(len(grouped1)))

# Comptage
counts1 = grouped1.groupby(['Année', 'categorie']).size().unstack(fill_value=0)

cumulative_counts1 = counts1.cumsum()

# Pourcentages
percentages1 = counts1.div(counts1.sum(axis=1), axis=0) * 100

##############################################################################################
start_year2 = 2023
end_year2 = 2025


#df_publications_filtered_touttypes = df_publications[df_publications['Type'].isin(liste_type_ORCID)]
df_publications_filtered_touttypes2 = df_publications[df_publications["Année"].between(start_year2, end_year2)][df_publications['Type'].isin(liste_type_ORCID)]
df_publications_non_dupliqués_touttypes2 = df_publications_filtered_touttypes2.drop_duplicates(subset='Titre')
df_publications_non_dupliqués_touttypes2['from']='ORCID'
df_publications_non_dupliqués_touttypes2['Titre_unique']=df_publications_non_dupliqués_touttypes2['Titre']

#df_HAL_filtered = df_hal[df_hal['Type de document'].isin(liste_type_HAL)]
df_HAL_filtered_touttypes2 = df_hal[df_hal["Année"].between(start_year2, end_year2)][df_hal['Type de document'].isin(liste_type_HAL)]
df_hal_non_dupliqués_touttypes2 = df_HAL_filtered_touttypes2.drop_duplicates(subset='Titre_unique')
df_hal_non_dupliqués_touttypes2['from']='HAL'


# Alignement des publications HAL et ORCID (prêts à être comparés)
df_to_be_compared2 = pd.concat([df_hal_non_dupliqués_touttypes2[['Année','Date complete depot','Premier_auteur','from','Titre_unique','In_FairCarboN']],df_publications_non_dupliqués_touttypes2[['Année','Premier_auteur','from','Titre_unique']]], axis=0)
df_to_be_compared2.reset_index(inplace=True)
df_to_be_compared2.drop(columns='index', inplace=True)

# Étape 1 : Pivot pour regrouper les sources
df_to_be_compared2['In_FairCarboN'] = df_to_be_compared2['In_FairCarboN'].fillna(False)
df_to_be_compared2['is_HAL'] = df_to_be_compared2['from'] == 'HAL'
df_to_be_compared2['is_ORCID'] = df_to_be_compared2['from'] == 'ORCID'

# Regrouper par titre
grouped2 = df_to_be_compared2.groupby(['Titre_unique','Année']).agg({
    'is_HAL': 'any',
    'is_ORCID': 'any',
    'In_FairCarboN': 'any'
}).reset_index()

grouped2['categorie'] = grouped2.apply(classify, axis=1)

# Comptage
counts2 = grouped2.groupby(['Année', 'categorie']).size().unstack(fill_value=0)

# prendre en compte le compte cumulatif 1 dans le 2
cumulative_counts2 = counts2.cumsum()

base = cumulative_counts1.iloc[-2]
base_df = pd.DataFrame([base], index=[counts2.index[0]])
base_df = base_df.reindex(columns=counts2.columns, fill_value=0)
base_df = pd.DataFrame([base_df.iloc[0]] * len(counts2), index=counts2.index)

cumulative_counts2_shifted = cumulative_counts2 + base_df

percentages2 = counts2.div(counts2.sum(axis=1), axis=0) * 100
# Palette personnalisée
couleurs = {
    'HAL + ORCID + Collection': '#1f77b4',
    'HAL + ORCID sans Collection': '#ff7f0e',
    'HAL seul sans Collection': '#2ca02c',
    'HAL avec Collection hors ORCID': '#d62728',
    'ORCID seul': '#9467bd',
    'Autre': '#8c564b'
}

categories_avec_collection = ['HAL + ORCID + Collection', 'HAL avec Collection hors ORCID']


##############################################################################################
start_year3 = 2024
end_year3 = 2025

############## PREMIER AFFICHAGE #########################################################

# Créer une figure avec deux colonnes et axes Y partagés
fig_combined = make_subplots(
    rows=1, cols=2,
    shared_yaxes=True,
    column_widths=[0.7, 0.3],
    horizontal_spacing=0.05,
    subplot_titles=("titre 1", "titre 2")
)

# Colonne 1 : valeurs brutes
for categorie in cumulative_counts1.columns:
    fig_combined.add_trace(go.Scatter(
        x=cumulative_counts1.index,
        y=cumulative_counts1[categorie],
        mode='lines+markers+text',
        name=categorie,
        stackgroup='one',
        line=dict(color=couleurs.get(categorie)),
        text=percentages1[categorie].round(1).astype(str) + '%' if categorie in categories_avec_collection else None,
        textposition='top center',
        hoverinfo='x+name+text',
        showlegend=False
    ), row=1, col=1)

for categorie in cumulative_counts2_shifted.columns:
    fig_combined.add_trace(go.Scatter(
        x=cumulative_counts2_shifted.index,
        y=cumulative_counts2_shifted[categorie],
        mode='lines+markers+text',
        name=categorie,
        stackgroup='one',
        line=dict(color=couleurs.get(categorie)),
        text=percentages2[categorie].round(1).astype(str) + '%' if categorie in categories_avec_collection else None,
        textposition='top center',
        hoverinfo='x+name+text',
    ), row=1, col=2)

fig_combined.update_layout(
    title='Évolution des catégories par année (valeurs brutes)',
    xaxis_title='Année',
    yaxis_title='Nombre de titres',
    legend_title='Catégorie',
    height=500
)

fig_combined.update_xaxes(
    tickmode='linear',
    tickformat='d'  # format entier
)
fig_combined.update_xaxes(
    range=[start_year - 0.1, end_year],row=1, col=1
)
fig_combined.update_xaxes(
    range=[start_year2 - 0.1, end_year2 + 0.1],row=1, col=2
)

st.plotly_chart(fig_combined, use_container_width=True)


############## DEUXIEME AFFICHAGE #########################################################

cumulative_counts1_zoom = cumulative_counts1[['HAL + ORCID sans Collection']]
cumulative_counts2_shifted_zoom = cumulative_counts2_shifted[['HAL + ORCID + Collection','HAL + ORCID sans Collection','HAL avec Collection hors ORCID']]

# Créer une figure avec deux colonnes et axes Y partagés
fig_combined2 = make_subplots(
    rows=1, cols=2,
    shared_yaxes=True,
    column_widths=[0.7, 0.3],
    horizontal_spacing=0.05,
    subplot_titles=("titre 1", "titre 2")
)

# Colonne 1 : valeurs brutes
for categorie in cumulative_counts1_zoom.columns:
    fig_combined2.add_trace(go.Scatter(
        x=cumulative_counts1_zoom.index,
        y=cumulative_counts1_zoom[categorie],
        mode='lines+markers+text',
        name=categorie,
        stackgroup='one',
        line=dict(color=couleurs.get(categorie)),
        text=percentages1[categorie].round(1).astype(str) + '%' if categorie in categories_avec_collection else None,
        textposition='top center',
        hoverinfo='x+name+text',
        showlegend=False
    ), row=1, col=1)

for categorie in cumulative_counts2_shifted_zoom.columns:
    fig_combined2.add_trace(go.Scatter(
        x=cumulative_counts2_shifted_zoom.index,
        y=cumulative_counts2_shifted_zoom[categorie],
        mode='lines+markers+text',
        name=categorie,
        stackgroup='one',
        line=dict(color=couleurs.get(categorie)),
        text=percentages2[categorie].round(1).astype(str) + '%' if categorie in categories_avec_collection else None,
        textposition='top center',
        hoverinfo='x+name+text',
    ), row=1, col=2)

fig_combined2.update_layout(
    title='Évolution des catégories par année (valeurs brutes)',
    xaxis_title='Année',
    yaxis_title='Nombre de titres',
    legend_title='Catégorie',
    height=500
)

fig_combined2.update_xaxes(
    tickmode='linear',
    tickformat='d'  # format entier
)
fig_combined2.update_xaxes(
    range=[start_year - 0.1, end_year],row=1, col=1
)
fig_combined2.update_xaxes(
    range=[start_year2 - 0.1, end_year2 + 0.1],row=1, col=2
)

st.plotly_chart(fig_combined2, use_container_width=True)

###################################################################################################

# Filtrer les lignes HAL avec une date complète
hal_dates = df_to_be_compared2[df_to_be_compared2['is_HAL'] & df_to_be_compared2['Date complete depot'].notna()]

# Créer un dictionnaire {Titre_unique: Date complète}
dict_hal_dates = hal_dates.set_index('Titre_unique')['Date complete depot'].to_dict()

def enrichir_date(row):
    titre = row['Titre_unique']
    if titre in dict_hal_dates:
        return dict_hal_dates[titre]  # Date HAL connue
    else:
        # Date artificielle : 1er juillet de l'année
        return pd.to_datetime(str(int(row['Année'])) + '-07-01')

df_to_be_compared2['Date_enrichie'] = df_to_be_compared2.apply(enrichir_date, axis=1)

grouped3 = df_to_be_compared2.groupby(['Titre_unique']).agg({
    'is_HAL': 'any',
    'is_ORCID': 'any',
    'In_FairCarboN': 'any',
    'Date_enrichie': 'first'  # On garde la date enrichie
}).reset_index()

grouped3['categorie'] = grouped3.apply(classify, axis=1)

counts3 = grouped3.groupby(['Date_enrichie', 'categorie']).size().unstack(fill_value=0)
counts3 = counts3.sort_index()
cumulative_counts3 = counts3.cumsum()

import plotly.graph_objects as go

fig3 = go.Figure()

for cat in cumulative_counts3.columns:
    fig3.add_trace(go.Scatter(
        x=cumulative_counts3.index,
        y=cumulative_counts3[cat],
        mode='lines+markers',
        name=cat
    ))

fig3.update_layout(
    title='Cumul temporel des titres par catégorie',
    xaxis_title='Date',
    yaxis_title='Nombre cumulé',
    template='plotly_white'
)

st.plotly_chart(fig3)