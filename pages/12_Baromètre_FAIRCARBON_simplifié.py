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

# Palette personnalisée
couleurs = {
    'HAL + ORCID + Collection': '#1f77b4',
    'HAL + ORCID sans Collection': '#ff7f0e',
    'HAL seul sans Collection': '#2ca02c',
    'HAL avec Collection hors ORCID': '#d62728',
    'ORCID seul': '#9467bd',
    'Autre': '#8c564b'
}

couleurs_depots = {
    'Zenodo':'#2ca02c',
    'Recherche Data Gouv':'#9467bd',
    'Data InDoRes':'#ff7f0e',
}

categories_avec_collection = ['HAL + ORCID + Collection', 'HAL avec Collection hors ORCID']

######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
d = datetime.date.today()
df_Contacts = read_data("Data\FairCarboN_Datas_Contacts")
#df_Labo_Site = read_data("Data\FairCarboN_Datas_Labo")

df_hal = st.session_state['df_hal']
df_publications = st.session_state['df_publications']

st.title(":grey[Baromètre FairCarboN]")

start_year = 2021
end_year = 2023

df_Contacts_ = df_Contacts.drop_duplicates(subset='Contact')
df_Contacts_.fillna(0,inplace=True)
#potentiel_global = sum(df_Contacts_['Potentiel'])

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(label='Nombre de contacts', value=len(df_Contacts['Contact'].unique()),border=True)
with col2:
    st.metric(label='Nombre de contacts avec ORCID', value=len(df_Contacts['ORCID'].unique()),border=True)
with col3:
    st.metric(label='Nombre de contacts avec ORCID non vide', value=len(df_publications['contact'].unique()),border=True)
with col4:
    st.metric(label='Nombre de contacts sollicités', value=len(df_Contacts_[df_Contacts_['Sollicitation']=='OUI']), border=True)
with col5:
    st.metric(label='Nombre de réponses', value=len(df_Contacts_[df_Contacts_['Réponse']=='OUI']), border=True)


######################################################################################################################

df_hal = df_hal.drop_duplicates(subset=['Année', 'Titre_unique'])
df_hal['Date complete depot'] = pd.to_datetime(df_hal['Date complete depot'], errors='coerce')
df_hal['Année'] = pd.to_numeric(df_hal['Année'], errors='coerce').astype('Int64')

# Étape 2 : comptage par année et catégorie
counts = df_hal.groupby(['Année', 'In_FairCarboN']).size().unstack(fill_value=0)
counts = counts.rename(columns={True: 'Dans la collection', False: 'Hors collection'})

# Étape 3 : cumul global
cumulative = counts.cumsum()

# Étape 4 : pourcentages cumulés
percentages = cumulative.divide(cumulative.sum(axis=1), axis=0) * 100
percentages = percentages.round(1).astype(str) + '%'

# Étape 5 : séparer les années pour les subplots
cumulative_avant = cumulative[(cumulative.index >= 2021) & (cumulative.index <= 2023)]
cumulative_apres = cumulative[(cumulative.index >= 2023) & (cumulative.index <= 2025)]
pct_avant = percentages.loc[cumulative_avant.index]
pct_apres = percentages.loc[cumulative_apres.index]


# Trace supplémentaire dans row=2, col=2
cumulative_collection = cumulative['Dans la collection']
cumulative_collection_apres = cumulative_collection[(cumulative_collection.index >= 2023) & (cumulative_collection.index <= 2025)]

df_collection = df_hal[df_hal['In_FairCarboN'] == True]
cumulative_by_date = df_collection.groupby('Date complete depot').size().sort_index().cumsum()


# Couleurs
couleurs = {
    'Dans la collection': '#2ca02c',
    'Hors collection': '#d62728'
}

# Création du graphique
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Avant FairCarboN (2021–2023)", "Après FairCarboN (2023–2025)"),
    shared_yaxes=True,
    horizontal_spacing=0.1
)

# Subplot Avant
for cat in cumulative_avant.columns:
    fig.add_trace(go.Scatter(
        x=cumulative_avant.index,
        y=cumulative_avant[cat],
        mode='lines+markers+text',
        name=cat,
        stackgroup='one',
        line=dict(color=couleurs[cat]),
        #text=pct_avant[cat],
        textposition='top center',
        hoverinfo='x+name+text'
    ), row=1, col=1)

# Subplot Après (le cumul continue)
for cat in cumulative_apres.columns:
    fig.add_trace(go.Scatter(
        x=cumulative_apres.index,
        y=cumulative_apres[cat],
        mode='lines+markers+text',
        name=cat,
        stackgroup='one',
        line=dict(color=couleurs[cat]),
        text=pct_apres[cat] if cat == 'Dans la collection' else None,
        textposition='top center',
        hoverinfo='x+name+text',
        showlegend=False
    ), row=1, col=2)


fig.add_trace(go.Scatter(
    x=cumulative_by_date.index,
    y=cumulative_by_date.values,
    mode='lines+markers',
    name='Cumul précis Dans la collection',
    line=dict(color='#2ca02c', width=3, dash='dot'),
    showlegend=False
), row=2, col=2)

# Mise en page
fig.update_layout(
    title="Évolution cumulée des titres avant et après FairCarboN",
    height=600,
    legend_title="Catégorie",
    template="plotly_white",
    xaxis_title="Année",
    yaxis_title="Nombre cumulé de titres",
    legend=dict(
        x=0.5,
        y=-0.2,
        xanchor='center',
        yanchor='top',
        orientation='h',
        font=dict(size=20)
    )
)
fig.update_xaxes(
    tickmode='linear',
    tickformat='d',  # format entier
    dtick=1,         # un tick par année
    row=1, col=1
)
fig.update_xaxes(
    tickmode='linear',
    tickformat='d',
    dtick=1,
    row=1, col=2
)
fig.update_yaxes(matches='y1', row=1, col=2)
fig.update_yaxes(range=[15000, 37000], row=1)
fig.update_xaxes(range=[2023-0.2, 2025+0.2], row=1, col=2)
fig.update_xaxes(
    tickformat="%Y-%m-%d",
    range=["2023-01-01",d + datetime.timedelta(days=15)],
    title_text="Date complète de dépôt",
    row=2, col=2
)
fig.update_yaxes(
    matches=None,
    showticklabels=True,
    ticks="outside",
    title_text="NB de titres",
    row=2, col=2
)


# Affichage dans Streamlit
st.plotly_chart(fig, use_container_width=True)