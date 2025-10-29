import streamlit as st
from PIL import Image
import pandas as pd
import folium
from folium.features import CustomIcon
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
from wordcloud import WordCloud
import plotly.graph_objects as go
import networkx as nx

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

def to_rgb_string(rgb_tuple):
    r, g, b = (int(255 * c) for c in rgb_tuple)
    return f"rgb({r}, {g}, {b})"

def rgb_to_hex(rgb_string):
    """Convertit une couleur CSS 'rgb(r,g,b)' en hexadécimal '#rrggbb'."""
    rgb = rgb_string.replace("rgb(", "").replace(")", "").split(",")
    return "#{:02x}{:02x}{:02x}".format(*[int(x.strip()) for x in rgb])

@st.cache_data
def read_data(path):
    # Chemin vers le fichier Excel
    #fichier_excel = "Data\FairCarboN_Datas_V2.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=0,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")

    return df

def create_pie_icon(projets, border_color, icon_size, color_map):
    """Crée une icône de camembert encodée en base64 pour Folium."""
    fig, ax = plt.subplots(figsize=(1, 1))
    projet_counts = [1] * len(projets)
    colors_used = [color_map.get(proj, "#cccccc") for proj in projets]  # couleur par défaut si projet absent
    ax.pie(projet_counts, colors=colors_used, wedgeprops={'edgecolor': border_color, 'linewidth': 5})
    plt.axis('off')

    img_data = BytesIO()
    plt.savefig(img_data, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    img_data.seek(0)
    encoded = base64.b64encode(img_data.read()).decode()

    icon_url = f"data:image/png;base64,{encoded}"
    return folium.CustomIcon(icon_image=icon_url, icon_size=icon_size)

st.cache_resource
def carto2(grouped_, avg_lat, avg_long, color_map2):
    """Crée une carte Folium avec des marqueurs camemberts pour chaque laboratoire ou site."""
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=5.5, tiles='CartoDB positron', control_scale=True)

    for _, row in grouped_.iterrows():
        projets = row['projet']
        latitude = row['Latitude']
        longitude = row['Longitude']
        type_data = row['Type_Data']
        laboratoire = row.get('laboratoire', 'Laboratoire inconnu')

        if type_data == "Labo":
            icon = create_pie_icon(projets, border_color="black", icon_size=(35, 35), color_map=color_map2)
        elif type_data == "Site":
            icon = create_pie_icon(projets, border_color="red", icon_size=(30, 30), color_map=color_map2)
        else:
            continue  # ignorer les types inconnus

        popup_html = "<b>Projets :</b><br>" + "<br>".join(projets)
        popup = folium.Popup(popup_html, max_width=250)
        tooltip = laboratoire

        folium.Marker(location=[latitude, longitude], popup=popup, tooltip=tooltip, icon=icon).add_to(m)

    return m

######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df_Labo_Site = read_data("Data\FairCarboN_Datas_Labo")
df_Contacts = read_data("Data\FairCarboN_Datas_Contacts")
df_Labo2 = read_data("Data\FairCarboN_Datas_Labo2")

# Couleurs associées à chaque projet
projects = sorted(df_Contacts['projet'].unique())
unites = sorted(df_Contacts['Sigle structure'].unique())
ordre_perso = ["ALAMOD","SLAM-B","RIFT","CrosyeN","CarboNium","CABESTAN","CANETE","DEEP-C","Drought for C",
                   "PEACE","TROPECOS","CLIM-FAS","CO2_CMPhi","GREENSCALE","PREFALIM","RhizoSeqC","PEPR"]

# Create a list of colors (one per project)
colors = [
"rgb(141,211,199)",
"rgb(255,255,179)",
"rgb(190,186,218)",
"rgb(251,128,114)",
"rgb(128,177,211)",
"rgb(253,180,98)",
"rgb(179,222,105)",
"rgb(252,205,229)",
"rgb(217,217,217)",
"rgb(188,128,189)",
"rgb(204,235,197)",
"rgb(255,237,111)",
"rgb(179,119,0)",
"rgb(179,255,191)",
"rgb(255,238,204)",
"rgb(204,255,238)",
"rgb(204,204,255)",
]

color_map2 = {"ALAMOD":"#8DD3C7",
              "SLAM-B":"#FFFFB3",
              "RIFT":"#BEBADA",
              "CrosyeN":"#FB8072",
              "CarboNium":"#80B1D3",
              "CABESTAN":"#FDB462",
              "CANETE":"#B3DE69",
              "DEEP-C":"#FCCDE5",
              "Drought for C":"#D9D9D9",
              "PEACE":"#BC80BD",
              "TROPECOS":"#CCEBC5",
              "CLIM-FAS":"#FFED6F",
              "CO2_CMPhi":"#B37700",
              "GREENSCALE":"#B3FFBF",
              "PREFALIM":"#FFEECC",
              "RhizoSeqC":"#CCFFEE",
              "PEPR":"#CCCCFF"}


######################################################################################################################
########### NOMBRE LABOS PAR PROJET ##################################################################################
######################################################################################################################

st.title(f":grey[Analyse générale des données de FAIRCARBON]")
col1 , col2, col3 = st.columns(3)
with col1:

    df_lab_counts = df_Contacts.groupby("projet")['Sigle structure'].nunique().reset_index()
    df_lab_counts.rename(columns={"Sigle structure": "Nombre d'Unités"}, inplace=True)
    df_lab_counts["projet"] = pd.Categorical(df_lab_counts["projet"], categories=ordre_perso, ordered=True)
    df_lab_counts = df_lab_counts.sort_values("projet")

    project_names = df_lab_counts['projet']
    color_map = {project: colors[i % len(colors)] for i, project in enumerate(project_names)}

    # Assign colors based on the project
    bar_colors = [color_map[project] for project in project_names]

    # Plot
    fig0 = go.Figure(go.Bar(
        x=df_lab_counts["Nombre d'Unités"],
        y=project_names,
        orientation='h',
        marker_color=bar_colors  # Assign custom colors
    ))

    fig0.update_layout(height=400,
                       margin=dict(t=0))

    st.subheader(f":grey[Nb d'unités]")
    st.metric(label='', value=len(set(df_Contacts['Sigle structure'].unique())))
    st.plotly_chart(fig0, use_container_width=True)

with col2:
    df_contacts_counts = df_Contacts.groupby("projet")['Contact'].nunique().reset_index()
    df_contacts_counts.rename(columns={"Contact": "Nombre de contacts"}, inplace=True)
    df_contacts_counts["projet"] = pd.Categorical(df_contacts_counts["projet"], categories=ordre_perso, ordered=True)
    df_contacts_counts = df_contacts_counts.sort_values("projet")

    # Plot
    fig0b = go.Figure(go.Bar(
        x=df_contacts_counts["Nombre de contacts"],
        y=project_names,
        orientation='h',
        marker_color=bar_colors  # Assign custom colors
    ))

    fig0b.update_layout(height=400,
                        margin=dict(t=0))

    st.subheader(f":grey[Nb de contacts]")
    st.metric(label='', value=len(set(df_Contacts['Contact'])))
    st.plotly_chart(fig0b, use_container_width=True)

with col3:
    df_sites_counts = df_Labo_Site["projet"][df_Labo_Site['Type_Data']=="Site"].value_counts().reset_index()
    df_sites_counts.rename(columns={"count": "Nombre de sites"}, inplace=True)
    df_projets_sans_sites = pd.DataFrame({"projet": ["CO2_CMPhi","GREENSCALE","PREFALIM","RhizoSeqC","PEPR"],"Nombre de sites": [0, 0, 0, 0, 0]})
    df_sites_counts_ = pd.concat([df_sites_counts,df_projets_sans_sites], axis=0)
    df_sites_counts_['projet'] = pd.Categorical(df_sites_counts_['projet'], categories=ordre_perso, ordered=True)
    df_sites_counts_ = df_sites_counts_.sort_values("projet")

    # Plot
    fig0c = go.Figure(go.Bar(
        x=df_sites_counts_["Nombre de sites"],
        y=project_names,
        orientation='h',
        marker_color=bar_colors  # Assign custom colors
    ))

    fig0c.update_layout(height=400,
                        margin=dict(t=0))

    st.subheader(f":grey[Nb de sites/Lieux étudiés]")
    st.metric(label='', value=len(set(df_Labo_Site['laboratoire'][df_Labo_Site['Type_Data']=="Site"])))
    st.plotly_chart(fig0c, use_container_width=True)

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################

#choix de visualisation
col1, col2, col3, col4 =st.columns([0.4,0.2,0.2,0.2])
with col1:
    st.subheader(f":grey[Choix de visualisation]")
    st.markdown("Choix obligatoire")
with col2:
    Unites = st.checkbox('Unités')
with col3:
    Sites = st.checkbox('Sites')
with col4:
    Unites_Sites = st.checkbox('Unités & Sites')


col1, col2 = st.columns([0.4,0.6])
with col1:
    st.subheader(f":grey[Choix du projet visualisé]")
    st.markdown("Pas de choix = tous les projets")
with col2:
    # Choix Projet
    Selection_projets = st.multiselect('',options=projects)

if len(Selection_projets)==0: #aucun choix
    df_selected = df_Labo_Site #le dataframe ne change pas, c'est l'original
    df_contacts_selected = df_Contacts
    projets_selected = projects
else:
    df_selected = df_Labo_Site[df_Labo_Site['projet'].isin(Selection_projets)]
    df_contacts_selected = df_Contacts[df_Contacts['projet'].isin(Selection_projets)]
    projets_selected = Selection_projets

laboratoires_select = df_selected[['laboratoire','Type_Data']]
laboratoires_bis_Unites = laboratoires_select[laboratoires_select['Type_Data']=='Labo']
laboratoires_bis_sites = laboratoires_select[laboratoires_select['Type_Data']=='Site']


# Regrouper par projet
grouped = df_selected.groupby(['laboratoire','Type_Data','Latitude', 'Longitude'])['projet'].apply(list).reset_index()
grouped_contacts = df_contacts_selected.groupby(['Contact','Sigle structure'])['projet'].apply(list).reset_index()


if Unites:
    grouped_ = grouped[grouped['Type_Data']=='Labo']
    grouped_contacts_ = grouped_contacts
    data_sigles = df_selected['laboratoire'][df_selected['Type_Data']=='Labo'].values
    data_projet = df_selected['projet'][df_selected['Type_Data']=='Labo'].values
elif Sites:
    grouped_ = grouped[grouped['Type_Data']=='Site']
    grouped_contacts_ = grouped_contacts
    data_sigles = df_selected['laboratoire'][df_selected['Type_Data']=='Site'].values
    data_projet = df_selected['projet'][df_selected['Type_Data']=='Site'].values
elif Unites_Sites:
    grouped_ = grouped[grouped['Type_Data'].isin(['Labo','Site'])]
    grouped_contacts_ = grouped_contacts
    data_sigles = df_selected['laboratoire'][df_selected['Type_Data'].isin(['Labo','Site'])].values
    data_projet = df_selected['projet'][df_selected['Type_Data'].isin(['Labo','Site'])].values
else:
    grouped_ = pd.DataFrame()
    grouped_contacts_ = pd.DataFrame()
    data_sigles = []
    data_projet = []


###############################################################################################
########### COMPTES GENERAUX ##################################################################
###############################################################################################

st.title(f":grey[Cartographie des projets de FairCarboN]")
col1, col2 = st.columns(2)
with col1:
    st.metric(label='Nombre lieux représentées',value=len(grouped_))
with col2:
    st.metric(label='Nombre de contacts associés',value=len(grouped_contacts_))

#Calcul de la position initiale de la carto
if len(grouped_)==0:
    avg_lat = 45
    avg_long = 5
else:
    avg_lat = sum(grouped_['Latitude'])/len(grouped_)
    avg_long = sum(grouped_['Longitude'])/len(grouped_)


###############################################################################################
########### CARTOGRAPHIE & NUAGE DE MOTS ######################################################
###############################################################################################


col1, col2, col3 = st.columns((0.1,0.75,0.15))
with col1:
    #Unités ou Sites
    if len(grouped_)==0:
        pass
    else:
        # Assign the same frequency to each name
        frequencies = {name: 1 for name in grouped_['laboratoire'].values}

        # Generate the word cloud
        wordcloud = WordCloud(width=200, height=200, background_color='white', colormap='viridis').generate_from_frequencies(frequencies)

        # Display in sidebar
        #st.subheader("Nuage des noms d'unités ou sites")
        fig_n0, ax = plt.subplots(figsize=(1, 1))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_n0)

    #Contacts
    if len(grouped_contacts_)==0:
        pass
    else:
        # Assign the same frequency to each name
        frequencies = {name: 1 for name in grouped_contacts_['Contact'].values}

        # Generate the word cloud
        wordcloud = WordCloud(width=100, height=100, background_color='white', colormap='viridis').generate_from_frequencies(frequencies)

        # Display in sidebar
        #st.subheader("Nuage des noms de contacts")
        fig_n0b, ax = plt.subplots(figsize=(1, 1))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_n0b)

with col2:
    m = carto2(grouped_, avg_lat, avg_long, color_map2)
    st_folium(m, width=800)

###############################################################################################
########### LEGENDE CARTO #####################################################################
###############################################################################################
with col3:
    st.subheader("Légende")
    for projet in ordre_perso:
        couleur = color_map2.get(projet, "#cccccc")  # couleur par défaut si projet 
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="width: 15px; height: 15px; background-color: {couleur}; border-radius: 3px; margin-right: 10px;"></div>
                <span>{projet}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


###############################################################################################
########### ANALYSE PARTICIPATIONS ET EXCLUSIVITE #############################################
###############################################################################################
st.title(f":grey[Analyse des participations dans FairCarboN]")
col1, col2 = st.columns(2)

with col1:
    # Compte du nombre de projets pour chaque labo
    df_units = df_Labo_Site[df_Labo_Site['Type_Data']=="Labo"]
    lab_project_counts = df_units.groupby('laboratoire')['projet'].nunique().reset_index()
    lab_project_counts.columns = ['laboratoire', 'num_projects']

    # Merge count retour dans l'original DataFrame
    df = df_Labo_Site.merge(lab_project_counts, on='laboratoire')
    df['num_other_projects'] = df['num_projects'] - 1

    # Pivot to count labs per project by number of other projects
    summary = df.groupby(['projet', 'num_other_projects']).size().unstack(fill_value=0)

    # Sort projects by total number of labs
    summary = summary.loc[summary.sum(axis=1).sort_values(ascending=False).index]

    # Normalize rows to get proportions
    summary_prop = summary.div(summary.sum(axis=1), axis=0)

    # Melt dataframe to long format for Plotly
    summary_prop = summary_prop.reset_index().melt(id_vars='projet', var_name='num_other_projects', value_name='proportion')

    # Convert 'num_other_projects' to string for consistent sorting in plot
    summary_prop['num_other_projects'] = summary_prop['num_other_projects'].astype(str)

    summary_prop['projet'] = pd.Categorical(summary_prop['projet'], categories=ordre_perso, ordered=True)

    # Liste des valeurs uniques
    projects = summary_prop['projet'].unique()
    other_project_levels = sorted(summary_prop['num_other_projects'].unique())

    # Créer une trace par niveau d'implication
    traces = []
    for level in other_project_levels:
        df_level = summary_prop[summary_prop['num_other_projects'] == level]
        traces.append(go.Bar(
            y=df_level['projet'],
            x=df_level['proportion'],
            name=f"{level} autres projets",
            orientation='h'
        ))

    # Créer la figure
    fig2 = go.Figure(data=traces)
    fig2.update_layout(
        barmode='stack',
        yaxis=dict(categoryorder='array',categoryarray=ordre_perso),
        legend_title="Nombre d'autres implications",
        margin=dict(l=100, r=20, t=60, b=40),
        xaxis_title="Proportion parmi les unités membres du projet",
        yaxis_title="Projets"
    )

    st.subheader(":grey[Proportion d'exclusivité des Unités]")
    st.plotly_chart(fig2, use_container_width=True)

with col2:

    Contacts_counts = df_Contacts.groupby('Contact')['projet'].nunique().reset_index()
    Contacts_counts.columns = ['Contact', 'num_projects']

    df2 = df_Contacts.merge(Contacts_counts, on='Contact')
    df2['num_other_projects'] = df2['num_projects'] - 1

    summary2 = df2.groupby(['projet', 'num_other_projects']).size().unstack(fill_value=0)
    summary2 = summary2.loc[summary.sum(axis=1).sort_values(ascending=False).index]
    summary_prop2 = summary2.div(summary2.sum(axis=1), axis=0)
    summary_prop2 = summary_prop2.reset_index().melt(id_vars='projet', var_name='num_other_projects', value_name='proportion')
    summary_prop2['num_other_projects'] = summary_prop2['num_other_projects'].astype(str)

    summary_prop2['projet'] = pd.Categorical(summary_prop2['projet'], categories=ordre_perso, ordered=True)
    
    # Filtrer les projets sélectionnés
    filtered_df = summary_prop2[summary_prop2['projet'].isin(projets_selected)]

    # Obtenir les niveaux d'implication et les projets
    other_project_levels = sorted(filtered_df['num_other_projects'].unique())
    projects = filtered_df['projet'].unique()

    # Créer une trace par niveau d'implication
    traces = []
    for level in other_project_levels:
        df_level = filtered_df[filtered_df['num_other_projects'] == level]
        traces.append(go.Bar(
            y=df_level['projet'],
            x=df_level['proportion'],
            name=f"{level} autres projets",
            orientation='h'
        ))

    # Créer la figure
    fig2b = go.Figure(data=traces)
    fig2b.update_layout(
        barmode='stack',
        yaxis=dict(categoryorder='array',categoryarray=ordre_perso),
        legend_title="Nombre d'autres implications",
        xaxis_title="Proportion parmi les contacts du projet",
        yaxis_title="Projets",
        margin=dict(l=100, r=20, t=60, b=40)
    )

    st.subheader(f":grey[Proportion d'exclusivité des Contacts]")
    st.plotly_chart(fig2b, use_container_width=True)

###############################################################################################
########### ANALYSE CONTACTS ##################################################################
###############################################################################################

df_Contacts_ok = df_contacts_selected #[df_contacts_selected['Confiance']=="ok"]

counts = df_Contacts_ok.groupby(['projet', 'Fonction']).size().reset_index(name='Nombre')

# Calculer le total par projet
totals = counts.groupby('projet')['Nombre'].transform('sum')

# Ajouter une colonne proportion (en %)
counts['Proportion'] = counts['Nombre'] / totals * 100

# Liste des fonctions et des projets
fonctions = sorted(counts['Fonction'].unique())
projects = counts['projet'].unique()

counts['projet'] = pd.Categorical(counts['projet'], categories=ordre_perso, ordered=True)

# Créer une trace par fonction
traces = []
for fonction in fonctions:
    df_fct = counts[counts['Fonction'] == fonction]
    traces.append(go.Bar(
        y=df_fct['projet'],
        x=df_fct['Proportion'],
        name=fonction,
        orientation='h',
        #text=df_fct['Nombre'],
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(size=20)
    ))

# Créer la figure
fig_fonction = go.Figure(data=traces)
fig_fonction.update_layout(
    barmode='stack',
    yaxis=dict(categoryorder='array', categoryarray=ordre_perso),
    xaxis=dict(title='Proportion (%)', ticksuffix='%'),
    yaxis_title='Projet',
    legend_title='Fonction',
    margin=dict(l=100, r=20, t=60, b=40)
)

# Affichage dans Streamlit
col1, col2 = st.columns(2)
with col1:
    st.subheader(":grey[Répartition des fonctions par projet]")
    st.plotly_chart(fig_fonction, use_container_width=True)

###############################################################################################
########### ANALYSE TUTELLES ##################################################################
###############################################################################################

# Étape 1 : transformer la colonne 'Tutelles' en liste
df_Labo2['tutelle_list'] = df_Labo2['Tutelles'].str.split(r'\s*/\s*')

# Étape 2 : exploser les tutelles
df_exploded = df_Labo2.explode('tutelle_list')

# Étape 3 : normaliser les tutelles
df_exploded['tutelle_list'] = df_exploded['tutelle_list'].str.strip().str.upper()

# Étape 4 : filtrer les tutelles ciblées
tutelles_cibles = ['CNRS', 'INRAE', 'IRD', 'CIRAD']
df_exploded = df_exploded[df_exploded['tutelle_list'].isin(tutelles_cibles)]

# Étape 5 : compter le nombre de laboratoires par projet et tutelle
counts_tutelles = df_exploded.groupby(['projet', 'tutelle_list'])['Sigle structure'].nunique().reset_index(name='count')

# Appliquer l’ordre comme catégorie ordonnée
counts_tutelles['projet'] = pd.Categorical(counts_tutelles['projet'], categories=ordre_perso, ordered=True)

# Étape 7 : créer une trace par tutelle
traces_tutelles = []
for tutelle in tutelles_cibles:
    df_tut = counts_tutelles[counts_tutelles['tutelle_list'] == tutelle]
    traces_tutelles.append(go.Bar(
        y=df_tut['projet'],
        x=df_tut['count'],
        name=tutelle,
        orientation='h',
        #text=df_tut['count'],
        textposition='auto',
        hovertemplate='%{x} labos<br>Projet: %{y}<br>Tutelle: %{trace.name}<extra></extra>'
    ))

# Étape 8 : créer la figure
fig_tutelles = go.Figure(data=traces_tutelles)
fig_tutelles.update_layout(
    barmode='stack',
    yaxis=dict(
        categoryorder='array',
        categoryarray=ordre_perso
    ),
    xaxis_title='Nombre de laboratoires',
    yaxis_title='Projet',
    legend_title='Tutelle',
    margin=dict(l=100, r=20, t=60, b=40),
    template='plotly_white'
)

# Étape 9 : afficher dans Streamlit
st.subheader(":grey[Nombre de laboratoires affiliés à chaque tutelle par projet]")
st.plotly_chart(fig_tutelles, use_container_width=True)

###############################################################################################
########### ANALYSE DOMAINES SCIENTIFIQUES ####################################################
###############################################################################################

df_Labo2['domaines_list'] = df_Labo2['Domaine scientifique'].str.split(r'\s*/\s*')
df_domains = df_Labo2.explode('domaines_list')
df_domains['domaines_list'] = df_domains['domaines_list'].str.strip()

counts_domaines = df_domains.groupby(['projet', 'domaines_list'])['Sigle structure'].nunique().reset_index(name='count')

counts_domaines['projet'] = pd.Categorical(counts_domaines['projet'], categories=ordre_perso, ordered=True)

domaines_uniques = sorted(counts_domaines['domaines_list'].unique())
traces_domaines = []

for domaine in domaines_uniques:
    df_dom = counts_domaines[counts_domaines['domaines_list'] == domaine]
    traces_domaines.append(go.Bar(
        y=df_dom['projet'],
        x=df_dom['count'],
        name=domaine,
        orientation='h',
        #text=df_dom['count'],
        textposition='auto',
        hovertemplate='%{x} labos<br>Projet: %{y}<br>Domaine: %{trace.name}<extra></extra>'
    ))

fig_domains = go.Figure(data=traces_domaines)
fig_domains.update_layout(
    barmode='stack',
    yaxis=dict(
        categoryorder='array',
        categoryarray=ordre_perso
    ),
    xaxis_title='Nombre de laboratoires',
    yaxis_title='Projet',
    legend_title='Domaines scientifiques',
    margin=dict(l=100, r=20, t=60, b=40),
    template='plotly_white'
)

st.subheader(":grey[Nombre de laboratoires associés aux domaines scientifiques]")
st.plotly_chart(fig_domains, use_container_width=True)

###############################################################################################
########### ANALYSE LIENS PAR VISU GRAPHE #####################################################
###############################################################################################
st.title(f":grey[Analyse des liens dans FairCarboN]")
data = {
    'Sigles': data_sigles,
    'Projet': data_projet
}
df = pd.DataFrame(data)

# Creation du graphe
G = nx.Graph()

# Ajout de noeuds et lignes
for _, row in df.iterrows():
    nom = row['Sigles']
    projet = row['Projet']
    G.add_node(nom, type='unité')
    G.add_node(projet, type='project')
    G.add_edge(nom, projet)

# Création de la couche du graphe
pos = nx.spring_layout(G, seed=1, iterations=100)

# Noeuds séparés pour les projets et pour les unités, pour un affichage spécifique
project_x, project_y, project_text = [], [], []
unit_x, unit_y, unit_text = [], [], []

for node in G.nodes():
    x, y = pos[node]
    if G.nodes[node]['type'] == 'project':
        project_x.append(x)
        project_y.append(y)
        project_text.append(f"<b>{node}</b>")
    else:
        unit_x.append(x)
        unit_y.append(y)
        unit_text.append(node)

# Création des lignes
edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Préparation des Noeuds
unit_trace = go.Scatter(
    x=unit_x, y=unit_y,
    mode='markers+text',
    text=unit_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        color='gold',
        size=20,
        line_width=2
    ),
    textfont=dict(
        size=12,
        color='black'
    )
)

project_trace = go.Scatter(
    x=project_x, y=project_y,
    mode='markers+text',
    text=project_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        color='green',
        size=25,
        line_width=2
    ),
    textfont=dict(
        size=16,
        color='darkgreen'
    )
)

# Préparation de la figure
fig3 = go.Figure(
    data=[edge_trace, unit_trace, project_trace],
    layout=go.Layout(
        width=600,
        height=600,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)

# Affichage
st.subheader(f":grey[Liens entre unités ou sites / et projets]")
st.plotly_chart(fig3, use_container_width=True)