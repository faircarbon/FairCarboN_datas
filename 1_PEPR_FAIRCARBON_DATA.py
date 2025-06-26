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

def to_rgb_string(rgb_tuple):
    r, g, b = (int(255 * c) for c in rgb_tuple)
    return f"rgb({r}, {g}, {b})"


###############################################################################################
########### TRANSFORMATION FICHIER XLS ########################################################
###############################################################################################
@st.cache_data
def read_data(path):
    # Chemin vers le fichier Excel
    #fichier_excel = "Data\FairCarboN_Datas_V2.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=1,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")

    ######## NETTOYAGES EVENTUELS ######################

    # filtrer les lignes incomplètes
    df_filtré = df.dropna(subset=["Latitude", "Longitude","projet","laboratoire"])
    # Renommer les colonnes
    #df_filtré_renommé = df_filtré.rename(columns={
    #    "Acronyme projet": "projet",
    #    "Acronyme unité": "laboratoire"
    #})
    #df_filtré_renommé.to_csv("Data\FairCarboN_Datas_V2_renommé.csv", index=False)

    return df_filtré

# Charger les données
df_Labo_Site = read_data("Data\FairCarboN_Datas_Labo")

# Couleurs associées à chaque projet
projects = sorted(df_Labo_Site['projet'].unique())
laboratoires = sorted(df_Labo_Site['laboratoire'].unique())
colors = plt.cm.tab20.colors  # Palette de couleurs
project_color_map = {project: colors[i % len(colors)] for i, project in enumerate(projects)}

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################

# Choix Projet
Selection_projets = st.sidebar.multiselect('Projets',options=projects)

if len(Selection_projets)==0: #aucun choix
    df_selected = df_Labo_Site #le dataframe ne change pas, c'est l'original
    projets_selected = projects
else:
    df_selected = df_Labo_Site[df_Labo_Site['projet'].isin(Selection_projets)]
    projets_selected = Selection_projets

laboratoires_select = df_selected[['laboratoire','Type_Data']]
laboratoires_bis_Unites = laboratoires_select[laboratoires_select['Type_Data']=='Labo']
laboratoires_bis_sites = laboratoires_select[laboratoires_select['Type_Data']=='Site']


# Regrouper par laboratoire
grouped = df_selected.groupby(['laboratoire','Type_Data','Latitude', 'Longitude'])['projet'].apply(list).reset_index()

col1, col2, col3 =st.sidebar.columns(3)
with col1:
    Unites = st.checkbox('Unités')
with col2:
    Sites = st.checkbox('Sites')
with col3:
    Unites_Sites = st.checkbox('Unités & Sites')

if Unites:
    grouped_ = grouped[grouped['Type_Data']=='Labo']
    data_sigles = df_selected['laboratoire'][df_selected['Type_Data']=='Labo'].values
    data_projet = df_selected['projet'][df_selected['Type_Data']=='Labo'].values
elif Sites:
    grouped_ = grouped[grouped['Type_Data']=='Site']
    data_sigles = df_selected['laboratoire'][df_selected['Type_Data']=='Site'].values
    data_projet = df_selected['projet'][df_selected['Type_Data']=='Site'].values
elif Unites_Sites:
    grouped_ = grouped[grouped['Type_Data'].isin(['Labo','Site'])]
    data_sigles = df_selected['laboratoire'][df_selected['Type_Data'].isin(['Labo','Site'])].values
    data_projet = df_selected['projet'][df_selected['Type_Data'].isin(['Labo','Site'])].values
else:
    grouped_ = pd.DataFrame()
    data_sigles = []
    data_projet = []

st.sidebar.metric(label='Nombre lieux représentées',value=len(grouped_))

if len(grouped_)==0:
    avg_lat = 45
    avg_long = 5
else:
    avg_lat = sum(grouped_['Latitude'])/len(grouped_)
    avg_long = sum(grouped_['Longitude'])/len(grouped_)



###############################################################################################
########### NUAGE DE MOTS #####################################################################
###############################################################################################

if len(grouped_)==0:
    pass
else:
    # Assign the same frequency to each name
    frequencies = {name: 1 for name in grouped_['laboratoire'].values}

    # Generate the word cloud
    wordcloud = WordCloud(width=300, height=300, background_color='white', colormap='viridis').generate_from_frequencies(frequencies)

    # Display in sidebar
    st.sidebar.title("Nuage des noms d'unités ou sites")
    fig0, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.sidebar.pyplot(fig0)


###############################################################################################
########### CARTOGRAPHIE ######################################################################
###############################################################################################

st.title("Carte interactive FAIRCARBON")
st.cache_resource
def carto(grouped_, avg_lat, avg_long):
    # Créer la carte
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=5, tiles='CartoDB positron',  # Ou 'Stamen Toner Lite'
        control_scale=True)  # barycentrée

    # Générer des marqueurs en camembert
    for _, row in grouped_.iterrows():
        projets = row['projet']
        latitude = row['Latitude']
        longitude = row['Longitude']
        type_data = row['Type_Data']

        if type_data == "Labo":
            # Créer un graphique en camembert
            fig, ax = plt.subplots(figsize=(1, 1))
            projet_counts = [1] * len(projets)  # égale pondération
            colors_used = [project_color_map[proj] for proj in projets]
            #ax.pie(projet_counts, colors=colors_used) version sans bordure
            wedges, _ = ax.pie(
                projet_counts,
                colors=colors_used,
                wedgeprops={'edgecolor': 'black', 'linewidth': 5}  # Bordure noire épaisse
            )
            plt.axis('off')

            # Sauvegarder en mémoire
            img_data = BytesIO()
            plt.savefig(img_data, format='png', bbox_inches='tight', transparent=True)
            plt.close(fig)
            img_data.seek(0)
            encoded = base64.b64encode(img_data.read()).decode()

            icon_url = f"data:image/png;base64,{encoded}"
            icon = folium.CustomIcon(icon_image=icon_url, icon_size=(35, 35))
        
        elif type_data == "Site":
            # Créer un graphique en camembert
            fig, ax = plt.subplots(figsize=(1, 1))
            projet_counts = [1] * len(projets)  # égale pondération
            colors_used = [project_color_map[proj] for proj in projets]
            #ax.pie(projet_counts, colors=colors_used) version sans bordure
            wedges, _ = ax.pie(
                projet_counts,
                colors=colors_used,
                wedgeprops={'edgecolor': 'red', 'linewidth': 5}  # Bordure rouge épaisse
            )
            plt.axis('off')

            # Sauvegarder en mémoire
            img_data = BytesIO()
            plt.savefig(img_data, format='png', bbox_inches='tight', transparent=True)
            plt.close(fig)
            img_data.seek(0)
            encoded = base64.b64encode(img_data.read()).decode()

            icon_url = f"data:image/png;base64,{encoded}"
            icon = folium.CustomIcon(icon_image=icon_url, icon_size=(30, 30))

        # Ajouter le marqueur
        popup = folium.Popup("<br>".join(projets), max_width=200)
        tooltip = row['laboratoire']
        folium.Marker(location=[latitude, longitude], popup=popup, tooltip=tooltip, icon=icon).add_to(m)

    return m

col1, col2 = st.columns((0.8,0.2))
with col1:
    m = carto(grouped_, avg_lat, avg_long)
    st_folium(m, width=800)

###############################################################################################
########### LEGENDE CARTO #####################################################################
###############################################################################################

with col2:
    st.subheader("Légende")
    for i in range(len(projects)):
        rgb_css = to_rgb_string(colors[i])
        st.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<div style="width: 15px; height: 15px; background-color: {rgb_css}; border-radius: 3px; margin-right: 10px;"></div>'
            f'<span>{projects[i]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )


###############################################################################################
########### NOMBRE LABOS PAR PROJET #######################################################
###############################################################################################
col1 , col2 = st.columns(2)
with col1:
    compte_labos_par_projet = df_Labo_Site["projet"][df_Labo_Site['Type_Data']=="Labo"].value_counts()

    # Create a list of colors (one per project)
    colors = px.colors.qualitative.Set3 # Or use px.colors.qualitative.* for more sets
    project_names = compte_labos_par_projet.index
    color_map = {project: colors[i % len(colors)] for i, project in enumerate(project_names)}

    # Assign colors based on the project
    bar_colors = [color_map[project] for project in project_names]

    # Plot
    fig0 = go.Figure(go.Bar(
        x=compte_labos_par_projet.values,
        y=project_names,
        orientation='h',
        marker_color=bar_colors  # Assign custom colors
    ))

    st.subheader("Nombre d'unités impliquées")
    st.plotly_chart(fig0, use_container_width=True)

with col2:
    compte_sites_par_projet = df_Labo_Site["projet"][df_Labo_Site['Type_Data']=="Site"].value_counts()

    # Create a list of colors (one per project)
    colors = px.colors.qualitative.Set3 # Or use px.colors.qualitative.* for more sets
    project_names = compte_sites_par_projet.index
    color_map = {project: colors[i % len(colors)] for i, project in enumerate(project_names)}

    # Assign colors based on the project
    bar_colors = [color_map[project] for project in project_names]

    # Plot
    fig0b = go.Figure(go.Bar(
        x=compte_sites_par_projet.values,
        y=project_names,
        orientation='h',
        marker_color=bar_colors  # Assign custom colors
    ))

    st.subheader("Nombre de sites étudiés")
    st.plotly_chart(fig0b, use_container_width=True)

###############################################################################################
########### ANALYSE LABOS MULTI PROJETS #######################################################
###############################################################################################

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

# Plotly stacked bar chart
fig2 = px.bar(
            summary_prop[summary_prop['projet'].isin(projets_selected)],
            x='proportion',
            y='projet',
            color='num_other_projects',
            orientation='h',
            #title="Proportion d'exclusivité des membres des projets de FairCarboN",
            labels={'proportion': 'Proportion parmi les unités membres du projet', 'projet': 'Projets', 'num_other_projects': 'Nombre autres projets'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )

fig2.update_layout(
            yaxis=dict(autorange="reversed"),  # Projects from top-down
            legend_title="Nombre d'autres implications",
            #height=25 * len(summary) + 200,  # Dynamically adjust height
            margin=dict(l=100, r=20, t=60, b=40)
        )

st.subheader("Proportion d'exclusivité des membres des projets de FairCarboN")
st.plotly_chart(fig2, use_container_width=True)

###############################################################################################
########### ANALYSE LABOS MULTI PROJETS BIS ###################################################
###############################################################################################


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
st.subheader("Liens entre unités et projets")
st.plotly_chart(fig3, use_container_width=True)