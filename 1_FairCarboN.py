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
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df_Labo_Site = read_data("Data\FairCarboN_Datas_Labo")
df_Contacts = read_data("Data\FairCarboN_Datas_Contacts")

# Couleurs associées à chaque projet
projects = sorted(df_Contacts['projet'].unique())
unites = sorted(df_Contacts['Sigle structure'].unique())
ordre_perso = ["ALAMOD","SLAM-B","RIFT","CrosyeN","CarboNium","CABESTAN","CANETE","DEEP-C","Drought for C",
                   "PEACE","TROPECOS","CLIM-FAS","CO2_CMPhi","GREENSCALE","PREFALIM","RhizoSeqC","PEPR"]

# Create a list of colors (one per project)
colors = px.colors.qualitative.Set3 # Or use px.colors.qualitative.* for more sets
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

st.write(colors)


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

st.cache_resource
def carto(grouped_, avg_lat, avg_long):
    # Créer la carte
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=1.5, tiles='CartoDB positron',  # Ou 'Stamen Toner Lite'
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
    m = carto(grouped_, avg_lat, avg_long)
    st_folium(m, width=800)

###############################################################################################
########### LEGENDE CARTO #####################################################################
###############################################################################################
colors2 = plt.cm.tab20.colors  # Palette de couleurs

with col3:
    st.subheader("Légende")
    for i in range(len(projects)):
        rgb_css = to_rgb_string(colors2[i])
        st.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<div style="width: 15px; height: 15px; background-color: {rgb_css}; border-radius: 3px; margin-right: 10px;"></div>'
            f'<span>{projects[i]}</span>'
            f'</div>',
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
        yaxis=dict(autorange='reversed'),
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
        yaxis=dict(autorange='reversed'),
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

# Créer une trace par fonction
traces = []
for fonction in fonctions:
    df_fct = counts[counts['Fonction'] == fonction]
    traces.append(go.Bar(
        y=df_fct['projet'],
        x=df_fct['Proportion'],
        name=fonction,
        orientation='h',
        text=df_fct['Nombre'],
        textposition='inside',
        insidetextanchor='start',
        textfont=dict(size=20)
    ))

# Créer la figure
fig_fonction = go.Figure(data=traces)
fig_fonction.update_layout(
    barmode='stack',
    yaxis=dict(categoryorder='total ascending'),
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