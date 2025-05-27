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
def read_data():
    # Chemin vers le fichier Excel
    fichier_excel = "Data\FairCarboN_RNSR_copie.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(fichier_excel, sheet_name=1,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv("Data\FairCarboN_RNSR_copie.csv", index=False, encoding="utf-8")

    ######## NETTOYAGES EVENTUELS ######################

    # filtrer les lignes incomplètes
    df_filtré = df.dropna(subset=["Latitude", "Longitude","Acronyme projet","Acronyme unité"])
    # Renommer les colonnes
    df_filtré_renommé = df_filtré.rename(columns={
        "Acronyme projet": "projet",
        "Acronyme unité": "laboratoire"
    })
    df_filtré_renommé.to_csv("Data\FairCarboN_RNSR_copie_filtré_renommé.csv", index=False)

    return df_filtré_renommé

# Charger les données
df = read_data()

# Couleurs associées à chaque projet
projects = sorted(df['projet'].unique())
laboratoires = sorted(df['laboratoire'].unique())
colors = plt.cm.tab20.colors  # Palette de couleurs
project_color_map = {project: colors[i % len(colors)] for i, project in enumerate(projects)}

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################
st.sidebar.title('Filtrage')
Selection_projets = st.sidebar.multiselect('Projets',options=projects)
if len(Selection_projets)==0:
    df_selected = df
    projets_selected = projects
else:
    df_selected = df[df['projet'].isin(Selection_projets)]
    projets_selected = Selection_projets

Selection_laboratoires = st.sidebar.multiselect('Unités',options=laboratoires)
if len(Selection_laboratoires)==0:
    df_selected_ = df_selected
else:
    df_selected_ = df[df['laboratoire'].isin(Selection_laboratoires)]

# Regrouper par laboratoire
grouped = df_selected_.groupby(['laboratoire','Type_Data','Latitude', 'Longitude'])['projet'].apply(list).reset_index()

avg_lat = sum(df_selected_['Latitude'])/len(df_selected_)
avg_long = sum(df_selected_['Longitude'])/len(df_selected_)

st.sidebar.metric(label='Nombre Unités représentées',value=len(grouped))

###############################################################################################
########### NUAGE DE MOTS #####################################################################
###############################################################################################

# Assign the same frequency to each name
frequencies = {name: 1 for name in df_selected_['laboratoire'].values}

# Generate the word cloud
wordcloud = WordCloud(width=300, height=300, background_color='white', colormap='viridis').generate_from_frequencies(frequencies)

# Display in sidebar
st.sidebar.title("Nuage des noms d'Unités")
fig0, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.sidebar.pyplot(fig0)


###############################################################################################
########### CARTOGRAPHIE ######################################################################
###############################################################################################

st.title("Carte interactive FAIRCARBON")
st.cache_resource
def carto(grouped, avg_lat, avg_long):
    # Créer la carte
    m = folium.Map(location=[avg_lat, avg_long], zoom_start=2, tiles='CartoDB positron',  # Ou 'Stamen Toner Lite'
        control_scale=True)  # barycentrée

    # Générer des marqueurs en camembert
    for _, row in grouped.iterrows():
        projets = row['projet']
        latitude = row['Latitude']
        longitude = row['Longitude']
        type_data = row['Type_Data']

        if type_data == "Analyses":
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
        
        elif type_data == "Modelisation":
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
    m = carto(grouped, avg_lat, avg_long)
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
########### ANALYSE LABOS MULTI PROJETS #######################################################
###############################################################################################

# Count number of unique projects per lab
lab_project_counts = df.groupby('laboratoire')['projet'].nunique().reset_index()
lab_project_counts.columns = ['laboratoire', 'num_projects']

# Merge count back into original DataFrame
df = df.merge(lab_project_counts, on='laboratoire')
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

st.title("Proportion d'exclusivité des membres des projets de FairCarboN")
st.plotly_chart(fig2, use_container_width=True)