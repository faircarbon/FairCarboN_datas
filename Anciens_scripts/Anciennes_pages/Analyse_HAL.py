import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from Publications import afficher_publications_hal
import datetime
import ast

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON HAL DATA",
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
def intersect_lists(row):
    return list(set(row['Labo_filter2']) & set(row['Labo_']))


def filtre_labo1(row):
    try:
        return [item for item in row['Labo_all'] if row['Auteur_recherché'] in item]
    except:
        return []

# Fonction pour extraire le suffixe après le dernier '_'
def filtre_labo2(liste):
    try:
        return [item.split('_')[-1] for item in liste]
    except:
        return []
    
def extraire_doi(cellule):
    morceaux = cellule.split(';')
    for m in morceaux:
        if m.strip().startswith('10.'):
            return m.strip()
    return ""  # ou "" si tu préfères une chaîne vide

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
df = read_data("Data\FairCarboN_Datas_Contacts")
d = datetime.date.today()
start_year=2024
end_year=d.year
#st.slider(label='Choix plage de dates',min_value=2020, max_value=2025)
liste_chercheurs = df['Contact']
liste_projet = df['projet']
liste_sollicitation = df['Sollicitation']
liste_labs = df['Sigle structure']


df_global_hal = pd.read_csv("Data/HAL/all_publications_hal_2025-11-12.csv")
filtered_df = df_global_hal[df_global_hal['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]
df_inter = pd.read_csv("Data/HAL/all_publications_hal_2025-11-12.csv")
st.session_state['df_hal'] = df_inter

###############################################################################################
########### VISUALISATION GENERALE ############################################################
###############################################################################################
st.title(f":grey[Etude des publications sur HAL]")
col1,col2 = st.columns(2)

with col1:
    st.metric(label="Nombre de contacts étudiés", value=len(set(liste_chercheurs)))
    st.metric(label="Nombre de dépôts HAL global", value=len(set(df_global_hal['Ids'].values)))
    st.metric(label="Nombre de dépôts HAL dans la collection FairCarboN", value=len(set(filtered_df['Ids'].values)))

with col2:
    st.metric(label="Nombre de contacts trouvés dans HAL", value=len(set(df_global_hal['Auteur_recherché'])))
    st.metric(label="Nombre d'articles global", value=len(set(df_global_hal['Ids'][df_global_hal['Type de document']=="ART"].values)))
    st.metric(label="Nombre d'articles dans la collection FairCarboN", value=len(set(filtered_df['Ids'][filtered_df['Type de document']=="ART"].values)))

df_unique = df_inter[['Ids','Date de publication','Value']].drop_duplicates()
# Agréger les valeurs par année
df_yearly = df_unique.groupby('Date de publication')['Value'].sum().reset_index()

# Créer le graphique
fig_dates = px.bar(df_yearly, x='Date de publication', y='Value', title='Dépôts rattachés aux contacts FaircarboN', color_discrete_sequence=['green'])
fig_dates.update_layout(
    xaxis_title="Date du dépôt sur HAL",
)
st.plotly_chart(fig_dates, use_container_width=True)

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################

projets = list(set(df_global_hal['Projet']))
auteurs = list(set(df_global_hal['Auteur_recherché']))
col1,col2 = st.columns(2)
with col1:
    st.subheader(f":grey[Choix du/des projet(s) visualisé(s)]")
    choix_projet = st.multiselect(label='', options=projets )
    if len(choix_projet)==0:
        choix_p = projets
    else:
        choix_p = choix_projet
with col2:
    st.subheader(f":grey[Choix de(s) l'auteur(e(s)) visualisé(e(s))]")
    choix_auteur = st.multiselect(label='', options=list(set(df_global_hal['Auteur_recherché'][df_global_hal['Projet'].isin(choix_p)])))
    if len(choix_auteur)==0:
        choix_a = df_global_hal['Auteur_recherché'][df_global_hal['Projet'].isin(choix_p)]
    else:
        choix_a = choix_auteur


df_global_hal_proj =df_global_hal[df_global_hal['Projet'].isin(choix_p)][df_global_hal['Auteur_recherché'].isin(choix_a)][df_global_hal['Date de publication']>=start_year]
df_global_hal_proj.reset_index(inplace=True)
df_global_hal_proj.drop(columns='index', inplace=True)
ifc = df_global_hal_proj['Ids'][df_global_hal_proj['In_FairCarboN']==True].drop_duplicates()
In_FC = len(ifc)

col1,col2,col3 = st.columns([0.25,0.25,0.5])
with col1:
    st.metric(label=f'Nombre de dépôts dans HAL depuis {start_year}',value=len(set(df_global_hal_proj['Ids'][df_global_hal_proj['Date de publication']>=start_year])))
with col2:
    st.metric(label="dans la collection FairCarboN", value=In_FC)
with col3:
    st.metric(label=f"Nombre d'auteur(e)s ayant publié depuis {start_year}", value=len(list(set(df_global_hal_proj['Auteur_recherché'][df_global_hal_proj['Date de publication']>=start_year]))))


st.dataframe(df_global_hal_proj)
###############################################################################################
########### PREPARATIONS PARETOS ############################################################
###############################################################################################

unique_projet_titles = df_global_hal_proj[['Projet','Titre_unique']].drop_duplicates(subset=['Projet','Titre_unique'])
projects_count = unique_projet_titles['Projet'].value_counts().reset_index()
projects_count.columns = ['Projet', 'compte']

unique_person_titles = df_global_hal_proj[['Auteur_recherché','Titre_unique']].drop_duplicates(subset=['Auteur_recherché','Titre_unique'])
row_counts = unique_person_titles['Auteur_recherché'].value_counts().reset_index()
row_counts.columns = ['Auteur', 'compte']

unique_labo_titles = df_global_hal_proj[['Sigle structure','Titre_unique']].drop_duplicates(subset=['Sigle structure','Titre_unique'])
labo_count = unique_labo_titles['Sigle structure'].value_counts().reset_index()
labo_count.columns = ['Labo', 'compte']
top10_labo_count = labo_count.sort_values(by='compte', ascending=False).head(20)

df_pareto = labo_count.sort_values(by='compte', ascending=False).reset_index(drop=True)
df_pareto['cum_percentage'] = df_pareto['compte'].cumsum() / df_pareto['compte'].sum() * 100

# 3. Limit to top N labs
top_n = 20
df_pareto_plot = df_pareto.head(top_n)

unique_auteurs_titles = df_global_hal_proj[['Sigle structure','Auteur_recherché']].drop_duplicates(subset=['Sigle structure','Auteur_recherché'])
labo_count2 = unique_auteurs_titles['Sigle structure'].value_counts().reset_index()
labo_count2.columns = ['Labo', 'compte']
top10_labo_count2 = labo_count2.sort_values(by='compte', ascending=False).head(20)

# Sort by count descending
df_pareto2 = labo_count2.sort_values(by='compte', ascending=False).reset_index(drop=True)
df_pareto2['cum_percentage'] = df_pareto2['compte'].cumsum() / df_pareto2['compte'].sum() * 100

# Limit to top N labs for readability
df_pareto_plot2 = df_pareto2.head(top_n)


###################################################################################################################################
fig = px.pie(
    projects_count,
    names='Projet',
    values='compte',
    title='Répartition des publications parmi les membres des projets',
    color_discrete_sequence=px.colors.qualitative.Set3,
    hole=0.3  
)

fig1 = px.pie(
    projects_count,
    names='Projet',
    values='compte',
    title='Participation aux projets',
    color_discrete_sequence=px.colors.qualitative.Set3,
    hole=0.3
)
fig1.update_traces(textinfo='label')
fig1.update_layout(showlegend=False)

# Box plot using Plotly
fig2 = px.box(row_counts, y='compte', points="all",hover_data=['Auteur'], title="Distribution du nombre de publications parmi ces membres")
fig2.update_traces(marker_color='tomato', line_color='tomato')

fig_pareto_pub = go.Figure()

# Bar: publication count
fig_pareto_pub.add_trace(go.Bar(
    x=df_pareto_plot['Labo'],
    y=df_pareto_plot['compte'],
    name='Nombre de publications',
    marker_color="green",
    yaxis='y1'
))

# Line: cumulative percentage
fig_pareto_pub.add_trace(go.Scatter(
    x=df_pareto_plot['Labo'],
    y=df_pareto_plot['cum_percentage'],
    name='Pourcentage cumulé',
    yaxis='y2',
    mode='lines+markers',
    marker=dict(color='black'),
    line=dict(width=2)
))

# 5. Layout with dual axes
fig_pareto_pub.update_layout(
    title='Pareto des publications par labo (Top 20)',
    xaxis=dict(title='Labo'),
    yaxis=dict(title='Nombre de publications', side='left'),
    yaxis2=dict(
        title='Pourcentage cumulé (%)',
        overlaying='y',
        side='right',
        range=[0, 110]
    ),
    legend=dict(x=1.1, y=0.85),
    margin=dict(l=40, r=40, t=60, b=80),
    height=500
)

for i, row in df_pareto_plot.iterrows():
    fig_pareto_pub.add_annotation(
        x=row['Labo'],
        y=row['cum_percentage'],
        yref='y2',
        text=f"{row['cum_percentage']:.1f}%",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-20,
        font=dict(size=10, color='black'),
        arrowcolor='black',
        align='center'
    )

# Create bar + line plot (Pareto)
fig_pareto = go.Figure()

# Bar for publication counts
fig_pareto.add_trace(go.Bar(
    x=df_pareto_plot2['Labo'],
    y=df_pareto_plot2['compte'],
    name='Nombre d\'auteurs',
    marker_color='slateblue',
    yaxis='y1'
))

# Line for cumulative %
fig_pareto.add_trace(go.Scatter(
    x=df_pareto_plot2['Labo'],
    y=df_pareto_plot2['cum_percentage'],
    name='Pourcentage cumulé',
    yaxis='y2',
    mode='lines+markers',
    marker=dict(color='tomato'),
    line=dict(width=2)
))

# Layout with dual axes
fig_pareto.update_layout(
    title="Pareto des publications avec nombre de contacts par labo (Top 20)",
    xaxis=dict(title='Labo'),
    yaxis=dict(title='Nombre d\'auteurs', side='left'),
    yaxis2=dict(
        title='Pourcentage cumulé (%)',
        overlaying='y',
        side='right',
        range=[0, 110]
    ),
    legend=dict(x=1.1, y=0.85),
    margin=dict(l=40, r=40, t=60, b=80),
    height=500
)

for i, row in df_pareto_plot2.iterrows():
    fig_pareto.add_annotation(
        x=row['Labo'],
        y=row['cum_percentage'],
        yref='y2',
        text=f"{row['cum_percentage']:.1f}%",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-20,
        font=dict(size=10, color='tomato'),
        arrowcolor='tomato',
        align='center'
    )


###############################################################################################
######################## AFFICHAGE ############################################################
###############################################################################################
col1,col2 = st.columns(2)
with col1:
    if len(choix_auteur)==0:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig1, use_container_width=True)
    
with col2:
    st.plotly_chart(fig2, use_container_width=True)

st.plotly_chart(fig_pareto_pub, use_container_width=True)
st.plotly_chart(fig_pareto, use_container_width=True)

# Exemple de requête
#Liste_chercheurs = ['Olivier Bornet']
#requete_api_hal = f'http://api.archives-ouvertes.fr/search/?q=text:"{Liste_chercheurs[0].lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s'
#reponse = requests.get(requete_api_hal, timeout=5)
#test_liste_coll=[]
#st.write(reponse.json()['response']['docs'][4])
#test_liste_coll.append(reponse.json()['response']['docs'][0]['collCode_s'])
#st.write(test_liste_coll)

# Définir les catégories et les valeurs
labels = [
    "Dépôt HAL depuis 2024",
    "Pas de dépôts HAL depuis 2024",
    "Aucun dépôt HAL identifié"
]

values = [
    412,              # Dépôts depuis 2024
    427 - 412,        # Dépôts uniquement avant 2024
    474 - 427         # Aucun dépôt HAL
]

# Couleurs personnalisées (tu peux les adapter)
custom_colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # vert, orange, rouge

# Créer le pie chart
fig_usagehal = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    textinfo='label+percent',
    hoverinfo='label+value',
    marker=dict(colors=custom_colors, line=dict(color='#000000', width=1))
)])

fig_usagehal.update_layout(
    title="Statistiques de dépôt HAL",
    template="plotly_white"
)

st.plotly_chart(fig_usagehal, use_container_width=True)