import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go


###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON DATA GLOBAL",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

###############################################################################################
########### RECUPERATION DES DATAFRAMES #######################################################
###############################################################################################

df_hal = st.session_state['df_hal']
df_rdg = st.session_state['df_rdg']
df_indores = st.session_state['df_InDoRes']
df_zenodo = st.session_state['df_zenodo']

mots_cles_recherches = ['pepr faircarbon','faircarbon','alamod','slam-b','rift','crosyen','greenscale','canete','carbonium','deep-c','climfas','rhizoseqc','cabestan','tropecos','peace','prefalim','co2cmphi']

df_hal_reduit = df_hal[['Nom_archive','Auteur_recherché','Uri','Titre_unique','Date de publication','Mots_clés','Type de document','ANR project acronyme','DOI sources','In_FairCarboN','Sollicitation']]
df_rdg_reduit = df_rdg[['Nom_archive','Auteur_recherché','Titre_unique','Mots_clés','DOI sources','Date de publication','Type de document']]
df_indores_reduit = df_indores[['Nom_archive','Auteur_recherché','Titre_unique','Mots_clés','DOI sources','Date de publication','Type de document']]
df_zenodo_reduit = df_zenodo[['Nom_archive','Auteur_recherché','Titre_unique','Date de publication','Type de document']]

df_concat = pd.concat([df_hal_reduit,df_rdg_reduit,df_indores_reduit,df_zenodo_reduit], axis=0)
df_concat['In_FairCarboN'] = df_concat['In_FairCarboN'].fillna(False)
df_concat['Type de document'] = df_concat['Type de document'].fillna('DATASET')
df_concat['Sollicitation'] = df_concat['Sollicitation'].fillna('NON')
df_concat['Mots_clés'] = df_concat['Mots_clés'].apply(lambda x: [] if not isinstance(x, list) else x)
df_concat['Mots_clés'] = df_concat['Mots_clés'].apply(lambda lst: [mot.lower() for mot in lst])
df_concat['ANR project acronyme'] = df_concat['ANR project acronyme'].apply(lambda x: [] if not isinstance(x, list) else x)
df_concat['ANR project acronyme'] = df_concat['ANR project acronyme'].apply(lambda lst: [mot.lower() for mot in lst])
df_concat['Référencement par mots clés'] = df_concat['Mots_clés'].apply(
    lambda liste: any(mot in liste for mot in mots_cles_recherches))
df_concat['Projet ANR dans FairCarboN'] = df_concat['ANR project acronyme'].apply(
    lambda liste: any(mot in liste for mot in mots_cles_recherches))
df_concat["Contient Zenodo"] = df_concat["DOI sources"].apply(
    lambda lst: any("zenodo" in s for s in lst) if isinstance(lst, list) else False
)
df_concat["Contient hal"] = df_concat["DOI sources"].apply(
    lambda lst: any("hal" in s for s in lst) if isinstance(lst, list) else False
)

def is_empty_list(x):
    return isinstance(x, list) and len(x) == 0

def extract_pattern(uri):
    if not isinstance(uri, str):
        return None
    match = re.match(r'https://([^/]+)', uri)
    return match.group(1) if match else None

df_concat["projet ANR à vérifier"] = df_concat["ANR project acronyme"].apply(is_empty_list)  & (df_concat["In_FairCarboN"]==False)
df_concat['Pattern'] = df_concat['Uri'].apply(extract_pattern)

###############################################################################################
########### VISUALISATIONS INDIVIDUELLES #######################################################
###############################################################################################
if 'count' not in st.session_state:
    st.session_state.count = 0
def increment_counter():
    st.session_state.count += 1
def reset_counter():
    st.session_state.count = 0


df_referencé = df_concat[df_concat['Référencement par mots clés']==True]

p = set(df_concat['Auteur_recherché'].values)


col1, col2, col3, col4 = st.columns([0.8,0.05,0.05,0.1])
with col1:
    try:
        Selection_p = st.selectbox(label='Selection', options=p, index=st.session_state.count)
    except:
        Selection_p = ""
        reset_counter()
with col2:
    st.markdown('')
    st.markdown('')
    button1 = st.button(':heavy_plus_sign:',on_click=increment_counter)
with col3:
    st.markdown('')
    st.markdown('')
    button2 =st.button('R',on_click=reset_counter)
with col4:
    if df_concat['Sollicitation'][df_concat['Auteur_recherché']==Selection_p].values[0]=='NON':
        st.image('Data/nok.png', width=80, caption="Sollicitation")
    else:
        st.image('Data/ok.png', width=80, caption="Sollicitation")

df_selected = df_concat[df_concat['Auteur_recherché']==Selection_p]
df_selected.reset_index(inplace=True)
df_selected.drop(columns='index', inplace=True)

if st.session_state.count > len(p):
        st.session_state.count = 0

colA, colB =st.columns(2)
with colA:
    col1, col2 =st.columns(2)
    with col1:
        st.subheader(f":grey[Publications HAL]")
        st.metric(label='', value=len(df_selected[df_selected['Nom_archive']=='HAL']))
    with col2:
        st.subheader(f":grey[Dans FairCarboN]")
        st.metric(label='', value=len(df_selected[df_selected['Nom_archive']=='HAL'][df_selected['In_FairCarboN']==True]))
    df_selected_hal = df_selected[df_selected['Nom_archive']=='HAL']
    if len(df_selected_hal)>0:
        row_counts_hal = df_selected_hal['Pattern'].value_counts().reset_index()
        row_counts_hal.columns = ['Archive HAL', 'compte']

                # Calcul du total et du pourcentage
        total = row_counts_hal['compte'].sum()
        # Calcul des pourcentages
        total = row_counts_hal['compte'].sum()
        row_counts_hal['pourcentage'] = (row_counts_hal['compte'] / total) * 100

        # Génération des étiquettes conditionnelles
        labels = row_counts_hal['Archive HAL']
        values = row_counts_hal['compte']
        text_labels = [
            f"{pct:.1f}%" if pct > 1 else "" 
            for label, pct in zip(labels, row_counts_hal['pourcentage'])
        ]

        # Création du graphique avec go.Figure
        fig_datasets = go.Figure(
            data=[go.Pie(
                labels=labels,
                values=values,
                text=text_labels,
                textinfo='text',  # N'affiche que text, donc rien si vide
                hoverinfo='percent+value',
                hole=0.3,
                marker=dict(colors=px.colors.qualitative.Set3)
            )]
        )

        fig_datasets.update_layout(
            title='Répartition des publications sur différents guichets HAL',
            showlegend=True
        )
        st.plotly_chart(fig_datasets,use_container_width=True)

        df_selected_hal['Value']=1 
        df_unique = df_selected_hal[['Titre_unique', 'Date de publication', 'Value', 'Type de document']].drop_duplicates()
        df_yearly = df_unique.groupby(['Date de publication', 'Type de document'])['Value'].sum().reset_index()

        # Créer le graphique
        title = f"Communications rattachées à {df_selected_hal['Auteur_recherché'].values[0]}"
        fig_dates = px.bar(
            df_yearly,
            x='Date de publication',
            y='Value',
            color='Type de document',
            title=title,
            barmode='stack'  # ou 'group' pour barres côte à côte
        )

        fig_dates.update_xaxes(
                                tickmode='linear',     
                                dtick=1,          # intervalle d’un an
                                tickformat='d'    # format entier (pas de virgule, ni décimales)
                            )

        # Afficher dans Streamlit
        st.plotly_chart(fig_dates, use_container_width=True)

    else:
        df_yearly = pd.DataFrame()
    

with colB:
    st.subheader(f":grey[Datasets ouverts]")
    liste_entrepots = ['Recherche Data Gouv','Zenodo', 'Data InDoRes']
    df_selected_datasets = df_selected[df_selected['Nom_archive'].isin(liste_entrepots)]
    st.metric(label='', value=len(df_selected_datasets))
    if len(df_selected_datasets)>0:
        row_counts_datasets = df_selected_datasets['Nom_archive'].value_counts().reset_index()
        row_counts_datasets.columns = ['Archive', 'compte']
        fig_datasets = px.pie(
                                row_counts_datasets,
                                names='Archive',
                                values='compte',
                                title='Répartition des datasets déposés',
                                color_discrete_sequence=px.colors.qualitative.Set3,
                                hole=0.3  
                            )
        st.plotly_chart(fig_datasets,use_container_width=True)

        df_selected_datasets['Value']=1    
        df_unique_datasets = df_selected_datasets[['Titre_unique','Date de publication','Value','Type de document']].drop_duplicates()
        # Agréger les valeurs par année
        df_yearly_datasets = df_unique_datasets.groupby(['Date de publication', 'Type de document'])['Value'].sum().reset_index()

        # Créer le graphique
        fig_dates_datasets = px.bar(df_yearly_datasets, 
                                    x='Date de publication', 
                                    y='Value', 
                                    title=f'Dépôts rattachés à {df_selected_datasets["Auteur_recherché"].values[0]}',
                                    color='Type de document')
        
        if len(df_yearly)>0:
            fig_dates_datasets.update_xaxes(
                            tickmode='linear',     
                            dtick=1,          # intervalle d’un an
                            tickformat='d',    # format entier (pas de virgule, ni décimales)
                            range=[min(df_yearly['Date de publication'].values)-0.5, 2025 + 0.5]
                        )
        
            fig_dates_datasets.update_yaxes(
                                range=[0,max(df_yearly.groupby('Date de publication')['Value'].sum())]
                            )
        
        st.plotly_chart(fig_dates_datasets, use_container_width=True)
    else:
        pass

st.dataframe(df_selected[['Auteur_recherché','Nom_archive', 'Date de publication','Titre_unique']], hide_index=True)
###############################################################################################
########### AUTRES ANALYSES ###################################################################
###############################################################################################

tous_doi_sources_hal = sum(df_hal_reduit['DOI sources'], [])
tous_doi_sources_rdg = sum(df_rdg_reduit['DOI sources'], [])

# Enlever les doublons en gardant l'ordre
liste_doi_sources_hal = list(dict.fromkeys(tous_doi_sources_hal))
liste_doi_sources_rdg = list(dict.fromkeys(tous_doi_sources_rdg))
liste3 = [element for element in liste_doi_sources_hal if element in liste_doi_sources_rdg]


