import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import pickle
from unidecode import unidecode
import datetime


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

def is_empty_list(val):
    # Si c'est une liste, vérifier si vide
    if isinstance(val, list):
        return len(val) == 0
    # Si c'est une chaîne vide
    elif isinstance(val, str):
        return val.strip() == ""
    # Si c'est NaN (float('nan')) ou None
    elif val is None or (isinstance(val, float) and pd.isna(val)):
        return True
    # Sinon, considérer que ce n'est pas vide
    else:
        return False

def extract_pattern(uri):
    if not isinstance(uri, str):
        return None
    match = re.match(r'https://([^/]+)', uri)
    return match.group(1) if match else None

def contient_mots_cles(val):
    if isinstance(val, str):
        # Recherche mots-clés dans la chaîne
        return any(mot in val for mot in mots_cles_recherches)
    elif isinstance(val, list):
        # Recherche mots-clés dans la liste (éléments convertis en str)
        return any(any(mot in str(elem) for mot in mots_cles_recherches) for elem in val)
    else:
        # Pour NaN, float, None, etc., on retourne False
        return False
    
def colored_text(text, color="black", bold=False, size="16px"):
    """
    Affiche du texte stylisé dans Streamlit.

    Args:
        text (str): le texte à afficher
        color (str): couleur CSS (ex: "red", "#ff8800", "rgb(0,200,100)")
        bold (bool): True pour mettre en gras
        size (str): taille de police (ex: "16px", "1.2em")
    """
    style = f"color:{color}; font-size:{size};display:flex; justify-content:center;align-items:center; height:5vh;"
    if bold:
        style += " font-weight:bold;"
    st.markdown(f"<span style='{style}'>{text}</span>", unsafe_allow_html=True)
    
###############################################################################################
########### RECUPERATION DES DATAFRAMES #######################################################
###############################################################################################
d = datetime.date.today()
df = read_data("Data\FairCarboN_Datas_Contacts")

df['DataInBrief'].fillna(0, inplace=True)
df['EarthSystemScienceData'].fillna(0, inplace=True)
df['Recensement'].fillna("NON", inplace=True)
df['Sollicitation'] = df['Sollicitation'].fillna('NON')
df["Data In Brief papers"] = df["DIB_titres"].apply(
    lambda x: x.split("/") if isinstance(x, str) and x else []
)
df["Earth System Science Data papers"] = df["ESSD_titres"].apply(
    lambda x: x.split("/") if isinstance(x, str) and x else []
)
df_long_DIB = df.explode("Data In Brief papers").dropna(subset=["Data In Brief papers"])
df_long_ESSD = df.explode("Earth System Science Data papers").dropna(subset=["Earth System Science Data papers"])

df_hal = st.session_state['df_hal']
df_rdg = st.session_state['df_rdg']
df_zenodo = st.session_state['df_zenodo']
df_publications = st.session_state['df_publications']

mots_cles_recherches = ['pepr faircarbon','faircarbon','alamod','slam-b','rift','crosyen','greenscale','canete','carbonium','deep-c','climfas','rhizoseqc','cabestan','tropecos','peace','prefalim','co2cmphi']

df_hal_reduit = df_hal[['Nom_archive','Auteur_recherché','Premier_auteur','Uri','Titre_unique','Date de publication','Mots_clés','Type de document','ANR project acronyme','EU project acronyme','Financement','DOI sources','In_FairCarboN','Sollicitation']]
df_rdg_reduit = df_rdg[['Nom_archive','Auteur_recherché','Titre_unique','Mots_clés','DOI sources','Date de publication','Type de document']]
df_zenodo_reduit = df_zenodo[['Nom_archive','Auteur_recherché','Titre_unique','Date de publication','Type de document']]

df_concat = pd.concat([df_hal_reduit,df_rdg_reduit,df_zenodo_reduit], axis=0)
df_concat['In_FairCarboN'] = df_concat['In_FairCarboN'].fillna(False)
df_concat['Type de document'] = df_concat['Type de document'].fillna('DATASET')
df_concat['Mots_clés'] = df_concat['Mots_clés'].apply(lambda x: [] if not isinstance(x, list) else x)
df_concat['Mots_clés'] = df_concat['Mots_clés'].apply(lambda lst: [mot.lower() for mot in lst])
df_concat['ANR project acronyme'] = df_concat['ANR project acronyme'].apply(lambda x: [] if not isinstance(x, list) else x)
df_concat['ANR project acronyme'] = df_concat['ANR project acronyme'].apply(lambda lst: [mot.lower() for mot in lst])
df_concat['Référencement par mots clés'] = df_concat['Mots_clés'].apply(
    lambda liste: any(mot in liste for mot in mots_cles_recherches))
df_concat['Projet ANR dans FairCarboN'] = df_concat['ANR project acronyme'].apply(contient_mots_cles)

df_concat["Financement_split"] = df_concat["Financement"].apply(
    lambda lst: [part for elem in lst for part in str(elem).split(" ")] if isinstance(lst, list) else []
)
df_concat["Financement_split"] = df_concat["Financement_split"].apply(lambda lst: [mot.lower() for mot in lst])
df_concat['Financement dans FairCarboN'] = df_concat['Financement_split'].apply(contient_mots_cles)
df_concat["Contient Zenodo"] = df_concat["DOI sources"].apply(
    lambda lst: any("zenodo" in s for s in lst) if isinstance(lst, list) else False
)
df_concat["Contient hal"] = df_concat["DOI sources"].apply(
    lambda lst: any("hal" in s for s in lst) if isinstance(lst, list) else False
)

df_concat["Sans_référencement"] = (
    df_concat["ANR project acronyme"].apply(is_empty_list) & df_concat["EU project acronyme"].apply(is_empty_list) & df_concat['Financement'].apply(is_empty_list))
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

st.title(":grey[Synthèse des analyses par individu]")

df_referencé = df_concat[df_concat['Référencement par mots clés']==True]

p = set(df_concat['Auteur_recherché'].values)

col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([0.29,0.05,0.05,0.185,0.125,0.125,0.125,0.10])
with col1:
    try:
        Selection_p = st.selectbox(label='Selection', options=p, index=st.session_state.count)
    except:
        Selection_p = ""
        reset_counter()
    st.subheader(f":grey[Fonction: {df['Fonction'][df['Contact']==Selection_p].values[0]}]")
    st.subheader(f":grey[ORCID n°: {df['ORCID'][df['Contact']==Selection_p].values[0]}]")
with col2:
    st.markdown('')
    st.markdown('')
    button1 = st.button(':heavy_plus_sign:',on_click=increment_counter)
with col3:
    st.markdown('')
    st.markdown('')
    button2 =st.button('R',on_click=reset_counter)
with col4:
    st.markdown('')
    st.markdown('')
    for i in range(len(df['projet'][df['Contact']==Selection_p].values)):
        colored_text(f"{df['projet'][df['Contact']==Selection_p].values[i]}",color="green",bold=True,size="30px")
with col5:
    if df['Recensement'][df['Contact']==Selection_p].values[0]=='NON':
        st.image('Data/nok.png', width=120, caption="Recensement")
    else:
        st.image('Data/ok.png', width=120, caption="Recensement")
with col6:
    if df['Sollicitation'][df['Contact']==Selection_p].values[0]=='NON':
        st.image('Data/nok.png', width=120, caption="Sollicitation")
    else:
        st.image('Data/ok.png', width=120, caption="Sollicitation")
with col7:
    if df['Réponse'][df['Contact']==Selection_p].values[0]=='NON':
        st.image('Data/nok.png', width=120, caption="Réponse")
    else:
        st.image('Data/ok.png', width=120, caption="Réponse")
with col8:
    #st.markdown(f"<span style='color:dimgray;font-weight:bold; font-size:25px;'>Data Papers</span>", unsafe_allow_html=True)
    st.subheader(f":grey[D. P.]")
    st.metric(label=' ',value=int(df['DataInBrief'][df['Contact']==Selection_p].values[0])+int(df['EarthSystemScienceData'][df['Contact']==Selection_p].values[0]),border=True)
    

df_selected = df_concat[df_concat['Auteur_recherché']==Selection_p]
df_selected.reset_index(inplace=True)
df_selected.drop(columns='index', inplace=True)

df_publications_selected = df_publications[df_publications['Auteur_recherché']==Selection_p]
df_publications_selected.reset_index(inplace=True)
df_publications_selected.drop(columns='index', inplace=True)
df_publications_selected.rename(columns={'from': 'Nom_archive', 'Année': 'Date de publication', 'Titre':'Titre_unique'}, inplace=True)
df_publications_selected = df_publications_selected[['Nom_archive','Date de publication','Titre_unique','Present_in_HAL']]

df_selected_solli = df_selected[['Auteur_recherché','Premier_auteur','Nom_archive', 'Date de publication','Titre_unique','ANR project acronyme','EU project acronyme','Financement','Financement dans FairCarboN','Sans_référencement','In_FairCarboN']][df_selected['Date de publication']>=2024][df_selected['Nom_archive']=='HAL']
df_selected_solli_ssref = df_selected_solli[['Nom_archive','Premier_auteur', 'Date de publication','Titre_unique','Sans_référencement','Financement dans FairCarboN']][df_selected['Sans_référencement']==True]
df_selected_solli_avecrefmalplacee = df_selected_solli[['Nom_archive','Premier_auteur', 'Date de publication','Titre_unique','Sans_référencement','Financement dans FairCarboN']][df_selected['Financement dans FairCarboN']==True]
df_averifier = pd.concat([df_selected_solli_ssref,df_selected_solli_avecrefmalplacee], axis=0)

if st.session_state.count > len(p):
        st.session_state.count = 0

colA, colB =st.columns(2)
with colA:
    col1, col2, col3 =st.columns(3)
    with col1:
        st.subheader(f":grey[Publ. ORCID]")
        st.metric(label='', value=len(df_publications_selected),border=True)
    with col2:
        st.subheader(f":grey[Publ. HAL]")
        st.metric(label='', value=len(df_selected[df_selected['Nom_archive']=='HAL']),border=True)
    with col3:
        st.subheader(f":grey[FairCarboN]")
        st.metric(label='', value=len(df_selected[df_selected['Nom_archive']=='HAL'][df_selected['In_FairCarboN']==True]),border=True)
    
    df_selected_hal = df_selected[df_selected['Nom_archive']=='HAL']
    if len(df_selected_hal)>0:
        row_counts_hal = df_selected_hal['Pattern'].value_counts().reset_index()
        row_counts_hal.columns = ['Archive HAL', 'compte']

                # Calcul du total et du pourcentage
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
    col1, col2, col3 =st.columns(3)
    with col1:
        st.subheader(f":grey[DS ouverts]")
        liste_entrepots = ['Recherche Data Gouv','Zenodo']
        df_selected_datasets = df_selected[df_selected['Nom_archive'].isin(liste_entrepots)]
        st.metric(label='', value=len(df_selected_datasets),border=True)
    with col2:
        st.subheader(f":grey[Pas de ref.]")
        st.metric(label='', value=len(df_selected[df_selected["Sans_référencement"]==True]),border=True)
    with col3:
        st.subheader(f":grey[à vérifier]")
        st.metric(label='', value=len(df_averifier),border=True)
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

#st.dataframe(df_averifier, hide_index=True)
#st.dataframe(df_publications_selected,hide_index=True)

df_averifier_final = pd.concat([df_averifier,df_publications_selected[df_publications_selected['Present_in_HAL']==False][df_publications_selected['Date de publication']>=2024]], axis=0)
df_averifier_final.rename({'Nom_archive':'Source'}, inplace=True)
df_averifier_final['Present_in_HAL'].fillna('A référencer', inplace=True)
df_averifier_final['Sans_référencement'].fillna('A déposer sur HAL', inplace=True)
# Suppression de la colonne 'Ville'
df_averifier_final = df_averifier_final.drop('Financement dans FairCarboN', axis=1)
#st.dataframe(df_averifier_final, hide_index=True)

# Remapper les valeurs pour plus de lisibilité
df_publications_selected['Présence'] = df_publications_selected['Present_in_HAL'].map({
    True: 'Présent dans HAL',
    False: 'Absent de HAL'
})

# Agréger les données
counts = df_publications_selected.groupby(['Date de publication', 'Présence']).size().unstack(fill_value=0)

# Ajouter les colonnes manquantes si elles n'existent pas
for col in ['Présent dans HAL', 'Absent de HAL']:
    if col not in counts.columns:
        counts[col] = 0

# Vérifier si le DataFrame est vide
if counts.empty or (counts['Présent dans HAL'].sum() == 0 and counts['Absent de HAL'].sum() == 0):
    st.warning("Aucune donnée disponible pour générer le graphique comparatif publis ORCID/HAL.")
else:
    # Calculer les pourcentages en évitant la division par zéro
    total = counts['Présent dans HAL'] + counts['Absent de HAL']
    percentages = (counts['Présent dans HAL'] / total.replace(0, 1) * 100).round(1)

    # Créer le graphe en barres empilées avec go.Bar
    fig4 = go.Figure()

    if counts['Présent dans HAL'].sum() > 0:
        fig4.add_bar(
            x=counts.index,
            y=counts['Présent dans HAL'],
            name='Présent dans HAL',
            marker_color='mediumseagreen',
            text=[f"{p}%" for p in percentages],
            textposition='inside'
        )

    if counts['Absent de HAL'].sum() > 0:
        fig4.add_bar(
            x=counts.index,
            y=counts['Absent de HAL'],
            name='Absent de HAL',
            marker_color='salmon'
        )

    fig4.update_layout(
        barmode='stack',
        title='Présence des titres ORCID dans HAL par année',
        xaxis_title='Date de publication',
        yaxis_title='Nombre de titres',
        template='plotly_white'
    )

    fig4.update_xaxes(
            tickmode='linear',
            tickformat='d',  # format entier
            dtick=1)

    st.plotly_chart(fig4, use_container_width=True)
    


Selection_p_ = unidecode(Selection_p).replace(" ", "_")

if int(df['DataInBrief'][df['Contact']==Selection_p].values[0])>0:
    st.dataframe(df_long_DIB["Data In Brief papers"][df_long_DIB['Contact']==Selection_p], hide_index=True)
if int(df['EarthSystemScienceData'][df['Contact']==Selection_p].values[0])>0:
    st.dataframe(df_long_ESSD["Earth System Science Data papers"][df_long_ESSD['Contact']==Selection_p], hide_index=True)

# Spécifie le dossier d'export
export_path = "./Data/Publications_sans_ref/"
os.makedirs(export_path, exist_ok=True)

# Nom des fichiers
csv_filename = os.path.join(export_path, f"CHECK_{Selection_p_}_{d}.csv")
pkl_filename = os.path.join(export_path, f"CHECK_{Selection_p_}_{d}.pkl")

if st.button("Sauvegarde et téléchargement"):
    col1, col2 = st.columns(2)
    # Proposer le téléchargement des fichiers

    with col1:
        df_averifier_final.to_csv(csv_filename, index=False)
        st.success(f"Fichiers enregistrés en csv dans `{export_path}` (côté serveur).")
        # CSV
        csv_buffer = io.StringIO()
        df_averifier_final.to_csv(csv_buffer, index=False)
        st.download_button(
                label="Télécharger le CSV",
                data=csv_buffer.getvalue().encode('utf-8'),
                file_name=f"CHECK_{Selection_p_}_{d}.csv",
                mime="text/csv"
            )
    with col2:
        with open(pkl_filename, "wb") as f:
            pickle.dump(df_averifier_final, f)
        st.success(f"Fichiers enregistrés en pickle dans `{export_path}` (côté serveur).")
        # Pickle
        pkl_buffer = io.BytesIO()
        pickle.dump(df_averifier_final, pkl_buffer)
        pkl_buffer.seek(0)
        st.download_button(
                label="Télécharger le fichier Pickle",
                data=pkl_buffer,
                file_name=f"CHECK_{Selection_p_}_{d}.pkl",
                mime="application/octet-stream"
            )

###############################################################################################
########### AUTRES ANALYSES ###################################################################
###############################################################################################

#tous_doi_sources_hal = sum(df_hal_reduit['DOI sources'], [])
#tous_doi_sources_rdg = sum(df_rdg_reduit['DOI sources'], [])

# Enlever les doublons en gardant l'ordre
#liste_doi_sources_hal = list(dict.fromkeys(tous_doi_sources_hal))
#liste_doi_sources_rdg = list(dict.fromkeys(tous_doi_sources_rdg))
#liste3 = [element for element in liste_doi_sources_hal if element in liste_doi_sources_rdg]