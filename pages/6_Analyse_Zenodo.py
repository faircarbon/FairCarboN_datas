import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import requests


###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON ZENODO DATA",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

###############################################################################################
########### FONCTIONS SUPPORT #################################################################
###############################################################################################
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
    df = pd.read_excel(f"{path}.xlsx", sheet_name=1,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")
    return df


def recuperation_zenodo(url_zenodo, params_zenodo, headers_zenodo):
    """
    Connexion à l'API Zenodo et récupération des résultats de recherche.

    Paramètres :
        url_zenodo (str) : URL de l'API Zenodo
        params_zenodo (dict) : paramètres de la requête
        headers_zenodo (dict) : en-têtes HTTP

    Retour :
        list : liste des éléments trouvés (dictionnaires)
    """
    try:
        response = requests.get(url_zenodo, params=params_zenodo, headers=headers_zenodo)
        response.raise_for_status()
        data = response.json()
        return data.get('hits', {}).get('hits', [])
    except requests.RequestException as e:
        print(f"[Erreur API Zenodo] {e}")
        return []

def extraire_valeur(dico, cle, default=""):
    """Extraction sécurisée d'une valeur dans un dictionnaire"""
    return dico.get(cle, default) if dico else default

def Recup_contenu_zenodo(url_zenodo, params_zenodo, headers_zenodo, auteur_recherche, projet):
    """
    Extraction des informations bibliographiques à partir de l'API Zenodo.

    Paramètres :
        url_zenodo (str) : URL de l'API Zenodo
        params_zenodo (dict) : paramètres de la requête
        headers_zenodo (dict) : en-têtes HTTP
        auteur_recherche (str) : nom de l'auteur à rechercher
        projet (str) : nom du projet associé

    Retour :
        pd.DataFrame : tableau des résultats formaté
    """
    contenu = recuperation_zenodo(url_zenodo, params_zenodo, headers_zenodo)

    donnees = {
        'Nom_archive': [],
        'Auteur_recherché': [],
        'Projet': [],
        'ID': [],
        'Titre_unique': [],
        'Auteur': [],
        'Résumé': [],
        'Date': [],
        'Publication Url': [],
        'Type de document':[]
    }

    for item in contenu:
        metadata = item.get('metadata', {})
        creators = metadata.get('creators', [{}])
        resource_type = metadata.get('resource_type',[{}])

        donnees['Nom_archive'].append('Zenodo')
        donnees['Auteur_recherché'].append(auteur_recherche)
        donnees['Projet'].append(projet)
        donnees['ID'].append(item.get('id', ''))
        donnees['Titre_unique'].append(item.get('title', ''))
        donnees['Auteur'].append(creators[0].get('name', '') if creators else '')
        donnees['Résumé'].append(metadata.get('description', ''))
        donnees['Date'].append(item.get('created', ''))
        donnees['Publication Url'].append(metadata.get('doi', ''))
        donnees['Type de document'].append(resource_type.get('type',''))

    return pd.DataFrame(donnees)

@st.cache_data
def acquisition_data_zenodo(liste_chercheurs,liste_chercheurs_bis, liste_projet):
    liste_columns = ['Nom_archive','Auteur_recherché','Projet','ID','Titre_unique','Auteur',"Résumé","Date","Publication Url",'Type de document']
    df_global_zenodo = pd.DataFrame(columns=liste_columns)
    for i, s in enumerate(liste_chercheurs_bis):
        print(i)
        params_zenodo = {'q': f'metadata.creators.person_or_org.name:"{s}"', # f'"{s.lower()}"'
                         'size':60,
                        'access_token': zenodo_token}
                    
        df = Recup_contenu_zenodo(url_zenodo,params_zenodo, headers_zenodo, liste_chercheurs[i], liste_projet[i])
        dfi = pd.concat([df_global_zenodo,df], axis=0)
        dfi.reset_index(inplace=True)
        dfi.drop(columns='index', inplace=True)
        df_global_zenodo = dfi
    df_global_zenodo["Date"] = pd.to_datetime(df_global_zenodo["Date"], errors="coerce")
    df_global_zenodo["Date de publication"]= df_global_zenodo["Date"].dt.year
    df_global_zenodo.sort_values(by='ID', inplace=True, ascending=False)
    df_global_zenodo.reset_index(inplace=True)
    df_global_zenodo.drop(columns='index', inplace=True)
    # 💾 Sauvegarde en CSV
    df_global_zenodo.to_csv(f"Data/Zenodo/all_datasets_zenodo_{d}.csv", index=False)

    return df_global_zenodo


######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################

#récupération des dataverses présents dans RDG
d = datetime.date.today()
start_year=2024
end_year=d.year

######################################################################################################################
######################## ZENODO ######################################################################################
url_zenodo = 'https://zenodo.org/api/records/'
zenodo_token = "OMMGEVUcApEKSt4JEkSK7OzpqZQPMvGKAlB2yP2MXG6APstRn2hWpiHfpjaA"
headers_zenodo = {"Content-Type": "application/json"}


st.title(":grey[Analyse des dépôts dans Zenodo]")

# Charger les données
df = read_data("Data\FairCarboN_Datas_Contacts")
# Séparer la chaîne en deux parties (Prénom et Nom)
df[['Prenom', 'Nom']] = df['Contact'].str.rsplit(' ', n=1, expand=True)
df['Contact_bis'] = df['Nom'] + ', ' + df['Prenom']
liste_chercheurs = df['Contact']
liste_chercheurs_bis = df['Contact_bis']
liste_projet = df['projet']

df_global_zenodo = acquisition_data_zenodo(liste_chercheurs, liste_chercheurs_bis, liste_projet)
df_global_zenodo['Value']=1

st.session_state['df_zenodo'] = df_global_zenodo

# Aggregate (e.g., sum) values by year
df_yearly = df_global_zenodo.groupby('Date de publication')['Value'].sum().reset_index()

# Plot aggregated data
fig_dates = px.bar(df_yearly, x='Date de publication', y='Value', title='Dépôts rattachés aux contacts FaircarboN')
st.plotly_chart(fig_dates, use_container_width=True)

###############################################################################################
########### FILTRAGE ##########################################################################
###############################################################################################
projets = list(set(df_global_zenodo['Projet']))
auteurs = list(set(df_global_zenodo['Auteur_recherché']))
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
    choix_auteur = st.multiselect(label='', options=list(set(df_global_zenodo['Auteur_recherché'][df_global_zenodo['Projet'].isin(choix_p)])))
    if len(choix_auteur)==0:
        choix_a = df_global_zenodo['Auteur_recherché'][df_global_zenodo['Projet'].isin(choix_p)]
    else:
        choix_a = choix_auteur

df_zenodo_proj =df_global_zenodo[df_global_zenodo['Projet'].isin(choix_p)][df_global_zenodo['Auteur_recherché'].isin(choix_a)][df_global_zenodo['Date de publication']>=start_year]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label='Nombre de datasets rattachés à nos contacts', value=len(df_global_zenodo))
with col2:
    st.metric(label=f'Nombre de datasets entre {start_year} et {end_year}', value=len(df_zenodo_proj))
with col3:
    st.metric(label='Nombre de contacts', value=len(set(df_zenodo_proj['Auteur_recherché'].values)))

unique_projet_titles = df_zenodo_proj[['Projet','Titre_unique']].drop_duplicates()
projects_count = unique_projet_titles['Projet'].value_counts().reset_index()
projects_count.columns = ['Projet', 'compte']

unique_person_titles = df_zenodo_proj[['Auteur_recherché','Titre_unique']].drop_duplicates()
row_counts = unique_person_titles['Auteur_recherché'].value_counts().reset_index()
row_counts.columns = ['Auteur', 'compte']

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

# Affichage
col1,col2 = st.columns(2)
with col1:
    if len(choix_auteur)==0:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(fig1, use_container_width=True)
    
with col2:
    st.plotly_chart(fig2, use_container_width=True)


params_zenodo_bis = {'q': f'metadata.creators.person_or_org.name:"Clivot, Hugues"', # f'"{s.lower()}"'
                         'size':50,
                        'access_token': zenodo_token}
liste_chercheurs_ = ['Hugues Clivot']
liste_projet_ = ['CANETE']
contenu_test = recuperation_zenodo(url_zenodo, params_zenodo_bis, headers_zenodo)
#st.write(contenu_test)                    
test = Recup_contenu_zenodo(url_zenodo,params_zenodo_bis, headers_zenodo, liste_chercheurs_[0], liste_projet_[0])
#st.dataframe(test)