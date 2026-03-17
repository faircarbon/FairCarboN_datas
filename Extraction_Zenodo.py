import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import requests

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
    df = pd.read_excel(f"{path}.xlsx", sheet_name=0,header=0, engine='openpyxl')
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


# Charger les données
df = read_data("Data\FairCarboN_Datas_Contacts")
# Séparer la chaîne en deux parties (Prénom et Nom)
df[['Prenom', 'Nom']] = df['Contact'].str.rsplit(' ', n=1, expand=True)
df['Contact_bis'] = df['Nom'] + ', ' + df['Prenom']
liste_chercheurs = df['Contact']
liste_chercheurs_bis = df['Contact_bis']
liste_projet = df['projet']


df_global_zenodo = acquisition_data_zenodo(liste_chercheurs, liste_chercheurs_bis, liste_projet)
print("Récupértion Zenodo réaliséee avec succès!")


