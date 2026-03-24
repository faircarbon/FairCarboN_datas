import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from Publications import afficher_publications_hal
import datetime
import ast

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
    # Chemin vers le fichier Excel
    #fichier_excel = "Data\FairCarboN_Datas_V2.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=0,header=0, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")
    return df

@st.cache_data
def acquisition_data(start_year,end_year,liste_chercheurs, liste_labs, liste_projet, liste_sollicitation):
    liste_columns_hal = ['Nom_archive',
                         'Auteur_recherché',
                         'Sigle structure',
                         'Projet',
                         'Ids',
                         'Titre et auteurs',
                         'Uri',
                         'Type',
                         'Type de document', 
                         'Date de publication',
                         'Date complete depot',
                         'Date complete production',
                         'Collection',
                         'Collection_code',
                         'Organisme',
                         'Auteur',
                         'Labo_all',
                         'Labo_',
                         'Titre',
                         'Langue',
                         'Mots_clés',
                         'Publication_source',
                         'ANR project acronyme',
                         'ANR project titre',
                         'EU project acronyme',
                         'EU project titre',
                         'Financement',
                         'Sollicitation']
    df_global_hal = pd.DataFrame(columns=liste_columns_hal)
    #progress = stqdm(total=len(liste_chercheurs))
    for i, s in enumerate(liste_chercheurs):
        print(i)
        #url_type = f'http://api.archives-ouvertes.fr/search/?q=text:"{s.lower().strip()}"&rows=1500&wt=json&fq=producedDateY_i:[{start_year} TO {end_year}]&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, producedDateY_i,authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s,labStructName_s,language_s,keyword_s,anrProjectAcronym_s,anrProjectTitle_s,europeanProjectAcronym_s,europeanProjectTitle_s,funding_s'
        url_type = f'http://api.archives-ouvertes.fr/search/?q=text:"{s.lower().strip()}"&rows=1500&wt=json&sort=docid asc&fl=docid,label_s,uri_s,submitType_s,docType_s, releasedDateY_i,releasedDate_s,producedDate_s, authLastNameFirstName_s,collName_s,collCode_s,instStructAcronym_s,collCode_s,authIdHasStructure_fs,title_s,labStructName_s,language_s,keyword_s,anrProjectAcronym_s,anrProjectTitle_s,europeanProjectAcronym_s,europeanProjectTitle_s,funding_s'
        df = afficher_publications_hal(url_type, s, liste_labs.iloc[i], liste_projet.iloc[i], liste_sollicitation[i])
        dfi = pd.concat([df_global_hal,df], axis=0)
        dfi.reset_index(inplace=True)
        dfi.drop(columns='index', inplace=True)
        df_global_hal = dfi
        #progress.update(i/len(liste_chercheurs))
    df_global_hal.sort_values(by='Ids', inplace=True, ascending=False)
    df_global_hal.reset_index(inplace=True)
    df_global_hal.drop(columns='index', inplace=True)

    df_global_hal['Labo_filter1'] = df_global_hal.apply(filtre_labo1, axis=1)
    df_global_hal['Labo_filter2'] = df_global_hal['Labo_filter1'].apply(filtre_labo2)


    # Colonne Auteur Labo qui est la résultante
    df_global_hal['Auteur_Labo'] = df_global_hal.apply(intersect_lists, axis=1)

    # On ne garde qu'un titre
    df_global_hal['Titre_unique'] = df_global_hal['Titre'].apply(lambda row: row[0])
    df_global_hal['Premier_auteur'] = df_global_hal['Auteur'].apply(lambda row: row[0])
    # On ne garde qu'une langue
    df_global_hal['Langue_unique'] = df_global_hal['Langue'].apply(lambda row: row[0])
    df_global_hal['Labo_unique'] = df_global_hal['Auteur_Labo'].apply(lambda row: row[0] if (len(row)>0) else None)
    #df_global_hal['Mots_Clés'] = df_global_hal['Mots_Clés'].apply(lambda x: ' '.join(x))
    #df_global_hal['combined'] = df_global_hal['Titre_bis'] + ' ' + df_global_hal['Mots_Clés']
    df_global_hal['DOI sources'] = df_global_hal['Publication_source'].apply(extraire_doi)
    df_global_hal['Mots_clés_'] = df_global_hal['Mots_clés'].apply(
    lambda x: '/'.join(x) if isinstance(x, list) else '')
    df_global_hal['ANR project acronyme_'] = df_global_hal['ANR project acronyme'].apply(
    lambda x: '/'.join(x) if isinstance(x, list) else '')
    df_global_hal['EU project acronyme_'] = df_global_hal['EU project acronyme'].apply(
    lambda x: '/'.join(x) if isinstance(x, list) else '')
    df_global_hal['Financement_'] = df_global_hal['Financement'].apply(
    lambda x: '/'.join(x) if isinstance(x, list) else '')
    return df_global_hal

######################################################################################################################
########### PARAMETRES ###############################################################################################
######################################################################################################################
d = datetime.date.today()
start_year=2024
end_year=d.year

######################################################################################################################
########### DONNEES ##################################################################################################
######################################################################################################################
# Charger les données
df = read_data("Data/FairCarboN_Datas_Contacts")

liste_chercheurs = df['Contact']
liste_projet = df['projet']
liste_sollicitation = df['Sollicitation']
liste_labs = df['Sigle structure']

df_global_hal = acquisition_data(start_year=start_year,end_year=end_year,liste_chercheurs=liste_chercheurs, liste_labs=liste_labs, liste_projet=liste_projet, liste_sollicitation=liste_sollicitation)

# Tableau de l'existant dans la collection FAIRCARBON
filtered_df = df_global_hal[df_global_hal['Collection_code'].apply(lambda names: 'FAIRCARBON' in names)]

# Ajout colonne In_FairCarboN
df_global_hal['In_FairCarboN'] = df_global_hal['Titre'].isin(filtered_df['Titre'])

###########################################################################################################################################
df_inter = df_global_hal[['Nom_archive',
                              'Projet',
                              'Auteur_recherché',
                              'Sigle structure',
                              'Premier_auteur',
                              'Ids','Uri',
                              'Titre_unique',
                              'Collection_code',
                              'Labo_unique',
                              'Langue_unique',
                              'DOI sources',
                              'Type de document',
                              'Date de publication',
                              'Date complete depot',
                              'Date complete production',
                              'Mots_clés_','ANR project acronyme_',
                              'EU project acronyme_','Financement_',
                              'In_FairCarboN','Sollicitation']].drop_duplicates(subset=['Auteur_recherché','Premier_auteur','Ids'])

df_inter['Mots_clés'] = df_inter['Mots_clés_'].apply(
        lambda x: x.split('/') if isinstance(x, str) and x else []
    )
df_inter['ANR project acronyme'] = df_inter['ANR project acronyme_'].apply(
        lambda x: x.split('/') if isinstance(x, str) and x else []
    )
df_inter['EU project acronyme'] = df_inter['EU project acronyme_'].apply(
        lambda x: x.split('/') if isinstance(x, str) and x else []
    )
df_inter['Financement'] = df_inter['Financement_'].apply(
        lambda x: x.split('/') if isinstance(x, str) and x else []
    )

df_inter['DOI sources'] = df_inter['DOI sources'].apply(lambda x: [x])
df_inter['Value']=1

df_inter.to_csv(f"Data/HAL/all_publications_hal_{d}.csv",index=False, encoding="utf-8")

print("récupération HAL réalisée avec succès!")