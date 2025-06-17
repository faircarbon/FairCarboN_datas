import requests
import json.decoder
from bs4 import BeautifulSoup as soup
import pandas as pd


def afficher_texte_reponse_api_hal(requete_api_hal: str):
    """Interroge l'API HAL et affiche le texte de la réponse
    Paramètre = requête API HAL"""
    reponse = requests.get(requete_api_hal, timeout=5)
    print(reponse.text)


def afficher_erreur_api(erreur):
    """Affiche les erreurs soulevées lors de l'interrogation de l'API HAL
    Paramètre = erreur"""
    print(f"Les résultats HAL n'ont pas pu être récupérés ({erreur}).")


def afficher_publications_hal(requete_api_hal: str, auteur):
    """Interroge l'API HAL et affiche les infos des documents de la réponse
    Paramètre = requête API HAL avec wt=json (str)"""
    try:
        reponse = requests.get(requete_api_hal, timeout=5)
        ids = []
        labels = []
        uris = []
        types = []
        docTypes = []
        date = []
        source = []
        author = []
        collection = []
        collection_code = []
        organisme = []
        labo_all = []
        labo = []
        titre = []
        language = []
        for doc in reponse.json()['response']['docs']:
            ids.append(int(doc['docid']))
            labels.append(soup(doc['label_s'], 'html.parser').text)
            uris.append(doc['uri_s'])
            types.append(doc['submitType_s'])
            docTypes.append(doc['docType_s'])
            date.append(doc['producedDateY_i'])
            author.append(doc['authLastNameFirstName_s'])#authIdHal_i
            try:
                collection.append(doc['collName_s'])
            except:
                collection.append('Collection_inexistante')
            try:
                collection_code.append(doc['collCode_s'])
            except:
                collection_code.append('Collection_code_inexistant')
            try:
                organisme.append(doc['instStructAcronym_s'])
            except:
                organisme.append('Organisme_non_mentionné')
            labo_all.append(doc['authIdHasStructure_fs'])
            titre.append(doc['title_s'])
            try:
                labo.append(doc['labStructName_s'])
            except:
                labo.append('Structure_non_mentionnée')
            try:
                language.append(doc['language_s'])
            except:
                language.append('Email_non_mentionné')
            source.append('HAL')

        reponse_df = pd.DataFrame({'Store':source,
                                   'Auteur_recherché':auteur,
                                   'Ids':ids,
                                   'Titre et auteurs':labels,
                                   'Uri':uris,
                                   'Type':types,
                                   'Type de document':docTypes,
                                   'Date de production':date,
                                   'Collection':collection,
                                   'Collection_code':collection_code,
                                   'Auteur_organisme':organisme,
                                   'Auteur':author,
                                   'Labo_all':labo_all,
                                   'Labo_':labo,
                                   'Titre':titre,
                                   'Langue':language})

    except requests.exceptions.HTTPError as errh:
        afficher_erreur_api(errh)
    except requests.exceptions.ConnectionError as errc:
        afficher_erreur_api(errc)
    except requests.exceptions.Timeout as errt:
        afficher_erreur_api(errt)
    except requests.exceptions.RequestException as err:
        afficher_erreur_api(err)
    except json.decoder.JSONDecodeError as errj:
        afficher_erreur_api(errj)

    return reponse_df