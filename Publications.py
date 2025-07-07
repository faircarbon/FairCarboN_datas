import requests
import json.decoder
from bs4 import BeautifulSoup as soup
import pandas as pd


def afficher_texte_reponse_api_hal(requete_api_hal: str):
    """Interroge l'API HAL et affiche le texte de la réponse
    Paramètre = requête API HAL"""
    reponse = requests.get(requete_api_hal, timeout=5)
    print(reponse.text)


"""def afficher_erreur_api(erreur):
    #Affiche les erreurs soulevées lors de l'interrogation de l'API HAL
    #Paramètre = erreur
    print(f"Les résultats HAL n'ont pas pu être récupérés ({erreur}).")


def afficher_publications_hal(requete_api_hal: str, auteur, projet):
    #Interroge l'API HAL et affiche les infos des documents de la réponse
    #Paramètre = requête API HAL avec wt=json (str)
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
        mots_cles = []
        for doc in reponse.json()['response']['docs']:
            ids.append(int(doc['docid']))
            labels.append(soup(doc['label_s'], 'html.parser').text)
            uris.append(doc['uri_s'])
            types.append(doc['submitType_s'])
            docTypes.append(doc['docType_s'])
            date.append(doc['producedDateY_i'])
            author.append(doc['authLastNameFirstName_s'])#authIdHal_i
            #try:
            #    collection.append(doc['collName_s'])
            #except:
            #    collection.append('Collection_inexistante')
            try:
                collection_code.append(doc['collCode_s'])
            except:
                collection_code.append('Collection_code_inexistant')
            #try:
            #    organisme.append(doc['instStructAcronym_s'])
            #except:
            #    organisme.append('Organisme_non_mentionné')
            try:
                labo_all.append(doc['authIdHasStructure_fs'])
            except:
                labo_all.append(['_pasdelabo'])
            titre.append(doc['title_s'])
            try:
                labo.append(doc['labStructName_s'])
            except:
                labo.append(['Structure_non_mentionnée'])
            try:
                language.append(doc['language_s'])
            except:
                language.append('Email_non_mentionné')
            try:
                mots_cles.append(doc['keyword_s'])
            except:
                mots_cles.append([])
            source.append('HAL')

        reponse_df = pd.DataFrame({'Store':source,
                                   'Auteur_recherché':auteur,
                                   'Projet':projet,
                                   'Ids':ids,
                                   'Titre et auteurs':labels,
                                   'Uri':uris,
                                   'Type':types,
                                   'Type de document':docTypes,
                                   'Date de production':date,
                                   #'Collection':collection,
                                   'Collection_code':collection_code,
                                   #'Auteur_organisme':organisme,
                                   'Auteur':author,
                                   'Labo_all':labo_all,
                                   'Labo_':labo,
                                   'Titre':titre,
                                   'Langue':language,
                                   'Mots_Clés':mots_cles})

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

    return reponse_df"""

def afficher_erreur_api(erreur):
    print(f"Erreur lors de l'appel à l'API HAL : {erreur}")

def afficher_publications_hal(requete_api_hal: str, auteur: str, projet: str) -> pd.DataFrame:
    """Interroge l'API HAL et retourne les publications dans un DataFrame"""
    reponse_df = pd.DataFrame()  # DataFrame vide par défaut

    def safe_get(doc, key, default=None):
        return doc.get(key, default)

    try:
        reponse = requests.get(requete_api_hal, timeout=5)
        reponse.raise_for_status()  # Déclenche une exception HTTP si erreur
        data = reponse.json()

        docs = data.get('response', {}).get('docs', [])
        if not docs:
            print("Aucun document trouvé.")
            return reponse_df

        # Listes pour stocker les valeurs
        ids, labels, uris, types, docTypes, dates = [], [], [], [], [], []
        authors, collection, collection_codes, labo_all, labos, titres = [], [], [], [], [], []
        languages, mots_cles, sources, organisme = [], [], [], []

        for doc in docs:
            ids.append(int(safe_get(doc, 'docid', 0)))
            labels.append(soup(safe_get(doc, 'label_s', ''), 'html.parser').text)
            uris.append(safe_get(doc, 'uri_s', ''))
            types.append(safe_get(doc, 'submitType_s', ''))
            docTypes.append(safe_get(doc, 'docType_s', ''))
            dates.append(safe_get(doc, 'producedDateY_i', None))
            authors.append(safe_get(doc, 'authLastNameFirstName_s', []))
            collection.append(safe_get(doc, 'collName_s', ['Collection_inexistante']))
            collection_codes.append(safe_get(doc, 'collCode_s', ['Code_inexistant']))
            organisme.append(safe_get(doc, 'instStructAcronym_s', ['organisme_inexistant']))
            labo_all.append(safe_get(doc, 'authIdHasStructure_fs', ['_pasdelabo']))
            labos.append(safe_get(doc, 'labStructName_s', ['Structure_non_mentionnée']))
            titres.append(safe_get(doc, 'title_s', ''))
            languages.append(safe_get(doc, 'language_s', 'Langue_non_mentionnée'))
            mots_cles.append(safe_get(doc, 'keyword_s', []))
            sources.append('HAL')

        # Construction du DataFrame
        reponse_df = pd.DataFrame({
            'Store': sources,
            'Auteur_recherché': [auteur] * len(ids),
            'Projet': [projet] * len(ids),
            'Ids': ids,
            'Titre et auteurs': labels,
            'Uri': uris,
            'Type': types,
            'Type de document': docTypes,
            'Date de production': dates,
            'Collection':collection,
            'Collection_code': collection_codes,
            'Organisme':organisme,
            'Auteur': authors,
            'Labo_all': labo_all,
            'Labo_': labos,
            'Titre': titres,
            'Langue': languages,
            'Mots_Clés': mots_cles
        })

    except (requests.RequestException, json.decoder.JSONDecodeError, ValueError) as err:
        afficher_erreur_api(err)

    return reponse_df