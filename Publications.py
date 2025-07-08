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
        languages, mots_cles, sources, organisme, publication = [], [], [], [], []

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
            titres.append(safe_get(doc, 'title_s', []))
            languages.append(safe_get(doc, 'language_s', ['Langue_non_mentionnée']))
            mots_cles.append(safe_get(doc, 'keyword_s', []))
            publication.append(safe_get(doc, 'label_s', []))
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
            'Mots_Clés': mots_cles,
            "Publication_source":publication
        })

    except (requests.RequestException, json.decoder.JSONDecodeError, ValueError) as err:
        afficher_erreur_api(err)

    return reponse_df