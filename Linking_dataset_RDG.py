import requests

# ─── Paramètres à renseigner ────────────────────────────────────────────────

SERVER_URL   = "https://entrepot.recherche.data.gouv.fr"  # URL de RechercheDataGouv
API_TOKEN    = "13b493ed-e02b-4e65-95de-d97d6896916a"     # Votre token API (Mon compte > Paramètres API)

# ID numérique du dataset à lier (récupérable via l'API ou l'URL du dataset)
# Si vous avez le DOI, utilisez la méthode get_dataset_id_from_doi() ci-dessous
#DATASET_ID   = 12345  # <-- à remplacer

# Alias de VOTRE collection (visible dans l'URL : .../dataverse/<alias>)
MY_COLLECTION_ALIAS = "faircarbon"  # <-- à remplacer

# ─── (Optionnel) Récupérer l'ID numérique depuis un DOI ────────────────────

def get_dataset_id_from_doi(server_url, doi, api_token):
    """Retourne l'ID numérique d'un dataset à partir de son DOI."""
    url = f"{server_url}/api/datasets/:persistentId/"
    headers = {"X-Dataverse-key": api_token}
    params  = {"persistentId": doi}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()["data"]["id"]

# Exemple :
# DATASET_ID = get_dataset_id_from_doi(
#     SERVER_URL,
#     doi="doi:10.57745/XXXXXX",
#     api_token=API_TOKEN
# )

# ─── Lier le dataset à votre collection ─────────────────────────────────────

def link_dataset_to_collection(server_url, api_token, dataset_id, collection_alias):
    """
    Crée un lien entre un dataset existant et une collection Dataverse.
    Le dataset reste dans sa collection d'origine et est rendu visible dans la vôtre.
    """
    url = f"{server_url}/api/datasets/{dataset_id}/link/{collection_alias}"
    headers = {"X-Dataverse-key": api_token}

    response = requests.put(url, headers=headers)

    if response.status_code == 200:
        print(f"✅ Dataset {dataset_id} lié avec succès à la collection '{collection_alias}'.")
        print(response.json())
    else:
        print(f"❌ Erreur {response.status_code} : {response.text}")

    return response

# ─── Exécution ───────────────────────────────────────────────────────────────

#link_dataset_to_collection(SERVER_URL, API_TOKEN, DATASET_ID, MY_COLLECTION_ALIAS)

DOI = "doi:10.23708/4FAPMG"
DATASET_ID = get_dataset_id_from_doi(server_url=SERVER_URL,doi=DOI,api_token=API_TOKEN)
print(DATASET_ID)