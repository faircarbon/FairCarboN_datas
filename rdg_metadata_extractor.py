"""
Extraction de métadonnées depuis RechercheDataGouv (Dataverse)
==============================================================
Récupère les métadonnées de TOUS les jeux de données d'une collection,
y compris ceux liés depuis d'autres collections (dataset linking).
Résultat exporté en CSV.

Dépendances : pip install pyDataverse requests
"""

import csv
import time
import requests
from pyDataverse.api import NativeApi

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL    = "https://entrepot.recherche.data.gouv.fr"  # URL de RechercheDataGouv
API_TOKEN   = "13b493ed-e02b-4e65-95de-d97d6896916a"         # Votre token API (optionnel pour le public)
COLLECTION  = "faircarbon"  # Alias de votre collection (ex: "monlabo")
OUTPUT_FILE = "datasets_metadata.csv"   # Fichier de sortie

# Colonnes du CSV — modifiez selon vos besoins
CSV_COLUMNS = [
    "pid",
    "title",
    "url",
    "published_at",
    "collection_parente",
    "is_linked",
    "authors",
    "description",
    "keywords",
    "subjects",
    "license",
    "version",
    "nb_fichiers",
]

# ── Initialisation ────────────────────────────────────────────────────────────
native_api = NativeApi(BASE_URL, API_TOKEN)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Recherche via Search API (inclut les datasets liés grâce à subtree)
# ══════════════════════════════════════════════════════════════════════════════

def search_all_datasets(collection_alias: str) -> list[dict]:
    """
    Utilise la Search API avec subtree=<alias> pour trouver tous les datasets
    visibles dans la collection, y compris les datasets liés depuis l'extérieur.
    """
    print(f"\n[1] Recherche de tous les datasets dans '{collection_alias}'...")

    datasets = []
    start    = 0
    per_page = 100  # max autorisé par l'API

    while True:
        resp = requests.get(
            f"{BASE_URL}/api/search",
            params={
                "q":        "*",
                "type":     "dataset",
                "subtree":  collection_alias,
                "per_page": per_page,
                "start":    start,
            },
            headers={"X-Dataverse-key": API_TOKEN} if API_TOKEN else {},
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        items     = data.get("items", [])
        total     = data.get("total_count", 0)
        datasets += items

        print(f"  Récupérés : {len(datasets)}/{total}")

        if len(datasets) >= total or not items:
            break
        start += per_page
        time.sleep(0.3)

    print(f"  → {len(datasets)} datasets trouvés.")
    return datasets


# ══════════════════════════════════════════════════════════════════════════════
# 2. Récupération et aplatissement des métadonnées complètes
# ══════════════════════════════════════════════════════════════════════════════

def _extract_text(value) -> str:
    """
    Aplatit une valeur de métadonnée Dataverse (scalaire, liste, dict imbriqué)
    en une chaîne lisible dans une cellule CSV.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(_extract_text(item))
        return " | ".join(filter(None, parts))
    if isinstance(value, dict):
        # Essaie les clés communes qui portent la valeur textuelle
        for key in ("value", "typeName", "name", "displayName"):
            if key in value:
                return _extract_text(value[key])
        # Sinon concatène toutes les valeurs du dict
        return " ; ".join(_extract_text(v) for v in value.values() if v)
    return str(value)


def fetch_and_flatten(pid: str, search_item: dict, collection_alias: str) -> dict | None:
    """
    Appelle la Native API pour récupérer les métadonnées complètes, puis
    les aplatit en une ligne CSV.
    """
    try:
        resp = native_api.get_dataset(pid)
        if resp.status_code != 200:
            print(f"  ⚠ HTTP {resp.status_code} pour {pid}")
            return None
        data = resp.json().get("data", {})
    except Exception as e:
        print(f"  ✗ Erreur pour {pid} : {e}")
        return None

    latest = data.get("latestVersion", {})

    # Extraire les champs du bloc "citation"
    citation_fields = {
        f["typeName"]: f.get("value")
        for f in latest.get("metadataBlocks", {})
                       .get("citation", {})
                       .get("fields", [])
    }

    # Auteurs : liste de dicts avec authorName, authorAffiliation…
    authors_raw = citation_fields.get("author", [])
    authors = " | ".join(
        _extract_text(a.get("authorName")) if isinstance(a, dict) else _extract_text(a)
        for a in (authors_raw if isinstance(authors_raw, list) else [authors_raw])
    )

    # Description : liste de dicts avec dsDescriptionValue
    desc_raw = citation_fields.get("dsDescription", [])
    descriptions = " | ".join(
        _extract_text(d.get("dsDescriptionValue")) if isinstance(d, dict) else _extract_text(d)
        for d in (desc_raw if isinstance(desc_raw, list) else [desc_raw])
    )

    # Mots-clés
    kw_raw = citation_fields.get("keyword", [])
    keywords = " | ".join(
        _extract_text(k.get("keywordValue")) if isinstance(k, dict) else _extract_text(k)
        for k in (kw_raw if isinstance(kw_raw, list) else [kw_raw])
    )

    # Sujets (liste simple de chaînes)
    subjects = _extract_text(citation_fields.get("subject"))

    # Licence
    license_info = latest.get("license", {})
    license_name = license_info.get("name", "") if isinstance(license_info, dict) else str(license_info)

    # Collection parente et statut lié
    collection_parente = search_item.get("identifier_of_dataverse", "")
    is_linked = collection_parente.lower() != collection_alias.lower()

    return {
        "pid":               pid,
        "title":             _extract_text(citation_fields.get("title")) or search_item.get("name", ""),
        "url":               data.get("persistentUrl") or search_item.get("url", ""),
        "published_at":      search_item.get("published_at", ""),
        "collection_parente": collection_parente,
        "is_linked":         is_linked,
        "authors":           authors,
        "description":       descriptions,
        "keywords":          keywords,
        "subjects":          subjects,
        "license":           license_name,
        "version":           f"{latest.get('versionNumber', '')}.{latest.get('versionMinorNumber', '')}",
        "nb_fichiers":       len(latest.get("files", [])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"Extraction des métadonnées — {BASE_URL}")
    print(f"Collection cible : {COLLECTION}")
    print("=" * 60)

    # Étape 1 : lister tous les datasets (natifs + liés)
    search_results = search_all_datasets(COLLECTION)

    if not search_results:
        print("Aucun dataset trouvé. Vérifiez l'alias de collection et le token API.")
        return

    # Étape 2 : récupérer et aplatir les métadonnées
    print(f"\n[2] Récupération des métadonnées complètes ({len(search_results)} datasets)...")

    rows = []
    for i, item in enumerate(search_results, 1):
        pid = item.get("global_id")
        if not pid:
            continue
        print(f"  [{i}/{len(search_results)}] {pid}")
        row = fetch_and_flatten(pid, item, COLLECTION)
        if row:
            rows.append(row)
        time.sleep(0.2)

    # Étape 3 : écriture du CSV
    print(f"\n[3] Écriture du CSV ({len(rows)} lignes)...")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        # utf-8-sig ajoute le BOM pour une ouverture correcte dans Excel
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Résumé
    nb_linked = sum(1 for r in rows if r["is_linked"])
    nb_native = len(rows) - nb_linked

    print(f"\n✓ Fichier CSV sauvegardé : '{OUTPUT_FILE}'")
    print(f"  Datasets natifs  : {nb_native}")
    print(f"  Datasets liés    : {nb_linked}")
    print(f"  Total            : {len(rows)}")

    if nb_linked:
        print(f"\n── Aperçu des datasets liés ──")
        for r in [r for r in rows if r["is_linked"]][:5]:
            print(f"  • {r['pid']} | collection parente : {r['collection_parente']}")
            print(f"    Titre : {r['title']}")


# ══════════════════════════════════════════════════════════════════════════════
# Bonus : accès rapide aux métadonnées d'un seul dataset
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset_metadata(pid: str) -> dict:
    """Utilitaire : récupère les métadonnées d'un dataset par son DOI."""
    item = {"identifier_of_dataverse": "", "published_at": "", "url": ""}
    return fetch_and_flatten(pid, item, "")


if __name__ == "__main__":
    main()

    # Exemple d'utilisation de la fonction utilitaire :
    # import json
    # meta = get_dataset_metadata("doi:10.57745/XXXXXX")
    # print(json.dumps(meta, ensure_ascii=False, indent=2))
