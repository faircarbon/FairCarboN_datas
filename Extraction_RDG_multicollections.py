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
import datetime
from pyDataverse.api import NativeApi

# ── Configuration ─────────────────────────────────────────────────────────────
d = datetime.date.today()
BASE_URL    = "https://entrepot.recherche.data.gouv.fr"  # URL de RechercheDataGouv
API_TOKEN   = "13b493ed-e02b-4e65-95de-d97d6896916a"         # Votre token API (optionnel pour le public)
COLLECTIONS  = ["faircarbon","alamod","slam-b",'crosyen','rift','carbonium','canete','peace','clim-fas','prefalim','rhizoseqc','greenscale','co2_cmphi','drought_forc','tropecos','deep-c','cabestan'] # Alias de votre collection (ex: "monlabo")
OUTPUT_FILE = f"Data/RechercheDataGouv/all_datasets_rdg_multi_{d}.csv"   # Fichier de sortie

# Colonnes du CSV — modifiez selon vos besoins
CSV_COLUMNS = [
    "Projet",
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
# 1. Recherche via Search API
# ══════════════════════════════════════════════════════════════════════════════

def search_all_datasets(collection_alias: str) -> list[dict]:
    print(f"\n[SEARCH] Collection '{collection_alias}'...")

    datasets = []
    start    = 0
    per_page = 100

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
# 2. Extraction et aplatissement des métadonnées
# ══════════════════════════════════════════════════════════════════════════════

def _extract_text(value) -> str:
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
        for key in ("value", "typeName", "name", "displayName"):
            if key in value:
                return _extract_text(value[key])
        return " ; ".join(_extract_text(v) for v in value.values() if v)
    return str(value)


def fetch_and_flatten(pid: str, search_item: dict, collection_alias: str) -> dict | None:
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

    citation_fields = {
        f["typeName"]: f.get("value")
        for f in latest.get("metadataBlocks", {})
                       .get("citation", {})
                       .get("fields", [])
    }

    authors_raw = citation_fields.get("author", [])
    authors = " | ".join(
        _extract_text(a.get("authorName")) if isinstance(a, dict) else _extract_text(a)
        for a in (authors_raw if isinstance(authors_raw, list) else [authors_raw])
    )

    desc_raw = citation_fields.get("dsDescription", [])
    descriptions = " | ".join(
        _extract_text(d.get("dsDescriptionValue")) if isinstance(d, dict) else _extract_text(d)
        for d in (desc_raw if isinstance(desc_raw, list) else [desc_raw])
    )

    kw_raw = citation_fields.get("keyword", [])
    keywords = " | ".join(
        _extract_text(k.get("keywordValue")) if isinstance(k, dict) else _extract_text(k)
        for k in (kw_raw if isinstance(kw_raw, list) else [kw_raw])
    )

    subjects = _extract_text(citation_fields.get("subject"))

    license_info = latest.get("license", {})
    license_name = license_info.get("name", "") if isinstance(license_info, dict) else str(license_info)

    collection_parente = search_item.get("identifier_of_dataverse", "")
    is_linked = collection_parente.lower() != collection_alias.lower()

    return {
        "Projet":   collection_alias,   # <-- nouveau
        "pid":                pid,
        "title":              _extract_text(citation_fields.get("title")) or search_item.get("name", ""),
        "url":                data.get("persistentUrl") or search_item.get("url", ""),
        "published_at":       search_item.get("published_at", ""),
        "collection_parente": collection_parente,
        "is_linked":          is_linked,
        "authors":            authors,
        "description":        descriptions,
        "keywords":           keywords,
        "subjects":           subjects,
        "license":            license_name,
        "version":            f"{latest.get('versionNumber', '')}.{latest.get('versionMinorNumber', '')}",
        "nb_fichiers":        len(latest.get("files", [])),
    }



# ══════════════════════════════════════════════════════════════════════════════
# 3. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print(f"Extraction des métadonnées — {BASE_URL}")
    print(f"Collections cibles : {', '.join(COLLECTIONS)}")
    print("=" * 60)

    all_rows   = []
    seen_pids  = set()   # évite les doublons si un dataset est lié dans plusieurs collections

    for collection in COLLECTIONS:
        search_results = search_all_datasets(collection)
        if not search_results:
            print(f"  Aucun dataset trouvé pour '{collection}'.")
            continue

        print(f"\n  Récupération des métadonnées pour '{collection}' ({len(search_results)} datasets)...")
        for i, item in enumerate(search_results, 1):
            pid = item.get("global_id")
            if not pid:
                continue

            # Si déjà vu dans une autre collection que faircarbon, on ignore
            if pid in seen_pids and collection != "faircarbon":
                # La collection courante a priorité : on remplace l'entrée faircarbon
                all_rows = [r for r in all_rows if r["pid"] != pid]
            elif pid in seen_pids and collection == "faircarbon":
                print(f"  [{i}/{len(search_results)}] {pid} — déjà traité, ignoré.")
                continue

            print(f"  [{i}/{len(search_results)}] {pid}")
            row = fetch_and_flatten(pid, item, collection)
            if row:
                all_rows.append(row)
                seen_pids.add(pid)
            time.sleep(0.2)

    # Écriture du CSV
    print(f"\n[CSV] Écriture ({len(all_rows)} lignes) → '{OUTPUT_FILE}'")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    # Résumé
    nb_linked = sum(1 for r in all_rows if r["is_linked"])
    print(f"\n✓ Terminé.")
    print(f"  Datasets natifs  : {len(all_rows) - nb_linked}")
    print(f"  Datasets liés    : {nb_linked}")
    print(f"  Total            : {len(all_rows)}")


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
