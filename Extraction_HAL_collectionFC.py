"""
Extraction des métadonnées HAL - Collection FairCarboN
API : http://api.archives-ouvertes.fr/search/
Sortie : fichier CSV avec les métadonnées principales
"""

import requests
import csv
import json
import time
import sys
import datetime


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
d = datetime.date.today()
API_BASE_URL = "http://api.archives-ouvertes.fr/search/"

COLLECTION = "FAIRCARBON"          # Nom de la collection HAL
OUTPUT_FILE = f"Data/HAL/all_publications_hal_FC_{d}.csv"
ROWS_PER_PAGE = 100                # Nombre de résultats par requête (max 200)


# Champs HAL à récupérer via l'API
HAL_FIELDS = [
    "docid",
    "halId_s",
    "title_s",
    "authFullName_s",
    "docType_s",
    "submittedDate_s",       # Date de dépôt
    "producedDate_s",        # Date de production
    "publicationDate_s",     # Date de publication
    "keyword_s",             # Mots-clés libres
    #"subject_s",             # Thématiques / domaines
    "domain_s",
    "anrProjectReference_s", # Référence projet ANR
    "anrProjectAcronym_s",   # Acronyme projet ANR
    "anrProjectTitle_s",     # Titre projet ANR
    "funding_s",             # Financement général
    "europeanProjectAcronym_s",  # Projet européen (ex. H2020)
    "journalTitle_s",        # Revue (si article)
    "conferenceTitle_s",     # Conférence (si communication)
    "abstract_s",            # Résumé
    "uri_s",                 # Lien HAL
    "openAccess_bool",       # Accès ouvert ?
    "language_s",            # Langue
]


# ─────────────────────────────────────────────
#  FONCTIONS
# ─────────────────────────────────────────────

def build_query_params(start: int) -> dict:
    """Construit les paramètres de la requête API HAL."""
    return {
        "q": f"collCode_s:{COLLECTION}",
        "fl": ",".join(HAL_FIELDS),
        "rows": ROWS_PER_PAGE,
        "start": start,
        "wt": "json",
        "sort": "submittedDate_s desc",
    }


def fetch_page(start: int, session: requests.Session) -> dict:
    """Récupère une page de résultats depuis l'API HAL."""
    params = build_query_params(start)
    response = session.get(API_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def join_list(value) -> str:
    """Convertit une liste en chaîne séparée par ' | '."""
    if isinstance(value, list):
        return " | ".join(str(v).strip() for v in value if v)
    return str(value).strip() if value else ""


def parse_record(doc: dict) -> dict:
    """Extrait et formate les métadonnées d'un document HAL."""
    return {
        "HAL ID":               doc.get("halId_s", ""),
        "Titre":                join_list(doc.get("title_s", "")),
        "Auteurs":              join_list(doc.get("authFullName_s", [])),
        "Type de document":     doc.get("docType_s", ""),
        "Date de dépôt":        doc.get("submittedDate_s", "")[:10] if doc.get("submittedDate_s") else "",
        "Date de production":   doc.get("producedDate_s", "")[:10] if doc.get("producedDate_s") else "",
        "Date de publication":  doc.get("publicationDate_s", "")[:10] if doc.get("publicationDate_s") else "",
        "Mots-clés":            join_list(doc.get("keyword_s", [])),
        #"Domaines":             join_list(doc.get("subject_s", [])),
        "Domaines":             join_list(doc.get("domain_s", [])),
        "Financement":          join_list(doc.get("funding_s", [])),
        "Acronyme projet ANR":  join_list(doc.get("anrProjectAcronym_s", [])),
        "Référence projet ANR": join_list(doc.get("anrProjectReference_s", [])),
        "Titre projet ANR":     join_list(doc.get("anrProjectTitle_s", [])),
        "Projet européen":      join_list(doc.get("europeanProjectAcronym_s", [])),
        "Revue":                join_list(doc.get("journalTitle_s", "")),
        "Conférence":           join_list(doc.get("conferenceTitle_s", "")),
        "Langue":               join_list(doc.get("language_s", [])),
        "Accès ouvert":         "Oui" if doc.get("openAccess_bool") else "Non",
        "Résumé":               join_list(doc.get("abstract_s", [])),
        "Lien HAL":             doc.get("uri_s", ""),
    }


def export_to_csv(records: list[dict], filepath: str):
    """Écrit les enregistrements dans un fichier CSV."""
    if not records:
        print("Aucun enregistrement à exporter.")
        return

    fieldnames = list(records[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(records)

    print(f"\n✅ Export terminé : {len(records)} dépôts enregistrés dans « {filepath} »")


# ─────────────────────────────────────────────
#  PROGRAMME PRINCIPAL
# ─────────────────────────────────────────────

def main():
    print(f"🔍 Interrogation de l'API HAL pour la collection : {COLLECTION}")
    print(f"   URL de base : {API_BASE_URL}\n")

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    all_records = []
    start = 0

    try:
        # Première requête pour connaître le nombre total
        data = fetch_page(start=0, session=session)
        total = data["response"]["numFound"]
        print(f"📄 Nombre total de dépôts trouvés : {total}\n")

        if total == 0:
            print("Aucun résultat. Vérifiez le nom de la collection.")
            sys.exit(0)

        # Pagination
        while start < total:
            print(f"   Récupération des enregistrements {start + 1} à {min(start + ROWS_PER_PAGE, total)} / {total}...")

            if start > 0:          # La première page est déjà chargée
                data = fetch_page(start=start, session=session)

            docs = data["response"]["docs"]
            for doc in docs:
                all_records.append(parse_record(doc))

            start += ROWS_PER_PAGE
            if start < total:
                time.sleep(0.5)    # Pause pour ne pas surcharger l'API

    except requests.exceptions.HTTPError as e:
        print(f"\n❌ Erreur HTTP : {e}")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("\n❌ Impossible de contacter l'API HAL. Vérifiez votre connexion internet.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Interruption manuelle. Export partiel en cours...")

    export_to_csv(all_records, OUTPUT_FILE)


if __name__ == "__main__":
    main()
