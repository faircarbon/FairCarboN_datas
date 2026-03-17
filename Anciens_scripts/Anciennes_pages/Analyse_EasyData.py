import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from owslib.csw import CatalogueServiceWeb
from owslib import fes
import sys
import xml.etree.ElementTree as ET
from io import BytesIO

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON EASYDATA DATA",
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


# Fonction pour extraire les auteurs depuis un enregistrement XML ISO 19115-3
def extract_authors_from_iso19115_3(xml_content):
    try:
        namespaces = {
            'cit': "http://standards.iso.org/iso/19115/-3/cit/2.0",
            'gco': "http://standards.iso.org/iso/19115/-3/gco/1.0",
        }
        root = ET.fromstring(xml_content)
        authors = []

        # Trouver tous les éléments <cit:individual> contenant <cit:CI_Individual>
        for indiv in root.findall(".//cit:individual/cit:CI_Individual", namespaces):
            name_el = indiv.find(".//gco:CharacterString", namespaces)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        return authors
    except Exception as e:
        return [f"[Erreur extraction auteurs: {e}]"]

st.title(":grey[Etude du contenu de EaSy Data]")

url_easydata = "https://www.easydata.earth/api/csw"

# Connexion au service CSW
csw = CatalogueServiceWeb(url_easydata, timeout=30)

# Afficher les informations du service
st.write(f"Service Title: {csw.identification.title}")
st.write(f"Service Version: {csw.identification.version}")

# Exemple de filtre par mot-clé
#keywords = ['climat']
#filter_keywords = fes.PropertyIsLike(propertyname='csw:AnyText', literal='*' + keywords[0] + '*')
#csw.getrecords2(constraints=[filter_keywords], maxrecords=5)

start = 1
step = 10  # nombre d'enregistrements par page
records_data = []

# Première requête pour obtenir le nombre total
csw.getrecords2(startposition=start, maxrecords=step, esn='summary', outputschema='http://www.opengis.net/cat/csw/2.0.2') #'http://www.isotc211.org/2005/gmd'
total_matches = csw.results["matches"]
st.write(f"🔎 Nombre total d'enregistrements : {total_matches}")

while start <= total_matches:
    #st.write(f"📥 Téléchargement de la page {start} à {start + step - 1}...")
    csw.getrecords2(startposition=start, maxrecords=step, esn='summary', outputschema='http://www.opengis.net/cat/csw/2.0.2')

    for rec_id, record in csw.records.items():
        try:
            csw.getrecordbyid(id=[rec_id], outputschema='http://www.opengis.net/cat/csw/2.0.2', esn='full')
            record_full = csw.records[rec_id]
            authors = extract_authors_from_iso19115_3(csw.response)
            records_data.append({
                "ID": rec_id,
                "Titre": getattr(record_full, "title", ""),
                "Résumé": getattr(record_full, "abstract", ""),
                "Date": getattr(record_full, "modified", "") or getattr(record_full, "date", ""),
                "Auteur(s)": ", ".join(authors),
                "Mots-clés": ", ".join([s for s in getattr(record_full, "subjects", []) if isinstance(s, str)]),
            })
        except Exception as e:
            st.write(f"⚠️ Erreur sur l'ID {rec_id}: {e}")

    start += step  # passer à la page suivante

# Créer le DataFrame
df = pd.DataFrame(records_data)

# Afficher un aperçu
st.dataframe(df)