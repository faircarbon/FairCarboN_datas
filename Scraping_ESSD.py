import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    WebDriverException, NoSuchElementException, TimeoutException
)
from webdriver_manager.firefox import GeckoDriverManager
import time, re, sys

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON CATALOGUE",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)
######################################################################################################################
########### FONCTIONS SUPPORTS #######################################################################################
######################################################################################################################

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

def init_driver(headless=True):
    """Initialise Firefox avec geckodriver via webdriver-manager."""
    options = webdriver.FirefoxOptions()
    if headless:
        options.add_argument("--headless")  # exécution sans interface graphique
    try:
        service = Service(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=options)
        driver.set_page_load_timeout(30)
        return driver
    except WebDriverException as e:
        st.write("❌ Erreur lors de l'initialisation du driver :", e)
        sys.exit(1)

def get_hits_for(driver, name):
    """Recherche un nom et retourne le nombre de 'hits' trouvé."""
    try:
        # Localise la barre de recherche (à adapter si le sélecteur change)
        search_input = driver.find_element(By.ID, "search_query_solr")
        search_input.clear()
        search_input.send_keys(f'"{name}"')
        search_input.send_keys(Keys.ENTER)
        time.sleep(2)  # attends l’ouverture de la fenêtre

        # Titre de la fenêtre modale
        hits_element = driver.find_element(By.ID, "templateSearchResultNr")
        hits = int(hits_element.text.strip())
        return hits

    except NoSuchElementException:
        st.write(f"⚠️ Élément introuvable pour '{name}' (vérifie le sélecteur CSS).")
        return None
    except TimeoutException:
        st.write(f"⚠️ Timeout pendant la recherche '{name}'.")
        return None
    except Exception as e:
        st.write(f"⚠️ Erreur inattendue pour '{name}':", e)
        return None
    
######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
# Charger les données
df_test = pd.read_csv("ESSD.csv")
if 'ESSD' not in df_test.columns:
    df_test['ESSD'] = pd.NA

driver = init_driver(headless=False)  # passe à True si tu veux sans fenêtre
results = []

try:
    driver.get("https://essd.copernicus.org/data_description_paper.html")
    time.sleep(2)

    for idx, row in df_test[df_test['ESSD'].isna()].iterrows():
        nom = row['Contact']
        hits = get_hits_for(driver, nom)
        df_test.at[idx, 'ESSD'] = hits  # remplissage uniquement si vide

finally:
    driver.quit()

df_test['ESSD'] = df_test['ESSD'].apply(lambda x: '' if pd.isna(x) else int(x))
st.dataframe(df_test)
df_test.to_csv("ESSD.csv", index=False)