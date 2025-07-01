import streamlit as st
from crossref.restful import Works
import pandas as pd
import numpy as np
from pyDataverse.models import Dataset
from pyDataverse.utils import read_file
from pyDataverse.api import NativeApi
import time
import requests

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="TEST",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)



base = "https://entrepot.recherche.data.gouv.fr"
rows = 10
start = 0
page = 1
condition = True # emulate do-while


response_init = requests.get(base + '/api/v1/search?q=*')
response_init.raise_for_status()  # Sécurité : stoppe si erreur
data_init = response_init.json().get("data", {})
total_count = data_init.get("total_count", 0)

all_items = []

while (condition):
    url = base + '/api/v1/search?q=*' + "&start=" + str(start)
    
    response = requests.get(url)
    response.raise_for_status()  # Sécurité : stoppe si erreur

    data = response.json().get("data", {})
    items = data.get("items", [])

    if not items:
        break

    all_items.extend(items)
    start = start + rows
    page += 1
    print(page)
    condition = start < total_count


# 🔍 Filtrer uniquement les datasets
dataset_items = [item for item in all_items if item.get("type") == "dataset"]

# 🎯 Extraction des champs souhaités
filtered_data = [
        {"name": item.get("name"), "global_id": item.get("global_id"), 'entrepot':item.get('publisher'), 'parent':item.get('storageIdentifier')}
        for item in dataset_items
]

# 📊 DataFrame
df = pd.DataFrame(filtered_data)

# 💾 Sauvegarde en CSV
df.to_csv("Data/all_datasets_rdg.csv", index=False)

# 🖥️ Affichage Streamlit
st.write(f"{len(df)} éléments de type 'dataset' récupérés.")
st.dataframe(df)