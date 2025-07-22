import streamlit as st
import pandas as pd
from pyDataverse.models import Dataset
from pyDataverse.utils import read_file
from pyDataverse.api import NativeApi
import datetime
import numpy as np
import re
import plotly.express as px
import requests
import os
import json

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON RDG DATA MINING",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "développé par Jérôme Dutroncy"}
)

###############################################################################################
########### RECUPERATION DES DATAFRAMES #######################################################
###############################################################################################

df_hal = st.session_state['df_hal']
df_rdg = st.session_state['df_rdg']
df_zenodo = st.session_state['df_zenodo']
st.dataframe(df_hal)
st.dataframe(df_rdg)
st.dataframe(df_zenodo)