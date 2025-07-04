import streamlit as st
from crossref.restful import Works
import pandas as pd
import numpy as np

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

works = Works()
results = works.query(author='Jérôme Demarty').filter(type="journal-article").sample(50)

df = pd.DataFrame(results)
st.dataframe(df)

df_ = pd.DataFrame()
for i in range(50):
    dfi = pd.DataFrame(df['author'].iloc[i])
    dfc = pd.concat([df_,dfi],axis=0)
    df_ = dfc

try:
    st.dataframe(df_[df_['given']=="Jérôme"][df_['family']=="Demarty"])
except:
    pass