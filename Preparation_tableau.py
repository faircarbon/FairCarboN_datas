import pandas as pd
import plotly.graph_objects as go

df1 = pd.read_csv("Data\DOCUMENTATION_TEST.csv", sep=";", encoding="utf-8")
df2 = pd.read_csv("Data\DESCRIPTION_GENERALE_TEST.csv", sep=";", encoding="utf-8")
df3 = pd.read_csv("Data\ACQUISITION_ET_STOCKAGE_TEST.csv", sep=";", encoding="utf-8")
df4 = pd.read_csv("Data\ETHIQUE__PARTAGE_ET_VALORISATION_TEST.csv", sep=";", encoding="utf-8")

common_cols = ["NOM", "INTITULE_DONNEES", "NATURE"]
merged_df = df1.merge(df2, on=common_cols, how="outer").merge(df3, on=common_cols, how="outer").merge(df4, on=common_cols, how="outer")
merged_df.to_csv("Data\DONNEES_GLOBALE.csv", index=False)