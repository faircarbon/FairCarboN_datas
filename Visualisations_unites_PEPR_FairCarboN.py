import pandas as pd
import plotly.graph_objects as go

unites = pd.read_csv("Data\LABOS.csv", encoding="utf-8", sep=";")

compte_labos_par_projet = unites["Acronyme projet"].value_counts()

fig = go.Figure(go.Bar(
    x=compte_labos_par_projet.values,
    y=compte_labos_par_projet.index,
    orientation='h'
))

# Personnalisation (optionnelle)
fig.update_layout(
    title='compte_labos_par_projet',
    xaxis_title='Nombre',
    yaxis_title='Projet',
    template='plotly_white'
)

fig.show()