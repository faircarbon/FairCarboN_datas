import pandas as pd
import plotly.graph_objects as go

# Charger le fichier csv
csv_file = "Data\QUESTIONNAIRE_PGD_FAIRCARBON_CANETE_2025-05-12_14-04-10_6821e3ba043eb9.25082439.csv"  
df_i = pd.read_csv(csv_file, encoding="utf-8", sep=";")

df_i_p = df_i[df_i["NATURE"]=="PRODUITES"]
df_i_e = df_i[df_i["NATURE"]=="PRE_EXISTANTES"]
df= pd.concat([df_i_p, df_i_e], axis=0)

#traitement source vide
df["CONDITIONS_REUTILISATION"].fillna("NON_CONCERNE_C", inplace=True)
df["RESTRICTIONS"].fillna("NON_CONCERNE_R", inplace=True)


# s'assurer que les colonnes existent bien
required_columns = {"INTITULE_DONNEES","NATURE","ETHIQUE","PERSONNEL","CONDITIONS_REUTILISATION","RESTRICTIONS","ENTREPOT_SPECIFIQUE","EMBARGO","LICENCE","VOLUMETRIE"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Convertir en numerique la valeur de volumétrie (au cas où)
df["VOLUMETRIE"] = pd.to_numeric(df["VOLUMETRIE"], errors="coerce").fillna(0)

# Creation des labels uniques pour les noeuds
all_labels = list(pd.concat([df["NATURE"],df["INTITULE_DONNEES"],df["ETHIQUE"], df["PERSONNEL"],df["CONDITIONS_REUTILISATION"],df["RESTRICTIONS"],df["ENTREPOT_SPECIFIQUE"],df["EMBARGO"],df["LICENCE"]]).unique())

# Mapping à partir des labels index
label_to_index = {label: i for i, label in enumerate(all_labels)}

# Def des liens du Diagram
sources = []
targets = []
values = []


# flow de "NATURE" à "INTITULE_DONNEES"
sources.extend(df["NATURE"].map(label_to_index))
targets.extend(df["INTITULE_DONNEES"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

colors=['lightblue','lightgreen']
colors_for_links = []
df_sources = pd.DataFrame(sources)
sources_init = df_sources.value_counts().values
colors_for_links_init=[]
for i, j in enumerate(sources_init):
    colors_for_links_init.extend(j * [colors[i]])


# flow de "INTITULE_DONNEES" à "ETHIQUE"
sources.extend(df["INTITULE_DONNEES"].map(label_to_index))
targets.extend(df["ETHIQUE"].map(label_to_index))
values.extend(df["VOLUMETRIE"])


# flow de "ETHIQUE" à "PERSONNEL"
sources.extend(df["ETHIQUE"].map(label_to_index))
targets.extend(df["PERSONNEL"].map(label_to_index))
values.extend(df["VOLUMETRIE"])


# flow de "PERSONNEL" à "CONDITIONS_REUTILISATION"
sources.extend(df["PERSONNEL"].map(label_to_index))
targets.extend(df["CONDITIONS_REUTILISATION"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

# flow de "CONDITIONS_REUTILISATION" à "RESTRICTIONS"
sources.extend(df["CONDITIONS_REUTILISATION"].map(label_to_index))
targets.extend(df["RESTRICTIONS"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

# flow de "RESTRICTIONS" à "ENTREPOT_SPECIFIQUE"
sources.extend(df["RESTRICTIONS"].map(label_to_index))
targets.extend(df["ENTREPOT_SPECIFIQUE"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

# flow de "ENTREPOT_SPECIFIQUE" à "EMBARGO"
sources.extend(df["ENTREPOT_SPECIFIQUE"].map(label_to_index))
targets.extend(df["EMBARGO"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

# flow de "EMBARGO" à "LICENCE"
sources.extend(df["EMBARGO"].map(label_to_index))
targets.extend(df["LICENCE"].map(label_to_index))
values.extend(df["VOLUMETRIE"])


#### GESTION DES COULEURS
colors_for_nodes = ["yellow"] * len(df['NATURE'].unique()) + ["white"] * (len(df["INTITULE_DONNEES"].unique())) + ["white"] * (len(df["ETHIQUE"].unique())) + ["white"] * (len(df["PERSONNEL"].unique())) + ["white"] * (len(df["CONDITIONS_REUTILISATION"].unique())) + ["white"] * (len(df["RESTRICTIONS"].unique())) + ["white"] * (len(df["ENTREPOT_SPECIFIQUE"].unique())) + ["white"] * (len(df["EMBARGO"].unique())) + ["white"] * (len(df["LICENCE"].unique()))
colors_for_links = colors_for_links_init * (len(required_columns)-1)


####################################
### Sankey diagram
####################################
fig = go.Figure(go.Sankey(
    arrangement='freeform',
    #orientation="v",
    node=dict(
        pad=80,
        thickness=15,
        line=dict(color="grey", width=1),
        label=all_labels),
    link=dict(
        #arrowlen=50,
        source=sources,
        target=targets,
        value=values
    )
))

fig.update_layout(
    hovermode = 'x',
    title=dict(text="<b> PARTAGE DES DONNEES GREENSCALE </b>", font=dict(color="black",size=18), x=0.3, y=0.01),
    font=dict(size = 15, color = 'black'),
    plot_bgcolor='black',
    paper_bgcolor='snow'
)

fig.update_traces(node_color = colors_for_nodes,
                  link_color = colors_for_links)

fig.add_annotation(dict(font=dict(color="black",size=18), x=0.01, y=1.1, showarrow=False, text='<b> NATURE </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.12, y=1.1, showarrow=False, text='<b> INTITULES </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.22, y=1.1, showarrow=False, text='<b> ETHIQUE </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.38, y=1.1, showarrow=False, text='<b> PERSONNEL </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.50, y=1.1, showarrow=False, text='<b> REUTILISATION </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.65, y=1.1, showarrow=False, text='<b> RESTRICTIONS </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.8, y=1.1, showarrow=False, text='<b> ENTREPOT </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.9, y=1.1, showarrow=False, text='<b> EMBARGO </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=1.05, y=1.1, showarrow=False, text='<b> LICENCE </b>'))

# Visualisation
fig.show()