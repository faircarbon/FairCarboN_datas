import pandas as pd
import plotly.graph_objects as go

# Charger le fichier csv
csv_file = r"Data\20250602_Questionnaire_PGD_CANETE_GL.csv" 
df_i = pd.read_csv(csv_file, encoding="utf-8", sep=";")

df_i_p = df_i[df_i["NATURE"]=="PRODUITES"]
df_i_e = df_i[df_i["NATURE"]=="PRE_EXISTANTES"]
df_= pd.concat([df_i_p, df_i_e], axis=0)

df__ = df_[df_['VOLUMETRIE']>1][df_['VOLUMETRIE']<1000][df_["NATURE"]=="PRODUITES"]

df = df__.sort_values(by='VOLUMETRIE', ascending=True)

# Optional: reset the index if you want a clean index after sorting
df = df.reset_index(drop=True)

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
all_labels = list(pd.concat([df["INTITULE_DONNEES"],df["CONDITIONS_REUTILISATION"],df["RESTRICTIONS"],df["ENTREPOT_SPECIFIQUE"],df["EMBARGO"],df["LICENCE"]]).unique())

# Mapping à partir des labels index
label_to_index = {label: i for i, label in enumerate(all_labels)}

# Def des liens du Diagram
sources = []
targets = []
values = []


# flow de "NATURE" à "INTITULE_DONNEES"
sources.extend(df["INTITULE_DONNEES"].map(label_to_index))
targets.extend(df["RESTRICTIONS"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

import random

def generate_colors(n=30):
    colors = []
    for _ in range(n):
        blue = random.randint(0, 255)  # Composante bleue élevée
        red = random.randint(0, 255)    # Composante rouge modérée
        green = random.randint(0, 255)   # Composante verte faible pour éviter le cyan
        colors.append(f'#{red:02x}{green:02x}{blue:02x}')
    return colors

colors = generate_colors(40)


colors_for_links = []
df_sources = pd.DataFrame(sources)
sources_init = df_sources.value_counts().values
colors_for_links_init=[]
for i, j in enumerate(sources):
    colors_for_links_init.extend([colors[j]])


"""# flow de "INTITULE_DONNEES" à "ETHIQUE"
sources.extend(df["INTITULE_DONNEES"].map(label_to_index))
targets.extend(df["RESTRICTIONS"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""


"""# flow de "ETHIQUE" à "PERSONNEL"
sources.extend(df["ETHIQUE"].map(label_to_index))
targets.extend(df["PERSONNEL"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""

"""
# flow de "PERSONNEL" à "CONDITIONS_REUTILISATION"
sources.extend(df["PERSONNEL"].map(label_to_index))
targets.extend(df["CONDITIONS_REUTILISATION"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""

"""# flow de "CONDITIONS_REUTILISATION" à "RESTRICTIONS"
sources.extend(df["CONDITIONS_REUTILISATION"].map(label_to_index))
targets.extend(df["RESTRICTIONS"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""

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
colors_for_nodes = ["yellow"] * (len(df["INTITULE_DONNEES"].unique())) + ["white"] * (len(df["RESTRICTIONS"].unique())) + ["white"] * (len(df["ENTREPOT_SPECIFIQUE"].unique())) + ["white"] * (len(df["EMBARGO"].unique())) + ["white"] * (len(df["LICENCE"].unique()))
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
    title=dict(text="<b> PARTAGE DES DONNEES CANETE </b>", font=dict(color="black",size=18), x=0.3, y=0.01),
    font=dict(size = 15, color = 'black'),
    plot_bgcolor='black',
    paper_bgcolor='snow'
)

fig.update_traces(node_color = colors_for_nodes,
                  link_color = colors_for_links)

fig.add_annotation(dict(font=dict(color="black",size=18), x=0.01, y=1.1, showarrow=False, text='<b> INTITULES </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.2, y=1.1, showarrow=False, text='<b> RESTRICTIONS </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.5, y=1.1, showarrow=False, text='<b> ENTREPOT </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.8, y=1.1, showarrow=False, text='<b> EMBARGO </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=1.05, y=1.1, showarrow=False, text='<b> LICENCE </b>'))

# Visualisation
fig.show()