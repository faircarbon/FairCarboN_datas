import pandas as pd
import plotly.graph_objects as go

# Charger le fichier csv
csv_file = r"Data\20250602_Questionnaire_PGD_CANETE_GL.csv" 
df_i = pd.read_csv(csv_file, encoding="utf-8", sep=";")

df_i_p = df_i[df_i["NATURE"]=="PRODUITES"]
df_i_e = df_i[df_i["NATURE"]=="PRE_EXISTANTES"]
df_= pd.concat([df_i_p, df_i_e], axis=0)



df = df_.sort_values(by='VOLUMETRIE', ascending=True)

# Optional: reset the index if you want a clean index after sorting
df = df.reset_index(drop=True)


for i in range(len(df)):
    if df.loc[i,"EXTENSION_FORMAT"]=="AUTRE_(FORMAT)":
        df.loc[i,"EXTENSION_FORMAT"]=df.loc[i,"AUTRE_EXTENSION_FORMAT"]
    if df.loc[i,"LOGICIEL"]=="AUTRE_(LOGICIEL)":
        df.loc[i,"LOGICIEL"]=df.loc[i,"AUTRE_LOGICIEL"]


# s'assurer que les colonnes existent bien
required_columns = {"INTITULE_DONNEES","NATURE","TYPE","EXTENSION_FORMAT","LOGICIEL","VOLUMETRIE"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Convertir en numerique la valeur de volumétrie (au cas où)
df["VOLUMETRIE"] = pd.to_numeric(df["VOLUMETRIE"], errors="coerce").fillna(0)

# Creation des labels uniques pour les noeuds
all_labels = list(pd.concat([df["INTITULE_DONNEES"],df["NATURE"]]).unique())

# Mapping à partir des labels index
label_to_index = {label: i for i, label in enumerate(all_labels)}


# Def des liens du Diagram
sources = []
targets = []
values = []


# flow de "INTITULE_DONNEES" à "NATURE"
sources.extend(df["INTITULE_DONNEES"].map(label_to_index))
targets.extend(df["NATURE"].map(label_to_index))
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


"""# flow de "NATURE" à "TYPE"
sources.extend(df["NATURE"].map(label_to_index))
targets.extend(df["TYPE"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

# flow de "TYPE" à "EXTENSION_FORMAT"
sources.extend(df["TYPE"].map(label_to_index))
targets.extend(df["EXTENSION_FORMAT"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

# flow de "EXTENSION_FORMAT" à "LOGICIEL"
sources.extend(df["EXTENSION_FORMAT"].map(label_to_index))
targets.extend(df["LOGICIEL"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""


#### GESTION DES COULEURS
colors_for_nodes = ["yellow"] * len(df["INTITULE_DONNEES"].unique()) + ["white"] * len(df["NATURE"].unique()) 
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
    title=dict(text="<b> DESCRIPTION DES DONNEES CANETE </b>", font=dict(color="black",size=18), x=0.3, y=0.01),
    font=dict(size = 11, color = 'black'),
    plot_bgcolor='black',
    paper_bgcolor='snow'
)

fig.update_traces(node_color = colors_for_nodes,
                  link_color = colors_for_links)

fig.add_annotation(dict(font=dict(color="black",size=18), x=0.01, y=1.1, showarrow=False, text='<b> INTITULES </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.95, y=1.1, showarrow=False, text='<b> NATURE </b>'))




# Visualisation
fig.show()