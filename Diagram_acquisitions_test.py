import pandas as pd
import plotly.graph_objects as go

# Charger le fichier csv
csv_file = r"Data\20250602_Questionnaire_PGD_CANETE_GL.csv"  
df_i = pd.read_csv(csv_file, encoding="utf-8", sep=";")

df_i_p = df_i[df_i["NATURE"]=="PRODUITES"]
df_i_e = df_i[df_i["NATURE"]=="PRE_EXISTANTES"]
df_= pd.concat([df_i_p, df_i_e], axis=0)

df__ = df_[df_['VOLUMETRIE']>100]

df = df__.sort_values(by='VOLUMETRIE', ascending=True)

# Optional: reset the index if you want a clean index after sorting
df = df.reset_index(drop=True)

"""#traitement source vide
df["SOURCE"].fillna("NON_CONCERNE_S", inplace=True)
df["MOYENS"].fillna("NON_CONCERNE_M", inplace=True)
df["REFERENCES"].fillna("NON_CONCERNE_R", inplace=True)
df["DOCS_ASSOCIES"].fillna("NON_CONCERNE_D", inplace=True)"""

"""#traitement periode autre
for i in range(len(df)):
    if df.loc[i,"ACQUISITION_AUTO_PERIODE"]=="OTHER_PERIODE":
        df.loc[i,"ACQUISITION_AUTO_PERIODE"]=df.loc[i,"PERIODE_AUTO_AUTRE"]

#stockage brut autre
for i in range(len(df)):
    if df.loc[i,"ACQUISITION_NON_AUTO_MODE"]=="FIXE":
        df.loc[i,"ACQUISITION_NON_AUTO_MODE"]=df.loc[i,"ACQUISITION_FIXEE"]

for i in range(len(df)):
    if df.loc[i,"STOCKAGE_BRUT"]=="OTHER_STOCKAGE_BRUT":
        df.loc[i,"STOCKAGE_BRUT"]=df.loc[i,"STOCKAGE_BRUT_AUTRE"]"""


"""#traitement brutes autre
for i in range(len(df)):
    if df.loc[i,"TRAITEMENT_DONNEES_BRUTES"]=="OTHER_TRAITEMENT":
        df.loc[i,"TRAITEMENT_DONNEES_BRUTES"]=df.loc[i,"TRAITEMENT_AUTRE"]"""


# s'assurer que les colonnes existent bien
required_columns = {"INTITULE_DONNEES","NATURE","ACQUISITION_AUTOMATIQUE","STOCKAGE_BRUT","ACCES_BRUTES","TRAITEMENT_DONNEES_BRUTES","MOYEN_CALCUL","VOLUMETRIE"}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Convertir en numerique la valeur de volumétrie (au cas où)
df["VOLUMETRIE"] = pd.to_numeric(df["VOLUMETRIE"], errors="coerce").fillna(0)

# Creation des labels uniques pour les noeuds
all_labels = list(pd.concat([df["INTITULE_DONNEES"],df["STOCKAGE_BRUT"],df["ACCES_BRUTES"]]).unique())

# Mapping à partir des labels index
label_to_index = {label: i for i, label in enumerate(all_labels)}

# Def des liens du Diagram
sources = []
targets = []
values = []


# flow de "NATURE" à "INTITULE_DONNEES"
sources.extend(df["INTITULE_DONNEES"].map(label_to_index))
targets.extend(df["STOCKAGE_BRUT"].map(label_to_index))
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


"""# flow de "INTITULE_DONNEES" à "ACQUISITION_AUTOMATIQUE"
sources.extend(df["INTITULE_DONNEES"].map(label_to_index))
targets.extend(df["ACQUISITION_AUTOMATIQUE"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""


"""# flow de "ACQUISITION_AUTOMATIQUE" à "STOCKAGE_BRUT"
sources.extend(df["ACQUISITION_AUTOMATIQUE"].map(label_to_index))
targets.extend(df["STOCKAGE_BRUT"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""


# flow de "STOCKAGE_BRUT" à "ACCES_BRUTES"
sources.extend(df["STOCKAGE_BRUT"].map(label_to_index))
targets.extend(df["ACCES_BRUTES"].map(label_to_index))
values.extend(df["VOLUMETRIE"])

"""# flow de "ACCES_BRUTES" à "TRAITEMENT_DONNEES_BRUTES"
sources.extend(df["ACCES_BRUTES"].map(label_to_index))
targets.extend(df["TRAITEMENT_DONNEES_BRUTES"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""

"""# flow de "TRAITEMENT_DONNEES_BRUTES" à "MOYEN_CALCUL"
sources.extend(df["TRAITEMENT_DONNEES_BRUTES"].map(label_to_index))
targets.extend(df["MOYEN_CALCUL"].map(label_to_index))
values.extend(df["VOLUMETRIE"])"""



#### GESTION DES COULEURS
colors_for_nodes = ["yellow"] * (len(df["INTITULE_DONNEES"].unique())) + ["white"] * (len(df["STOCKAGE_BRUT"].unique())) + ["white"] * (len(df["ACCES_BRUTES"].unique())) 
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
    title=dict(text="<b> ACQUISITIONS DES DONNEES CANETE </b>", font=dict(color="black",size=18), x=0.3, y=0.01),
    font=dict(size = 15, color = 'black'),
    plot_bgcolor='black',
    paper_bgcolor='snow'
)

fig.update_traces(node_color = colors_for_nodes,
                  link_color = colors_for_links)


fig.add_annotation(dict(font=dict(color="black",size=18), x=0.01, y=1.1, showarrow=False, text='<b> INTITULES </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.5, y=1.1, showarrow=False, text='<b> STOCKAGE </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=1.01, y=1.1, showarrow=False, text='<b> ACCES </b>'))


# Visualisation
fig.show()