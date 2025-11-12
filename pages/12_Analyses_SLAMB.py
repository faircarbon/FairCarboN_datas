import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np
from plotly.subplots import make_subplots

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON SLAMB",
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


######################################################################################################################
########### DONNEES INITIALES ########################################################################################
######################################################################################################################
file1 = "Data\Questionnaires\Questionnaires_SLAM_B"

df = read_data(file1)

st.dataframe(df)


df["NOM_PRENOM"] = df["NOM"]+ "_" +df["PRENOM"]

df["VOLUMETRIE_MAX"]=0.1


# s'assurer que les colonnes existent bien
required_columns = {'LABO', 'TUTELLE', 'STATUT','NOM_PRENOM','NUM_TACHE','NATURE','VOLUMETRIE_MAX'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

# Convertir en numerique la valeur de volumétrie (au cas où)
df["VOLUMETRIE_MAX"] = pd.to_numeric(df["VOLUMETRIE_MAX"], errors="coerce").fillna(0)

# Creation des labels uniques pour les noeuds
all_labels = list(pd.concat([df["NOM_PRENOM"],df["LABO"],df["TUTELLE"], df["STATUT"], df['NUM_TACHE'], df['NATURE']]).unique())

# Mapping à partir des labels index
label_to_index = {label: i for i, label in enumerate(all_labels)}

# Def des liens du Diagram
sources = []
targets = []
values = []


# flow de "NOM_PRENOM" à "LABO"
sources.extend(df["NOM_PRENOM"].map(label_to_index))
targets.extend(df["LABO"].map(label_to_index))
values.extend(df["VOLUMETRIE_MAX"])

import random

def generate_colors(n=30):
    colors = []
    for _ in range(n):
        blue = random.randint(0, 50)  # Composante bleue élevée
        red = random.randint(0, 255)    # Composante rouge modérée
        green = random.randint(0, 255)   # Composante verte faible pour éviter le cyan
        colors.append(f'#{red:02x}{green:02x}{blue:02x}')
    return colors

# Générer et afficher la liste

colors =['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
            'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
            'blueviolet', 'brown', 'burlywood', 'cadetblue',
            'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
            'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
           'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
            'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
            'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
            'darkslateblue', 'darkslategray', 'darkslategrey',
            'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
            'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
            'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
            'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
            'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
            'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
            'lemonchiffon', 'lightblue', 'lightcoral','lightcyan',
            'lightgoldenrodyellow', 'lightgray', 'lightgrey',
            'lightgreen', 'lightpink','lightsalmon', 'lightseagreen',
            'lightskyblue', 'lightslategray','lightslategrey',
            'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
            'linen', 'magenta','maroon', 'mediumaquamarine',
            'mediumblue', 'mediumorchid', 'mediumpurple',
            'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
            'mediumturquoise', 'mediumvioletred', 'midnightblue',
            'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
            'oldlace', 'olive','olivedrab', 'orange', 'orangered',
            'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
            'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
            'plum', 'powderblue', 'purple', 'red', 'rosybrown',
            'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon',
            'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
            'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
            'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
            'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
            'yellow', 'yellowgreen']

#colors = plt.cm.tab20.colors

#colors = generate_colors(30)

colors_for_links = []
df_sources = pd.DataFrame(sources)
sources_init = df_sources.value_counts().values
colors_for_links_init=[]
for i, j in enumerate(sources):
    colors_for_links_init.extend([colors[j]])


# flow de "LABO" à "AFFILIATION"
sources.extend(df["LABO"].map(label_to_index))
targets.extend(df["TUTELLE"].map(label_to_index))
values.extend(df["VOLUMETRIE_MAX"])

# flow de "AFFILIATION" à "STATUT"
sources.extend(df["TUTELLE"].map(label_to_index))
targets.extend(df["STATUT"].map(label_to_index))
values.extend(df["VOLUMETRIE_MAX"])

# flow de "STATUT" à "COORDINATION"
sources.extend(df["STATUT"].map(label_to_index))
targets.extend(df['NUM_TACHE'].map(label_to_index))
values.extend(df["VOLUMETRIE_MAX"])

# flow de "COORDINATION" à "WORKPACKAGE_RESP"
sources.extend(df['NUM_TACHE'].map(label_to_index))
targets.extend(df['NATURE'].map(label_to_index))
values.extend(df["VOLUMETRIE_MAX"])



# GESTION DES COULEURS

colors_for_nodes = ["yellow"] * len(df["NOM_PRENOM"].unique()) + ["white"] * (len(df["LABO"].unique())) + ["white"] * (len(df["TUTELLE"].unique())) + ["white"] * (len(df["STATUT"].unique())) + ["white"] * (len(df['NUM_TACHE'].unique())) + ["white"] * (len(df['NATURE'].unique()))
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
    title=dict(text="<b> CONTRIBUTEURS/TRICES SLAM-B </b>", font=dict(color="black",size=18), x=0.3, y=0.01),
    font=dict(size = 14, color = 'black'),
    plot_bgcolor='black',
    paper_bgcolor='snow'
)

fig.update_traces(node_color = colors_for_nodes,
                  link_color = colors_for_links)

fig.add_annotation(dict(font=dict(color="black",size=18), x=0.01, y=1.1, showarrow=False, text='<b> NOM </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.15, y=1.1, showarrow=False, text='<b> LABO </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.3, y=1.1, showarrow=False, text='<b> TUTELLE </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.5, y=1.1, showarrow=False, text='<b> STATUT </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.73, y=1.1, showarrow=False, text='<b> NUM TACHE </b>'))
fig.add_annotation(dict(font=dict(color="black",size=18), x=0.93, y=1.1, showarrow=False, text='<b> NATURE </b>'))


voir_diagram = st.checkbox('Voir Diagram')
if voir_diagram:
    # Visualisation
    fig.show()