import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
import textwrap

pio.templates.default = "plotly"

###############################################################################################
########### TITRE DE L'ONGLET #################################################################
###############################################################################################
st.set_page_config(
    page_title="FAIRCARBON DATA",
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
    # Chemin vers le fichier Excel
    #fichier_excel = "Data\FairCarboN_Datas_V2.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=0,header=1, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")

    return df

@st.cache_data
def read_data_code(path):
    # Chemin vers le fichier Excel
    #fichier_excel = "Data\FairCarboN_Datas_V2.xlsx"
    # Lecture du fichier Excel dans un DataFrame
    df = pd.read_excel(f"{path}.xlsx", sheet_name=1,header=1, engine='openpyxl')
    # Transformation du fichier en csv
    df.to_csv(f"{path}.csv", index=False, encoding="utf-8")

    return df

def parse_numero(val):
    if pd.isna(val) or str(val).strip() == "":
        return "Non renseigné", "Non renseigné"
    
    val = str(val).strip()
    
    # Cas "GT1", "GT2", etc. → seulement un groupe de tâches
    match_gt = re.match(r'^GT(\d+)$', val, re.IGNORECASE)
    if match_gt:
        return f"Groupe {match_gt.group(1)}", "Non renseigné"
    
    # Cas "T3.1", "T3.2", etc. → groupe + tâche
    match_t = re.match(r'^T(\d+)\.(\d+)$', val, re.IGNORECASE)
    if match_t:
        return f"Groupe {match_t.group(1)}", f"Tâche {match_t.group(2)}"
    
    # Cas non reconnu → valeur brute
    return val, "Non renseigné"

color_map = {
    "Groupe 1": "#636EFA",
    "Groupe 2": "#EF553B",
    "Groupe 3": "#00CC96",
}
###############################################################################################
########### DATA DONNEES ##############################################################################
###############################################################################################

data = read_data("Data\cartomodeles_V2_040626")

data_columns = {"n° Tâche SLAM-B (sélectionner dans le menu déroulant)":"Numero",
                "Prénom Nom":"Prenom_Nom",
                "Partenaire":"Partenaire",
                "Nom du jeu de données":"Nom",
                "Type de données (sélectionner dans le menu déroulant, sauf si autre catégorie plus adéquate)":"Type",
                "Origine des données (sélectionner dans le menu déroulant, sauf si autre catégorie plus adéquate)":"Origine",
                "Accès et Diffusion (actuellement)":"Acces",
                "Variables principales":"Variables",
                "Emprise spatiale":"Emprise_spatiale",
                "Résolution spatiale":"Resolution_spatiale",
                "Période temporelle":"Periode_temporelle",
                "Résolution temporelle":"Resolution_temporelle",
                "Domaine du jeu de données (sélectionner dans le menu déroulant, sauf si autre catégorie plus adéquate)":"Domaine",
                "Sous-domaine du jeu de données (sélectionner dans le menu déroulant, sauf si autre catégorie plus adéquate)":"Sous_domaine"}

data = data.rename(columns=data_columns)

data['Recensement']="Données"

data[["Groupe de tâches", "Tâche"]] = data["Numero"].apply(
    lambda x: pd.Series(parse_numero(x))
)
data["GT - Tâche"]= data["Groupe de tâches"] + data["Tâche"]

data = data.fillna("Non renseigné")

df_agg = (
    data.groupby(["Groupe de tâches", "Tâche","Prenom_Nom"])
    .size()
    .reset_index(name="count")
)

df_agg["label_nom"] = df_agg.apply(
    lambda row: '<br>'.join(str(row["Prenom_Nom"]).split(" ")),
    axis=1
)

fig_organigramme = px.treemap(
    df_agg,
    path=["Groupe de tâches","Tâche","label_nom"],
    values="count",
    color="Groupe de tâches",
    color_discrete_map=color_map,
    title="Treemap Données"
)

fig_organigramme.update_traces(
    texttemplate="%{label} (%{value})",  # affiche nom + count
    textfont=dict(size=22)
)

fig_organigramme.update_layout(
    uniformtext=dict(
        minsize=16,     # ← taille minimum du texte
        mode='show'     # ← 'show' force l'affichage même si ça dépasse la case
                        #    'hide' masque le texte si la case est trop petite
    )
)

st.title("Données")
st.plotly_chart(fig_organigramme, use_container_width=True)

st.write(f"nombre de lignes : {len(data)}")
st.write(f"Nombre de répondants : {len(data["Prenom_Nom"].unique())}")
st.write(f"Nombre de tâches identifiées : {len(data["GT - Tâche"].unique())}")
st.write(df_agg['count'].mean())

fig_histo= px.bar(
    df_agg,
    x="Prenom_Nom",
    y="count",
    text="count",
    color="Groupe de tâches",   # ← colorie par groupe de tâches
    barmode="group",             # ← 'group' côte à côte, 'stack' empilé
    title="Répartition par personne",
    width=1200,
    color_discrete_map=color_map,
)

fig_histo.update_traces(
    width=0.8  # ← largeur de chaque barre, entre 0 et 1
)
# Trier par ordre décroissant
fig_histo.update_layout(
    xaxis={"categoryorder": "total descending"}
)


with st.container(border=True):
    st.plotly_chart(fig_histo,use_container_width=False)

#fig_organigramme3.write_html("organigramme_arbre.html")

###############################################################################################
########### DATA CODE ##############################################################################
###############################################################################################

data_code = read_data_code("Data\cartomodeles_V2_040626")

data_code_columns = {"n° Tâche SLAM-B (sélectionner dans le menu déroulant)":"Numero",
                    "Prénom Nom":"Prenom_Nom",
                    "Partenaire":"Partenaire",
                    "Nom du modèle / code / algorithme / logiciel":"Nom_logiciel",
                    "Description brève , objectifs principaux":"Objectifs",
                    "Domaine principal (sélectionner dans le menu déroulant, sauf si autre catégorie plus adéquate)":"Domaine",
                    "Caractéristiques techniques":"Caracteristiques",
                    "Domaine de validité":"Validite",
                    "Principales limites":"Limites",
                    "Acces et Diffusion (actuellement)":"Acces",
                    "Maturité (sélectionner dans le menu déroulant)":"Maturite"}

data_code = data_code.rename(columns=data_code_columns)

data_code['Recensement']="Codes"

data_code[["Groupe de tâches", "Tâche"]] = data_code["Numero"].apply(
    lambda x: pd.Series(parse_numero(x))
)
data_code["GT - Tâche"]= data_code["Groupe de tâches"] + data_code["Tâche"]

data_code = data_code.fillna("")

df_code_agg = (
    data_code.groupby(["Groupe de tâches", "Tâche","Prenom_Nom"])
    .size()
    .reset_index(name="count")
)

df_code_agg["label_nom"] = df_code_agg.apply(
    lambda row: '<br>'.join(str(row["Prenom_Nom"]).split(" ")),
    axis=1
)

fig_organigramme_code = px.treemap(
    df_code_agg,
    path=["Groupe de tâches","Tâche", "Prenom_Nom"],
    values="count",
    title="Treemap Codes",
    color="Groupe de tâches",
    color_discrete_map=color_map,
)

fig_organigramme_code.update_traces(
    texttemplate="%{label} (%{value})",  # affiche nom + count
    textfont=dict(size=22)
)

fig_organigramme_code.update_layout(
    uniformtext=dict(
        minsize=16,     # ← taille minimum du texte
        mode='show'     # ← 'show' force l'affichage même si ça dépasse la case
                        #    'hide' masque le texte si la case est trop petite
    )
)

fig_histo_code= px.bar(
    df_code_agg,
    x="Prenom_Nom",
    y="count",
    text="count",
    color="Groupe de tâches",   # ← colorie par groupe de tâches
    barmode="group",             # ← 'group' côte à côte, 'stack' empilé
    title="Répartition par personne",
    width=1200,
    color_discrete_map=color_map,
)

fig_histo_code.update_traces(
    width=0.8  # ← largeur de chaque barre, entre 0 et 1
)
# Trier par ordre décroissant
fig_histo_code.update_layout(
    xaxis={"categoryorder": "total descending"}
)

st.title("Codes")
st.write(f"nombre de lignes : {len(data_code)}")
st.write(f"Nombre de répondants : {len(data_code["Prenom_Nom"].unique())}")
st.write(f"Nombre de tâches identifiées : {len(data_code["GT - Tâche"].unique())}")
st.write(df_code_agg['count'].mean())
st.plotly_chart(fig_organigramme_code, use_container_width=True)

with st.container(border=True):
    st.plotly_chart(fig_histo_code,use_container_width=False)


liste1 = [nom.lower().strip() for nom in data["Prenom_Nom"].unique()]
liste2 = [nom.lower().strip() for nom in data_code["Prenom_Nom"].unique()]

uniquement_liste1 = set(liste1) - set(liste2)
uniquement_liste2 = set(liste2) - set(liste1)
communs           = set(liste1) & set(liste2)

###############################################################################################
########### DATA UNIFIE ##############################################################################
###############################################################################################


df = pd.concat([data, data_code], ignore_index=True)

# Tri : Groupe de tâches → Tâche → Recensement
df = df.sort_values(["Groupe de tâches", "Tâche", "Recensement", "Domaine"]).reset_index(drop=True)

st.dataframe(df)
df.fillna("", inplace=True)

NIVEAUX = [
    "Groupe de tâches",
    "Tâche",
    "Recensement",
    ["Type", "Maturite"],
    ["Nom", "Nom_logiciel"]
]

# ─────────────────────────────────────────────
# 3. Fonction utilitaire : valeur du niveau pour une ligne
# ─────────────────────────────────────────────
def get_val(row, niveau):
    if isinstance(niveau, list):
        for col in niveau:
            if col in row.index and pd.notna(row[col]):
                return str(row[col]), col
        return None, None
    else:
        val = row[niveau]
        return (str(val), niveau) if pd.notna(val) else (None, None)

# ─────────────────────────────────────────────
# 4. Couleurs par groupe (niveau 1)
# ─────────────────────────────────────────────
color_map2 = {
    "Groupe 1": "#636EFA",
    "Groupe 2": "#EF553B",
    "Groupe 3": "#00CC96",
}
DEFAULT_COLOR = "#AAAAAA"

color_map = {
    "1.environnement": "#125A27",
    "2.agriculture": "#E98A0D",
    "3.bioéconomie": "#B8187B",
    "4.socio-économie": "#BAD459",
}

# ─────────────────────────────────────────────
# 5. Construction générique des nœuds
# ─────────────────────────────────────────────
ids      = []
labels   = []
parents  = []
values   = []
colors   = []
feuilles_data = {}  # ← stockage des données des feuilles
noeuds_vus = set()

for _, row in df.iterrows():

    path_ids = []

    grp_val, _ = get_val(row, NIVEAUX[0])
    node_color  = color_map2.get(grp_val, DEFAULT_COLOR)

    for depth, niveau in enumerate(NIVEAUX):

        val, col = get_val(row, niveau)
        if val is None:
            break

        node_id   = "|".join(path_ids + [col + "=" + val])
        parent_id = "|".join(path_ids) if path_ids else ""
        path_ids.append(col + "=" + val)

        if node_id not in noeuds_vus:
            noeuds_vus.add(node_id)

            mask = pd.Series([True] * len(df))
            for d2, n2 in enumerate(NIVEAUX[:depth + 1]):
                v2, c2 = get_val(row, n2)
                if c2:
                    mask &= (df[c2] == row[c2])
            nb = mask.sum()

            ids.append(node_id)
            labels.append(val)
            parents.append(parent_id)
            values.append(int(nb))
            colors.append(node_color)

            # ← stocker les données si c'est une feuille (niveau 5)
            if depth == len(NIVEAUX) - 1:
                feuilles_data[node_id] = row.to_dict()

# ─────────────────────────────────────────────
# Colonnes supplémentaires à afficher dans les cases
# ─────────────────────────────────────────────
COLS_EXTRA = ["Nature","Description","Domaine","Sous_domaine", "Objectifs","Variables","Emprise_spatiale","Periode_temporelle","Acces","Validite","Limites"]  # ← adaptez à vos colonnes

import textwrap

def format_texte(row_dict):
    lignes = ["<b>" + str(row_dict.get("Nom", row_dict.get("Nom_logiciel", ""))) + "</b>"]
    for col in COLS_EXTRA:
        val = row_dict.get(col, "")
        if val and str(val).strip() not in ("", "nan", "None"):
            # Couper les valeurs longues toutes les N caractères
            val_str = str(val)
            val_wrapped = "<br>".join(textwrap.wrap(val_str, width=200))
            lignes.append(f"<b>{col}</b> : {val_wrapped}")
    return "<br>".join(lignes)

customdata = [
    [format_texte(feuilles_data[nid])] if nid in feuilles_data else [""]
    for nid in ids
]

# texttemplate simplifié : on affiche juste la chaîne déjà construite
texttemplate = "%{customdata[0]}"

hovertemplate = "<b>%{label}</b><br>" + "<br>".join(
    f"<b>{col}</b> : %{{customdata[{i}]}}"
    for i, col in enumerate(COLS_EXTRA)
) + "<extra></extra>"

# Palette pour la colonne de coloration
PALETTE_FEUILLES = {
    "Produites": "#FF6B6B",
    "Réutilisées": "#4ECDC4",
    "Simulées": "#FFE66D",
    "Aggrégées": "#294286"
}
DEFAULT_FEUILLE_COLOR = "#CCCCCC"

# Réécrire les couleurs pour les feuilles
colors_final = []
for nid, c in zip(ids, colors):
    if nid in feuilles_data:
        val_col = str(feuilles_data[nid].get(COLS_EXTRA[0], ""))
        colors_final.append(PALETTE_FEUILLES.get(val_col, DEFAULT_FEUILLE_COLOR))
    else:
        colors_final.append(c)  # nœuds intermédiaires gardent leur couleur d'origine

# ─────────────────────────────────────────────
# 6. Figure Plotly
# ─────────────────────────────────────────────
fig = go.Figure(go.Treemap(
    ids=ids,
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    marker=dict(colors=colors_final, line=dict(width=2, color="white")),
    textfont=dict(size=16, family="Arial"),
    customdata=customdata,
    texttemplate=texttemplate,
    hovertemplate=hovertemplate,
    pathbar=dict(visible=True, side="top", thickness=28),
))

# ── Légende : couleurs des groupes (nœuds intermédiaires) ──
for nom, couleur in color_map2.items():
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=12, color=couleur, symbol="square"),
        name=nom,
        legendgroup="groupes",
        legendgrouptitle=dict(text="Groupes") if nom == list(color_map2.keys())[0] else None,
        showlegend=True,
    ))

# ── Légende : couleurs des feuilles (COLS_EXTRA[0]) ──
for nom, couleur in PALETTE_FEUILLES.items():
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(size=12, color=couleur, symbol="square"),
        name=nom,
        legendgroup="feuilles",
        legendgrouptitle=dict(text=COLS_EXTRA[0]) if nom == list(PALETTE_FEUILLES.keys())[0] else None,
        showlegend=True,
    ))

# ── Couleur par défaut si valeur inconnue ──
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(size=12, color=DEFAULT_FEUILLE_COLOR, symbol="square"),
    name="Non renseigné",
    legendgroup="feuilles",
    showlegend=True,
))

fig.update_layout(
    title=dict(
        text="Organigramme",
        font=dict(size=20, family="Arial Black"),
        x=0.5, xanchor="center",
    ),
    legend=dict(
        orientation="v",
        x=1.01,
        y=0.5,
        xanchor="left",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#CCCCCC",
        borderwidth=1,
        tracegroupgap=15,
    ),
    margin=dict(t=80, l=10, r=10, b=10),
    height=650,
)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# ─────────────────────────────────────────────
# 7. Affichage Streamlit
# ─────────────────────────────────────────────
st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# 8. Export HTML autonome
# ─────────────────────────────────────────────
fig.write_html("organigramme.html", full_html=True)

###############################################################################################
########### NATURES ##############################################################################
###############################################################################################

# Décompte des natures
nature_counts = df["Nature"].value_counts().reset_index()
nature_counts.columns = ["Nature", "Count"]

# Couleurs selon la palette
colors = [PALETTE_FEUILLES.get(n, DEFAULT_FEUILLE_COLOR) for n in nature_counts["Nature"]]

fig_nature = go.Figure(data=[go.Pie(
    labels=nature_counts["Nature"],
    values=nature_counts["Count"],
    marker=dict(colors=colors, line=dict(color="white", width=2)),
    textinfo="label+percent+value",
    hovertemplate="<b>%{label}</b><br>Nombre : %{value}<br>Part : %{percent}<extra></extra>"
)])

fig_nature.update_layout(
    title="Répartition par Nature",
    legend=dict(orientation="v", x=1.05, y=0.5)
)

st.plotly_chart(fig_nature, use_container_width=True)

###############################################################################################
########### CATE ##############################################################################
###############################################################################################

# --- Extraction du nom et du type (use/dev) ---
def extract_info(categorie):
    cat = str(categorie).strip()
    if cat.endswith("use"):
        return cat[:-3].strip(), "use"
    elif cat.endswith("dev"):
        return cat[:-3].strip(), "dev"
    else:
        return cat, "autre"

data_code[["nom", "type"]] = data_code["Catégorie"].apply(
    lambda x: pd.Series(extract_info(x))
)

# --- Comptage par nom ---
counts = data_code["nom"].value_counts().reset_index()
counts.columns = ["nom", "count"]

# --- Type majoritaire par nom (pour la couleur) ---
type_par_nom = data_code.groupby("nom")["type"].agg(
    lambda x: x.value_counts().idxmax()
).reset_index()
type_par_nom.columns = ["nom", "type_majoritaire"]

counts = counts.merge(type_par_nom, on="nom")

# --- Palette de couleurs ---
palette = {
    "use": "#636EFA",   # bleu
    "dev": "#EF553B",   # rouge
    "autre": "#AAAAAA", # gris
}
couleurs = counts["type_majoritaire"].map(palette).tolist()

# --- Graphique Pie ---
fig_cat = go.Figure(go.Pie(
    labels=counts["nom"],
    values=counts["count"],
    marker=dict(colors=couleurs),
    textinfo="label+value",
    hovertemplate="<b>%{label}</b><br>Occurrences : %{value}<br>Part : %{percent}<extra></extra>"
))


fig_cat.update_layout(
    title="Répartition des noms (bleu = use, rouge = dev)",
    legend_title="Type",
    showlegend=True,
)

# Ajoute des traces fantômes juste pour la légende
for type_label, couleur in palette.items():
    if type_label != "autre":  # on exclut "autre" si tu n'en as pas besoin
        fig_cat.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=10, color=couleur, symbol="square"),
            name=type_label,
            showlegend=True,
        ))

# Masquer la légende auto du Pie
fig_cat.update_traces(showlegend=False, selector=dict(type="pie"))

fig_cat.update_xaxes(visible=False)
fig_cat.update_yaxes(visible=False)

# --- Affichage Streamlit ---
st.plotly_chart(fig_cat, use_container_width=True)