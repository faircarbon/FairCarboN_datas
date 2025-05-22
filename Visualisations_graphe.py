import pandas as pd
import plotly.graph_objects as go
import networkx as nx


unites = pd.read_csv("Data\LABOS.csv", encoding="utf-8", sep=";")
print(unites.columns)

data_projet = unites["Acronyme projet"].values
data_sigles = unites["Sigle structure"].values

data= {
    'Sigles': data_sigles,
    'Projet': data_projet
}

df = pd.DataFrame(data)

# Création d'un graphe
G = nx.Graph()

# Ajout des nœuds et des arêtes
for _, row in df.iterrows():
    nom = row['Sigles']
    projet = row['Projet']
    G.add_node(nom, type='person')
    G.add_node(projet, type='project')
    G.add_edge(nom, projet)

# Positionnement des nœuds avec spring layout
pos = nx.spring_layout(G, seed=42)

# Séparer les nœuds personnes et projets pour des couleurs différentes
node_x = []
node_y = []
node_text = []
node_color = []

for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    node_color.append('blue' if G.nodes[node]['type'] == 'person' else 'green')

# Lignes entre les nœuds
edge_x = []
edge_y = []

for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1, color='#888'),
    hoverinfo='none',
    mode='lines')

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=node_text,
    textposition="top center",
    hoverinfo='text',
    marker=dict(
        showscale=False,
        color=node_color,
        size=20,
        line_width=2))

# Affichage final
fig = go.Figure(data=[edge_trace, node_trace],
         layout=go.Layout(
            title='<br>Réseau Sigles - Projets',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        )

fig.show()