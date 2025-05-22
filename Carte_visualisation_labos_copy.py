import pandas as pd
import folium
from folium.features import CustomIcon
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Chemin vers le fichier Excel
fichier_excel = "Data\FairCarboN_RNSR_copie.xlsx"
# Lecture du fichier Excel dans un DataFrame
df = pd.read_excel(fichier_excel, sheet_name=1,header=0, engine='openpyxl')
df.to_csv("Data\FairCarboN_RNSR_copie.csv", index=False, encoding="utf-8")

# filtrer les lignes incoimplètes
df_filtré = df.dropna(subset=["Latitude", "Longitude"])
# Renommer les colonnes
df_filtré_renommé = df_filtré.rename(columns={
    "Acronyme projet": "projet",
    "Acronyme unité": "laboratoire"
})
df_filtré_renommé.to_csv("Data\FairCarboN_RNSR_copie_filtré_renommé.csv", index=False)

# Charger les données
df = pd.read_csv("Data\FairCarboN_RNSR_copie_filtré_renommé.csv")

# Regrouper par laboratoire
grouped = df.groupby(['laboratoire','Type_Data','Latitude', 'Longitude'])['projet'].apply(list).reset_index()

# Couleurs associées à chaque projet
projects = sorted(df['projet'].unique())
colors = plt.cm.tab20.colors  # Palette de couleurs
project_color_map = {project: colors[i % len(colors)] for i, project in enumerate(projects)}

# Créer la carte
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles='CartoDB positron',  # Ou 'Stamen Toner Lite'
    control_scale=True)  # Centrée sur la France

# Générer des marqueurs en camembert
for _, row in grouped.iterrows():
    projets = row['projet']
    latitude = row['Latitude']
    longitude = row['Longitude']
    type_data = row['Type_Data']

    if type_data == "Analyses":
        # Créer un graphique en camembert
        fig, ax = plt.subplots(figsize=(1, 1))
        projet_counts = [1] * len(projets)  # égale pondération
        colors_used = [project_color_map[proj] for proj in projets]
        #ax.pie(projet_counts, colors=colors_used) version sans bordure
        wedges, _ = ax.pie(
            projet_counts,
            colors=colors_used,
            wedgeprops={'edgecolor': 'black', 'linewidth': 5}  # Bordure noire épaisse
        )
        plt.axis('off')

        # Sauvegarder en mémoire
        img_data = BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        img_data.seek(0)
        encoded = base64.b64encode(img_data.read()).decode()

        icon_url = f"data:image/png;base64,{encoded}"
        icon = folium.CustomIcon(icon_image=icon_url, icon_size=(20, 20))
    
    elif type_data == "Modelisation":
        # Créer un graphique en camembert
        fig, ax = plt.subplots(figsize=(1, 1))
        projet_counts = [1] * len(projets)  # égale pondération
        colors_used = [project_color_map[proj] for proj in projets]
        #ax.pie(projet_counts, colors=colors_used) version sans bordure
        wedges, _ = ax.pie(
            projet_counts,
            colors=colors_used,
            wedgeprops={'edgecolor': 'red', 'linewidth': 5}  # Bordure rouge épaisse
        )
        plt.axis('off')

        # Sauvegarder en mémoire
        img_data = BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        img_data.seek(0)
        encoded = base64.b64encode(img_data.read()).decode()

        icon_url = f"data:image/png;base64,{encoded}"
        icon = folium.CustomIcon(icon_image=icon_url, icon_size=(20, 20))

    # Ajouter le marqueur
    popup = folium.Popup("<br>".join(projets), max_width=200)
    tooltip = row['laboratoire']
    folium.Marker(location=[latitude, longitude], popup=popup, tooltip=tooltip, icon=icon).add_to(m)

# Générer la légende HTML
legend_html = '''
<div style="position: fixed; 
            bottom: 30px; left: 30px; width: 200px; height: auto; 
            z-index:9999; font-size:14px;
            background-color: white;
            border:2px solid grey;
            border-radius:6px;
            padding: 10px;">
<b>Légende des projets</b><br>
'''
for project, color in project_color_map.items():
    hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in color[:3])
    legend_html += f'<i style="background:{hex_color};width:12px;height:12px;display:inline-block;margin-right:6px;"></i>{project}<br>'

legend_html += '</div>'

# Ajouter la légende à la carte
m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarder la carte
m.save("carte_laboratoires.html")