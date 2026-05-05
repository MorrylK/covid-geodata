import json
import glob
import os
import pandas as pd
import plotly.express as px

# ==========================================
# 1. CHARGEMENT ET EXTRACTION DES DONNÉES
# ==========================================
# Chemin vers votre dossier
path = 'data/departements-france-polygons/*.json'
files = sorted(glob.glob(path))

all_data = []
combined_features = []

print(f"Traitement de {len(files)} fichiers...")

for file in files:
    # Extraire la date du nom du fichier (ex: 2020-03-01)
    filename = os.path.basename(file)
    date_str = filename.replace("departements-france-polygons-", "").replace(".json", "")
    
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for feature in data['features']:
            # Aplatissement des propriétés (Population, Beds, Emergencies)
            props = pd.json_normalize(feature['properties']).to_dict(orient='records')[0]
            
            # Ajout des métadonnées temporelles
            props['Date'] = date_str
            all_data.append(props)
            
            # Stockage des géométries (uniquement pour le premier fichier pour alléger la mémoire)
            if file == files[0]:
                combined_features.append(feature)

# Création du DataFrame global
df = pd.DataFrame(all_data).fillna(0)
geojson_france = {"type": "FeatureCollection", "features": combined_features}

# ==========================================
# 2. GÉNÉRATION DE LA CARTE DYNAMIQUE
# ==========================================
fig = px.choropleth_mapbox(
    df,
    geojson=geojson_france,
    locations="Code",                # Clé dans le DataFrame[cite: 1]
    featureidkey="properties.Code",  # Clé dans le GeoJSON[cite: 1]
    color="Emergencies.Total",       # Variable à afficher[cite: 1]
    animation_frame="Date",          # LE CURSEUR TEMPOREL
    hover_name="Province/State",     # Nom affiché au survol[cite: 1]
    color_continuous_scale="Reds",
    range_color=[0, df["Emergencies.Total"].max()],
    mapbox_style="carto-positron",
    zoom=4.5, 
    center={"lat": 46.2276, "lon": 2.2137},
    opacity=0.6,
    title="Évolution de l'épidémie COVID-19 en France"
)

# Optimisation de la mise en page
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

# Sauvegarde au format HTML (interactif dans le navigateur)
fig.write_html("carte_dynamique_covid.html")

print("✅ Carte générée avec succès : 'carte_dynamique_covid.html'")
# fig.show() # Décommenter pour afficher directement si vous utilisez un notebook
