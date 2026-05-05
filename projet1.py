import json
import glob
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# =================================================================
# 1. CHARGEMENT ET TRAITEMENT DES DONNÉES MANQUANTES
# =================================================================

def load_and_clean_data(json_path):
    files = sorted(glob.glob(json_path))
    all_records = []
    geometries = []

    print(f"📦 Chargement de {len(files)} fichiers GeoJson...")

    for i, file in enumerate(files):
        # Extraction de la date depuis le nom du fichier
        date_str = os.path.basename(file).split('-')[-3:] # Ajuster selon le format exact
        date_str = "-".join(date_str).replace('.json', '')
        
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for feature in data['features']:
                # Aplatissement des dictionnaires imbriqués (Population, Beds, etc.)
                props = pd.json_normalize(feature['properties']).to_dict(orient='records')[0]
                props['Date'] = date_str
                all_records.append(props)
                
                # On stocke la géométrie une seule fois pour la carte
                if i == 0:
                    geometries.append(feature)

    df = pd.DataFrame(all_records)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # --- GESTION DES DONNÉES MANQUANTES (Point critique de l'énoncé) ---
    # Calcul du taux de remplissage
    missing_rate = df.isnull().mean() * 100
    
    # 1. Suppression des colonnes avec > 40% de données vides (ex: MedicalTests.Confirmed à 91%)
    cols_to_drop = missing_rate[missing_rate > 40].index
    df = df.drop(columns=cols_to_drop)
    print(f"🗑️ Colonnes supprimées (>40% vide) : {list(cols_to_drop)}")

    # 2. Imputation pour les colonnes restantes
    # Pour les séries temporelles, l'interpolation linéaire est préférable à la moyenne
    df = df.sort_values(['Code', 'Date'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby('Code')[numeric_cols].transform(lambda x: x.interpolate().fillna(0))

    geojson = {"type": "FeatureCollection", "features": geometries}
    return df, geojson

# =================================================================
# 2. INGÉNIERIE DES DONNÉES (VACCIN & NORMALISATION)
# =================================================================

def enrich_with_vaccination(df):
    """
    Simule ou intègre les données de vaccination. 
    Pour un projet réel, on mergerait avec 'vacsi-dep-fra.csv' de data.gouv.fr
    """
    # Ici, nous créons une corrélation théorique pour l'exercice 
    # (Taux croissant avec le temps à partir de 2021)
    df['Vaccination_Rate'] = 0.0
    mask_2021 = df['Date'] > '2021-01-01'
    # Simulation d'une progression logistique de la vaccination
    days_since_start = (df['Date'] - df['Date'].min()).dt.days
    df['Vaccination_Rate'] = 1 / (1 + np.exp(-0.02 * (days_since_start - 350)))
    
    return df

# =================================================================
# 3. CLASSIFICATION NON SUPERVISÉE (CLUSTERING)
# =================================================================

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

def dynamic_temporal_clustering(df, train_end_date='2020-06-01', n_clusters=4):
    """
    Entraîne le modèle sur une période de référence (ex: 1ère vague) 
    et classe les départements de manière dynamique jour après jour.
    """
    print(f"🧠 Entraînement du modèle sur les données jusqu'au {train_end_date}...")
    
    # 1. Sélection des indicateurs QUOTIDIENS (on normalise par la population)
    # On évite les cumuls ici, on veut la "photographie" du jour
    df['Urgences_Quotidiennes_100k'] = (df['Emergencies.Total'] / df['Population.Total']) * 100000
    features = ['Urgences_Quotidiennes_100k'] 
    
    # 2. Création du jeu d'entraînement (La période de référence)
    # On prend toutes les observations avant notre date limite
    df_train = df[df['Date'] <= train_end_date].dropna(subset=features)
    
    # 3. Normalisation et Entraînement (Fit)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train)
    
    # 4. Prédiction sur l'ENSEMBLE des 378 fichiers (Predict)
    # On transforme tout le dataset avec le même scaler
    X_all = scaler.transform(df[features].fillna(0)) 
    df['Cluster_Dynamique'] = kmeans.predict(X_all)
    
    # 5. Tri des clusters pour la lisibilité (du moins grave au plus grave)
    # On calcule le centre de chaque cluster pour les ordonner
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_clusters = cluster_centers.argsort()
    mapping = {sorted_clusters[i]: f"État {i+1} (Gravité {i+1}/4)" for i in range(n_clusters)}
    
    df['Etat_Epidemique'] = df['Cluster_Dynamique'].map(mapping)
    
    print("✅ Clustering dynamique terminé. Les départements voyageront entre les clusters au fil du temps.")
    return df

# Utilisation :
# df_covid = dynamic_temporal_clustering(df_covid, train_end_date='2020-06-01')

# =================================================================
# 4. VISUALISATION DYNAMIQUE ET SYNCHRONISÉE
# =================================================================

def create_dynamic_map(df, geojson):
    # On s'assure que la date est au format string pour Plotly
    df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.sort_values('Date')

    # 1. ASTUCE D'OPTIMISATION : On extrait le numéro de l'état (1, 2, 3 ou 4) en format numérique
    # Cela permet à Plotly de rester sur une échelle continue, beaucoup plus rapide à animer.
    df['Gravite_Num'] = df['Etat_Epidemique'].str.extract(r'(\d)').astype(int)

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations="Code",
        featureidkey="properties.Code",
        color="Gravite_Num", # On utilise la colonne NUMÉRIQUE pour la vitesse
        animation_frame="Date_Str",
        hover_name="Province/State",
        hover_data={
            "Gravite_Num": False,       # On cache le numéro brut
            "Etat_Epidemique": True,    # On affiche le beau texte
            "Emergencies.Total": True, 
            "Date_Str": False
        }, 
        # On définit nos 4 couleurs dans l'ordre
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"],
        range_color=[1, 4], # On fixe les bornes de 1 à 4
        mapbox_style="carto-positron",
        zoom=4.5,
        center={"lat": 46.2276, "lon": 2.2137},
        opacity=0.7,
        title="<b>Évolution COVID-19 : Départements selon leur État Épidémique (Clusters)</b>"
    )

    # 2. MAQUILLAGE DE LA LÉGENDE : On transforme la barre continue en catégories
    fig.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        coloraxis_colorbar=dict(
            title="Gravité (Cluster)",
            tickvals=[1, 1.75, 2.5, 3.25, 4], # Positionnement des textes sur la barre
            ticktext=[
                "État 1 (Vert)", 
                "", # Espace vide pour centrer
                "État 2 (Jaune)", 
                "État 3 (Orange)", 
                "État 4 (Rouge)"
            ],
            ticks="outside"
        )
    )
    
    # Export
    fig.write_html("covid_france_analysis.html")
    print("✅ Carte dynamique générée et OPTIMISÉE : covid_france_analysis.html")

# =================================================================
# EXÉCUTION DU SCRIPT
# =================================================================

if __name__ == "__main__":
    # Chemin vers vos fichiers JSON locaux
    DATA_PATH = 'data/departements-france-polygons/*.json'

    # 1. ETL & Nettoyage
    df_covid, france_geojson = load_and_clean_data(DATA_PATH)

    # 2. Enrichissement
    #df_covid = enrich_with_vaccination(df_covid)

    # 3. Machine Learning
    df_covid = dynamic_temporal_clustering(df_covid, train_end_date='2020-06-01', n_clusters=4)
    # 4. Analyse de corrélation
    #corr = df_covid[['Vaccination_Rate', 'Emergencies.Total']].corr().iloc[0,1]
    #print(f"📊 Corrélation Vaccination / Urgences : {corr:.4f}")

    # 5. Visualisation
    create_dynamic_map(df_covid, france_geojson)