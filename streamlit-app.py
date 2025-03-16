#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface Streamlit pour l'Analyseur d'Émagramme pour Parapentistes
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime
from retry_requests import retry
import json
from streamlit_geolocation import streamlit_geolocation

# Importer les classes et fonctions du fichier principal
from emagramme_analyzer import (
    EmagrammeAnalyzer,
    EmagrammeDataFetcher,
    THERMAL_TRIGGER_DELTA
)

# Importer l'analyse améliorée IA pour les parapentistes
# Ajouter cette nouvelle importation
from enhanced_emagramme_analysis import (
    EnhancedEmagrammeAgent,
    analyze_emagramme_for_pilot
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("streamlit_emagramme_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('StreamlitEmagrammeApp')

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyseur d'Émagramme pour Parapentistes",
    page_icon="🪂",
    layout="wide"
)

# Définition des sites de parapente prédéfinis
PRESET_SITES = [
    {"name": "Cuges pey gros", "lat": 43.271766, "lon": 5.669529, "altitude": 436, "model": "arome"},
    {"name": "Cuges Sud ouest", "lat": 43.283557, "lon": 5.689622, "altitude": 380, "model": "arome"},
    {"name": "Sainte Victoire", "lat": 43.5333, "lon": 5.5667, "altitude": 900, "model": "arome"},
    {"name": "Garlaban", "lat": 43.332935, "lon": 5.553419, "altitude": 701, "model": "arome"},
    {"name": "Gourdon", "lat": 43.7242, "lon": 6.9792, "altitude": 760, "model": "arome"},
    {"name": "Col de Bleyne", "lat": 43.8502, "lon": 6.7639, "altitude": 1550, "model": "arome"},
    {"name": "Saint André les Alpes", "lat": 43.9809, "lon": 6.5039, "altitude": 950, "model": "arome"},
    {"name": "Ceillac", "lat": 44.6700, "lon": 6.7800, "altitude": 1640, "model": "arome"},
    {"name": "Annecy - Montmin", "lat": 45.8119, "lon": 6.2458, "altitude": 1240, "model": "arome"},
    {"name": "Chamonix - Planpraz", "lat": 45.9270, "lon": 6.8700, "altitude": 2000, "model": "arome"},
    {"name": "Saint Hilaire du Touvet", "lat": 45.2950, "lon": 5.8864, "altitude": 1000, "model": "arome"},
]

# Dictionnaire des aides contextuelles pour les différents concepts
help_texts = {
    "emagramme": "Un émagramme est un graphique météorologique qui représente la température et l'humidité de l'atmosphère à différentes altitudes. Il permet d'analyser la stabilité de l'air et de prédire les conditions de vol.",
    
    "couche_convective": "La couche convective est la portion d'atmosphère où se produisent les mouvements verticaux d'air (thermiques). Plus elle est épaisse, plus le plafond des thermiques sera élevé et meilleur sera le potentiel de vol.",
    
    "inversion": "Une inversion est une couche où la température augmente avec l'altitude (contrairement à la normale). Les inversions bloquent souvent les thermiques et limitent la hauteur de vol.",
    
    "gradient_thermique": "Le gradient thermique mesure la diminution de température avec l'altitude. Un gradient fort (>0.7°C/100m) favorise les thermiques puissants, tandis qu'un gradient faible (<0.5°C/100m) produit des thermiques plus doux.",
    
    "thermique": "Une colonne d'air ascendante générée par le réchauffement du sol. Les thermiques se détachent du sol et s'élèvent jusqu'au sommet de la couche convective, permettant aux parapentistes de gagner de l'altitude.",
    
    "vent_anabatique": "Un vent qui remonte une pente sous l'effet du réchauffement de celle-ci par le soleil. Contrairement aux thermiques, il reste collé à la pente et sa vitesse est généralement plus faible.",
    
    "cumulus": "Nuage à développement vertical qui se forme au sommet d'un thermique lorsque l'air atteint son point de condensation. Les cumulus marquent souvent les meilleurs thermiques.",
    
    "base_nuages": "L'altitude à laquelle se forme la base des cumulus. Elle correspond au niveau où la température de l'air et le point de rosée sont identiques dans un thermique ascendant.",
    
    "point_de_rosee": "Température à laquelle l'air doit être refroidi pour atteindre la saturation en vapeur d'eau. L'écart entre température et point de rosée permet d'estimer l'humidité de l'air.",
    
    "stabilite": "Mesure de la résistance de l'atmosphère aux mouvements verticaux. Une atmosphère instable favorise les thermiques puissants, tandis qu'une atmosphère stable les inhibe.",
    
    "subsidence": "Mouvement descendant de l'air à grande échelle, souvent associé aux anticyclones. La subsidence comprime et réchauffe l'air, créant souvent des inversions qui limitent le développement vertical des thermiques."
}


# Fonction de géolocalisation automatique
def get_user_location():
    """
    Tente d'obtenir la position géographique de l'utilisateur via une API de géolocalisation IP.
    Retourne les coordonnées par défaut en cas d'échec.
    """
    default_location = {
        "latitude": 43.271766,
        "longitude": 5.669529,
        "altitude": 436,
        "city": "PCuges pey gros"
    }
    
    try:
        # Utiliser ipinfo.io pour obtenir la géolocalisation approximative
        response = requests.get('https://ipinfo.io/json', timeout=3)
        if response.status_code == 200:
            data = response.json()
            if 'loc' in data:
                lat, lon = map(float, data['loc'].split(','))
                city = data.get('city', 'Ville inconnue')
                
                # Obtenir l'altitude approximative via Open-Elevation API ou service similaire
                try:
                    elevation_response = requests.get(
                        f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}",
                        timeout=3
                    )
                    if elevation_response.status_code == 200:
                        elevation_data = elevation_response.json()
                        altitude = elevation_data.get('results', [{}])[0].get('elevation', 500)
                    else:
                        altitude = 500  # Valeur par défaut si l'API d'élévation échoue
                except:
                    altitude = 500  # Valeur par défaut en cas d'erreur
                
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": altitude,
                    "city": city
                }
    except Exception as e:
        logger.warning(f"Erreur lors de la géolocalisation: {e}")
    
    return default_location

def search_ffvl_sites(lat, lon, radius=20, api_key="VOTRE_CLE_API"):
    """
    Recherche les sites de vol à proximité d'une position donnée
    en utilisant l'API FFVL.
    
    Args:
        lat: Latitude du point central
        lon: Longitude du point central
        radius: Rayon de recherche en km
        api_key: Clé API FFVL
        
    Returns:
        Liste des sites trouvés
    """
    try:
        # URL de l'API FFVL pour les terrains
        url = f"https://data.ffvl.fr/api?base=terrains&mode=json&key={api_key}"
        
        # Afficher l'URL pour le débogage
        logger.info(f"Requête FFVL: {url}")
        
        response = requests.get(url, timeout=10)
        
        # Vérifier le statut et le contenu de la réponse
        logger.info(f"Statut: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
        
        # Afficher les 100 premiers caractères pour le débogage
        content_preview = response.text[:100].replace('\n', ' ')
        logger.info(f"Aperçu du contenu: {content_preview}...")
        
        if response.status_code != 200:
            st.error(f"Erreur API FFVL: {response.status_code}")
            return []
        
        # Vérifier si la réponse est vide
        if not response.text.strip():
            st.warning("L'API FFVL a renvoyé une réponse vide")
            return []
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            st.error(f"Erreur de décodage JSON: {str(e)}")
            st.code(response.text[:500], language="json")  # Afficher le début de la réponse
            return []
        
        # Vérifier si la structure est correcte
        if not isinstance(data, dict) or "terrains" not in data:
            st.warning(f"Structure de données inattendue: {type(data)}")
            if isinstance(data, dict):
                st.json(data)  # Afficher le JSON reçu
            return []
        
        # Filtrer les sites en fonction de leur distance par rapport au point central
        # (approximation à vol d'oiseau)
        sites = []
        for terrain in data.get("terrains", []):
            # Vérifier que les coordonnées sont valides
            if not terrain.get("latitude") or not terrain.get("longitude"):
                continue
                
            site_lat = float(terrain["latitude"])
            site_lon = float(terrain["longitude"])
            
            # Calcul de distance approximatif (Haversine)
            from math import radians, cos, sin, asin, sqrt
            
            def haversine(lat1, lon1, lat2, lon2):
                # Convertir en radians
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                
                # Formule haversine
                dlon = lon2 - lon1 
                dlat = lat2 - lat1 
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a)) 
                r = 6371  # Rayon de la Terre en km
                return c * r
            
            distance = haversine(lat, lon, site_lat, site_lon)
            
            if distance <= radius:
                # Récupérer les informations pertinentes
                site_info = {
                    "name": terrain.get("nom", "Site sans nom"),
                    "type": terrain.get("type", "Type non spécifié"),
                    "latitude": site_lat,
                    "longitude": site_lon,
                    "distance": round(distance, 1),
                    "altitude": terrain.get("altitude_deco", ""),
                    "orientation": terrain.get("orientations", ""),
                    "status": terrain.get("statut", ""),
                    "difficulty": terrain.get("difficulte", ""),
                    "ffvl_id": terrain.get("id", ""),
                    "icon_url": f"https://data.ffvl.fr/api/?base=terrains&mode=icon&tid={terrain.get('id', '')}"
                }
                
                # Ne garder que les sites de parapente/delta
                if "parapente" in site_info["type"].lower() or "delta" in site_info["type"].lower():
                    sites.append(site_info)
        
        # Trier par distance
        sites.sort(key=lambda x: x["distance"])
        
        return sites
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des sites FFVL: {e}")
        return []
         
# Fonction pour afficher l'émagramme dans Streamlit
def display_emagramme(analyzer, analysis, llm_analysis=None):
    """Affiche l'émagramme et les résultats de l'analyse dans Streamlit"""
    # Utilise la nouvelle fonction fusionnée qui gère l'analyse IA en option
    fig = analyzer.plot_emagramme(analysis=analysis, llm_analysis=llm_analysis, show=False)
    st.pyplot(fig)

def calculate_convective_layer_thickness(analyzer, analysis):
    """Calcule et visualise l'épaisseur de la couche convective"""
    
    # La couche convective se situe entre le sol et le plafond thermique
    ground_altitude = analysis.ground_altitude
    thermal_ceiling = analysis.thermal_ceiling
    thickness = thermal_ceiling - ground_altitude
    
    # Déterminer la qualité des ascendances basée sur l'épaisseur
    quality_description = ""
    if thickness < 1000:
        quality_description = "Faible - Vol de distance difficile"
    elif thickness < 2000:
        quality_description = "Moyenne - Vol local principalement"
    elif thickness < 3000:
        quality_description = "Bonne - Vol de distance possible mais technique"
    else:
        quality_description = "Excellente - Conditions idéales pour le vol de distance"
    
    return {
        "thickness": thickness,
        "description": quality_description,
        "ground_altitude": ground_altitude,
        "thermal_ceiling": thermal_ceiling
    }

def create_convective_layer_plot(analysis, inversions=None):
    """Crée un graphique pour visualiser la couche convective"""
    ground_alt = analysis.ground_altitude
    thermal_ceiling = analysis.thermal_ceiling
    anabatic_zone_top = min(ground_alt + 500, thermal_ceiling)
    
    # Réduire la taille de la figure
    fig, ax = plt.subplots(figsize=(3, 4))  # Taille réduite
    
    # Zone convective complète
    ax.axhspan(ground_alt, thermal_ceiling, alpha=0.1, color='green', 
            label="Couche convective")
    
    # Zone anabatique
    ax.axhspan(ground_alt, anabatic_zone_top, alpha=0.1, color='blue', 
            label="Zone anabatique")
    
    # Lignes horizontales
    ax.axhline(y=ground_alt, color='brown', linestyle='-', linewidth=1)  # Réduire l'épaisseur des lignes
    ax.axhline(y=thermal_ceiling, color='purple', linestyle='--', linewidth=1)
    ax.axhline(y=anabatic_zone_top, color='blue', linestyle='--', linewidth=0.8)
    
    # Réduire la taille du texte des annotations
    ax.text(0.05, ground_alt + 50, f"Sol: {ground_alt:.0f}m", 
            fontsize=6, ha='left', va='bottom')  # Réduire la taille du texte
    ax.text(0.05, thermal_ceiling - 50, f"Plafond: {thermal_ceiling:.0f}m", 
            fontsize=6, ha='left', va='top')
    ax.text(0.05, anabatic_zone_top - 20, f"Limite anab.: {anabatic_zone_top:.0f}m", 
            fontsize=6, ha='left', va='top', color='blue')  # Abréger le texte
    
    # Information sur l'épaisseur avec texte plus petit
    thickness = thermal_ceiling - ground_alt
    ax.text(0.5, (ground_alt + thermal_ceiling) / 2, 
            f"Épaisseur: {thickness:.0f}m", 
            fontsize=7, ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Inversions avec texte plus petit
    if inversions:
        for i, (base, top) in enumerate(inversions):
            if base < thermal_ceiling + 500:
                ax.axhspan(base, top, alpha=0.15, color='red', 
                       label=f"Inversion {i+1}" if i == 0 else "")
                ax.text(0.05, (base + top) / 2, f"Inv{i+1}: {base:.0f}-{top:.0f}m", 
                       fontsize=6, ha='left', va='center', color='darkred')  # Texte plus compact
    
    # Configuration des axes
    ax.set_ylim(max(0, ground_alt - 200), thermal_ceiling + 500)
    ax.set_xlim(0, 1)
    ax.set_ylabel("Altitude (m)", fontsize=7)  # Réduire la taille du label
    ax.set_xticks([])
    ax.tick_params(axis='y', labelsize=6)  # Réduire la taille des graduations
    
    # Titre plus petit
    ax.set_title("Structure verticale de l'atmosphère", fontsize=8)
    
    # Légende plus petite et plus compacte
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
           fancybox=True, shadow=True, ncol=2, fontsize=6)  # Réduire la taille de la légende
    
    # Réduire les marges internes
    plt.tight_layout(pad=1.0)
    
    return fig

# Fonction pour obtenir et analyser les données
def fetch_and_analyze(lat, lon, model, site_altitude, api_key=None, openai_key=None, delta_t=THERMAL_TRIGGER_DELTA, 
                     data_source="open-meteo", timestep=0, fetch_evolution=False, evolution_hours=24, evolution_step=3):
    """
    Récupère les données et effectue l'analyse pour un pas de temps spécifique, 
    avec option pour récupérer l'évolution temporelle
    """
    
    # Mettre à jour le delta de température si nécessaire
    global THERMAL_TRIGGER_DELTA
    if delta_t != THERMAL_TRIGGER_DELTA:
        THERMAL_TRIGGER_DELTA = delta_t
    
    # Initialiser le récupérateur de données
    fetcher = EmagrammeDataFetcher(api_key=api_key)
    
    # Attribuer l'altitude du site au fetcher
    fetcher.site_altitude = site_altitude
    
    # Récupérer les données avec gestion des erreurs
    try:
        if data_source == "open-meteo":
            with st.spinner(f"Récupération des données météo via Open-Meteo (pour H+{timestep})..."):
                st.info("Utilisation d'Open-Meteo (sans clé API)")
                
                levels = None
                evolution_data = None
                
                try:
                    if fetch_evolution:
                        levels, evolution_data = fetcher.fetch_from_openmeteo(
                            lat, lon, model=model, timestep=timestep,
                            fetch_evolution=True, evolution_hours=evolution_hours, evolution_step=evolution_step
                        )
                    else:
                        levels = fetcher.fetch_from_openmeteo(
                            lat, lon, model=model, timestep=timestep
                        )
                except Exception as e:
                    st.error(f"Erreur avec Open-Meteo: {str(e)}")
                    if fetch_evolution:
                        return None, None, None, None
                    else:
                        return None, None, None
                
                if levels is None:
                    st.error("Impossible de récupérer les données météorologiques.")
                    if fetch_evolution:
                        return None, None, None, None
                    else:
                        return None, None, None

        # Récupérer les informations sur les nuages et précipitations si disponibles
        cloud_info = getattr(fetcher, 'cloud_info', None)
        precip_info = getattr(fetcher, 'precip_info', None)
        
        # Initialiser l'analyseur avec toutes les informations
        if timestep > 0:
            model_name = f"{model} H+{timestep}"
        else:
            model_name = model
            
        analyzer = EmagrammeAnalyzer(levels, site_altitude=site_altitude, 
                                   cloud_info=cloud_info, precip_info=precip_info,
                                   model_name=model_name)
        
        # IMPORTANT : Copier les informations directement dans l'objet analyzer
        analyzer.cloud_info = cloud_info
        analyzer.precip_info = precip_info
        
        # Effectuer l'analyse
        analysis = analyzer.analyze()
        
        # Analyse IA si une clé OpenAI est fournie
        detailed_analysis = None
        if openai_key:
            try:
                with st.spinner("Génération de l'analyse par l'IA..."):
                    agent = EnhancedEmagrammeAgent(openai_api_key=openai_key)
                    detailed_analysis = agent.analyze_conditions(analysis)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse IA: {str(e)}")
                logger.error(f"Erreur lors de l'analyse IA: {e}")
                # En cas d'erreur avec OpenAI, utiliser notre analyse interne améliorée
                with st.spinner("Utilisation de l'analyse IA améliorée interne..."):
                    detailed_analysis = analyze_emagramme_for_pilot(analysis)
        else:
            # Si pas de clé OpenAI, utiliser automatiquement notre analyse interne améliorée
            with st.spinner("Génération de l'analyse détaillée..."):
                detailed_analysis = analyze_emagramme_for_pilot(analysis)
        
        # Si fetch_evolution est activé, renvoyer également les données d'évolution
        if fetch_evolution:
            return analyzer, analysis, detailed_analysis, evolution_data
        else:
            return analyzer, analysis, detailed_analysis
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération ou de l'analyse des données: {str(e)}")
        logger.error(f"Erreur lors de la récupération/analyse: {e}", exc_info=True)
        # Retourner les valeurs appropriées selon fetch_evolution
        if fetch_evolution:
            return None, None, None, None
        else:
            return None, None, None

def create_evolution_plots(evolution_data, site_altitude):
    """
    Crée les graphiques d'évolution basés sur les données temporelles
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from datetime import datetime
    import pandas as pd
    
    # Vérifier si les données d'évolution existent
    if not evolution_data or not all(key in evolution_data for key in ["timestamps", "thermal_ceilings", "thermal_gradients"]):
        st.error("Aucune donnée d'évolution disponible.")
        # Retourner un dictionnaire avec les clés attendues mais vide
        empty_best_period = {
            "time": "N/A",
            "score": 0,
            "ceiling": 0,
            "gradient": 0,
            "clouds": 0,
            "rain": 0,
            "temp": 0,
            "wind": 0
        }
        return {}, empty_best_period, pd.DataFrame()
    
    # Créer un DataFrame pour faciliter la manipulation
    try:
        df = pd.DataFrame({
            "timestamp": evolution_data["timestamps"],
            "thermal_ceiling": evolution_data["thermal_ceilings"],
            "thermal_gradient": evolution_data["thermal_gradients"],
            "thermal_strength": evolution_data["thermal_strengths"],
            "temperature": evolution_data["temperatures"],
            "wind_speed": evolution_data["wind_speeds"],
            "wind_direction": evolution_data["wind_directions"],
            "cloud_cover_low": [cloud["low"] for cloud in evolution_data["cloud_covers"]],
            "cloud_cover_mid": [cloud["mid"] for cloud in evolution_data["cloud_covers"]],
            "cloud_cover_high": [cloud["high"] for cloud in evolution_data["cloud_covers"]],
            "cloud_cover_total": [cloud["total"] for cloud in evolution_data["cloud_covers"]],
            "rain": [precip["rain"] for precip in evolution_data["precipitation"]],
            "precip_probability": [precip["probability"] for precip in evolution_data["precipitation"]]
        })
    except Exception as e:
        st.error(f"Erreur lors de la création du DataFrame d'évolution: {e}")
        empty_best_period = {
            "time": "N/A",
            "score": 0,
            "ceiling": 0,
            "gradient": 0,
            "clouds": 0,
            "rain": 0,
            "temp": 0,
            "wind": 0
        }
        return {}, empty_best_period, pd.DataFrame()
    
    # Vérifier si le DataFrame est vide
    if df.empty:
        st.error("Aucune donnée d'évolution disponible.")
        return {}, {}, pd.DataFrame()
    
    # Convertir le plafond thermique relatif (au-dessus du niveau de la mer) à relatif au site
    df["thermal_ceiling_relative"] = df["thermal_ceiling"] - site_altitude
    
    # Convertir les timestamps en format lisible
    df["time_str"] = [ts.strftime("%d/%m %Hh") for ts in df["timestamp"]]
    
    # Créer les graphiques
    graphs = {}
    
    # 1. Évolution du plafond thermique
    fig_ceiling = px.line(df, x="time_str", y="thermal_ceiling_relative", 
                         labels={"thermal_ceiling_relative": "Plafond thermique (m au-dessus du site)", 
                                "time_str": "Date/Heure"},
                         title="Évolution du plafond thermique",
                         markers=True)
    fig_ceiling.update_layout(hovermode="x unified")
    graphs["ceiling"] = fig_ceiling
    
    # 2. Évolution du gradient thermique
    fig_gradient = px.line(df, x="time_str", y="thermal_gradient",
                          labels={"thermal_gradient": "Gradient (°C/1000m)", 
                                 "time_str": "Date/Heure"},
                          title="Évolution du gradient thermique",
                          markers=True)
    fig_gradient.update_layout(hovermode="x unified")
    graphs["gradient"] = fig_gradient
    
    # 3. Évolution de la couverture nuageuse (graphique empilé)
    fig_clouds = go.Figure()
    fig_clouds.add_trace(go.Bar(x=df["time_str"], y=df["cloud_cover_low"],
                              name="Nuages bas", marker_color="royalblue"))
    fig_clouds.add_trace(go.Bar(x=df["time_str"], y=df["cloud_cover_mid"],
                              name="Nuages moyens", marker_color="lightblue"))
    fig_clouds.add_trace(go.Bar(x=df["time_str"], y=df["cloud_cover_high"],
                              name="Nuages hauts", marker_color="lightskyblue"))
    fig_clouds.update_layout(barmode="stack", 
                           title="Évolution de la couverture nuageuse",
                           xaxis_title="Date/Heure", 
                           yaxis_title="Couverture (%)",
                           hovermode="x unified")
    graphs["clouds"] = fig_clouds
    
    # 4. Précipitations et probabilité
    fig_precip = go.Figure()
    fig_precip.add_trace(go.Bar(x=df["time_str"], y=df["rain"],
                              name="Pluie (mm)", marker_color="blue"))
    fig_precip.add_trace(go.Scatter(x=df["time_str"], y=df["precip_probability"],
                                  name="Probabilité (%)", mode="lines+markers",
                                  marker_color="red", line_color="red", yaxis="y2"))
    fig_precip.update_layout(
        title="Prévisions de précipitations",
        xaxis_title="Date/Heure",
        yaxis=dict(title="Pluie (mm)", side="left", showgrid=False),
        yaxis2=dict(title="Probabilité (%)", overlaying="y", side="right", range=[0, 100]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graphs["precipitation"] = fig_precip
    
    # 5. Météogramme combiné (température, vent et plafond)
    fig_meteo = go.Figure()
    
    # Température
    fig_meteo.add_trace(go.Scatter(x=df["time_str"], y=df["temperature"],
                                 name="Température (°C)", mode="lines+markers",
                                 line_color="orange", marker_color="orange"))
    
    # Vent
    fig_meteo.add_trace(go.Scatter(x=df["time_str"], y=df["wind_speed"],
                                 name="Vent (km/h)", mode="lines+markers",
                                 line_color="green", marker_color="green", yaxis="y2"))
    
    # Plafond thermique (échelle secondaire)
    fig_meteo.add_trace(go.Scatter(x=df["time_str"], y=df["thermal_ceiling_relative"],
                                 name="Plafond (m)", mode="lines+markers", line=dict(dash="dash"),
                                 line_color="purple", marker_color="purple", yaxis="y3"))
    
    fig_meteo.update_layout(
        title="Météogramme: Température, Vent et Plafond",
        xaxis_title="Date/Heure",
        yaxis=dict(title="Température (°C)", side="left"),
        yaxis2=dict(title="Vent (km/h)", overlaying="y", side="right"),
        yaxis3=dict(title="Plafond (m)", overlaying="y", side="right", position=0.85),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    graphs["meteo"] = fig_meteo
    
    # Analyse pour trouver les périodes optimales de vol
    df["vol_score"] = (
        df["thermal_ceiling_relative"] / 3000 * 0.4 +  # Plafond (40% du score)
        (df["thermal_gradient"] / 10) * 0.3 +  # Gradient (30% du score)
        (1 - df["cloud_cover_total"] / 100) * 0.1 +  # Ciel clair (10% du score)
        (1 - df["rain"]) * 0.2  # Absence de pluie (20% du score)
    )
    
    # Normaliser les scores entre 0 et 1
    df["vol_score"] = df["vol_score"].clip(0, 1)
    
    # Convertir en pourcentage
    df["vol_score"] = (df["vol_score"] * 100).round(1)
    
    # Graphique des scores de vol
    fig_score = px.bar(df, x="time_str", y="vol_score",
                     labels={"vol_score": "Score (%)", "time_str": "Date/Heure"},
                     title="Évaluation des conditions de vol",
                     color="vol_score",
                     color_continuous_scale=["red", "yellow", "green"],
                     range_color=[0, 100])
    fig_score.update_layout(hovermode="x unified")
    graphs["vol_score"] = fig_score
    
    # Initialiser with default values in case of empty data
    best_period = {
        "time": "N/A",
        "score": 0,
        "ceiling": 0,
        "gradient": 0,
        "clouds": 0,
        "rain": 0,
        "temp": 0,
        "wind": 0
    }
    
    # Trouver la meilleure période (gérer le cas où df est vide)
    if not df.empty and len(df["vol_score"]) > 0:
        try:
            best_index = df["vol_score"].idxmax()
            best_time = df.loc[best_index, "time_str"]
            best_score = df.loc[best_index, "vol_score"]
            
            # Récupérer des informations sur la meilleure période
            best_period = {
                "time": best_time,
                "score": best_score,
                "ceiling": df.loc[best_index, "thermal_ceiling_relative"],
                "gradient": df.loc[best_index, "thermal_gradient"],
                "clouds": df.loc[best_index, "cloud_cover_total"],
                "rain": df.loc[best_index, "rain"],
                "temp": df.loc[best_index, "temperature"],
                "wind": df.loc[best_index, "wind_speed"]
            }
        except Exception as e:
            st.warning(f"Impossible de déterminer la meilleure période: {e}")
    
    return graphs, best_period, df

def analyze_inversions_impact(analysis):
    """Analyse l'impact des inversions sur les ascendances thermiques"""
    
    inversions = analysis.inversion_layers
    if not inversions:
        return {"has_inversions": False, "message": "Aucune inversion détectée"}
    
    # Analyser l'impact de chaque inversion
    impacts = []
    for i, (base, top) in enumerate(inversions):
        # Calculer la force de l'inversion
        strength = top - base
        
        # Déterminer l'impact sur les ascendances
        if base < 1500:
            impact = "Forte limitation des ascendances thermiques à basse altitude"
            severity = "critical"
        elif base < analysis.thermal_ceiling:
            impact = f"Possible limitation du plafond thermique à {base:.0f}m"
            severity = "warning"
        else:
            impact = "Au-dessus de la couche convective, peu d'impact sur le vol"
            severity = "info"
            
        impacts.append({
            "base": base,
            "top": top,
            "strength": strength,
            "impact": impact,
            "severity": severity
        })
    
    return {"has_inversions": True, "inversions": impacts}

def analyze_anabatic_vs_thermal(analysis, site_altitude, site_slope=None):
    """
    Distingue entre les vents anabatiques et les thermiques proprement dits
    
    Args:
        analysis: L'analyse météo complète
        site_altitude: Altitude du site de décollage
        site_slope: Orientation de la pente (en degrés, 0=N, 90=E, etc.) si disponible
    """
    
    # Données de base
    ground_temp = analysis.ground_temperature
    thermal_gradient = analysis.thermal_gradient
    
    # Estimation des vents anabatiques
    anabatic_info = {
        "strength": 0,  # en m/s
        "development": "",
        "time_window": ""
    }
    
    # Si la pente est exposée à l'est/sud-est
    if site_slope and (45 <= site_slope <= 135):
        anabatic_info["time_window"] = "9h00-13h00"
    # Si la pente est exposée au sud/sud-ouest
    elif site_slope and (135 <= site_slope <= 225):
        anabatic_info["time_window"] = "11h00-16h00"
    else:
        anabatic_info["time_window"] = "10h00-15h00"
    
    # Estimer la force du vent anabatique basée sur la température et le gradient
    if ground_temp > 25:
        anabatic_strength = min(3.0, 1.0 + (ground_temp - 25) * 0.1)
        anabatic_info["development"] = "Bien développés mais peuvent être masqués par l'air instable"
    elif ground_temp > 15:
        anabatic_strength = min(2.0, 0.5 + (ground_temp - 15) * 0.1)
        anabatic_info["development"] = "Modérément développés"
    else:
        anabatic_strength = max(0.5, (ground_temp - 5) * 0.05)
        anabatic_info["development"] = "Faiblement développés"
    
    # Ajustement pour le gradient thermique
    anabatic_strength *= (thermal_gradient / 7.0)
    anabatic_info["strength"] = anabatic_strength
    
    # Informations sur les thermiques proprement dits
    thermal_info = {
        "strength": analysis.thermal_strength,
        "formation_altitude": site_altitude + 300,  # Estimation de détachement des thermiques
        "detachment_description": ""
    }
    
    # Description du détachement
    if thermal_gradient > 7.0:
        thermal_info["detachment_description"] = "Détachement rapide de la pente, thermiques distincts"
    elif thermal_gradient > 5.0:
        thermal_info["detachment_description"] = "Détachement modéré, mélange possible avec les anabatiques"
    else:
        thermal_info["detachment_description"] = "Détachement lent, difficile de distinguer des anabatiques"
    
    return {
        "anabatic": anabatic_info,
        "thermal": thermal_info,
        "recommendation": _generate_flight_strategy(anabatic_info, thermal_info, analysis)
    }

def _generate_flight_strategy(anabatic, thermal, analysis):
    """Génère des recommandations de stratégie de vol"""
    
    if thermal["strength"] == "Faible":
        if anabatic["strength"] > 1.5:
            return "Privilégier le vol près des pentes pour exploiter les vents anabatiques"
        else:
            return "Conditions difficiles, rester près des meilleures sources thermiques"
    elif thermal["strength"] in ["Modérée", "Forte"]:
        return "Thermiques bien formés, possible de s'éloigner des pentes après avoir gagné de l'altitude"
    else:  # Très Forte
        return "Attention aux thermiques puissants, prévoir une marge de sécurité par rapport au relief"

def show_glossary():
    """Affiche un glossaire visuel des termes de météorologie aérologique"""
    
    st.header("Glossaire visuel")
    
    # Utiliser des colonnes pour organiser le glossaire
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Émagramme et ses composantes")
        st.markdown("""
        **Les principales courbes d'un émagramme:**
        - **Courbe d'état (rouge)** : Température réelle de l'atmosphère à différentes altitudes
        - **Courbe du point de rosée (bleue)** : Température à laquelle l'air se condense
        - **Chemin du thermique (vert pointillé)** : Évolution d'une particule d'air qui s'élève
        - **Adiabatique sèche** : Refroidissement d'un thermique avant condensation (-1°C/100m)
        - **Adiabatique humide** : Refroidissement après condensation (-0,6°C/100m)
        """)
        
        st.subheader("Structures thermiques")
        st.markdown("""
        **Différents types de thermiques:**
        - **Thermiques bleus** : Sans formation de nuage, invisibles
        - **Thermiques à cumulus** : Avec condensation au sommet, marqués par un nuage
        - **Thermiques organisés en rues** : Alignés dans le sens du vent
        - **Thermiques continus** : Forme un "mur" de portance le long d'un relief
        - **Brise de pente** : Courant ascendant laminaire lié au réchauffement d'un versant
        """)

    with col2:
        st.subheader("Formation des cumulus")
        st.markdown("""
        **Formation d'un cumulus au sommet d'un thermique:**
        1. L'air chaud s'élève depuis le sol en se refroidissant (adiabatique sèche)
        2. Lorsque la température atteint le point de rosée, la vapeur d'eau se condense
        3. La condensation libère de la chaleur latente, ralentissant le refroidissement
        4. Le nuage continue à se développer tant que l'air du thermique reste plus chaud que l'air environnant
        5. La base des cumulus est plane et marque le niveau de condensation
        """)
        
        st.subheader("Inversions et leur effet")
        st.markdown("""
        **Comment une inversion bloque les thermiques:**
        - Une inversion est une couche où la température augmente avec l'altitude (contrairement à la normale)
        - Elle agit comme un "couvercle" thermique empêchant les mouvements verticaux
        - Les thermiques ralentissent ou s'arrêtent en atteignant cette couche stable
        - En vol, on ressent une soudaine diminution de la portance
        - Les inversions peuvent créer des plafonds bas et une atmosphère polluée en vallée
        - Elles sont souvent visibles par une ligne horizontale de brume ou un étalement des nuages
        """)
    
    # Conseils d'interprétation
    st.subheader("Conseils pour interpréter un émagramme")
    st.markdown("""
    1. **Regardez d'abord le gradient** - Un gradient proche de 1°C/100m dans les basses couches indique une bonne couche convective
    2. **Cherchez les inversions** - Elles limitent la hauteur des thermiques
    3. **Estimez l'humidité** - L'écart entre température et point de rosée vous indique si des nuages vont se former
    4. **Vérifiez le vent** - Un vent fort en altitude peut rendre le vol difficile même avec de bons thermiques
    """)

# Interface principale
def main():
    # Initialiser l'état de la géolocalisation
    if 'geolocation_attempted' not in st.session_state:
        st.session_state.geolocation_attempted = False
        st.session_state.user_location = None

    # Section pour la géolocalisation
    with st.expander("📱 Utiliser la géolocalisation de mon appareil", expanded=False):
        st.info("Cette fonction utilise le GPS de votre appareil pour obtenir votre position précise.")
        
        # Utiliser streamlit_geolocation pour récupérer la position
        location = streamlit_geolocation()
        
        # Afficher l'état de la géolocalisation après avoir essayé de récupérer la position
        st.write("État de la géolocalisation :")
        if location and 'latitude' in location and location['latitude'] is not None and 'longitude' in location and location['longitude'] is not None:
            st.success("📱 Géolocalisation activée")
        else:
            st.warning("📱 En attente de géolocalisation... Si vous ne voyez pas d'invite d'autorisation, vérifiez les paramètres de votre navigateur.")

        if st.checkbox("Mode débogage"):
            st.write("Informations de débogage :")
            st.write(f"Location object: {location}")
            st.write(f"Session state: {st.session_state}")
            if hasattr(st, 'request_headers'):
                st.write(f"User agent: {st.request_headers['User-Agent']}")
        
        if location and 'latitude' in location and location['latitude'] is not None and 'longitude' in location and location['longitude'] is not None:
            # La géolocalisation a réussi, mettre à jour l'état de session
            
            # Tenter d'obtenir le nom de la ville
            city = "Position actuelle"
            try:
                geocode_response = requests.get(
                    f"https://nominatim.openstreetmap.org/reverse?format=json&lat={location['latitude']}&lon={location['longitude']}&zoom=14",
                    headers={"User-Agent": "EmagrammeParapente/1.0"},
                    timeout=3
                )
                if geocode_response.status_code == 200:
                    geocode_data = geocode_response.json()
                    city = geocode_data.get('address', {}).get('city', 
                          geocode_data.get('address', {}).get('town', 
                          geocode_data.get('address', {}).get('village', "Lieu inconnu")))
            except:
                pass
            
            # Si l'altitude n'est pas disponible, essayer de l'obtenir
            altitude = location.get('altitude')
            if not altitude:
                try:
                    elevation_response = requests.get(
                        f"https://api.open-elevation.com/api/v1/lookup?locations={location['latitude']},{location['longitude']}",
                        timeout=3
                    )
                    if elevation_response.status_code == 200:
                        elevation_data = elevation_response.json()
                        altitude = elevation_data.get('results', [{}])[0].get('elevation', 500)
                    else:
                        altitude = 500  # Valeur par défaut
                except:
                    altitude = 500  # Valeur par défaut
            
            # Mettre à jour les informations de localisation
            st.session_state.user_location = {
                "latitude": location['latitude'],
                "longitude": location['longitude'],
                "altitude": altitude,
                "accuracy": location.get('accuracy', 0),
                "city": city,
                "source": "GPS"
            }
            st.session_state.geolocation_attempted = True
            
            # Afficher les informations de localisation
            st.success(f"Géolocalisation réussie ! Vous êtes à {city}")
            
            # Afficher les coordonnées et l'altitude
            col1, col2 = st.columns(2)
            with col1:
                if location['latitude'] is not None:
                    st.write(f"**Latitude:** {location['latitude']:.6f}")
                else:
                    st.write("**Latitude:** Non disponible")
                
                if location['longitude'] is not None:
                    st.write(f"**Longitude:** {location['longitude']:.6f}")
                else:
                    st.write("**Longitude:** Non disponible")

            with col2:
                if altitude is not None:
                    st.write(f"**Altitude:** {altitude:.0f} m")
                else:
                    st.write("**Altitude:** Non disponible")
                
                accuracy = location.get('accuracy', 0)
                if accuracy is not None:
                    st.write(f"**Précision:** ±{accuracy:.0f} m")
                else:
                    st.write("**Précision:** Non disponible")
            
            # Afficher une carte avec la position
            import folium
            from streamlit_folium import st_folium
            
            if location['latitude'] is not None and location['longitude'] is not None:
                m = folium.Map(location=[location['latitude'], location['longitude']], zoom_start=13)
                
                folium.Marker(
                    [location['latitude'], location['longitude']],
                    popup=f"Votre position<br>Altitude: {altitude if altitude is not None else 'Non disponible'} m",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)
                
                accuracy = location.get('accuracy', 0)
                if accuracy is not None and accuracy > 0:
                    folium.Circle(
                        radius=accuracy,
                        location=[location['latitude'], location['longitude']],
                        popup="Précision",
                        color="#3186cc",
                        fill=True,
                        fill_color="#3186cc"
                    ).add_to(m)
                
                st.subheader("Votre position")
                st_folium(m)
            
            # Bouton pour utiliser cette position dans l'application
            if st.button("Analyser l'émagramme à cette position"):
                if location['latitude'] is not None and location['longitude'] is not None:
                    st.session_state.site_selection = {
                        "latitude": location["latitude"],
                        "longitude": location["longitude"],
                        "altitude": altitude if altitude is not None else 500,  # Valeur par défaut si None
                        "model": st.session_state.site_selection.get("model", "meteofrance_arome_france_hd")
                    }
                    st.session_state.run_analysis = True
                    st.rerun()
                else:
                    st.error("Coordonnées GPS non disponibles. Veuillez réessayer.")

    # Si la géolocalisation GPS a réussi, afficher un bouton dans la sidebar
    if st.session_state.user_location and st.session_state.geolocation_attempted and st.session_state.user_location.get("source") == "GPS":
        accuracy_info = ""
        if "accuracy" in st.session_state.user_location and st.session_state.user_location["accuracy"] is not None:
            try:
                accuracy_info = f" (précision: ±{st.session_state.user_location['accuracy']:.0f}m)"
            except:
                accuracy_info = " (précision: non disponible)"
        
        st.sidebar.success(f"📱 Géolocalisation précise: {st.session_state.user_location['city']}{accuracy_info}")
        
        # Bouton pour utiliser la position géolocalisée
        if st.sidebar.button("Utiliser ma position GPS"):
            # Vérifier que latitude et longitude ne sont pas None
            if st.session_state.user_location.get("latitude") is not None and st.session_state.user_location.get("longitude") is not None:
                st.session_state.site_selection = {
                    "latitude": st.session_state.user_location["latitude"],
                    "longitude": st.session_state.user_location["longitude"],
                    "altitude": st.session_state.user_location["altitude"],
                    "model": st.session_state.site_selection.get("model", "meteofrance_arome_france_hd")
                }
                st.session_state.run_analysis = True
                st.rerun()
            else:
                st.sidebar.error("Coordonnées GPS non disponibles.")

    # Dans le corps principal de l'application
    if st.session_state.get("tutorial_mode", False):
        st.info("""
        ## Bienvenue dans le tutoriel d'initiation !
        
        Cette application analyse les conditions atmosphériques pour le vol en parapente.
        
        **Comment l'utiliser** :
        1. Sélectionnez un site ou entrez des coordonnées
        2. Cliquez sur "Analyser l'émagramme"
        3. Consultez les résultats dans les différents onglets
        
        Les sections avec le symbole ℹ️ contiennent des informations supplémentaires.
        """)
        
        # Ajouter une explication des codes couleurs
        st.markdown("""
        ### Comprendre les indicateurs
        
        🟢 **Vert** - Conditions favorables
        🟡 **Jaune** - Conditions acceptables avec précautions
        🔴 **Rouge** - Conditions défavorables ou dangereuses
        """)
    
    # Initialiser l'état de session si nécessaire
    if 'site_selection' not in st.session_state:
        if st.session_state.user_location:
            st.session_state.site_selection = {
                "latitude": st.session_state.user_location["latitude"],
                "longitude": st.session_state.user_location["longitude"],
                "altitude": st.session_state.user_location["altitude"],
                "model": "meteofrance_arome_france_hd"
            }
    
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
    
    # Titre et introduction
    st.title("🪂 Analyseur d'Émagramme pour Parapentistes")
    st.markdown("""
    Cet outil analyse les conditions météorologiques pour le vol en parapente à partir des données d'émagramme.
    Il calcule le plafond des thermiques, la formation de nuages, le gradient thermique, et fournit des 
    recommandations adaptées aux conditions.
    """)
    
    # Sidebar pour les entrées utilisateur
    st.sidebar.header("Configuration")
    
    # Section pour la source des données
    st.sidebar.header("Source des données")
    data_source = st.sidebar.radio(
        "Sélectionnez une source de données",
        options=["Open-Meteo (sans clé API)"],
        index=0,  # Option par défaut : Open-Meteo
        help="Choisissez la source pour récupérer les données météorologiques"
    )

    use_openmeteo = (data_source == "Open-Meteo (sans clé API)")

    # Modèle météo en fonction de la source de données
    if use_openmeteo:
        model_options = [
            "meteofrance_arome_france_hd", 
            "meteofrance_arpege_europe",
            "meteofrance_arpege_world",
            "ecmwf_ifs04",
            "gfs_seamless"
        ]
        model_descriptions = {
            "meteofrance_arome_france_hd": "AROME HD (France ~2km)",
            "meteofrance_arpege_europe": "ARPEGE (Europe ~11km)",
            "meteofrance_arpege_world": "ARPEGE (Mondial ~40km)",
            "ecmwf_ifs04": "ECMWF IFS (Mondial ~9km)",
            "gfs_seamless": "GFS (Mondial ~25km)"
        }
        model_labels = [model_descriptions[m] for m in model_options]
        model_index = st.sidebar.selectbox(
            "Modèle météo",
            options=range(len(model_options)),
            format_func=lambda i: model_labels[i],
            index=0
        )
        model = model_options[model_index]
        
        # Informations supplémentaires selon le modèle sélectionné
        if model == "meteofrance_arome_france_hd":
            st.sidebar.info("AROME HD: Haute résolution (~2km) sur la France, précis pour les reliefs")
        elif model == "meteofrance_arpege_europe":
            st.sidebar.info("ARPEGE Europe: Résolution moyenne (~11km), bonne couverture européenne")
        elif model == "meteofrance_arpege_world":
            st.sidebar.info("ARPEGE Mondial: Résolution plus grossière (~40km), disponible partout dans le monde")
        elif model == "ecmwf_ifs04":
            st.sidebar.info("ECMWF IFS: Résolution fine pour un modèle global (~9km), performance élevée")
        elif model == "gfs_seamless":
            st.sidebar.info("GFS: Modèle américain, disponible mondialement, résolution ~25km")
        
        # Pas besoin de clé API pour Open-Meteo
        api_key = None
    
    # Option pour clé OpenAI (analyse IA)
    use_ai = st.sidebar.checkbox("Utiliser l'analyse IA externe (OpenAI)", value=False,
                               help="Utilise OpenAI pour générer une analyse détaillée en complément de l'analyse intégrée")
    
    if use_ai:
        openai_key = st.sidebar.text_input("Clé API OpenAI", type="password")
    else:
        openai_key = None
        # Message indiquant que l'analyse intégrée sera utilisée
        st.sidebar.info("L'analyse détaillée intégrée sera utilisée")
    
    # Paramètres avancés
    with st.sidebar.expander("Paramètres avancés"):
        delta_t = st.slider("Delta T de déclenchement (°C)", 
                         min_value=1.0, max_value=6.0, value=3.0, step=0.5,
                         help="Différence de température requise pour déclencher un thermique")

        # Nouvelle option pour l'évolution temporelle
        fetch_evolution_enabled = st.checkbox("Afficher l'évolution des conditions", value=True,
                                           help="Récupère les données pour plusieurs heures et affiche des graphiques d'évolution")
        
        if fetch_evolution_enabled:
            col1, col2 = st.columns(2)
            with col1:
                evolution_hours = st.slider("Période d'évolution (heures)", 
                                        min_value=6, max_value=48, value=24, step=6,
                                        help="Durée totale de la période d'évolution à analyser")
            with col2:
                evolution_step = st.slider("Pas de temps (heures)", 
                                        min_value=1, max_value=6, value=3, step=1,
                                        help="Intervalle entre chaque point d'analyse")
        else:
            evolution_hours = 24
            evolution_step = 3
        
        # Ajouter la configuration FFVL
        st.subheader("Paramètres FFVL")
        ffvl_api_key = st.text_input("Clé API FFVL", 
                                value=st.session_state.get("ffvl_api_key", ""),
                                type="password",
                                help="Clé API FFVL pour la recherche de sites. Contactez informatique@ffvl.fr pour l'obtenir.")
        
        # Sauvegarder la clé API dans session_state
        if ffvl_api_key:
            st.session_state.ffvl_api_key = ffvl_api_key

    # Section pour le pas de temps de prévision (nouveau)
    st.sidebar.header("Temps de prévision")
    
    # Déterminer la plage de temps disponible selon le modèle
    if model.startswith("meteofrance_arome_france_hd"):
        max_timestep = 36
        timestep = st.sidebar.slider("Heure de prévision", 0, max_timestep, 0, 
                                    help=f"0 = analyse actuelle, 1-{max_timestep} = prévision en heures")
        st.sidebar.info(f"AROME: prévisions disponibles jusqu'à H+{max_timestep}")
    else:  # ARPEGE
        max_timestep = 96
        timestep = st.sidebar.slider("Heure de prévision", 0, max_timestep, 0, 
                                    help=f"0 = analyse actuelle, 1-{max_timestep} = prévision en heures")
        st.sidebar.info(f"ARPÈGE: prévisions disponibles jusqu'à H+{max_timestep}")
    
    # Convertir le timestep en jours et heures pour l'affichage
    days = timestep // 24
    hours = timestep % 24
    if timestep > 0:
        if days > 0:
            forecast_text = f"Prévision pour J+{days}, {hours}h"
        else:
            forecast_text = f"Prévision pour H+{hours}"
        st.sidebar.success(forecast_text)
        analyze_clicked = st.sidebar.button("Analyser l'émagramme", key="sidebar_analyser_emagramme")
    
    # Section pour les sites prédéfinis
    st.sidebar.header("Sites prédéfinis")
    st.sidebar.markdown("Cliquez sur un bouton pour charger un site et lancer l'analyse")
    
    # Fonction pour définir le site et déclencher l'analyse
    def set_site_and_analyze(site_data):
        # Convertir le modèle au nouveau format si nécessaire
        model_conversion = {
            "arome": "meteofrance_arome_france_hd",
            "arpege": "meteofrance_arpege_europe",
        }
        
        site_model = model_conversion.get(site_data["model"], site_data["model"])
        
        st.session_state.site_selection = {
            "latitude": site_data["lat"],
            "longitude": site_data["lon"],
            "altitude": site_data["altitude"],
            "model": site_model
        }
        st.session_state.run_analysis = True
    
    # Organiser les sites par région
    regions = {}
    for site in PRESET_SITES:
        # Classification simple basée sur la latitude/longitude
        if site["lat"] < 44.0:
            region = "Sud-Est"
        elif site["lat"] < 45.0:
            region = "Alpes du Sud"
        elif site["lat"] < 46.0:
            region = "Alpes du Nord"
        else:
            region = "Autre"
            
        if region not in regions:
            regions[region] = []
        regions[region].append(site)
    
    # Afficher les sites par région dans des expanders
    for region, sites in regions.items():
        with st.sidebar.expander(f"Sites {region}"):
            # Créer un tableau de boutons
            cols = st.columns(2)
            for i, site in enumerate(sites):
                cols[i % 2].button(site["name"], key=f"site_{region}_{i}", 
                                 on_click=set_site_and_analyze, 
                                 args=(site,))
    
    # Section des paramètres de localisation
    st.subheader("Localisation")
    col1, col2, col3 = st.columns([1, 1, 1])

    if location and 'latitude' in location and location['latitude'] is not None and 'longitude' in location and location['longitude'] is not None:
        with col1:
            latitude = st.number_input("Latitude", 
                                    min_value=-90.0, max_value=90.0, 
                                    value=st.session_state.site_selection["latitude"], 
                                    step=0.0001,
                                    format="%.4f")
        
        with col2:
            longitude = st.number_input("Longitude", 
                                    min_value=-180.0, max_value=180.0, 
                                    value=st.session_state.site_selection["longitude"], 
                                    step=0.0001,
                                    format="%.4f")
        
        with col3:
            # Altitude du site
            site_altitude = st.number_input("Altitude (m)", 
                                min_value=0.0,  # Changé de int à float
                                max_value=5000.0,  # Changé de int à float
                                value=float(st.session_state.site_selection["altitude"]),  # S'assurer que c'est un float
                                step=10.0,  # Changé de int à float
                                format="%.1f")  # Format avec un chiffre après la virgule
                                      
    
    # Section pour la recherche de décollages proches
    with st.expander("🪂 Recherche de décollages proches", expanded=False):
        search_radius = st.slider(
            "Rayon de recherche (km)", 
            min_value=5, 
            max_value=100, 
            value=50,
            step=5,
            help="Distance maximale des sites à rechercher"
        )
        
        if st.button("Rechercher les décollages"):
            with st.spinner(f"Recherche des sites FFVL à proximité dans un rayon de {search_radius} km..."):
                sites = search_ffvl_sites(
                    latitude, 
                    longitude, 
                    radius=search_radius,
                    api_key=st.session_state.get("ffvl_api_key", "DEMO_KEY")
                )
            
            if sites:
                st.success(f"{len(sites)} sites trouvés à proximité")
                
                # Afficher les sites dans un tableau
                for i, site in enumerate(sites[:10]):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.markdown(f"**{site.get('name', 'Site sans nom')}**")
                        # Vérifier l'existence des clés avant d'y accéder
                        if 'type' in site and site['type']:
                            st.markdown(f"Type: {site['type']}")
                        if 'orientation' in site and site['orientation']:
                            st.markdown(f"Orientation: {site['orientation']}")
                        if 'difficulty' in site and site['difficulty']:
                            st.markdown(f"Difficulté: {site['difficulty']}")
                    
                    with col2:
                        if 'distance' in site:
                            st.markdown(f"Distance: {site['distance']} km")
                        if 'altitude' in site and site['altitude']:
                            st.markdown(f"Altitude: {site['altitude']} m")
                        
                    with col3:
                        if st.button(f"🪂", key=f"site_ffvl_{i}"):
                            try:
                                # Déboguer les valeurs
                                st.write(f"Debug - latitude: {site.get('latitude')}, longitude: {site.get('longitude')}")
                                
                                # S'assurer que les coordonnées sont des nombres
                                lat = float(site.get("latitude", 0))
                                lon = float(site.get("longitude", 0))
                                alt_str = site.get("altitude", "")
                                
                                # Convertir l'altitude en nombre si possible
                                try:
                                    alt = float(alt_str) if alt_str else st.session_state.site_selection["altitude"]
                                except (ValueError, TypeError):
                                    alt = st.session_state.site_selection["altitude"]
                                
                                # Vérifier que les coordonnées sont valides
                                if -90 <= lat <= 90 and -180 <= lon <= 180:
                                    st.session_state.site_selection = {
                                        "latitude": lat,
                                        "longitude": lon,
                                        "altitude": alt,
                                        "model": st.session_state.site_selection["model"]
                                    }
                                    st.experimental_rerun()
                                else:
                                    st.error(f"Coordonnées hors limites: lat={lat}, lon={lon}")
                            except Exception as e:
                                st.error(f"Erreur lors de l'utilisation du site: {str(e)}")
            else:
                st.warning("Aucun site de vol trouvé à proximité")
                st.info("Essayez d'augmenter le rayon de recherche ou de vérifier votre position")

    # Ajouter la possibilité de recherche par nom de lieu
    with st.expander("🔍 Rechercher un lieu"):
        search_query = st.text_input("Nom du lieu (ville, montagne, site de vol...)")
        search_button = st.button("Rechercher")
        
        if search_button and search_query:
            try:
                with st.spinner(f"Recherche de {search_query}..."):
                    # Utiliser Nominatim (OpenStreetMap) pour la recherche de lieux
                    search_url = f"https://nominatim.openstreetmap.org/search?q={search_query}&format=json&limit=5"
                    response = requests.get(search_url, headers={"User-Agent": "EmagrammeParapente/1.0"})
                    
                    if response.status_code == 200:
                        results = response.json()
                        if results:
                            # Créer un tableau pour afficher les résultats
                            st.write("Résultats de recherche:")
                            
                            for i, result in enumerate(results):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"{result.get('display_name', 'Lieu inconnu')}")
                                with col2:
                                    if st.button(f"Sélectionner", key=f"select_{i}"):
                                        lat = float(result.get('lat', 0))
                                        lon = float(result.get('lon', 0))
                                        
                                        # Tenter d'obtenir l'altitude via une API d'élévation
                                        try:
                                            elev_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
                                            elev_response = requests.get(elev_url, timeout=3)
                                            if elev_response.status_code == 200:
                                                elev_data = elev_response.json()
                                                altitude = elev_data.get('results', [{}])[0].get('elevation', 500)
                                            else:
                                                altitude = 500
                                        except:
                                            altitude = 500
                                        
                                        st.session_state.site_selection = {
                                            "latitude": lat,
                                            "longitude": lon,
                                            "altitude": altitude,
                                            "model": st.session_state.site_selection["model"]
                                        }
                                        st.rerun()
                        else:
                            st.warning("Aucun résultat trouvé")
                    else:
                        st.error("Erreur lors de la recherche")
            except Exception as e:
                st.error(f"Erreur: {e}")

    # Bouton pour lancer l'analyse (IMPORTANT: définir 'analyze_clicked' AVANT de l'utiliser)
    analyze_clicked = st.button("Analyser l'émagramme")
    
    # Maintenant on peut utiliser analyze_clicked
    should_run_analysis = analyze_clicked or st.session_state.run_analysis

    if should_run_analysis:
        # Réinitialiser le flag pour éviter des analyses en boucle
        st.session_state.run_analysis = False
        
        if use_openmeteo and not api_key:
            # Déterminer la source de données pour la fonction fetch_and_analyze
            data_source_str = "open-meteo"
            
            # Récupérer et analyser les données (modifié pour l'évolution)
            if fetch_evolution_enabled:
                with st.spinner(f"Récupération des données d'évolution sur {evolution_hours}h..."):
                    analyzer, analysis, detailed_analysis, evolution_data = fetch_and_analyze(
                        latitude, longitude, model, site_altitude, api_key, openai_key, delta_t, 
                        data_source=data_source_str, timestep=timestep,
                        fetch_evolution=True, evolution_hours=evolution_hours, evolution_step=evolution_step
                    )
            else:
                analyzer, analysis, detailed_analysis = fetch_and_analyze(
                    latitude, longitude, model, site_altitude, api_key, openai_key, delta_t, 
                    data_source=data_source_str, timestep=timestep
                )
                evolution_data = None
            
            # Si l'analyse est réussie, afficher les résultats
            if analyzer and analysis:
                # Afficher l'émagramme
                st.subheader("Émagramme")
                display_emagramme(analyzer, analysis)
                
                # Calculer l'information sur la couche convective
                convective_layer = calculate_convective_layer_thickness(analyzer, analysis)
                
                # Créer et afficher la visualisation de la couche convective directement sous l'émagramme
                st.subheader("Visualisation de la couche convective")
                
                # Explication de la couche convective (dans un expander pour ne pas prendre trop de place)
                with st.expander("📚 Qu'est-ce que la couche convective ?"):
                    st.markdown("""
                    La couche convective est la partie de l'atmosphère où se produisent les mouvements verticaux 
                    (ascendants et descendants) de l'air. C'est dans cette couche que se forment les thermiques
                    exploitables pour le vol en parapente.
                    
                    Caractéristiques principales:
                    - S'étend du sol jusqu'au plafond thermique
                    - Présente un gradient de température d'environ 1°C/100m
                    - La turbulence y est plus importante qu'en dehors
                    - Plus elle est épaisse, plus le plafond des thermiques est élevé
                    """)
                
                # Afficher les informations de la couche convective
                cols = st.columns([2, 1])
                with cols[0]:
                    st.metric("Épaisseur de la couche convective", f"{convective_layer['thickness']:.0f} m")
                    st.write(f"**Qualité des ascendances**: {convective_layer['description']}")
                
                with cols[1]:
                    # Créer et afficher le graphique de la couche convective
                    fig = create_convective_layer_plot(analysis, analysis.inversion_layers)
                    st.pyplot(fig)
                
                # Onglets pour le reste des informations
                if fetch_evolution_enabled and evolution_data:
                    tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Évolution et Données brutes", "Sites FFVL", "Aide"])
                else:
                    tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Données brutes", "Sites FFVL", "Aide"])
                
                with tab1:
                    st.subheader("Analyse des mouvements d'air verticaux")

                    # Option pour l'orientation de la pente
                    site_slope = st.slider("Orientation de la pente de décollage (degrés)", 0, 359, 135, 
                                        help="0° = Nord, 90° = Est, 180° = Sud, 270° = Ouest")

                    air_movement = analyze_anabatic_vs_thermal(analysis, site_altitude, site_slope)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Vents anabatiques**")
                        
                        # Remplacer le bouton par un expander
                        with st.expander("📚 Différence entre vent anabatique et thermique"):
                            st.markdown("""
                            **Vents anabatiques** : Mouvements d'air qui remontent les pentes, généralement faibles (1-3 m/s), 
                            restent collés au relief et suivent précisément le contour de la montagne.
                            
                            **Thermiques** : Colonnes d'air ascendantes qui se détachent du sol, peuvent être beaucoup plus 
                            puissantes (jusqu'à 5-8 m/s) et montent verticalement jusqu'au sommet de la couche convective.
                            """)
                            
                        st.metric("Force estimée", f"{air_movement['anabatic']['strength']:.1f} m/s")
                        st.write(f"Développement: {air_movement['anabatic']['development']}")
                        st.write(f"Période favorable: {air_movement['anabatic']['time_window']}")

                    with col2:
                        st.write("**Thermiques**")
                        st.metric("Force", analysis.thermal_strength)
                        st.write(f"Détachement: {air_movement['thermal']['detachment_description']}")
                        st.write(f"Altitude estimation formation: {air_movement['thermal']['formation_altitude']:.0f}m")

                    st.info(f"**Stratégie recommandée**: {air_movement['recommendation']}")

                    # Vérifier d'abord si le vol est impossible
                    vol_impossible = (analysis.precipitation_type is not None and analysis.precipitation_type != 0)

                    # Vérifier si le vent dans la zone de vol est trop fort (déjà calculé dans l'analyseur)
                    if hasattr(analyzer, 'vol_impossible_wind') and analyzer.vol_impossible_wind:
                        vol_impossible = True
                        raisons = [f"Vent trop fort dans la zone de vol ({analyzer.max_wind_in_vol_zone:.1f} km/h)"]
                    elif (analysis.precipitation_type is not None and analysis.precipitation_type != 0):
                        raisons = [analysis.precipitation_description]
                    else:
                        raisons = []

                    if vol_impossible:
                        # Utiliser un style d'alerte visuelle différent
                        st.error("⚠️ VOL IMPOSSIBLE - Conditions météorologiques dangereuses")
                        
                        # Afficher la raison principale
                        st.error(f"Raison: {', '.join(raisons)}")
                        
                        # Limiter ce qui est affiché dans l'interface
                        with st.expander("Détails des conditions"):
                            st.warning(analysis.flight_conditions)
                            st.warning(analysis.wind_conditions)
                            
                            if analysis.hazards:
                                st.subheader("⚠️ Dangers spécifiques")
                                for hazard in analysis.hazards:
                                    st.warning(hazard)
                    else:
                        # Affichage normal pour les conditions volables
                        st.subheader("Informations générales")
                        
                        # Afficher le modèle et l'heure de prévision
                        forecast_time = ""
                        if "H+" in analysis.model_name:
                            model_parts = analysis.model_name.split("H+")
                            model_name = model_parts[0].strip()
                            hour = int(model_parts[1])
                            days = hour // 24
                            remaining_hours = hour % 24
                            if days > 0:
                                forecast_time = f" - Prévision pour J+{days}, {remaining_hours}h"
                            else:
                                forecast_time = f" - Prévision pour H+{hour}"

                        model_name = analysis.model_name if hasattr(analysis, 'model_name') and analysis.model_name else "inconnu"
                        st.info(f"Modèle météo utilisé: {model_name.upper()}{forecast_time}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Altitude du site", f"{analysis.ground_altitude:.0f} m")
                            st.metric("Température au sol", f"{analysis.ground_temperature:.1f} °C")
                            st.metric("Point de rosée", f"{analysis.ground_dew_point:.1f} °C")
                            
                        with col2:
                            st.metric("Plafond thermique", f"{analysis.thermal_ceiling:.0f} m")
                            st.metric("Gradient thermique", f"{analysis.thermal_gradient:.1f} °C/1000m")
                            st.metric("Force des thermiques", analysis.thermal_strength)
                            
                        with col3:
                            st.metric("Stabilité", analysis.stability)
                            if analysis.thermal_type == "Cumulus":
                                st.metric("Base des nuages", f"{analysis.cloud_base:.0f} m")
                                st.metric("Sommet des nuages", f"{analysis.cloud_top:.0f} m")
                            else:
                                st.info("Thermiques bleus (pas de condensation)")
                    
                    # Analyse des inversions (si présentes)
                    if analysis.inversion_layers:
                        st.subheader("Analyse des inversions")
                        
                        # Explication des inversions dans un expander
                        with st.expander("📚 Comment les inversions affectent-elles le vol ?"):
                            st.markdown("""
                            Une **inversion thermique** est une couche d'air où la température augmente avec l'altitude, 
                            contrairement à la situation normale où la température diminue en montant.
                            
                            ### Impact sur le vol en parapente :
                            
                            - **Blocage des thermiques** : Les inversions agissent comme un "couvercle" qui stoppe 
                            l'ascension des thermiques, limitant ainsi la hauteur maximale de vol.
                            
                            - **Stabilisation de l'air** : L'air est plus stable dans une inversion, réduisant 
                            la probabilité de formation de turbulences et de thermiques.
                            
                            - **Accumulation d'humidité** : Les inversions peuvent piéger l'humidité sous elles, 
                            créant des couches de nuages stratiformes.
                            
                            - **Position critique** : Une inversion basse (< 1500m) est particulièrement limitante 
                            car elle réduit considérablement le volume d'air exploitable pour le vol.
                            """)
                        
                        # Analyse des inversions
                        inversion_analysis = analyze_inversions_impact(analysis)
                        
                        for i, inv in enumerate(inversion_analysis["inversions"]):
                            if inv["severity"] == "critical":
                                st.error(f"Inversion {i+1}: De {inv['base']:.0f}m à {inv['top']:.0f}m - {inv['impact']}")
                            elif inv["severity"] == "warning":
                                st.warning(f"Inversion {i+1}: De {inv['base']:.0f}m à {inv['top']:.0f}m - {inv['impact']}")
                            else:
                                st.info(f"Inversion {i+1}: De {inv['base']:.0f}m à {inv['top']:.0f}m - {inv['impact']}")
                        
                        # Afficher les informations sur les nuages si disponibles
                        if (analysis.low_cloud_cover is not None or 
                            analysis.mid_cloud_cover is not None or 
                            analysis.high_cloud_cover is not None):
                            
                            st.subheader("Couverture nuageuse")
                            cols = st.columns(3)
                            with cols[0]:
                                if analysis.low_cloud_cover is not None:
                                    st.metric("Nuages bas", f"{analysis.low_cloud_cover:.0f}%")
                                    
                            with cols[1]:
                                if analysis.mid_cloud_cover is not None:
                                    st.metric("Nuages moyens", f"{analysis.mid_cloud_cover:.0f}%")
                                    
                            with cols[2]:
                                if analysis.high_cloud_cover is not None:
                                    st.metric("Nuages hauts", f"{analysis.high_cloud_cover:.0f}%")
                        
                        # Afficher les informations sur les précipitations si disponibles
                        if analysis.precipitation_type is not None:
                            st.subheader("Précipitations")
                            st.info(f"{analysis.precipitation_description}")
                        
                        # Conditions de vol
                        st.subheader("Conditions de vol")
                        st.write(analysis.flight_conditions)
                        
                        # Conditions de vent
                        st.subheader("Conditions de vent")
                        st.write(analysis.wind_conditions)
                        
                        # Risques
                        if analysis.hazards:
                            st.subheader("⚠️ Risques identifiés")
                            for hazard in analysis.hazards:
                                st.warning(hazard)
                        
                        # Équipement recommandé
                        if analysis.recommended_gear:
                            st.subheader("Équipement recommandé")
                            for gear in analysis.recommended_gear:
                                st.write(f"- {gear}")
                
                if fetch_evolution_enabled and evolution_data and "tab2" in locals():
                    with tab2:
                        st.header("Évolution des conditions sur la période")
                        
                        # Créer tous les graphiques d'évolution
                        with st.spinner("Génération des graphiques d'évolution..."):
                            graphs, best_period, evolution_df = create_evolution_plots(evolution_data, site_altitude)
                        
                        # Vérifier que best_period contient les clés attendues
                        if graphs and best_period and "time" in best_period:
                            # Afficher le résumé des meilleures périodes
                            st.subheader("Meilleure période de vol")
                            summary_cols = st.columns(4)
                            with summary_cols[0]:
                                st.metric("Meilleure période", best_period["time"])
                                st.metric("Score global", f"{best_period['score']:.0f}%")
                            
                            with summary_cols[1]:
                                st.metric("Plafond thermique", f"{best_period['ceiling']:.0f}m")
                                st.metric("Gradient", f"{best_period['gradient']:.1f}°C/1000m")
                            
                            with summary_cols[2]:
                                st.metric("Température", f"{best_period['temp']:.1f}°C")
                                st.metric("Nuages", f"{best_period['clouds']:.0f}%")
                            
                            with summary_cols[3]:
                                st.metric("Vent", f"{best_period['wind']:.1f}km/h")
                                if best_period['rain'] > 0:
                                    st.metric("Pluie", f"{best_period['rain']:.1f}mm")
                                else:
                                    st.metric("Pluie", "0mm")
                            
                            # Afficher les graphiques seulement s'ils existent
                            if "meteo" in graphs:
                                # Afficher le météogramme simplifié
                                st.subheader("Météogramme")
                                st.plotly_chart(graphs["meteo"], use_container_width=True)
                            
                            if "vol_score" in graphs:
                                # Afficher le graphique des scores de vol
                                st.subheader("Évaluation des conditions de vol")
                                st.plotly_chart(graphs["vol_score"], use_container_width=True)
                            
                            # Autres graphiques d'évolution dans des expanders
                            if "ceiling" in graphs:
                                with st.expander("Évolution du plafond thermique"):
                                    st.plotly_chart(graphs["ceiling"], use_container_width=True)
                            
                            if "gradient" in graphs:
                                with st.expander("Évolution du gradient thermique"):
                                    st.plotly_chart(graphs["gradient"], use_container_width=True)
                            
                            if "clouds" in graphs:
                                with st.expander("Évolution de la couverture nuageuse"):
                                    st.plotly_chart(graphs["clouds"], use_container_width=True)
                            
                            if "precipitation" in graphs:
                                with st.expander("Prévisions de précipitations"):
                                    st.plotly_chart(graphs["precipitation"], use_container_width=True)
                            
                            # Option pour télécharger les données d'évolution si disponibles
                            if not evolution_df.empty:
                                csv = evolution_df.to_csv(index=False)
                                st.download_button(
                                    label="Télécharger les données d'évolution (CSV)",
                                    data=csv,
                                    file_name=f"evolution_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.warning("Pas assez de données pour afficher l'évolution des conditions et déterminer la meilleure période de vol.")

                with tab2:
                    # Afficher les niveaux atmosphériques
                    st.subheader("Niveaux atmosphériques")
                    data = {
                        "Altitude (m)": [level.altitude for level in analyzer.levels],
                        "Pression (hPa)": [level.pressure for level in analyzer.levels],
                        "Température (°C)": [level.temperature for level in analyzer.levels],
                        "Point de rosée (°C)": [level.dew_point for level in analyzer.levels]
                    }
                    
                    # Ajouter les données de vent si disponibles
                    if analyzer.levels[0].wind_direction is not None:
                        data["Direction du vent (°)"] = [level.wind_direction for level in analyzer.levels]
                        data["Vitesse du vent (km/h)"] = [level.wind_speed for level in analyzer.levels]
                    
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    # Afficher DIRECTEMENT les informations de couverture nuageuse depuis l'analyse
                    st.subheader("Couverture nuageuse")
                    if (analysis.low_cloud_cover is not None or 
                        analysis.mid_cloud_cover is not None or 
                        analysis.high_cloud_cover is not None):
                        
                        cloud_data = {
                            "Type": ["Nuages bas", "Nuages moyens", "Nuages hauts"],
                            "Couverture (%)": [
                                analysis.low_cloud_cover if analysis.low_cloud_cover is not None else "Non disponible",
                                analysis.mid_cloud_cover if analysis.mid_cloud_cover is not None else "Non disponible",
                                analysis.high_cloud_cover if analysis.high_cloud_cover is not None else "Non disponible"
                            ]
                        }
                        st.dataframe(pd.DataFrame(cloud_data))
                    else:
                        st.info("Aucune information sur la couverture nuageuse disponible")
                    
                    # Afficher DIRECTEMENT les informations sur les précipitations depuis l'analyse
                    st.subheader("Précipitations")
                    if analysis.precipitation_type is not None:
                        st.info(f"Type: {analysis.precipitation_type} - {analysis.precipitation_description}")
                    else:
                        st.info("Aucune précipitation")
                    
                    # Afficher les calculs de spread
                    st.subheader("Calculs de spread (T° - Td)")
                    st.write(f"Spread au sol: {analysis.ground_spread:.1f}°C")
                    
                    if analysis.spread_levels:
                        spread_data = []
                        for level_name, spread_value in analysis.spread_levels.items():
                            spread_data.append({
                                "Niveau": level_name.capitalize(),
                                "Spread (°C)": f"{spread_value:.1f}"
                            })
                        st.dataframe(pd.DataFrame(spread_data))
                    
                    # Option pour télécharger les données
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les données (CSV)",
                        data=csv,
                        file_name=f"emagramme_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

                with tab3:
                    st.header("Recherche de sites FFVL")
                    
                    # Explication
                    st.markdown("""
                    Cette fonctionnalité vous permet de rechercher des sites de vol officiels dans la base de données 
                    de la Fédération Française de Vol Libre (FFVL).
                    """)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        ffvl_lat = st.number_input("Latitude", value=latitude, format="%.4f", key="ffvl_lat")
                        ffvl_lon = st.number_input("Longitude", value=longitude, format="%.4f", key="ffvl_lon")
                    
                    with col2:
                        search_radius = st.slider("Rayon de recherche (km)", 5, 100, 20, 5)
                        st.info("Une plus grande distance augmente le temps de recherche")
                    
                    if st.button("Rechercher des sites FFVL", key="search_ffvl"):
                        with st.spinner("Recherche des sites FFVL..."):
                            sites = search_ffvl_sites(
                                ffvl_lat, 
                                ffvl_lon, 
                                radius=search_radius, 
                                api_key=st.session_state.get("ffvl_api_key", "DEMO_KEY")
                            )
                            
                            if sites:
                                st.success(f"{len(sites)} sites trouvés")
                                
                                # Créer un DataFrame pour affichage
                                sites_df = pd.DataFrame(sites)
                                sites_display = sites_df[["name", "type", "distance", "altitude", "orientation", "difficulty"]].copy()
                                sites_display.columns = ["Nom", "Type", "Distance (km)", "Altitude (m)", "Orientation", "Difficulté"]
                                
                                st.dataframe(sites_display)
                                
                                # Afficher une carte avec les sites
                                import folium
                                from streamlit_folium import folium_static
                                
                                m = folium.Map(location=[ffvl_lat, ffvl_lon], zoom_start=10)
                                
                                # Ajouter le point central
                                folium.Marker(
                                    [ffvl_lat, ffvl_lon],
                                    popup="Position de référence",
                                    icon=folium.Icon(color="red", icon="info-sign")
                                ).add_to(m)
                                
                                # Ajouter les sites
                                for site in sites:
                                    icon_color = "green"
                                    if "difficile" in site.get("difficulty", "").lower():
                                        icon_color = "red"
                                    elif "confirmé" in site.get("difficulty", "").lower():
                                        icon_color = "orange"
                                    
                                    folium.Marker(
                                        [site["latitude"], site["longitude"]],
                                        popup=f"<b>{site['name']}</b><br>Type: {site['type']}<br>Altitude: {site['altitude']}m<br>Orientation: {site['orientation']}<br>Difficulté: {site['difficulty']}",
                                        icon=folium.Icon(color=icon_color, icon="flag")
                                    ).add_to(m)
                                
                                # Afficher la carte
                                folium_static(m)
                                
                                # Bouton pour sélectionner un site
                                selected_site = st.selectbox("Sélectionner un site pour l'analyse", 
                                                        options=range(len(sites)),
                                                        format_func=lambda i: f"{sites[i]['name']} ({sites[i]['distance']} km)")
                                
                                if st.button("Utiliser ce site"):
                                    site = sites[selected_site]
                                    st.session_state.site_selection = {
                                        "latitude": site["latitude"],
                                        "longitude": site["longitude"],
                                        "altitude": float(site["altitude"]) if site["altitude"] else st.session_state.site_selection["altitude"],
                                        "model": st.session_state.site_selection["model"]
                                    }
                                    st.rerun()
                            else:
                                st.warning("Aucun site trouvé dans ce rayon")
                                st.info("Essayez d'augmenter le rayon de recherche ou de vérifier votre position")

                with tab4:
                    st.header("Guide de la météorologie aérologique")
                    
                    st.write("""
                    Cette section vous aide à comprendre les concepts clés utilisés dans l'analyse 
                    des conditions de vol en parapente.
                    """)
                    
                    # Organiser les concepts par catégories
                    categories = {
                        "Concepts de base": ["emagramme", "couche_convective", "gradient_thermique", "stabilite"],
                        "Mouvements d'air": ["thermique", "vent_anabatique", "subsidence"],
                        "Humidité et nuages": ["cumulus", "base_nuages", "point_de_rosee"],
                        "Phénomènes limitants": ["inversion"]
                    }
                    
                    # Créer des expanders pour chaque catégorie
                    for category, concepts in categories.items():
                        with st.expander(category):
                            for concept in concepts:
                                st.subheader(concept.replace("_", " ").title())
                                st.markdown(help_texts[concept])
                                st.markdown("---")
                    
                    show_glossary()

# Point d'entrée principal
if __name__ == "__main__":
    main()