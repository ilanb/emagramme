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
from datetime import datetime, timedelta
from retry_requests import retry
import json
from streamlit_geolocation import streamlit_geolocation
import datetime
import time
from streamlit_calendar import calendar

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

from enhanced_emagramme_analysis import (
    EnhancedEmagrammeAgent,
    analyze_emagramme_for_pilot,
    analyze_terrain_effect,
    detect_convergence_zones,
    calculate_adaptive_trigger_delta,
    identify_cloud_types,
    analyze_valley_breeze,
    interpolate_missing_data,
    calculate_advanced_stability,
    analyze_wind_profile,
    recommend_best_takeoff_sites,
    predict_flight_duration
)

if 'previous_model' not in st.session_state:
    st.session_state.previous_model = None

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

def display_recommended_ffvl_sites(sites, wind_direction, wind_speed, thermal_ceiling):
    """
    Affiche les sites FFVL recommandés en fonction des conditions météo
    
    Args:
        sites: Liste des sites FFVL
        wind_direction: Direction du vent en degrés
        wind_speed: Vitesse du vent en km/h
        thermal_ceiling: Plafond thermique en mètres
    """
    # Obtenir les recommandations
    site_recommendations = recommend_best_takeoff_sites(
        sites, wind_direction, wind_speed, thermal_ceiling
    )
    
    if not site_recommendations["sites"]:
        st.warning("Aucun site trouvé ou données insuffisantes pour la recommandation")
        return
    
    # Afficher le résumé des conditions
    st.subheader("Conditions pour les sites de vol")
    
    col1, col2 = st.columns(2)
    with col1:
        if site_recommendations["wind_too_strong"]:
            st.warning(f"⚠️ Vent trop fort ({wind_speed:.1f} km/h) pour un décollage optimal")
        else:
            st.success(f"✅ Vent favorable ({wind_speed:.1f} km/h)")
    
    with col2:
        if site_recommendations["thermal_ceiling_adequate"]:
            st.success(f"✅ Plafond thermique suffisant ({thermal_ceiling:.0f}m)")
        else:
            st.warning(f"⚠️ Plafond thermique bas ({thermal_ceiling:.0f}m)")
    
    # Afficher les sites recommandés
    st.subheader("Sites recommandés")
    
    for i, site in enumerate(site_recommendations["sites"]):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {i+1}. {site['site_name']}")
            st.progress(site['score'] / 100)
            st.markdown(f"**Score:** {site['score']:.0f}/100")
            st.markdown(f"**Commentaire:** {site['comment']}")
        
        with col2:
            st.markdown(f"**Altitude:** {site['altitude']}m")
            st.markdown(f"**Distance:** {site['distance']} km")
            st.markdown(f"**Difficulté:** {site['difficulty']}")
            
            # Bouton pour analyser à ce site
            if st.button(f"Utiliser ce site", key=f"use_recommended_site_{i}"):
                st.session_state.site_selection = {
                    "latitude": site["latitude"],
                    "longitude": site["longitude"],
                    "altitude": float(site["altitude"]) if site["altitude"] else st.session_state.site_selection["altitude"],
                    "model": st.session_state.site_selection["model"]
                }
                st.rerun()

def is_paragliding_takeoff(terrain):
    """Détermine si un terrain est un décollage de parapente"""
    # Vérifier les différents champs où l'info peut se trouver
    possible_usages = terrain.get("possible_usages_text", "")
    functions = terrain.get("flying_functions_text", "")
    description = terrain.get("description", "")
    terrain_type = terrain.get("type", "")  # Ajouter ce champ pour vérifier le type
    
    # S'assurer que chaque champ est une chaîne, même si la valeur est None
    possible_usages = str(possible_usages).lower() if possible_usages is not None else ""
    functions = str(functions).lower() if functions is not None else ""
    description = str(description).lower() if description is not None else ""
    terrain_type = str(terrain_type) if terrain_type is not None else ""
    
    # D'abord vérifier si c'est un site de parapente
    is_paragliding = False
    
    # Si aucun de ces champs ne contient "parapente", ce n'est pas un site de parapente
    if any("parapente" in field for field in [possible_usages, functions, description]):
        is_paragliding = True
    
    # Si le site est explicitement marqué pour le delta mais pas le parapente, l'exclure
    if ("delta" in possible_usages and "parapente" not in possible_usages) or \
       ("delta" in functions and "parapente" not in functions):
        is_paragliding = False
    
    # Maintenant vérifier si c'est un décollage (t=1)
    is_takeoff = False
    
    # Vérifier si le type indique un décollage (t=1)
    if terrain_type == "1" or terrain_type == 1:
        is_takeoff = True
    
    # Chercher des mots-clés de décollage dans les descriptions
    takeoff_keywords = ["décollage", "decollage", "déco", "deco"]
    if any(keyword in description.lower() for keyword in takeoff_keywords):
        is_takeoff = True
        
    # Si c'est un atterrissage, l'exclure même s'il contient un mot-clé de décollage
    landing_keywords = ["atterrissage", "atterro"]
    if any(keyword in description.lower() for keyword in landing_keywords) and not is_takeoff:
        return False
    
    # Si le site est marqué comme un atterrissage (t=2), l'exclure
    if terrain_type == "2" or terrain_type == 2:
        return False
    
    # Retourner True uniquement si c'est à la fois un site de parapente ET un décollage
    return is_paragliding and is_takeoff

def set_ffvl_site_and_analyze(site_data):
    """Fonction de callback pour définir un site FFVL et déclencher l'analyse"""
    st.session_state.site_selection = {
        "latitude": site_data["latitude"],
        "longitude": site_data["longitude"],
        "altitude": site_data["altitude"],
        "model": st.session_state.site_selection.get("model", "meteofrance_arome_france_hd")
    }
    st.session_state.run_analysis = True

def search_ffvl_sites(lat, lon, radius=20, api_key="79254946b01975fec7933ffc2a644dd7"):
    """
    Recherche les sites de vol à proximité d'une position donnée
    en utilisant l'API FFVL.
    """
    try:
        # URL de l'API FFVL pour les terrains
        url = f"https://data.ffvl.fr/api?base=terrains&mode=json&key={api_key}"
        
        logger.info(f"Requête FFVL: {url}")
        
        response = requests.get(url, timeout=10)
        
        logger.info(f"Statut: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")
        
        if response.status_code != 200:
            st.error(f"Erreur API FFVL: {response.status_code}")
            return []
        
        if not response.text.strip():
            st.warning("L'API FFVL a renvoyé une réponse vide")
            return []
        
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            st.error(f"Erreur de décodage JSON: {str(e)}")
            return []
        
        # Adapter le traitement en fonction du format réel de la réponse
        terrains = data if isinstance(data, list) else data.get("terrains", [])
        
        #st.info(f"Nombre total de terrains reçus: {len(terrains)}")
        
        # Si les terrains sont vides, afficher un message et retourner
        if not terrains:
            st.warning("Aucun terrain trouvé dans la réponse de l'API")
            return []
        
        # Afficher un exemple de terrain pour comprendre sa structure (sans utiliser d'expander)
        #if len(terrains) > 0:
            #st.info("Exemple des champs disponibles dans un terrain:" + 
                   #", ".join(list(terrains[0].keys())[:10]) + "...")
        
        # Filtrer les sites en fonction de leur distance et de leur usage pour le vol
        sites = []
        sites_count = 0
        takeoff_sites_count = 0
        
        for terrain in terrains:
            # Vérifier que les coordonnées sont présentes
            if not terrain.get("latitude") or not terrain.get("longitude"):
                continue
                
            try:
                site_lat = float(terrain["latitude"])
                site_lon = float(terrain["longitude"])
            except (ValueError, TypeError):
                continue
            
            # Calcul de distance (Haversine)
            from math import radians, cos, sin, asin, sqrt
            
            def haversine(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlon = lon2 - lon1 
                dlat = lat2 - lat1 
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a)) 
                r = 6371  # Rayon de la Terre en km
                return c * r
            
            distance = haversine(lat, lon, site_lat, site_lon)
            
            if distance <= radius:
                sites_count += 1
                
                # Utiliser la nouvelle fonction pour vérifier si c'est un décollage de parapente
                if is_paragliding_takeoff(terrain):
                    takeoff_sites_count += 1
                    
                    # Récupérer les informations pertinentes
                    site_info = {
                        "name": terrain.get("toponym", "Site sans nom"),
                        "type": "Décollage parapente",  # Spécifier explicitement que c'est un décollage
                        "latitude": site_lat,
                        "longitude": site_lon,
                        "distance": round(distance, 1),
                        "altitude": terrain.get("altitude", ""),
                        "orientation": terrain.get("wind_orientations_ok", ""),
                        "status": terrain.get("status", ""),
                        "difficulty": terrain.get("terrain_experience_conseillee", ""),
                        "ffvl_id": terrain.get("suid", ""),
                        "description": terrain.get("description", "") or "",
                        "icon_url": f"https://data.ffvl.fr/api/?base=terrains&mode=icon&tid={terrain.get('suid', '')}"
                    }
                    
                    sites.append(site_info)
        
        # Trier par distance
        sites.sort(key=lambda x: x["distance"])
        
        #st.info(f"Sites dans le rayon de {radius}km: {sites_count}")
        #st.info(f"Décollages de parapente identifiés: {takeoff_sites_count}")
        
        # Affichage sur la carte et tableau
        if sites:
            st.success(f"{len(sites)} décollages trouvés à proximité")
            
            # Créer une carte centrée sur la position recherchée
            import folium
            from streamlit_folium import folium_static
            
            m = folium.Map(location=[lat, lon], zoom_start=13)
            
            # Ajouter un marqueur pour la position de référence
            folium.Marker(
                [lat, lon],
                popup="Position de référence",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            # Ajouter un marqueur pour chaque site trouvé
            for i, site in enumerate(sites):
                # Déterminer la couleur du marqueur en fonction de la difficulté IPPI
                icon_color = "green"  # Vert = Niveau 3 (débutant) par défaut

                difficulty = site.get("difficulty", "")
                if difficulty is not None:  # Vérifier si la difficulté n'est pas None
                    difficulty_lower = str(difficulty).lower()
                    # Extraire les numéros IPPI éventuels (3, 4, 5)
                    if "ippi 5" in difficulty_lower or "ippi5" in difficulty_lower or "niveau 5" in difficulty_lower or "niveau5" in difficulty_lower or "confirmé" in difficulty_lower:
                        icon_color = "darkred"  # Marron/Rouge foncé = Niveau 5 (confirmé)
                    elif "ippi 4" in difficulty_lower or "ippi4" in difficulty_lower or "niveau 4" in difficulty_lower or "niveau4" in difficulty_lower or "pilote" in difficulty_lower:
                        icon_color = "blue"     # Bleu = Niveau 4 (pilote)
                    
                    # Détections supplémentaires par mots-clés
                    if "difficile" in difficulty_lower or "confirmé" in difficulty_lower or "confirme" in difficulty_lower:
                        icon_color = "darkred"  # Rouge foncé/Marron = Difficile/Confirmé
                    elif "moyen" in difficulty_lower:
                        icon_color = "blue"     # Bleu = Moyen
                
                # Préparer les informations pour le popup avec sécurité contre les None
                site_name = site.get("name", "Site sans nom")
                site_type = site.get("type", "Type non spécifié")
                
                # Altitude avec sécurité
                altitude_display = "Altitude non spécifiée"
                if site.get("altitude"):
                    altitude_display = f"{site['altitude']}m"
                
                # Construction du popup HTML avec des vérifications pour chaque valeur
                popup_html = f"""
                <div style="min-width: 200px;">
                    <h4 style="margin-bottom: 5px;">{site['name']}</h4>
                    <b>Type:</b> {site['type']}<br>
                    <b>Altitude:</b> {site['altitude']}m<br>
                    <b>Orientation:</b> {site['orientation']}<br>
                    <b>Difficulté:</b> <span style="color: {icon_color}; font-weight: bold;">{site['difficulty']}</span>
                </div>
                """
                
                # Créer le marqueur avec le popup
                folium.Marker(
                    [site["latitude"], site["longitude"]],
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=site_name,
                    icon=folium.Icon(color=icon_color, icon="flag")
                ).add_to(m)
            
            # Afficher la carte
            st.subheader("Carte des sites de décollage")

            # Légende pour les couleurs des sites
            st.markdown("""
            **Légende des niveaux de difficulté :**
            - 🟢 **Vert** : Niveau facile / Brevet Pilote Initial (IPPI 3)
            - 🔵 **Bleu** : Niveau intermédiaire / Brevet Pilote (IPPI 4)
            - 🟤 **Rouge/Marron** : Niveau confirmé / Brevet Pilote Confirmé (IPPI 5)
            """)

            folium_static(m, width=1300, height=600)
            
            # Afficher les sites dans un tableau pour sélection
            st.subheader("Sélectionner un site pour l'analyse")
            
            # Utiliser 3 colonnes pour l'affichage compact
            cols = st.columns(3)
            for i, site in enumerate(sites):
                col = cols[i % 3]
                with col:
                    # Sécuriser toutes les valeurs
                    site_name = site.get("name", "Site sans nom")
                    
                    altitude_display = ""
                    if site.get("altitude"):
                        altitude_display = f"{site['altitude']}m"
                    else:
                        altitude_display = "Alt. N/A"

                    # Déterminer la couleur pour l'affichage dans la liste
                    difficulty = site.get("difficulty", "")
                    difficulty_emoji = "🟢"  # Emoji vert par défaut
                    icon_color = "green"     # Couleur verte par défaut

                    # Appliquer la même logique de coloration que pour les marqueurs
                    if difficulty is not None:  # Vérifier si la difficulté n'est pas None
                        difficulty_lower = str(difficulty).lower()
                        # Extraire les numéros IPPI éventuels (3, 4, 5)
                        if "ippi 5" in difficulty_lower or "ippi5" in difficulty_lower or "niveau 5" in difficulty_lower or "niveau5" in difficulty_lower or "confirmé" in difficulty_lower:
                            icon_color = "darkred"  # Marron/Rouge foncé = Niveau 5 (confirmé)
                            difficulty_emoji = "🟤"  # Emoji marron
                        elif "ippi 4" in difficulty_lower or "ippi4" in difficulty_lower or "niveau 4" in difficulty_lower or "niveau4" in difficulty_lower or "pilote" in difficulty_lower:
                            icon_color = "blue"     # Bleu = Niveau 4 (pilote)
                            difficulty_emoji = "🔵"  # Emoji bleu
                        
                        # Détections supplémentaires par mots-clés
                        if "difficile" in difficulty_lower or "confirmé" in difficulty_lower or "confirme" in difficulty_lower:
                            icon_color = "darkred"  # Rouge foncé/Marron = Difficile/Confirmé
                            difficulty_emoji = "🟤"  # Emoji marron
                        elif "moyen" in difficulty_lower:
                            icon_color = "blue"     # Bleu = Moyen
                            difficulty_emoji = "🔵"  # Emoji bleu

                    # Afficher le nom du site avec l'indicateur de niveau
                    st.markdown(f"{difficulty_emoji} **{site_name}**")
                    st.markdown(f"{altitude_display} | {site['distance']}km | {difficulty}")
                    
                    # Préparer les données du site pour le callback
                    site_data = {
                        "latitude": float(site.get("latitude", 0)),
                        "longitude": float(site.get("longitude", 0)), 
                        "altitude": float(site.get("altitude", 0)) if site.get("altitude") else 0,
                        "name": site_name
                    }
                    
                    # Utiliser on_click avec la fonction callback
                    st.button(f"🪂 Utiliser", key=f"site_ffvl_{i}", 
                             on_click=set_ffvl_site_and_analyze, 
                             args=(site_data,))
            
            return sites
        else:
            st.warning("Aucun site de vol trouvé à proximité")
            st.info("Essayez d'augmenter le rayon de recherche ou de vérifier votre position")
            return []
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des sites FFVL: {e}", exc_info=True)
        st.error(f"Erreur: {str(e)}")
        return []
         
# Fonction pour afficher l'émagramme dans Streamlit
def display_emagramme(analyzer, analysis, llm_analysis=None):
    """Affiche l'émagramme et les résultats de l'analyse dans Streamlit"""
    # Utilise la nouvelle fonction fusionnée qui gère l'analyse IA en option
    fig = analyzer.plot_emagramme(analysis=analysis, llm_analysis=llm_analysis, show=False)
    st.pyplot(fig)

def display_multi_tab_emagrammes(lat, lon, model, site_altitude, api_key=None, openai_key=None, 
                            delta_t=None, data_source="open-meteo", max_hours=24, hour_step=3):
    """
    Affiche plusieurs émagrammes dans des onglets séparés pour différentes heures de prévision.
    Charge automatiquement TOUS les émagrammes dès le début.
    
    Args:
        lat, lon: Coordonnées du site
        model: Modèle météo à utiliser
        site_altitude: Altitude du site en mètres
        api_key: Clé API pour les services météo si nécessaire
        openai_key: Clé API OpenAI pour l'analyse IA si disponible
        delta_t: Delta de température pour le déclenchement des thermiques
        data_source: Source de données météo
        max_hours: Nombre maximum d'heures à récupérer
        hour_step: Pas de temps entre chaque prévision (en heures)
    """
    
    # 1. Générer la liste des heures disponibles
    available_hours = list(range(0, max_hours + 1, hour_step))
    
    # 2. Initialiser l'état de session pour les données multi-tab si pas déjà fait
    if 'multi_tab_data' not in st.session_state:
        # Initialiser avec toutes les clés nécessaires dès le début
        st.session_state.multi_tab_data = {
            'analyzers': {},
            'analyses': {},
            'detailed_analyses': {},
            'last_params': {
                'lat': lat,
                'lon': lon,
                'model': model,
                'site_altitude': site_altitude,
                'data_source': data_source
            },
            'total_hours': len(available_hours),
            'hours_loaded': 0,
            'loading_complete': False  # Initialiser à False pour déclencher le chargement
        }
    
    # 3. Vérifier si les paramètres ont changé
    params_changed = False
    current_params = {
        'lat': lat,
        'lon': lon,
        'model': model,
        'site_altitude': site_altitude,
        'data_source': data_source
    }
    
    if current_params != st.session_state.multi_tab_data['last_params']:
        params_changed = True
        # Réinitialiser avec toutes les clés
        st.session_state.multi_tab_data = {
            'analyzers': {},
            'analyses': {},
            'detailed_analyses': {},
            'last_params': current_params,
            'total_hours': len(available_hours),
            'hours_loaded': 0,
            'loading_complete': False  # Réinitialiser à False pour déclencher le chargement
        }
    
    # 4. Créer les onglets pour chaque heure
    tab_names = [f"H+{h}" for h in available_hours]
    tabs = st.tabs(tab_names)
    
    # 5. Fonction pour charger les données d'une heure
    def load_hour_data(hour):
        """Charge les données pour une heure spécifique"""
        try:
            # Récupérer les données depuis la source appropriée
            analyzer, analysis, detailed_analysis = fetch_and_analyze(
                lat, lon, model, site_altitude, api_key, openai_key, delta_t, 
                data_source=data_source, timestep=hour
            )
            
            # Vérifier si l'analyse a réussi
            if analyzer is None or analysis is None:
                return None, None, None
            
            # Stocker les résultats dans session_state
            st.session_state.multi_tab_data['analyzers'][hour] = analyzer
            st.session_state.multi_tab_data['analyses'][hour] = analysis
            st.session_state.multi_tab_data['detailed_analyses'][hour] = detailed_analysis
            st.session_state.multi_tab_data['hours_loaded'] += 1
            
            return analyzer, analysis, detailed_analysis
                
        except Exception as e:
            return None, None, None
    
    # 6. Charger toutes les heures si ce n'est pas déjà fait
    progress_placeholder = st.empty()
    progress_bar_placeholder = st.empty()
    
    # Vérifier si le chargement est nécessaire - utiliser get() pour éviter KeyError
    loading_complete = st.session_state.multi_tab_data.get('loading_complete', False)
    
    if not loading_complete or params_changed:
        # Afficher une barre de progression
        hours_to_load = [h for h in available_hours if h not in st.session_state.multi_tab_data['analyzers']]
        if hours_to_load:
            progress_placeholder.info(f"Chargement des émagrammes pour toutes les heures... ({len(hours_to_load)} restants)")
            progress_bar = progress_bar_placeholder.progress(0)
            
            # Charger les données pour chaque heure
            for i, hour in enumerate(hours_to_load):
                progress = int((i / len(hours_to_load)) * 100)
                progress_bar.progress(progress)
                progress_placeholder.info(f"Chargement de l'émagramme pour H+{hour} ({i+1}/{len(hours_to_load)})...")
                
                # Charger les données
                load_hour_data(hour)
                
            # Finaliser
            progress_bar.progress(100)
            st.session_state.multi_tab_data['loading_complete'] = True
            
    # Effacer les messages de progrès une fois terminé
    if st.session_state.multi_tab_data.get('loading_complete', False):
        progress_placeholder.empty()
        progress_bar_placeholder.empty()
    
    # 7. Initialiser un analyzer par défaut pour le retour
    analyzer_0 = None
    analysis_0 = None 
    detailed_analysis_0 = None
    
    # 8. Afficher le contenu dans chaque onglet
    for i, hour in enumerate(available_hours):
        with tabs[i]:
            now = datetime.now()
            forecast_time = now + timedelta(hours=hour)
            st.info(f"Prévision pour le {forecast_time.strftime('%d/%m/%Y à %H:%M')} (H+{hour})")
            
            # Afficher l'émagramme si disponible
            if hour in st.session_state.multi_tab_data['analyzers']:
                analyzer = st.session_state.multi_tab_data['analyzers'][hour]
                analysis = st.session_state.multi_tab_data['analyses'][hour]
                detailed_analysis = st.session_state.multi_tab_data['detailed_analyses'][hour]
                
                # Mémoriser les données de la première heure pour le retour
                if i == 0:
                    analyzer_0 = analyzer
                    analysis_0 = analysis
                    detailed_analysis_0 = detailed_analysis
                
                # Afficher l'émagramme
                st.subheader(f"Émagramme pour H+{hour}")
                display_emagramme(analyzer, analysis, detailed_analysis)
                
                # Afficher d'autres informations supplémentaires
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Plafond thermique", f"{analysis.thermal_ceiling:.0f} m")
                
                with col2:
                    st.metric("Force des thermiques", analysis.thermal_strength)
                
                with col3:
                    if hasattr(analysis, 'thermal_type') and analysis.thermal_type == "Cumulus":
                        st.metric("Base des nuages", f"{analysis.cloud_base:.0f} m")
                    else:
                        st.metric("Type de thermiques", "Bleus (sans condensation)")
                
                # Expander pour l'analyse détaillée
                with st.expander("Analyse détaillée", expanded=False):
                    if detailed_analysis:
                        st.markdown(detailed_analysis)
                    else:
                        st.info("Pas d'analyse détaillée disponible")
            else:
                # Afficher un message d'erreur si les données ne sont pas disponibles
                st.error(f"Impossible de charger les données pour H+{hour}. Essayez de rafraîchir la page.")
    
    # Bouton de rafraîchissement en bas des onglets
    if st.button("Rafraîchir toutes les données", key="refresh_all_emagrammes"):
        # Réinitialiser les données
        st.session_state.multi_tab_data = {
            'analyzers': {},
            'analyses': {},
            'detailed_analyses': {},
            'last_params': current_params,
            'total_hours': len(available_hours),
            'hours_loaded': 0,
            'loading_complete': False
        }
        st.rerun()
    
    # Retourner les données de la première heure pour compatibilité
    return analyzer_0, analysis_0, detailed_analysis_0

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
    
    # Ajouter ce code de débogage pour vérifier les valeurs
    #st.write("Valeurs du plafond thermique:", df["thermal_ceiling_relative"].tolist())
    #st.write("Valeurs du gradient thermique:", df["thermal_gradient"].tolist())

    # 1. CRÉER D'ABORD les graphiques, PUIS ajouter les annotations
    # Évolution du plafond thermique - Modifier le style du graphique
    fig_ceiling = px.line(df, x="time_str", y="thermal_ceiling_relative", 
                        labels={"thermal_ceiling_relative": "Plafond thermique (m au-dessus du site)", 
                                "time_str": "Date/Heure"},
                        title="Évolution du plafond thermique",
                        markers=True,
                        color_discrete_sequence=["blue"])  # Couleur bleue
    fig_ceiling.update_layout(hovermode="x unified")
    fig_ceiling.update_traces(line=dict(width=3))  # Ligne plus épaisse
    
    # MAINTENANT ajouter les rectangles au graphique du plafond
    fig_ceiling.add_hrect(y0=0, y1=500, line_width=0, fillcolor="red", opacity=0.1,
                        annotation_text="Faible", annotation_position="right")
    fig_ceiling.add_hrect(y0=500, y1=1500, line_width=0, fillcolor="yellow", opacity=0.1,
                        annotation_text="Moyen", annotation_position="right")
    fig_ceiling.add_hrect(y0=1500, y1=3000, line_width=0, fillcolor="green", opacity=0.1,
                        annotation_text="Bon", annotation_position="right")
    
    graphs["ceiling"] = fig_ceiling

    # 2. Évolution du gradient thermique - Style différent
    fig_gradient = px.line(df, x="time_str", y="thermal_gradient",
                        labels={"thermal_gradient": "Gradient (°C/1000m)", 
                                "time_str": "Date/Heure"},
                        title="Évolution du gradient thermique",
                        markers=True,
                        color_discrete_sequence=["red"])  # Couleur rouge
    fig_gradient.update_layout(hovermode="x unified")
    
    # Ajouter des annotations au graphique du gradient
    fig_gradient.add_hline(y=6.5, line_dash="dash", line_color="green", 
                        annotation_text="Bon gradient", annotation_position="right")
    
    # Ajouter les rectangles au graphique du gradient
    fig_gradient.add_hrect(y0=0, y1=4, line_width=0, fillcolor="red", opacity=0.1,
                        annotation_text="Faible", annotation_position="right")
    fig_gradient.add_hrect(y0=4, y1=6.5, line_width=0, fillcolor="yellow", opacity=0.1,
                        annotation_text="Moyen", annotation_position="right")
    fig_gradient.add_hrect(y0=6.5, y1=10, line_width=0, fillcolor="green", opacity=0.1,
                        annotation_text="Fort", annotation_position="right")
    
    graphs["gradient"] = fig_gradient
    
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
    
    # Graphique des scores de vol - Utiliser toutes les données disponibles
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
    
    # Trouver la meilleure période sur TOUTE la plage de données
    # (et non pas seulement sur la plage affichée)
    if not df.empty and len(df["vol_score"]) > 0:
        try:
            # MODIFICATION ICI: Utiliser toutes les données pour trouver le meilleur score
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
    sidebar_analyze_clicked = False
    main_analyze_clicked = False
    analyzer = None
    analysis = None
    detailed_analysis = None
    evolution_data = None
    timestep = 0

    # Initialiser l'état de la géolocalisation
    if 'geolocation_attempted' not in st.session_state:
        st.session_state.geolocation_attempted = False
        st.session_state.user_location = None

    # Vérification de la géolocalisation (exemple avec un bouton)
    expander_message = "‼️ IMPORTANT Géolocalisez-vous !" if not st.session_state.geolocation_attempted else "✅ Géolocalisation réussie !"
    with st.sidebar.expander(expander_message, expanded=not st.session_state.geolocation_attempted):

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
    with st.sidebar.expander("Source des données"):
        data_source = st.radio(
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
                "ecmwf_ifs025",  # Corriger le nom du modèle (au lieu de ecmwf_ifs04)
                "gfs_seamless"
            ]
            model_descriptions = {
                "meteofrance_arome_france_hd": "AROME HD (France ~2km)",
                "meteofrance_arpege_europe": "ARPEGE (Europe ~11km)",
                "meteofrance_arpege_world": "ARPEGE (Mondial ~40km)",
                "ecmwf_ifs025": "ECMWF IFS (Mondial ~25km)",  # Mise à jour du nom et de la résolution
                "gfs_seamless": "GFS (Mondial ~25km)"
            }
            model_labels = [model_descriptions[m] for m in model_options]
            model_index = st.selectbox(
                "Modèle météo",
                options=range(len(model_options)),
                format_func=lambda i: model_labels[i],
                index=0
            )
            model = model_options[model_index]
            
            model_changed = st.session_state.previous_model != model
            st.session_state.previous_model = model  # Mettre à jour le modèle précédent

            # Pas besoin de clé API pour Open-Meteo
            api_key = None
        
        # Ajouter la configuration FFVL
        st.subheader("Paramètres FFVL")
        # CSS pour masquer le bouton d'affichage du mot de passe
        hide_password_eye_css = """
        <style>
        div[data-testid="stTextInput"] button {
            display: none;
        }
        </style>
        """

        # Injecter le CSS dans l'application
        st.markdown(hide_password_eye_css, unsafe_allow_html=True)

        # Ensuite, utilisez votre text_input comme avant :
        ffvl_api_key = st.text_input("Clé API FFVL", 
                                    value=st.session_state.get("ffvl_api_key", "79254946b01975fec7933ffc2a644dd7"),
                                    type="password",
                                    help="Clé API FFVL pour la recherche de sites. Contactez informatique@ffvl.fr pour l'obtenir.")
            
        # Sauvegarder la clé API dans session_state
        if ffvl_api_key:
                st.session_state.ffvl_api_key = ffvl_api_key

        # Option pour clé OpenAI (analyse IA)
        use_ai = st.checkbox("Utiliser l'analyse IA externe (OpenAI)", value=False,
                                help="Utilise OpenAI pour générer une analyse détaillée en complément de l'analyse intégrée")
        
        if use_ai:
            openai_key = st.text_input("Clé API OpenAI", type="password")
        else:
            openai_key = None
            # Message indiquant que l'analyse intégrée sera utilisée
            st.info("L'analyse détaillée intégrée sera utilisée")
    
    # Paramètres avancés
    with st.sidebar.expander("Paramètres"):
        delta_t = st.slider("Delta T de déclenchement (°C)", 
                        min_value=1.0, max_value=6.0, value=3.0, step=0.5,
                        help="Différence de température requise pour déclencher un thermique")

        # Option pour l'évolution temporelle
        fetch_evolution_enabled = st.checkbox("Afficher l'évolution des conditions", value=True,
                                help="Récupère les données depuis l'heure actuelle jusqu'à l'heure de prévision sélectionnée")
        
        # Ajouter cette nouvelle option spécifique pour le mode multi-horaire avec slider
        use_multi_hour = st.checkbox("Mode multi-horaires", value=False,
                                help="Affiche H+ pour naviguer entre les émagrammes")
        
        if fetch_evolution_enabled or use_multi_hour:
            evolution_hours = st.slider("Période d'évolution (heures)", 
                                        min_value=6, max_value=48, value=24, step=6,
                                        help="Durée totale de la période d'évolution à analyser")
            evolution_step = st.slider("Pas de temps (heures)", 
                                        min_value=1, max_value=6, value=3, step=1,
                                        help="Intervalle entre chaque point d'analyse")

                
        else:
            evolution_hours = 24
            evolution_step = 3
        
        st.subheader("Type de surface")
        surface_type = st.selectbox(
            "Type de terrain dominant",
            options=["urban", "dark_rock", "light_rock", "dry_soil", "grass", "forest", "water", "sand", "snow"],
            format_func=lambda x: {
                "urban": "Zone urbaine", 
                "dark_rock": "Roches sombres", 
                "light_rock": "Roches claires", 
                "dry_soil": "Sol sec", 
                "grass": "Prairie/Végétation", 
                "forest": "Forêt", 
                "water": "Plan d'eau", 
                "sand": "Sable", 
                "snow": "Neige"
            }[x],
            index=4  # Grass par défaut
        )
        
        # Calculer le delta_t adaptatif
        if 'ground_temperature' in locals() and 'wind_speed' in locals() and 'cloud_cover' in locals():
            adaptive_delta = calculate_adaptive_trigger_delta(
                surface_type, 
                ground_temperature, 
                wind_speed,
                cloud_cover
            )
            
            # Afficher le calcul adaptatif
            st.info(f"Delta T adaptatif calculé: {adaptive_delta:.1f}°C")
            
            # Permettre à l'utilisateur de choisir entre valeur fixe et adaptative
            use_adaptive = st.checkbox("Utiliser le delta T adaptatif", value=True)
            
            if use_adaptive:
                delta_t = adaptive_delta
            else:
                delta_t = st.slider("Delta T de déclenchement (°C)", 
                                min_value=1.0, max_value=6.0, value=3.0, step=0.5, key="slider_delta_t",
                                help="Différence de température requise pour déclencher un thermique")
    
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
    
    def set_ffvl_site_and_analyze(site_data):
        """Fonction de callback pour définir un site FFVL et déclencher l'analyse"""
        st.session_state.site_selection = {
            "latitude": site_data["latitude"],
            "longitude": site_data["longitude"],
            "altitude": site_data["altitude"],
            "model": st.session_state.site_selection.get("model", "meteofrance_arome_france_hd")
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

    # Section pour le pas de temps de prévision (nouveau)
    # Déterminer la plage de temps disponible selon le modèle
    if model == "meteofrance_arome_france_hd":
                max_timestep = 36
                step_hours = 1
                model_name = "AROME"
                info_text = f"AROME: prévisions disponibles jusqu'à H+{max_timestep}"
    elif model == "meteofrance_arpege_europe" or model == "meteofrance_arpege_world":
                max_timestep = 96
                step_hours = 1
                model_name = "ARPÈGE"
                info_text = f"ARPÈGE: prévisions disponibles jusqu'à H+{max_timestep}"
    elif model == "ecmwf_ifs025":
                max_timestep = 120
                step_hours = 3  # Pas de 3h pour ECMWF
                model_name = "ECMWF IFS"
                info_text = f"ECMWF IFS: prévisions disponibles jusqu'à H+{max_timestep} par pas de 3h"
                st.warning("⚠️ Le modèle ECMWF IFS fournit des données à résolution 3-horaire")
    elif model == "gfs_seamless":
                max_timestep = 120
                step_hours = 1
                model_name = "GFS"
                info_text = f"GFS: prévisions disponibles jusqu'à H+{max_timestep}"
    else:
                max_timestep = 72
                step_hours = 1
                model_name = "Modèle"
                info_text = f"Modèle: prévisions disponibles jusqu'à H+{max_timestep}"

            # 5. Forcer la réinitialisation du calendrier si le modèle a changé
    if model_changed:
                # Réinitialiser le timestep quand le modèle change
                timestep = 0
                
                # Supprimer l'état du calendrier des sessions précédentes
                if "main_window_calendar" in st.session_state:
                    del st.session_state.main_window_calendar

                # Effacer toutes les clés de session liées au calendrier
                calendar_keys = [key for key in st.session_state.keys() if "calendar" in key]
                for key in calendar_keys:
                    del st.session_state[key]
                
                # Optionnel: réinitialiser aussi l'événement sélectionné
                if "selected_forecast_event" in st.session_state:
                    del st.session_state.selected_forecast_event

            # 6. Le reste du code du calendrier reste le même
    from datetime import datetime, timedelta
    now = datetime.now()
    end_date = now + timedelta(hours=max_timestep)

    st.header(f"📅 Sélection de l'heure de prévision - {model_name}")
    st.write(info_text)

            # Configurations du calendrier
    calendar_options = {
                "headerToolbar": {
                    "left": "today prev,next",
                    "center": "title",
                    "right": "timeGridDay,timeGridWeek"
                },
                "initialView": "timeGridWeek",  # Changer de "timeGridDay" à "timeGridWeek"
                "initialDate": now.strftime("%Y-%m-%d"),
                "slotMinTime": "00:00:00",
                "slotMaxTime": "24:00:00",
                "slotDuration": f"01:00:00",
                "expandRows": True,
                "height": "600px",
                "selectable": True,
                "editable": False,
                "navLinks": True
            }

            # Créer une liste d'événements pour les heures disponibles du modèle actuel
    calendar_events = []

            # Générer un événement pour chaque pas de temps disponible
    for hour in range(0, max_timestep + 1, step_hours):
                forecast_time = now + timedelta(hours=hour)
                
                # Formater l'heure pour l'affichage
                if hour == 0:
                    title = "Analyse actuelle"
                else:
                    # Pour les modèles à pas variable, spécifier le pas de temps dans le titre
                    if step_hours > 1:
                        days = hour // 24
                        hours_of_day = hour % 24
                        if days > 0:
                            title = f"J+{days}, {hours_of_day}h"
                        else:
                            title = f"H+{hour}"
                    else:
                        title = f"H+{hour}"
                
                # Créer l'événement
                event = {
                    "id": str(hour),
                    "title": title,
                    "start": forecast_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end": (forecast_time + timedelta(hours=step_hours)).strftime("%Y-%m-%dT%H:%M:%S"),
                    "resourceId": str(hour),  # Pour identifier facilement l'heure choisie
                    "backgroundColor": "#4285F4" if hour == 0 else "#34A853"  # Couleur différente pour l'analyse actuelle
                }
                calendar_events.append(event)

            # CSS personnalisé pour améliorer l'apparence du calendrier
    custom_css = """
                .fc-event-title {
                    font-weight: bold;
                }
                .fc-event-past {
                    opacity: 0.85;
                }
                .fc-toolbar-title {
                    font-size: 1.5rem;
                }
                .fc-timegrid-event {
                    cursor: pointer;
                }
                .fc-timegrid-event:hover {
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                }
            """
    calendar_key = f"calendar_{model.replace('-', '_')}_{int(time.time() / 100)}"

            # Afficher le calendrier avec une clé unique
    calendar_state = calendar(
                events=calendar_events,
                options=calendar_options,
                custom_css=custom_css,
                key=calendar_key  # Utiliser une clé dynamique au lieu d'une clé fixe
            )

            # 7. Stocker l'état du calendrier et l'événement sélectionné dans session_state
    if calendar_state and "eventClick" in calendar_state:
                st.session_state.selected_forecast_event = calendar_state["eventClick"]
                selected_event = calendar_state["eventClick"]
    elif "selected_forecast_event" in st.session_state:
                # Récupérer l'événement sélectionné précédemment (sauf si le modèle a changé)
                selected_event = st.session_state.selected_forecast_event
    else:
                selected_event = None

            # 8. Le reste du code reste le même
    if selected_event:
                # Extraire l'ID de l'événement (qui correspond au pas de temps)
                timestep = int(selected_event["event"]["id"])
                
                # Vérifier si le timestep est valide pour le modèle actuel
                if timestep > max_timestep:
                    timestep = 0
                    st.warning(f"L'heure précédemment sélectionnée n'est pas disponible avec ce modèle. Veuillez sélectionner une nouvelle heure.")
                    selected_event = None
                    if "selected_forecast_event" in st.session_state:
                        del st.session_state.selected_forecast_event
                else:
                    # Afficher la confirmation avec design amélioré
                    st.success(f"✅ Prévision sélectionnée: {selected_event['event']['title']}")
                    
                    # Calculer le format lisible (jours/heures)
                    days = timestep // 24
                    hours = timestep % 24
                    
                    if timestep > 0:
                        forecast_text = f"Prévision pour H+{hours}" if days == 0 else f"Prévision pour J+{days}, {hours}h"
                        st.info(forecast_text)
    else:
                # Message plus clair quand aucune sélection n'est faite
                timestep = 0
                st.info("👆 Veuillez cliquer sur une heure dans le calendrier pour sélectionner une prévision")
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
    with st.expander("🪂 Recherche de décollages proches FFVL", expanded=False):
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
                    api_key=st.session_state.get("ffvl_api_key", "79254946b01975fec7933ffc2a644dd7")
                )    


    # Bouton pour lancer l'analyse (IMPORTANT: définir 'analyze_clicked' AVANT de l'utiliser)
    main_analyze_clicked = st.button("Analyser l'émagramme")
    
    # Maintenant on peut utiliser analyze_clicked
    should_run_analysis = main_analyze_clicked or sidebar_analyze_clicked or st.session_state.get('run_analysis', False)

    if should_run_analysis:
        # Réinitialiser le flag pour éviter des analyses en boucle
        st.session_state.run_analysis = False

        # S'assurer que toutes les valeurs nécessaires sont présentes
        if latitude is None or longitude is None or site_altitude is None:
            st.error("Les coordonnées ou l'altitude sont manquantes. Veuillez les saisir.")
        else:
            # Mettre à jour la session avec les valeurs actuelles
            st.session_state.site_selection = {
                "latitude": latitude,
                "longitude": longitude,
                "altitude": site_altitude,
                "model": model
            }
        
        if use_openmeteo and not api_key:
            # Déterminer la source de données
            data_source_str = "open-meteo"
            
            # Vérifier si on utilise le mode multi-horaire avec slider
            if use_multi_hour:
                # Générer automatiquement tous les émagrammes
                st.subheader("Prévisions par heure")
                analyzer, analysis, detailed_analysis = display_multi_tab_emagrammes(
                    latitude, longitude, model, site_altitude, 
                    api_key, openai_key, delta_t, 
                    data_source=data_source_str, 
                    max_hours=evolution_hours, 
                    hour_step=evolution_step
                )
                
                # Définir evolution_data à None puisque nous utilisons une autre approche
                evolution_data = None
            else:
                # Mode standard avec ou sans évolution - conservé tel quel
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
                # Pour le mode multi-horaire, ne pas afficher à nouveau l'émagramme
                # car il est déjà affiché dans la fonction display_multi_hour_emagramme
                if not use_multi_hour:
                    # Afficher l'émagramme (uniquement en mode standard)
                    st.subheader("Émagramme")
                    display_emagramme(analyzer, analysis)
                    
                    # Calculer l'information sur la couche convective
                    convective_layer = calculate_convective_layer_thickness(analyzer, analysis)
                    
                    # Créer et afficher la visualisation de la couche convective
                    st.subheader("Visualisation de la couche convective")
                    
                    # Explication de la couche convective (dans un expander)
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
                
                # Création des onglets (commun à tous les modes)
                if fetch_evolution_enabled and evolution_data:
                    # Vérifier si une erreur est signalée
                    if "error" in evolution_data:
                        st.error(evolution_data["message"])
                        st.error("Veuillez essayer d'augmenter ou réduire la période de prévision dans la sidebar : 'Paramètres'-'Heure de prévision'.")
                        # Ne pas continuer avec l'analyse d'évolution
                    else:
                        # Poursuivre avec votre code existant pour créer les graphiques
                        with st.spinner("Génération des graphiques d'évolution..."):
                            graphs, best_period, evolution_df = create_evolution_plots(evolution_data, site_altitude)
                        
                        # Vérifier si les graphiques ont été créés avec succès
                        if not graphs or not best_period or "time" not in best_period:
                            st.warning("Impossible de générer les graphiques d'évolution. Données insuffisantes.")
                    
                    if not use_multi_hour:
                        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Résultats", "Évolution et Données brutes", "Sites FFVL", "Aide", "Analyse avancée"])
                        # Variable pour suivre si les tabs sont créés
                        has_tab1 = True
                    else:
                        # En mode multi-horaire, ne pas créer le premier onglet
                        tab2, tab3, tab4, tab5 = st.tabs(["Évolution et Données brutes", "Sites FFVL", "Aide", "Analyse avancée"])
                        # Indiquer que tab1 n'est pas disponible
                        has_tab1 = False
                else:
                    if not use_multi_hour:
                        tab1, tab2, tab3, tab4 = st.tabs(["Résultats", "Données brutes", "Sites FFVL", "Aide"])
                        has_tab1 = True
                    else:
                        tab2, tab3, tab4 = st.tabs(["Données brutes", "Sites FFVL", "Aide"])
                        has_tab1 = False
 
                if 'has_tab1' in locals() and has_tab1:
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
                    
                            # Analyse des types de nuages
                            if (analysis.low_cloud_cover is not None or 
                                analysis.mid_cloud_cover is not None or 
                                analysis.high_cloud_cover is not None):
                                
                                st.subheader("Analyse des nuages")
                                
                                # Identifier les types de nuages
                                cloud_analysis = identify_cloud_types(
                                    analysis.low_cloud_cover, 
                                    analysis.mid_cloud_cover, 
                                    analysis.high_cloud_cover,
                                    analysis.thermal_ceiling,
                                    analysis.ground_temperature,
                                    analysis.ground_dew_point,
                                    analysis.precipitation_type
                                )
                                
                                # Afficher les types de nuages identifiés
                                cloud_types = cloud_analysis["identified_types"]
                                
                                if cloud_types:
                                    for cloud in cloud_types:
                                        if cloud["severity"] == "extreme":
                                            st.error(f"**{cloud['type']}** ({cloud['coverage']}%) - {cloud['impact']}")
                                        elif cloud["severity"] == "high":
                                            st.warning(f"**{cloud['type']}** ({cloud['coverage']}%) - {cloud['impact']}")
                                        elif cloud["severity"] == "medium":
                                            st.info(f"**{cloud['type']}** ({cloud['coverage']}%) - {cloud['impact']}")
                                        else:
                                            st.success(f"**{cloud['type']}** ({cloud['coverage']}%) - {cloud['impact']}")
                                    
                                    # Afficher le risque d'orage si pertinent
                                    if cloud_analysis["thunderstorm_risk"]:
                                        st.error(f"⚠️ **Risque d'orages {cloud_analysis['thunderstorm_proximity']}** - Soyez extrêmement vigilant")
                                else:
                                    st.info("Aucun nuage significatif identifié")

                if fetch_evolution_enabled and evolution_data and "tab2" in locals():
                    with tab2:
                        st.header("Évolution des conditions sur la période")
                        st.info(f"La meilleure période a été calculée sur la plage de H+0 à H+{timestep}.")

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
                    
                    # Option pour choisir le mode d'affichage
                    display_mode = st.radio(
                        "Mode d'affichage des sites",
                        options=["Standard", "Recommandations basées sur les conditions actuelles"],
                        index=0
                    )

                    if st.button("Rechercher des sites FFVL", key="search_ffvl"):
                        with st.spinner("Recherche des sites FFVL..."):
                            sites = search_ffvl_sites(
                                ffvl_lat, 
                                ffvl_lon, 
                                radius=search_radius, 
                                api_key=st.session_state.get("ffvl_api_key", "79254946b01975fec7933ffc2a644dd7")
                            )

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
                
                if fetch_evolution_enabled and evolution_data and 'tab5' in locals():
                    with tab5:
                        st.header("Analyse aérologique avancée")
                        
                        # Introduction à l'analyse avancée
                        st.markdown("""
                        Cette section fournit une analyse détaillée des conditions aérologiques basée sur 
                        des modèles avancés. Ces informations sont particulièrement utiles pour les pilotes 
                        expérimentés cherchant à optimiser leur stratégie de vol.
                        """)
                        
                        # Créer plusieurs sections pour les différentes analyses
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Analyse du terrain")
                            
                            with st.expander("ℹ️ Comprendre l'analyse du terrain", expanded=False):
                                st.markdown("""
                                **L'analyse du terrain** évalue comment la topographie locale influence les mouvements d'air:
                                
                                - **Facteur d'ensoleillement**: Mesure l'efficacité de l'exposition au soleil selon l'orientation et l'heure.
                                _Valeur idéale_: > 0.7 (exposition optimale au rayonnement solaire)
                                
                                - **Effet venturi**: Quantifie l'accélération du vent due au relief.
                                _Valeur idéale_: 1.0-1.5 (une valeur de 1.5 indique une amplification de 50%)
                                
                                - **Amplification thermique**: Combine tous les facteurs pour estimer l'intensification thermique.
                                _Valeur idéale_: > 1.5 (les thermiques sont 50% plus puissants qu'en terrain plat)
                                """)
                            
                            # Interface pour l'analyse du terrain
                            slope_angle = st.slider("Angle de la pente (°)", 0, 60, 30, 
                                                help="Angle moyen de la pente à proximité du décollage")
                            
                            aspect = st.select_slider("Orientation de la pente", 
                                                options=["N", "NE", "E", "SE", "S", "SO", "O", "NO"],
                                                value="S")
                            
                            # Convertir l'orientation en degrés
                            aspect_degrees = {"N": 0, "NE": 45, "E": 90, "SE": 135, 
                                            "S": 180, "SO": 225, "O": 270, "NO": 315}[aspect]
                            
                            # Calculer l'effet du relief
                            terrain_effect = analyze_terrain_effect(
                                latitude, longitude, analysis.ground_altitude, slope_angle, aspect_degrees
                            )
                            
                            # Ajouter des indicateurs de qualité avec des emojis
                            insolation_emoji = "🟢" if terrain_effect['insolation_factor'] > 0.7 else "🟡" if terrain_effect['insolation_factor'] > 0.4 else "🔴"
                            venturi_emoji = "🟢" if 1.0 <= terrain_effect['venturi_factor'] <= 1.5 else "🟡" if terrain_effect['venturi_factor'] < 2.0 else "🔴"
                            multiplier_emoji = "🟢" if terrain_effect['thermal_multiplier'] > 1.5 else "🟡" if terrain_effect['thermal_multiplier'] > 1.0 else "🔴"
                            
                            st.info(f"{insolation_emoji} **Facteur d'ensoleillement:** {terrain_effect['insolation_factor']:.2f}")
                            st.info(f"{venturi_emoji} **Effet venturi:** {terrain_effect['venturi_factor']:.2f}x")
                            st.info(f"{multiplier_emoji} **Amplification thermique:** {terrain_effect['thermal_multiplier']:.2f}x")
                            
                            # Explication détaillée du calcul
                            with st.expander("Comment ce calcul est-il effectué?", expanded=False):
                                st.markdown("""
                                **Méthode de calcul:**
                                
                                1. **Facteur d'ensoleillement**: 
                                - Calcul de l'angle d'élévation solaire basé sur la date
                                - Détermination de l'azimut solaire (position horizontale)
                                - Calcul de l'angle d'incidence des rayons sur la pente
                                - Normalisation entre 0 et 1
                                
                                2. **Effet venturi**:
                                - Formule: 1 + (angle_pente / 45) * 0.5
                                - Une pente de 45° amplifie le vent de 50%
                                
                                3. **Effet de compression orographique**:
                                - Prend en compte l'altitude du site
                                - Formule: 1 + (altitude / 3000) * 0.3
                                
                                4. **Amplification thermique totale**:
                                - Multiplication des trois facteurs ci-dessus
                                """)
                        
                        with col2:
                            st.subheader("Analyse des vents")
                            
                            with st.expander("ℹ️ Comprendre l'analyse des vents", expanded=False):
                                st.markdown("""
                                **L'analyse du profil de vent** examine comment le vent varie avec l'altitude:
                                
                                - **Niveau de turbulence**: Estimation basée sur les cisaillements (changements de vitesse/direction)
                                _Interprétation_: "Faible" est idéal, "Modérée" demande de la vigilance, "Forte" ou plus est problématique
                                
                                - **Score de turbulence**: Valeur numérique (0-1) quantifiant les turbulences
                                _Interprétation_: < 0.4 = conditions confortables, > 0.6 = conditions difficiles
                                
                                - **Phénomènes détectés**: Identifie des structures spécifiques comme les jets de basse couche,
                                les convergences ou les zones de cisaillement important
                                """)
                            
                            if hasattr(analyzer, 'wind_speeds') and analyzer.wind_speeds is not None:
                                # Analyse du profil de vent
                                wind_analysis = analyze_wind_profile(
                                    analyzer.altitudes, 
                                    analyzer.wind_speeds, 
                                    analyzer.wind_directions, 
                                    analysis.ground_altitude, 
                                    analysis.thermal_ceiling
                                )
                                
                                if wind_analysis["valid"]:
                                    # Ajouter des emojis selon le niveau de turbulence
                                    turbulence_emoji = "🟢" if wind_analysis["turbulence_level"] in ["Très faible", "Faible"] else "🟡" if wind_analysis["turbulence_level"] == "Modérée" else "🔴"
                                    
                                    st.metric("Niveau de turbulence", f"{turbulence_emoji} {wind_analysis['turbulence_level']}")
                                    
                                    # Ajout d'une jauge pour visualiser le score de turbulence
                                    st.markdown(f"**Score de turbulence: {wind_analysis['turbulence_score']:.2f}**")
                                    
                                    # Créer une barre de progression pour visualiser le score
                                    turbulence_color = "green" if wind_analysis['turbulence_score'] < 0.4 else "orange" if wind_analysis['turbulence_score'] < 0.7 else "red"
                                    st.progress(float(wind_analysis['turbulence_score']))
                                    
                                    with st.expander("Comment ce score est-il calculé?", expanded=False):
                                        st.markdown("""
                                        **Calcul du score de turbulence:**
                                        
                                        Le score combine 3 facteurs principaux:
                                        1. **Gradient de vitesse du vent** (40% du score): mesure les variations de vitesse avec l'altitude
                                        2. **Gradient de direction du vent** (40% du score): mesure les rotations du vent avec l'altitude
                                        3. **Vitesse moyenne du vent** (20% du score): les vents plus forts génèrent plus de turbulence
                                        
                                        Chaque facteur est normalisé entre 0 et 1, puis combiné avec sa pondération.
                                        """)
                                    
                                    # Afficher les phénomènes de vent significatifs
                                    if wind_analysis["wind_phenomena"]:
                                        st.markdown("**Phénomènes de vent détectés:**")
                                        for phenomenon in wind_analysis["wind_phenomena"]:
                                            st.warning(f"⚠️ {phenomenon['description']} - {phenomenon['impact']}")
                                            
                                            # Ajouter des explications pour chaque type de phénomène
                                            if phenomenon['type'] == "jet_basse_couche":
                                                with st.expander("Qu'est-ce qu'un jet de basse couche?"):
                                                    st.markdown("""
                                                    Un **jet de basse couche** est une accélération localisée du vent à une altitude 
                                                    relativement basse. Il peut créer des rotors (turbulences importantes) sous le jet 
                                                    et des zones de cisaillement au-dessus et en-dessous.
                                                    
                                                    **Impact sur le vol:** 
                                                    - Turbulence importante en dessous du jet
                                                    - Dérive rapide si vous entrez dans la couche du jet
                                                    - Difficulté à maintenir un cap constant
                                                    """)
                                            elif phenomenon['type'] == "convergence":
                                                with st.expander("Qu'est-ce qu'une convergence?"):
                                                    st.markdown("""
                                                    Une **convergence** se produit lorsque deux masses d'air se rencontrent, 
                                                    créant une zone d'ascendance. Elle est caractérisée par un changement 
                                                    important de la direction du vent sur une faible épaisseur d'altitude.
                                                    
                                                    **Impact sur le vol:**
                                                    - Peut créer une ligne d'ascendance exploitable
                                                    - Permet parfois le vol dynamique même sans thermiques
                                                    - Peut être turbulente mais prédictible
                                                    """)
                                            elif phenomenon['type'] == "couche_vent_fort":
                                                with st.expander("Qu'est-ce qu'une couche de vent fort?"):
                                                    st.markdown("""
                                                    Une **couche de vent fort** est une strate d'atmosphère où la vitesse 
                                                    du vent est significativement plus élevée qu'aux altitudes adjacentes.
                                                    
                                                    **Impact sur le vol:**
                                                    - Dérive importante dans cette couche
                                                    - Difficulté à progresser face au vent
                                                    - Risque de ne pas pouvoir rentrer au terrain si vous dérivez trop
                                                    - Peut limiter le plafond pratique même si le plafond thermique est plus haut
                                                    """)
                                    else:
                                        st.success("✅ Aucun phénomène de vent particulier détecté")
                                else:
                                    st.warning(wind_analysis["message"])
                            else:
                                st.warning("Données de vent insuffisantes pour l'analyse détaillée")
                        
                        # Section pour l'analyse de la stabilité
                        st.subheader("Stabilité atmosphérique avancée")
                        
                        with st.expander("ℹ️ Comprendre la stabilité atmosphérique", expanded=False):
                            st.markdown("""
                            **La stabilité atmosphérique** détermine la tendance de l'air à rester en place ou à se déplacer verticalement:
                            
                            - **Gradient thermique**: Taux de diminution de la température avec l'altitude
                            _Interprétation_: Idéal entre 6-8°C/km, < 5°C/km = stable, > 9°C/km = instable
                            
                            - **Indice K**: Mesure de l'instabilité et du potentiel orageux
                            _Interprétation_: < 15 = stable, 15-25 = quelques orages possibles, > 25 = risque d'orages important
                            
                            - **Stabilité**: Évaluation qualitative basée sur le Lifted Index
                            _Interprétation_: "Légèrement instable" est idéal pour le vol thermique
                            
                            - **Qualité thermique**: Évaluation de la qualité des thermiques basée sur le gradient
                            _Interprétation_: "Bonne" = thermiques bien formés et prévisibles
                            """)
                        
                        # Estimer les niveaux de pression à partir des altitudes
                        pressure_levels = 1013.25 * (1 - (analyzer.altitudes / 44330)) ** 5.255
                        
                        stability_analysis = calculate_advanced_stability(
                            analyzer.temperatures, 
                            analyzer.dew_points, 
                            analyzer.altitudes, 
                            pressure_levels
                        )
                        
                        if stability_analysis["stability_valid"]:
                            # Ajouter des explications et des indicateurs de qualité
                            gradient_emoji = "🟢" if 6.0 <= stability_analysis['overall_lapse_rate'] <= 8.0 else "🟡" if 5.0 <= stability_analysis['overall_lapse_rate'] <= 9.0 else "🔴"
                            stability_emoji = "🟢" if stability_analysis['stability'] in ["Légèrement instable", "Modérément instable"] else "🟡" if stability_analysis['stability'] == "Stable" else "🔴"
                            thermal_emoji = "🟢" if stability_analysis['thermal_quality'] == "good" else "🟡" if stability_analysis['thermal_quality'] == "moderate" else "🔴"
                            
                            # Déterminer l'emoji pour l'indice K
                            if stability_analysis['k_index'] is not None:
                                if stability_analysis['k_index'] < 15:
                                    k_emoji = "🟢"  # Peu de risque d'orage
                                elif stability_analysis['k_index'] < 25:
                                    k_emoji = "🟡"  # Quelques orages possibles
                                else:
                                    k_emoji = "🔴"  # Risque d'orages important
                            else:
                                k_emoji = "⚪"  # Indice K non disponible
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Gradient thermique", f"{gradient_emoji} {stability_analysis['overall_lapse_rate']:.1f}°C/km")
                                with st.expander("Qu'est-ce que le gradient thermique?"):
                                    st.markdown("""
                                    Le **gradient thermique** mesure la diminution de la température avec l'altitude.
                                    
                                    **Interprétation:**
                                    - **< 5°C/km**: Atmosphère très stable, thermiques faibles
                                    - **5-6°C/km**: Assez stable, thermiques modérés
                                    - **6-7°C/km**: Gradient standard, bonnes conditions
                                    - **7-8°C/km**: Légèrement instable, thermiques puissants
                                    - **> 9°C/km**: Très instable, thermiques puissants mais turbulents
                                    
                                    Le gradient standard dans l'atmosphère est d'environ 6.5°C/km.
                                    """)
                                
                                st.metric("Indice K", f"{k_emoji} {stability_analysis['k_index']:.1f}" if stability_analysis['k_index'] is not None else "⚪ N/A")
                                with st.expander("Qu'est-ce que l'indice K?"):
                                    st.markdown("""
                                    L'**indice K** est un indicateur météorologique du potentiel d'orages.
                                    
                                    Il est calculé à partir des différences de température et d'humidité 
                                    à différents niveaux d'altitude.
                                    
                                    **Interprétation:**
                                    - **< 15**: Air sec, peu de risque d'orage
                                    - **15-20**: Quelques orages isolés possibles
                                    - **20-25**: Orages épars possibles
                                    - **25-30**: Nombreux orages possibles
                                    - **> 30**: Orages généralisés
                                    """)
                            
                            with col2:
                                st.metric("Stabilité", f"{stability_emoji} {stability_analysis['stability']}")
                                with st.expander("Comment interpréter la stabilité?"):
                                    st.markdown("""
                                    La **stabilité** indique la résistance de l'atmosphère aux mouvements verticaux.
                                    
                                    **Interprétation:**
                                    - **Très stable**: Peu ou pas de thermiques, air calme
                                    - **Stable**: Thermiques faibles et prévisibles
                                    - **Légèrement instable**: Conditions idéales pour le vol thermique
                                    - **Modérément instable**: Thermiques puissants et bien formés
                                    - **Très instable**: Thermiques puissants mais turbulents, risque d'orages
                                    
                                    Pour le vol en parapente, une atmosphère légèrement à modérément instable 
                                    offre généralement les meilleures conditions.
                                    """)
                                
                                st.metric("Risque d'orage", f"{k_emoji} {stability_analysis['k_interpretation']}")
                                with st.expander("Comment est évalué le risque d'orage?"):
                                    st.markdown("""
                                    Le **risque d'orage** est évalué principalement à partir de l'indice K et d'autres paramètres 
                                    de stabilité.
                                    
                                    L'instabilité atmosphérique, combinée à une humidité suffisante à différents niveaux,
                                    crée les conditions favorables au développement orageux.
                                    
                                    Les orages présentent des risques majeurs pour le vol en parapente:
                                    - Rafales violentes et imprévisibles
                                    - Mouvements verticaux extrêmes
                                    - Grêle et précipitations intenses
                                    - Foudre
                                    
                                    **En cas de risque d'orage, il est recommandé de ne pas voler ou d'atterrir rapidement.**
                                    """)
                            
                            with col3:
                                st.metric("Qualité thermique", f"{thermal_emoji} {stability_analysis['thermal_quality'].capitalize()}")
                                with st.expander("Comment est évaluée la qualité thermique?"):
                                    st.markdown("""
                                    La **qualité thermique** évalue la structure et la prévisibilité des thermiques.
                                    
                                    **Facteurs pris en compte:**
                                    - Gradient thermique
                                    - Force des inversions
                                    - Stabilité globale de l'atmosphère
                                    
                                    **Interprétation:**
                                    - **Faible**: Thermiques discontinus, difficiles à exploiter
                                    - **Modérée**: Thermiques utilisables mais irréguliers
                                    - **Bonne**: Thermiques bien formés, réguliers et prévisibles
                                    
                                    Une bonne qualité thermique facilite la prise d'altitude et le maintien du vol.
                                    """)
                                
                                st.metric("Force des inversions", f"{stability_analysis['inversion_strength'].capitalize()}")
                                with st.expander("Qu'est-ce que la force des inversions?"):
                                    st.markdown("""
                                    La **force des inversions** évalue l'intensité des couches où la température 
                                    augmente avec l'altitude (inversions).
                                    
                                    **Interprétation:**
                                    - **Faible**: Inversions peu marquées, facilement traversables par les thermiques
                                    - **Modérée**: Inversions notables qui peuvent limiter la hauteur des thermiques
                                    - **Forte**: Inversions importantes qui bloquent efficacement les thermiques
                                    
                                    Les inversions agissent comme des "couvercles" qui limitent le développement vertical 
                                    des thermiques et peuvent définir le plafond thermique.
                                    """)
                        else:
                            st.warning(stability_analysis["message"])
                        
                        # Section pour la prédiction de la durée de vol
                        st.subheader("Prédiction de vol")
                        
                        with st.expander("ℹ️ Comprendre la prédiction de vol", expanded=False):
                            st.markdown("""
                            **La prédiction de vol** estime les caractéristiques du vol possible dans les conditions actuelles:
                            
                            - **Durée de vol estimée**: Temps de vol typique dans ces conditions
                            _Interprétation_: > 3h = excellentes conditions, 1-2h = conditions standards
                            
                            - **Taux de montée moyen**: Vitesse verticale moyenne des thermiques
                            _Interprétation_: < 1 m/s = faible, 1-2 m/s = moyen, > 2 m/s = fort
                            
                            - **Potentiel Cross-Country**: Évaluation de la possibilité de faire des vols de distance
                            _Interprétation_: "High" = conditions favorables pour des vols de distance
                            
                            - **Thermiques par heure**: Fréquence estimée des thermiques exploitables
                            _Interprétation_: > 6 = bonne fréquence, < 3 = thermiques rares
                            """)
                        
                        import numpy as np

                        # Prédiction de la durée de vol
                        flight_prediction = predict_flight_duration(
                            analysis.thermal_ceiling,
                            analysis.ground_altitude,
                            analysis.thermal_strength,
                            20 if analyzer.wind_speeds is None else np.nanmean(analyzer.wind_speeds),  # Vent moyen
                            analysis.low_cloud_cover,
                            analysis.thermal_gradient
                        )
                        
                        # Ajouter des indicateurs de qualité
                        duration_emoji = "🟢" if flight_prediction['flight_duration_hours'] > 3 else "🟡" if flight_prediction['flight_duration_hours'] > 1.5 else "🔴"
                        climb_emoji = "🟢" if flight_prediction['avg_climb_rate'] > 2 else "🟡" if flight_prediction['avg_climb_rate'] > 1 else "🔴"
                        xc_emoji = "🟢" if flight_prediction['xc_potential'] == "High" else "🟡" if flight_prediction['xc_potential'] == "Medium" else "🔴"
                        thermals_emoji = "🟢" if flight_prediction['thermals_per_hour'] > 6 else "🟡" if flight_prediction['thermals_per_hour'] > 3 else "🔴"
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Durée de vol estimée", f"{duration_emoji} {flight_prediction['flight_duration_hours']:.1f} heures")
                            with st.expander("Comment est calculée la durée de vol?"):
                                st.markdown("""
                                La **durée de vol estimée** combine plusieurs facteurs:
                                
                                1. **Hauteur exploitable** (plafond - altitude du site)
                                2. **Force des thermiques** (selon l'analyse)
                                3. **Conditions de vent** (vents modérés sont optimaux)
                                4. **Couverture nuageuse** (affecte l'ensoleillement)
                                5. **Gradient thermique** (indicateur de l'activité thermique)
                                
                                Une durée supérieure à 3 heures indique des conditions exceptionnelles,
                                tandis qu'une durée inférieure à 1 heure suggère des conditions marginales.
                                """)
                            
                            st.metric("Taux de montée moyen", f"{climb_emoji} {flight_prediction['avg_climb_rate']:.1f} m/s")
                            with st.expander("Comment est estimé le taux de montée?"):
                                st.markdown("""
                                Le **taux de montée moyen** est estimé principalement à partir:
                                
                                - De la force des thermiques indiquée dans l'analyse
                                - Du gradient thermique
                                - De la stabilité atmosphérique
                                
                                **Valeurs typiques:**
                                - < 1 m/s: Thermiques faibles, gain d'altitude lent
                                - 1-2 m/s: Thermiques modérés, bon pour le vol local
                                - 2-3 m/s: Thermiques forts, idéal pour le cross
                                - > 3 m/s: Thermiques très puissants, conditions potentiellement turbulentes
                                
                                Ces valeurs représentent des moyennes - les pics peuvent être plus élevés.
                                """)
                        
                        with col2:
                            st.metric("Potentiel Cross-Country", f"{xc_emoji} {flight_prediction['xc_potential']}")
                            with st.expander("Comment est évalué le potentiel cross?"):
                                st.markdown("""
                                Le **potentiel cross-country** évalue la possibilité de réaliser des vols de distance:
                                
                                **Facteurs pris en compte:**
                                - Durée de vol possible
                                - Plafond thermique
                                - Force des thermiques
                                - Régularité des ascendances
                                
                                **Interprétation:**
                                - **Low**: Vol local recommandé, conditions limitées
                                - **Medium**: Petit cross possible, vigilance requise
                                - **High**: Bonnes conditions pour vol de distance
                                
                                La planification d'un vol cross doit toujours prendre en compte d'autres facteurs
                                comme les zones aériennes, le terrain, et vos options d'atterrissage.
                                """)
                            
                            st.metric("Thermiques par heure", f"{thermals_emoji} {flight_prediction['thermals_per_hour']:.1f}")
                            with st.expander("Comment est calculée la fréquence des thermiques?"):
                                st.markdown("""
                                La **fréquence des thermiques** estime combien de thermiques exploitables 
                                un pilote peut rencontrer chaque heure.
                                
                                **Facteurs pris en compte:**
                                - Force des thermiques (thermiques plus forts = plus détectables)
                                - Gradient thermique (gradient plus fort = déclenchements plus fréquents)
                                - Stabilité de l'atmosphère
                                
                                **Interprétation:**
                                - < 3 thermiques/heure: Ascendances rares, vol difficile
                                - 3-6 thermiques/heure: Fréquence moyenne, vol standard
                                - > 6 thermiques/heure: Haute fréquence, vol facilité
                                
                                Une fréquence élevée permet de maintenir le vol plus facilement, même si
                                les thermiques individuels sont moins puissants.
                                """)
                        
                        # Heures optimales
                        st.info(f"**Heures optimales de vol:** Matin: {int(flight_prediction['optimal_start_morning'])}h{int((flight_prediction['optimal_start_morning'] % 1) * 60):02d} | Après-midi: {int(flight_prediction['optimal_start_afternoon'])}h{int((flight_prediction['optimal_start_afternoon'] % 1) * 60):02d}")
                        with st.expander("Comment sont calculées les heures optimales?"):
                            st.markdown("""
                            Les **heures optimales de vol** sont déterminées en fonction:
                            
                            - De la force des thermiques (pour les thermiques plus forts, on privilégie le début et la fin de journée)
                            - De la couverture nuageuse (influence l'ensoleillement et donc le cycle thermique)
                            - De la saison et de l'orientation du site
                            
                            **Stratégies selon les conditions:**
                            - **Thermiques forts**: Voler tôt le matin ou en fin d'après-midi pour éviter les survitesses
                            - **Thermiques modérés**: Voler en milieu de journée quand l'activité est maximale
                            - **Thermiques faibles**: Voler au moment le plus chaud, généralement entre 12h et 14h
                            
                            Ces heures sont des estimations et peuvent varier selon les conditions locales.
                            """)
                        
                        # Nouvelle section pour l'analyse des convergences de vent
                        st.subheader("Analyse des convergences")

                        with st.expander("ℹ️ Comprendre les convergences de vent", expanded=False):
                            st.markdown("""
                            **Les convergences de vent** sont des zones où des vents de directions différentes se rencontrent, 
                            créant souvent des ascendances exploitables:
                            
                            - **Zones de convergence**: Situations où les vents de directions différentes s'affrontent
                            - **Force de convergence**: Intensité de l'ascendance générée par la convergence
                            - **Potentiel d'ascendance dynamique**: Possibilité d'exploiter les convergences pour le vol
                            
                            Les convergences peuvent créer des lignes d'ascendance puissantes et persistantes,
                            parfois exploitables même en l'absence de thermiques.
                            """)

                        # Vérifier si les données de vent sont disponibles
                        if hasattr(analyzer, 'wind_speeds') and analyzer.wind_speeds is not None:
                            # Analyse des convergences
                            convergence_analysis = detect_convergence_zones(
                                analyzer.wind_directions, 
                                analyzer.wind_speeds, 
                                analyzer.altitudes
                            )
                            
                            if convergence_analysis["has_convergence"]:
                                st.success("✅ Zones de convergence détectées!")
                                
                                # Afficher les détails de chaque convergence
                                for i, conv in enumerate(convergence_analysis["convergences"]):
                                    st.info(f"**Convergence {i+1}:** Altitude {conv['altitude']:.0f}m - Type: {conv['type']}")
                                    
                                    # Afficher plus de détails dans un expander
                                    with st.expander(f"Détails de la convergence {i+1}"):
                                        st.markdown(f"""
                                        - **Altitude:** {conv['altitude']:.0f}m
                                        - **Force:** {conv['strength']:.2f} (0-1)
                                        - **Type:** {conv['type']}
                                        - **Direction en dessous:** {conv['lower_direction']:.0f}°
                                        - **Direction au-dessus:** {conv['upper_direction']:.0f}°
                                        """)
                                        
                                        # Créer un schéma simple pour illustrer la convergence
                                        import matplotlib.pyplot as plt
                                        import numpy as np
                                        
                                        fig, ax = plt.subplots(figsize=(4, 3))
                                        
                                        # Dessiner les flèches représentant les vents
                                        arrow_length = 0.4
                                        
                                        # Flèche inférieure (vent en dessous)
                                        lower_dx = arrow_length * np.sin(np.radians(conv['lower_direction']))
                                        lower_dy = arrow_length * np.cos(np.radians(conv['lower_direction']))
                                        ax.arrow(0.5 - lower_dx, 0.25, lower_dx, lower_dy, 
                                                head_width=0.05, head_length=0.1, fc='blue', ec='blue', linewidth=2)
                                        
                                        # Flèche supérieure (vent au-dessus)
                                        upper_dx = arrow_length * np.sin(np.radians(conv['upper_direction']))
                                        upper_dy = arrow_length * np.cos(np.radians(conv['upper_direction']))
                                        ax.arrow(0.5 - upper_dx, 0.75, upper_dx, upper_dy, 
                                                head_width=0.05, head_length=0.1, fc='red', ec='red', linewidth=2)
                                        
                                        # Flèche verticale pour l'ascendance
                                        ax.arrow(0.5, 0.3, 0, 0.4, head_width=0.05, head_length=0.1, 
                                                fc='green', ec='green', linestyle='--', linewidth=1)
                                        
                                        # Texte explicatif
                                        ax.text(0.2, 0.2, f"Vent {conv['lower_direction']:.0f}°", color='blue')
                                        ax.text(0.2, 0.8, f"Vent {conv['upper_direction']:.0f}°", color='red')
                                        ax.text(0.6, 0.5, "Ascendance", color='green')
                                        
                                        # Configuration des axes
                                        ax.set_xlim(0, 1)
                                        ax.set_ylim(0, 1)
                                        ax.set_title(f"Convergence à {conv['altitude']:.0f}m")
                                        ax.axis('off')
                                        
                                        st.pyplot(fig)
                                
                                # Évaluation du potentiel d'ascendance dynamique
                                if convergence_analysis["potential_dynamic_lift"]:
                                    st.success("✅ **Bon potentiel d'ascendance dynamique** - Ces convergences peuvent générer des ascendances exploitables")
                                    st.markdown("""
                                    **Conseils pour exploiter les convergences:**
                                    1. Volez à l'altitude où la convergence est détectée
                                    2. Cherchez des indices visuels comme des alignements de nuages ou de poussière/débris
                                    3. Les convergences créent souvent des "lignes" d'ascendance - volez le long de ces lignes
                                    4. L'ascendance peut être moins turbulente qu'un thermique mais plus faible - soyez patient
                                    """)
                                else:
                                    st.info("ℹ️ **Potentiel d'ascendance dynamique limité** - Ces convergences sont probablement trop faibles pour générer des ascendances significatives")
                            else:
                                st.info("ℹ️ Aucune zone de convergence significative détectée dans le profil de vent")
                        else:
                            st.warning("Données de vent insuffisantes pour l'analyse des convergences")

                        # Nouvelle section pour l'analyse des brises de vallée
                        st.subheader("Analyse des brises de vallée")

                        with st.expander("ℹ️ Comprendre les brises de vallée", expanded=False):
                            st.markdown("""
                            **Les brises de vallée** sont des vents thermiques locaux qui se développent dans les régions montagneuses:
                            
                            - **Brise montante (anabatique)**: Pendant la journée, l'air réchauffé remonte les pentes et vallées
                            - **Brise descendante (catabatique)**: La nuit, l'air refroidi descend les pentes et vallées
                            - **Cycle diurne**: Ces brises suivent un cycle quotidien prévisible
                            
                            Les brises de vallée sont particulièrement importantes pour le vol en région montagneuse,
                            car elles peuvent faciliter les décollages, créer des ascendances dynamiques le long des pentes,
                            et influencer la direction et la force du vent sur site.
                            """)

                        # Interface pour les paramètres de vallée
                        col1, col2 = st.columns(2)

                        with col1:
                            # Récupérer l'heure actuelle comme valeur par défaut
                            current_hour = datetime.now().hour
                            hour = st.slider("Heure de la journée", 0, 23, current_hour, 
                                            help="L'heure influence fortement le régime des brises")
                            
                            valley_depth = st.slider("Profondeur de la vallée (m)", 
                                                    100, 2000, 800, 100,
                                                    help="Distance verticale entre le fond de vallée et les crêtes")

                        with col2:
                            valley_width = st.slider("Largeur de la vallée (m)", 
                                                    200, 5000, 1000, 100,
                                                    help="Distance horizontale entre les versants opposés")
                            
                            valley_orientation = st.slider("Orientation de la vallée (°)", 
                                                        0, 359, 180, 
                                                        help="Direction dans laquelle la vallée monte (0=N, 90=E, etc.)")

                        # Analyse des brises de vallée
                        valley_breeze_analysis = analyze_valley_breeze(
                            hour, 
                            analysis.ground_altitude,
                            valley_depth, 
                            valley_width, 
                            valley_orientation
                        )

                        # Afficher les résultats
                        st.subheader(f"Régime de brise: {valley_breeze_analysis['regime'].capitalize()}")

                        # Ajouter une visualisation simple
                        import matplotlib.pyplot as plt
                        import numpy as np

                        fig, ax = plt.subplots(figsize=(6, 4))

                        # Dessiner le profil de la vallée
                        x = np.linspace(0, 10, 100)
                        left_slope = 5 + valley_depth/500 * np.exp(-((x-2)**2)/1.5)
                        right_slope = 5 + valley_depth/500 * np.exp(-((x-8)**2)/1.5)
                        valley_floor = np.minimum(left_slope, right_slope)

                        ax.fill_between(x, 0, left_slope, color='brown', alpha=0.3)
                        ax.fill_between(x, 0, right_slope, color='brown', alpha=0.3)

                        # Dessiner les flèches de brise
                        if valley_breeze_analysis['regime'] == 'montante':
                            # Flèches montantes sur les pentes
                            for i in range(2, 9, 2):
                                slope_height = left_slope[i*10] if i < 5 else right_slope[i*10]
                                arrow_size = min(1.0, valley_breeze_analysis['intensity_kmh'] / 15)
                                ax.arrow(i, slope_height/3, 0, arrow_size, 
                                        head_width=0.2, head_length=0.3, fc='red', ec='red')
                            
                            # Flèche centrale
                            ax.arrow(5, 1, 0, 1.5, head_width=0.3, head_length=0.5, fc='red', ec='red')
                        else:
                            # Flèches descendantes sur les pentes
                            for i in range(2, 9, 2):
                                slope_height = left_slope[i*10] if i < 5 else right_slope[i*10]
                                arrow_size = min(1.0, valley_breeze_analysis['intensity_kmh'] / 15)
                                ax.arrow(i, slope_height*2/3, 0, -arrow_size, 
                                        head_width=0.2, head_length=0.3, fc='blue', ec='blue')
                            
                            # Flèche centrale
                            ax.arrow(5, 2.5, 0, -1.5, head_width=0.3, head_length=0.5, fc='blue', ec='blue')

                        # Configuration des axes
                        ax.set_xlim(0, 10)
                        ax.set_ylim(0, max(np.max(left_slope), np.max(right_slope))*1.2)
                        ax.set_title(f"Brise {valley_breeze_analysis['regime']} - {valley_breeze_analysis['phase']}")
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['bottom'].set_visible(False)
                        ax.spines['left'].set_visible(False)

                        st.pyplot(fig)

                        # Afficher les détails de la brise
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Intensité de la brise", f"{valley_breeze_analysis['intensity_kmh']:.1f} km/h")
                            st.metric("Direction", f"{valley_breeze_analysis['direction']}°")

                        with col2:
                            st.metric("Phase actuelle", valley_breeze_analysis['phase'].capitalize())
                            st.metric("Pic d'intensité à", f"{valley_breeze_analysis['peak_hour']}h00")

                        # Conseils pour le vol basés sur le régime de brise
                        if valley_breeze_analysis['regime'] == 'montante':
                            st.success("""
                            **Conseils pour la brise montante:**
                            - Privilégiez les faces orientées au soleil pour trouver les meilleures ascendances
                            - Le milieu de journée offre généralement les brises montantes les plus fortes
                            - Combinez la brise de vallée avec les thermiques pour optimiser votre gain d'altitude
                            - Attention aux changements de vent lors du passage d'une vallée à l'autre
                            """)
                        else:
                            st.info("""
                            **Conseils pour la brise descendante:**
                            - Évitez de voler près des pentes qui peuvent générer des rabattants
                            - Les conditions sont généralement moins favorables au vol thermique
                            - Le vol du matin ou du soir peut nécessiter plus de prudence
                            - Si vous volez, restez à distance du relief et prévoyez des marges de sécurité
                            """)

                        # Nouvelle section pour l'interpolation des données manquantes
                        st.subheader("Analyse de la qualité des données")

                        with st.expander("ℹ️ Comprendre l'interpolation des données", expanded=False):
                            st.markdown("""
                            **L'interpolation des données** permet de compléter les informations manquantes dans le profil vertical:
                            
                            - **Données manquantes**: Niveaux où certaines mesures ne sont pas disponibles
                            - **Méthodes d'interpolation**: Techniques mathématiques pour estimer les valeurs absentes
                            - **Qualité de l'analyse**: L'interpolation améliore la fiabilité de l'analyse globale
                            
                            Une bonne interpolation est particulièrement importante pour les analyses qui dépendent
                            de profils verticaux complets, comme le calcul du plafond thermique ou de la stabilité.
                            """)

                        # Vérifier si des données sont manquantes
                        temperature_nan = np.isnan(analyzer.temperatures).sum() if hasattr(analyzer, 'temperatures') else 0
                        dew_point_nan = np.isnan(analyzer.dew_points).sum() if hasattr(analyzer, 'dew_points') else 0
                        wind_speed_nan = np.isnan(analyzer.wind_speeds).sum() if hasattr(analyzer, 'wind_speeds') else 0
                        wind_dir_nan = np.isnan(analyzer.wind_directions).sum() if hasattr(analyzer, 'wind_directions') else 0

                        total_points = len(analyzer.altitudes)
                        missing_data = temperature_nan > 0 or dew_point_nan > 0 or wind_speed_nan > 0 or wind_dir_nan > 0

                        if missing_data:
                            st.warning(f"⚠️ Données incomplètes détectées dans le profil vertical")
                            
                            # Afficher un résumé des données manquantes
                            missing_data_df = pd.DataFrame({
                                "Type de données": ["Température", "Point de rosée", "Vitesse du vent", "Direction du vent"],
                                "Points manquants": [temperature_nan, dew_point_nan, wind_speed_nan, wind_dir_nan],
                                "Pourcentage": [
                                    f"{temperature_nan/total_points*100:.1f}%", 
                                    f"{dew_point_nan/total_points*100:.1f}%", 
                                    f"{wind_speed_nan/total_points*100:.1f}%", 
                                    f"{wind_dir_nan/total_points*100:.1f}%"
                                ]
                            })
                            
                            st.dataframe(missing_data_df)
                            
                            # Bouton pour interpoler les données manquantes
                            if st.button("Interpoler les données manquantes"):
                                with st.spinner("Interpolation des données..."):
                                    # Créer des copies des données pour l'interpolation
                                    interpolated_temps = analyzer.temperatures.copy()
                                    interpolated_dew_points = analyzer.dew_points.copy()
                                    interpolated_wind_speeds = analyzer.wind_speeds.copy()
                                    interpolated_wind_directions = analyzer.wind_directions.copy()
                                    
                                    # Appliquer l'interpolation
                                    interpolated_temps, interpolated_dew_points, interpolated_wind_speeds, interpolated_wind_directions = interpolate_missing_data(
                                        analyzer.altitudes,
                                        interpolated_temps,
                                        interpolated_dew_points,
                                        interpolated_wind_speeds,
                                        interpolated_wind_directions
                                    )
                                    
                                    # Compter les NaN restants après interpolation
                                    temp_nan_after = np.isnan(interpolated_temps).sum()
                                    dew_nan_after = np.isnan(interpolated_dew_points).sum()
                                    wspd_nan_after = np.isnan(interpolated_wind_speeds).sum()
                                    wdir_nan_after = np.isnan(interpolated_wind_directions).sum()
                                    
                                    # Afficher les résultats
                                    st.success("✅ Interpolation terminée")
                                    
                                    results_df = pd.DataFrame({
                                        "Type de données": ["Température", "Point de rosée", "Vitesse du vent", "Direction du vent"],
                                        "Points manquants avant": [temperature_nan, dew_point_nan, wind_speed_nan, wind_dir_nan],
                                        "Points manquants après": [temp_nan_after, dew_nan_after, wspd_nan_after, wdir_nan_after],
                                        "Points récupérés": [
                                            temperature_nan - temp_nan_after,
                                            dew_point_nan - dew_nan_after,
                                            wind_speed_nan - wspd_nan_after,
                                            wind_dir_nan - wdir_nan_after
                                        ]
                                    })
                                    
                                    st.dataframe(results_df)
                                    
                                    # Visualiser l'effet de l'interpolation
                                    if temperature_nan > 0 or dew_point_nan > 0:
                                        fig, ax = plt.subplots(figsize=(8, 6))
                                        
                                        # Tracer les données originales avec marquage des points manquants
                                        ax.plot(analyzer.temperatures, analyzer.altitudes, 'r-', label='Température originale')
                                        ax.plot(analyzer.dew_points, analyzer.altitudes, 'b-', label='Point de rosée original')
                                        
                                        # Marquer les points où des données étaient manquantes
                                        temp_nan_mask = np.isnan(analyzer.temperatures)
                                        dew_nan_mask = np.isnan(analyzer.dew_points)
                                        
                                        # Tracer les données interpolées
                                        ax.plot(interpolated_temps, analyzer.altitudes, 'r--', label='Température interpolée')
                                        ax.plot(interpolated_dew_points, analyzer.altitudes, 'b--', label='Point de rosée interpolé')
                                        
                                        # Marquer les points interpolés
                                        ax.scatter(interpolated_temps[temp_nan_mask], analyzer.altitudes[temp_nan_mask], 
                                                c='red', marker='o', s=30, label='Température interpolée')
                                        ax.scatter(interpolated_dew_points[dew_nan_mask], analyzer.altitudes[dew_nan_mask], 
                                                c='blue', marker='o', s=30, label='Point de rosée interpolé')
                                        
                                        ax.set_xlabel('Température (°C)')
                                        ax.set_ylabel('Altitude (m)')
                                        ax.set_title('Effet de l\'interpolation sur les données de température')
                                        ax.legend()
                                        ax.grid(True)
                                        
                                        st.pyplot(fig)
                                        
                                    # Option pour utiliser les données interpolées
                                    st.info("""
                                    Note: Les données interpolées sont affichées à titre informatif uniquement.
                                    Pour utiliser ces données dans l'analyse complète, vous devriez relancer l'analyse
                                    avec l'option d'interpolation activée dans les paramètres avancés.
                                    """)
                        else:
                            st.success("✅ Profil vertical complet - Aucune donnée manquante détectée")
                            
                            # Afficher un résumé de la qualité des données
                            st.info(f"""
                            **Résumé de la qualité des données:**
                            - Nombre total de niveaux: {total_points}
                            - Profil de température: Complet
                            - Profil du point de rosée: Complet
                            - Profil de vent: Complet
                            
                            L'analyse devrait être fiable sans nécessiter d'interpolation supplémentaire.
                            """)

# Point d'entrée principal
if __name__ == "__main__":
    main()