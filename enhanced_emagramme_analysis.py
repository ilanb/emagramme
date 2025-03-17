#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyse aérologique améliorée pour l'application d'émagramme parapente
Contient des fonctions avancées d'analyse météorologique pour le vol en parapente
"""

import numpy as np
import pandas as pd
import math
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_analysis_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedEmagrammeAnalysis')

# Classe pour gérer l'analyse améliorée
class EnhancedEmagrammeAgent:
    """
    Agent d'analyse améliorée qui utilise un LLM pour analyser et commenter un émagramme
    Version améliorée avec plus de fonctionnalités spécialisées pour les parapentistes
    """
    
    def __init__(self, openai_api_key=None):
        """
        Initialise l'agent avec une clé API pour le LLM
        
        Args:
            openai_api_key: Clé API pour OpenAI (ou autre service)
        """
        self.api_key = openai_api_key
        self.has_openai = False
        
        # Tenter d'importer OpenAI si une clé est fournie
        if openai_api_key:
            try:
                import openai
                self.openai = openai
                self.openai.api_key = openai_api_key
                self.has_openai = True
                logger.info("Module OpenAI initialisé avec succès")
            except ImportError:
                logger.warning("Module OpenAI non disponible. Installez-le avec: pip install openai")
    
    def analyze_conditions(self, analysis):
        """
        Utilise le LLM pour générer une analyse détaillée des conditions de vol
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Description détaillée en langage naturel
        """
        try:
            if not self.has_openai:
                # Si pas d'API OpenAI, utiliser l'analyse interne
                return analyze_emagramme_for_pilot(analysis)

            # Construire le prompt pour l'API
            prompt = self._build_prompt(analysis)
            
            # Appeler l'API
            response = self.openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Ou un autre modèle adapté
                messages=[
                    {"role": "system", "content": """Tu es un expert en parapente et météorologie qui analyse 
                    les émagrammes pour fournir des conseils de vol précis et utiles.
                    Utilise un ton pédagogique mais direct, et concentre-toi sur les informations 
                    pratiques pour les pilotes.
                    N'hésite pas à mentionner clairement si le vol est impossible ou dangereux."""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API LLM: {e}")
            # En cas d'erreur, utiliser l'analyse interne
            return analyze_emagramme_for_pilot(analysis)
    
    def _build_prompt(self, analysis):
        """
        Construit le prompt pour le LLM basé sur l'analyse de l'émagramme
        """
        # Déterminer si le vol est impossible
        vol_impossible = False
        raisons_impossibilite = []
        
        # Vérifier les conditions qui rendent le vol impossible
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            vol_impossible = True
            raisons_impossibilite.append(f"{analysis.precipitation_description}")
        
        # Vérifier si le vent est trop fort en utilisant la description textuelle
        if "fort au sol" in analysis.wind_conditions.lower() or "critique" in analysis.wind_conditions.lower():
            vol_impossible = True
            raisons_impossibilite.append("Vent trop fort")
        
        # Construction du prompt de base
        prompt = f"""Analyse cet émagramme pour le vol en parapente:

    Site d'altitude: {analysis.ground_altitude}m
    Température au sol: {analysis.ground_temperature:.1f}°C
    Point de rosée au sol: {analysis.ground_dew_point:.1f}°C
    Plafond thermique: {analysis.thermal_ceiling:.0f}m
    Gradient thermique: {analysis.thermal_gradient:.1f}°C/1000m
    Force des thermiques: {analysis.thermal_strength}
    Stabilité de l'atmosphère: {analysis.stability}
    """
        
        # Ajout des précipitations si présentes
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            prompt += f"Précipitations: {analysis.precipitation_description}\n"
        
        # Ajout d'autres informations
        if analysis.thermal_type == "Cumulus":
            prompt += f"Base des nuages: {analysis.cloud_base:.0f}m\n"
            prompt += f"Sommet des nuages: {analysis.cloud_top:.0f}m\n"
        else:
            prompt += "Thermiques bleus (pas de condensation)\n"
        
        if analysis.inversion_layers:
            prompt += "\nCouches d'inversion:\n"
            for i, (base, top) in enumerate(analysis.inversion_layers):
                prompt += f"- De {base:.0f}m à {top:.0f}m\n"
                
        prompt += f"\nConditions générales: {analysis.flight_conditions}\n"
        prompt += f"\nConditions de vent: {analysis.wind_conditions}\n"
        
        if analysis.hazards:
            prompt += "\nRisques identifiés:\n"
            for hazard in analysis.hazards:
                prompt += f"- {hazard}\n"
        
        # Instructions spécifiques selon que le vol est possible ou non
        if vol_impossible:
            raisons = ", ".join(raisons_impossibilite)
            prompt += f"""
    IMPORTANT: Le vol est ABSOLUMENT IMPOSSIBLE en raison de: {raisons}.

    Ta réponse doit:
    1. Indiquer clairement que le vol est IMPOSSIBLE et expliquer pourquoi
    2. Détailler les dangers spécifiques de ces conditions
    3. Ne PAS suggérer d'heures optimales de vol
    4. Ne PAS suggérer de stratégies pour exploiter les thermiques
    5. Ne PAS mentionner de niveau de difficulté
    6. Ne PAS suggérer d'équipement (sauf pour indiquer qu'aucun équipement n'est recommandé car le vol est impossible)
    7. Recommander explicitement de ne pas voler et d'attendre de meilleures conditions

    Format ta réponse comme un avertissement clair et direct.
    """
        else:
            prompt += """
    Fournis une analyse détaillée et structurée avec:
    1. Une évaluation globale des conditions pour le vol en parapente
    2. Les heures optimales pour voler et les phases de la journée à privilégier
    3. Des stratégies pour exploiter au mieux les thermiques dans ces conditions
    4. Des mises en garde sur les dangers potentiels
    5. Des suggestions concrètes adaptées au type de vol possible (local, vol thermique, cross, etc.)
    6. Une évaluation du niveau de difficulté recommandé (débutant, intermédiaire, avancé)
    7. Des conseils sur le matériel adapté (sellette, voile, etc.)

    Ton analyse doit être:
    - Concise et pratique
    - Directement utilisable par un pilote
    - Avec des sections clairement définies
    - Sans rappeler les données brutes que je t'ai fournies
    - Orientée vers les actions et décisions du pilote
    """
        return prompt


# 1. Analyse de l'influence du relief
def analyze_terrain_effect(lat, lon, site_altitude, slope_angle, aspect):
    """
    Analyse l'influence du relief sur les thermiques
    
    Args:
        lat, lon: Coordonnées du site
        site_altitude: Altitude du site en mètres
        slope_angle: Angle de la pente en degrés
        aspect: Orientation de la pente en degrés (0=N, 90=E, etc.)
    
    Returns:
        Dict contenant les facteurs d'influence du relief
    """
    # Calculer l'exposition solaire en fonction de l'heure, de la saison et de l'orientation
    current_date = datetime.now()
    day_of_year = current_date.timetuple().tm_yday
    hour = current_date.hour + current_date.minute / 60
    
    # Angle d'élévation solaire approximatif
    declination = 23.45 * math.sin(math.radians((360/365) * (day_of_year - 81)))
    elevation = 90 - lat + declination
    
    # Azimut solaire approximatif (simplification)
    azimuth = 15 * (hour - 12)  # Mouvement de 15° par heure par rapport au sud
    if hour < 12:
        azimuth = 180 - azimuth
    else:
        azimuth = 180 + azimuth
    azimuth = azimuth % 360
    
    # Calculer l'angle d'incidence solaire sur la pente
    incidence_angle = math.cos(math.radians(slope_angle)) * math.sin(math.radians(elevation)) + \
                      math.sin(math.radians(slope_angle)) * math.cos(math.radians(elevation)) * \
                      math.cos(math.radians(azimuth - aspect))
    
    # Normaliser le facteur d'insolation (0-1)
    insolation_factor = max(0, min(1, incidence_angle))
    
    # Effet venturi en fonction de la pente
    venturi_factor = 1 + (slope_angle / 45) * 0.5  # Amplification jusqu'à 50% pour une pente à 45°
    
    # Effet de compression orographique
    compression_factor = 1 + (site_altitude / 3000) * 0.3  # Amplification jusqu'à 30% à 3000m
    
    return {
        "insolation_factor": insolation_factor,
        "venturi_factor": venturi_factor,
        "compression_factor": compression_factor,
        "thermal_multiplier": insolation_factor * venturi_factor * compression_factor
    }

# 2. Détection des zones de convergence
def detect_convergence_zones(wind_directions, wind_speeds, altitudes):
    """
    Détecte les zones de convergence potentielles basées sur les changements de direction du vent
    
    Args:
        wind_directions: Array des directions du vent à différentes altitudes
        wind_speeds: Array des vitesses du vent
        altitudes: Array des altitudes correspondantes
    
    Returns:
        Dict contenant les informations sur les convergences détectées
    """
    convergences = []
    
    # Vérifier si les entrées sont des arrays numpy, sinon les convertir
    if not isinstance(wind_directions, np.ndarray):
        wind_directions = np.array(wind_directions)
    if not isinstance(wind_speeds, np.ndarray):
        wind_speeds = np.array(wind_speeds)
    if not isinstance(altitudes, np.ndarray):
        altitudes = np.array(altitudes)
        
    # Calculer les différences de direction
    for i in range(1, len(wind_directions)):
        if np.isnan(wind_directions[i-1]) or np.isnan(wind_directions[i]):
            continue
            
        # Calculer la différence angulaire (tenant compte du cercle)
        diff = min(abs(wind_directions[i] - wind_directions[i-1]), 
                  360 - abs(wind_directions[i] - wind_directions[i-1]))
        
        # Si différence importante et vitesses significatives
        if diff > 30 and wind_speeds[i] > 5 and wind_speeds[i-1] > 5:
            # Calculer l'altitude de la convergence (moyenne des deux niveaux)
            convergence_altitude = (altitudes[i] + altitudes[i-1]) / 2
            
            # Déterminer la force de la convergence
            convergence_strength = diff * min(wind_speeds[i], wind_speeds[i-1]) / 100
            
            # Déterminer le type de convergence
            if abs(wind_directions[i] - wind_directions[i-1]) > 120:
                convergence_type = "Forte"
            else:
                convergence_type = "Modérée"
                
            convergences.append({
                "altitude": convergence_altitude,
                "strength": convergence_strength,
                "type": convergence_type,
                "lower_direction": wind_directions[i-1],
                "upper_direction": wind_directions[i]
            })
    
    return {
        "has_convergence": len(convergences) > 0,
        "convergences": convergences,
        "potential_dynamic_lift": any(c["strength"] > 0.5 for c in convergences)
    }

# 3. Calcul du déclenchement thermique adaptatif
def calculate_adaptive_trigger_delta(surface_type, ground_temperature, wind_speed, cloud_cover):
    """
    Calcule un déclenchement thermique adaptatif selon le type de terrain et conditions actuelles
    
    Args:
        surface_type: Type de surface ("urban", "forest", "water", "rock", "grass", etc.)
        ground_temperature: Température au sol en °C
        wind_speed: Vitesse du vent au sol en km/h
        cloud_cover: Couverture nuageuse en %
    
    Returns:
        Valeur de déclenchement thermique adapté en °C
    """
    # Facteurs de base par type de surface
    base_factors = {
        "urban": 2.0,      # Zones urbaines: chaleur stockée dans le béton/asphalte
        "dark_rock": 1.8,  # Roches sombres: forte absorption thermique
        "light_rock": 2.5, # Roches claires: absorption modérée
        "dry_soil": 2.2,   # Sol sec: bon déclenchement
        "grass": 2.8,      # Prairie: déclenchement moyen
        "forest": 3.2,     # Forêt: amortissement thermique
        "water": 5.0,      # Eau: très mauvais déclenchement
        "sand": 2.3,       # Sable: bon déclenchement mais déperdition rapide
        "snow": 4.0        # Neige: très mauvais déclenchement (sauf effet albédo)
    }
    
    # Valeur de base selon le type de surface
    delta_base = base_factors.get(surface_type, 3.0)  # Valeur par défaut si type non reconnu
    
    # Ajustements selon les conditions météo
    
    # 1. Température: plus il fait chaud, plus le déclenchement est facile
    temp_factor = max(0.7, min(1.3, 1 - (ground_temperature - 15) / 50))
    
    # 2. Vent: vent modéré facilite le déclenchement, vent fort le perturbe
    if wind_speed < 5:
        wind_factor = 1.2  # Vent trop faible: stabilisation
    elif wind_speed < 15:
        wind_factor = 0.9  # Vent idéal: amélioration
    else:
        wind_factor = 0.9 + (wind_speed - 15) / 50  # Vent fort: détérioration progressive
    
    # 3. Couverture nuageuse: affecte le rayonnement solaire
    cloud_factor = 1 + (cloud_cover / 100) * 0.5
    
    # Calculer le delta final
    delta_adjusted = delta_base * temp_factor * wind_factor * cloud_factor
    
    # Borner la valeur dans une plage raisonnable
    return max(1.5, min(5.0, delta_adjusted))

# 4. Identification des types de nuages
def identify_cloud_types(low_cover, mid_cover, high_cover, thermal_ceiling, 
                       ground_temperature, dew_point, precipitation_type):
    """
    Identifie les types de nuages probables selon les conditions
    
    Args:
        low_cover, mid_cover, high_cover: Couverture nuageuse en %
        thermal_ceiling: Plafond thermique en mètres
        ground_temperature, dew_point: Températures au sol en °C
        precipitation_type: Type de précipitation (0 = aucune)
    
    Returns:
        Dict contenant les types de nuages identifiés et leurs impacts
    """
    cloud_types = []
    
    # Gestion des valeurs None
    low_cover = 0 if low_cover is None else low_cover
    mid_cover = 0 if mid_cover is None else mid_cover
    high_cover = 0 if high_cover is None else high_cover
    precipitation_type = 0 if precipitation_type is None else precipitation_type
    
    # Calcul de la stabilité basique
    stability_index = ground_temperature - dew_point
    
    # Nuages bas
    if low_cover > 10:
        if thermal_ceiling < 1500 and stability_index < 5:
            cloud_types.append({
                "type": "Stratus",
                "altitude": "basse",
                "coverage": low_cover,
                "impact": "Stabilise l'atmosphère, limite le développement thermique",
                "severity": "high" if low_cover > 60 else "medium"
            })
        elif precipitation_type > 0:
            cloud_types.append({
                "type": "Nimbostratus",
                "altitude": "basse-moyenne",
                "coverage": low_cover,
                "impact": "Associé à des précipitations continues, vol impossible",
                "severity": "high"
            })
        else:
            cloud_types.append({
                "type": "Cumulus",
                "altitude": "basse",
                "coverage": low_cover,
                "impact": "Marqueurs thermiques, généralement favorables au vol",
                "severity": "low" if low_cover < 60 else "medium"
            })
    
    # Nuages moyens
    if mid_cover > 10:
        if precipitation_type > 0:
            cloud_types.append({
                "type": "Altostratus",
                "altitude": "moyenne",
                "coverage": mid_cover,
                "impact": "Réduit l'ensoleillement, affaiblit les thermiques",
                "severity": "medium" if mid_cover < 70 else "high"
            })
        elif stability_index < 3 and low_cover > 40:
            cloud_types.append({
                "type": "Stratocumulus",
                "altitude": "basse-moyenne",
                "coverage": mid_cover,
                "impact": "Peut indiquer une inversion, souvent associé à une stabilisation",
                "severity": "medium"
            })
        else:
            cloud_types.append({
                "type": "Altocumulus",
                "altitude": "moyenne",
                "coverage": mid_cover,
                "impact": "Peut indiquer des ascendances à altitudes moyennes, généralement peu impactant",
                "severity": "low"
            })
    
    # Nuages hauts
    if high_cover > 10:
        if high_cover > 70:
            cloud_types.append({
                "type": "Cirrostratus",
                "altitude": "haute",
                "coverage": high_cover,
                "impact": "Voile uniforme, réduit légèrement l'activité thermique",
                "severity": "low"
            })
        else:
            cloud_types.append({
                "type": "Cirrus",
                "altitude": "haute",
                "coverage": high_cover,
                "impact": "Impact négligeable sur les conditions de vol",
                "severity": "low"
            })
    
    # Analyse des risques d'orages
    thunderstorm_risk = False
    thunderstorm_proximity = "lointain"
    
    if low_cover > 30 and stability_index < 4 and ground_temperature > 25 and precipitation_type > 0:
        thunderstorm_risk = True
        thunderstorm_proximity = "proche" if low_cover > 60 else "lointain"
        cloud_types.append({
            "type": "Cumulonimbus",
            "altitude": "basse-haute",
            "coverage": low_cover,
            "impact": "Très dangereux, vol impossible à proximité, risque d'orages",
            "severity": "extreme"
        })
    
    return {
        "identified_types": cloud_types,
        "thunderstorm_risk": thunderstorm_risk,
        "thunderstorm_proximity": thunderstorm_proximity,
        "flight_impact": "high" if any(c["severity"] in ["high", "extreme"] for c in cloud_types) else 
                         "medium" if any(c["severity"] == "medium" for c in cloud_types) else "low"
    }

# 5. Analyse des brises de vallée
def analyze_valley_breeze(hour, site_altitude, valley_depth, valley_width, valley_orientation):
    """
    Analyse les brises de vallée en fonction de l'heure et de la topographie
    
    Args:
        hour: Heure de la journée (0-24)
        site_altitude: Altitude du site en mètres
        valley_depth: Profondeur de la vallée en mètres
        valley_width: Largeur de la vallée en mètres
        valley_orientation: Orientation de la vallée en degrés
    
    Returns:
        Dict contenant les prévisions de brise de vallée
    """
    # Déterminer le régime de brise
    regime = "montante" if 9 <= hour <= 18 else "descendante"
    
    # Intensité de base selon la période
    if regime == "montante":
        if hour < 11:
            phase = "établissement"
            base_intensity = (hour - 9) / 2 * 10  # 0-10 km/h entre 9h et 11h
        elif hour < 16:
            phase = "maximum"
            base_intensity = 10 + (hour - 11) / 5 * 5  # 10-15 km/h entre 11h et 16h
        else:
            phase = "affaiblissement"
            base_intensity = 15 - (hour - 16) / 2 * 10  # 15-5 km/h entre 16h et 18h
    else:  # descendante
        if hour < 3 or hour > 21:
            phase = "maximum"
            base_intensity = 10  # 10 km/h en pleine nuit
        elif hour < 9:
            phase = "affaiblissement"
            base_intensity = 10 - (hour - 3) / 6 * 10  # 10-0 km/h entre 3h et 9h
        else:  # 18-21
            phase = "établissement"
            base_intensity = (hour - 18) / 3 * 10  # 0-10 km/h entre 18h et 21h
    
    # Facteurs topographiques
    valley_factor = min(1.5, max(0.5, valley_depth / 1000))  # Plus profonde = plus forte
    width_factor = min(1.5, max(0.5, 1000 / valley_width))   # Plus étroite = plus forte
    
    # Intensité finale
    intensity = base_intensity * valley_factor * width_factor
    
    # Direction de la brise (dans l'axe de la vallée)
    if regime == "montante":
        # Brise montante: dans le sens de la vallée qui monte
        direction = valley_orientation
    else:
        # Brise descendante: sens opposé
        direction = (valley_orientation + 180) % 360
    
    return {
        "regime": regime,
        "phase": phase,
        "intensity_kmh": intensity,
        "direction": direction,
        "start_hour": 9 if regime == "montante" else 18,
        "peak_hour": 14 if regime == "montante" else 0,
        "end_hour": 18 if regime == "montante" else 9
    }

# 6. Interpolation des données manquantes
def interpolate_missing_data(altitudes, temperatures, dew_points, wind_speeds, wind_directions):
    """
    Interpole les données manquantes dans les profils verticaux
    
    Args:
        altitudes, temperatures, dew_points, wind_speeds, wind_directions: Arrays numpy
        
    Returns:
        Arrays nettoyés avec données interpolées
    """
    # Convertir en arrays numpy si nécessaire
    if not isinstance(altitudes, np.ndarray):
        altitudes = np.array(altitudes)
    if not isinstance(temperatures, np.ndarray):
        temperatures = np.array(temperatures)
    if not isinstance(dew_points, np.ndarray):
        dew_points = np.array(dew_points)
    if not isinstance(wind_speeds, np.ndarray):
        wind_speeds = np.array(wind_speeds)
    if not isinstance(wind_directions, np.ndarray):
        wind_directions = np.array(wind_directions)
    
    # Créer des masques pour les valeurs NaN dans chaque array
    temp_mask = np.isnan(temperatures)
    dew_mask = np.isnan(dew_points)
    wspd_mask = np.isnan(wind_speeds)
    wdir_mask = np.isnan(wind_directions)
    
    # Vérifier s'il y a des valeurs manquantes
    if np.any(temp_mask) and not np.all(temp_mask):
        # Interpolation linéaire pour les températures
        valid_indices = ~temp_mask
        temperatures[temp_mask] = np.interp(
            altitudes[temp_mask], 
            altitudes[valid_indices], 
            temperatures[valid_indices]
        )
    
    # Interpolation linéaire pour les points de rosée
    if np.any(dew_mask) and not np.all(dew_mask):
        valid_indices = ~dew_mask
        dew_points[dew_mask] = np.interp(
            altitudes[dew_mask], 
            altitudes[valid_indices], 
            dew_points[valid_indices]
        )
    
    # Interpolation linéaire pour les vitesses de vent
    if np.any(wspd_mask) and not np.all(wspd_mask):
        valid_indices = ~wspd_mask
        wind_speeds[wspd_mask] = np.interp(
            altitudes[wspd_mask], 
            altitudes[valid_indices], 
            wind_speeds[valid_indices]
        )
    
    # Interpolation circulaire pour les directions de vent
    if np.any(wdir_mask) and not np.all(wdir_mask):
        valid_indices = ~wdir_mask
        # Convertir en coordonnées cartésiennes pour l'interpolation
        u = np.cos(np.radians(wind_directions[valid_indices]))
        v = np.sin(np.radians(wind_directions[valid_indices]))
        
        # Interpoler u et v
        u_interp = np.interp(altitudes[wdir_mask], altitudes[valid_indices], u)
        v_interp = np.interp(altitudes[wdir_mask], altitudes[valid_indices], v)
        
        # Reconvertir en degrés
        wind_directions[wdir_mask] = (np.degrees(np.arctan2(v_interp, u_interp)) + 360) % 360
    
    # Assurer la cohérence physique (point de rosée <= température)
    dew_points = np.minimum(temperatures, dew_points)
    
    return temperatures, dew_points, wind_speeds, wind_directions

# 7. Calcul avancé de la stabilité atmosphérique
def calculate_advanced_stability(temperatures, dew_points, altitudes, pressure_levels):
    """
    Calcule des indices de stabilité atmosphérique avancés
    
    Args:
        temperatures, dew_points: Arrays des températures et points de rosée (°C)
        altitudes: Array des altitudes (m)
        pressure_levels: Array des niveaux de pression (hPa)
    
    Returns:
        Dict contenant divers indices de stabilité
    """
    # Convertir en arrays numpy si nécessaire
    if not isinstance(temperatures, np.ndarray):
        temperatures = np.array(temperatures)
    if not isinstance(dew_points, np.ndarray):
        dew_points = np.array(dew_points)
    if not isinstance(altitudes, np.ndarray):
        altitudes = np.array(altitudes)
    if not isinstance(pressure_levels, np.ndarray):
        pressure_levels = np.array(pressure_levels)
    
    # Vérifier que les données sont suffisantes
    if len(temperatures) < 3 or len(dew_points) < 3:
        return {"stability_valid": False, "message": "Données insuffisantes pour l'analyse de stabilité"}
    
    # Calculer le gradient moyen sur toute la colonne d'air
    height_diff = altitudes[-1] - altitudes[0]
    temp_diff = temperatures[0] - temperatures[-1]
    overall_lapse_rate = (temp_diff / height_diff) * 1000  # °C/km
    
    # Calculer le Lifted Index simplifié (différence entre température observée et adiabatique)
    # Une valeur négative indique de l'instabilité
    idx_500 = np.searchsorted(pressure_levels, 500)
    if idx_500 >= len(temperatures):
        idx_500 = len(temperatures) - 1
        
    # Estimation simplifiée du Lifted Index
    surface_temp = temperatures[0]
    surface_pressure = pressure_levels[0]
    dry_adiabatic_temp = surface_temp - (6.5 * (altitudes[idx_500] - altitudes[0]) / 1000)
    lifted_index = temperatures[idx_500] - dry_adiabatic_temp
    
    # Calculer le CAPE simplifié
    # CAPE positif indique une instabilité
    cape = 0
    
    # Calculer l'indice K
    # K-index = (T_850 - T_500) + TD_850 - (T_700 - TD_700)
    # On utilise les indices les plus proches des niveaux standard
    idx_850 = np.searchsorted(pressure_levels, 850)
    idx_700 = np.searchsorted(pressure_levels, 700)
    
    if idx_850 >= len(temperatures) or idx_700 >= len(temperatures) or idx_500 >= len(temperatures):
        k_index = None
    else:
        k_index = (temperatures[idx_850] - temperatures[idx_500]) + \
                 dew_points[idx_850] - (temperatures[idx_700] - dew_points[idx_700])
    
    # Interpréter les indices
    if lifted_index is not None:
        if lifted_index > 2:
            stability = "Très stable"
        elif lifted_index > 0:
            stability = "Stable"
        elif lifted_index > -3:
            stability = "Légèrement instable"
        elif lifted_index > -6:
            stability = "Modérément instable"
        else:
            stability = "Très instable"
    else:
        stability = "Indéterminé"
        
    if k_index is not None:
        if k_index < 15:
            k_interpretation = "Air sec, peu de risque d'orage"
        elif k_index < 20:
            k_interpretation = "Quelques orages isolés possibles"
        elif k_index < 25:
            k_interpretation = "Orages épars possibles"
        elif k_index < 30:
            k_interpretation = "Nombreux orages possibles"
        else:
            k_interpretation = "Orages généralisés"
    else:
        k_interpretation = "Indéterminé"
    
    return {
        "stability_valid": True,
        "overall_lapse_rate": overall_lapse_rate,
        "lifted_index": lifted_index,
        "stability": stability,
        "k_index": k_index,
        "k_interpretation": k_interpretation,
        "estimated_cape": cape,
        "inversion_strength": "strong" if overall_lapse_rate < 4 else "moderate" if overall_lapse_rate < 6 else "weak",
        "thermal_quality": "poor" if overall_lapse_rate < 5 else "moderate" if overall_lapse_rate < 7 else "good"
    }

# 8. Analyse détaillée du profil de vent
def analyze_wind_profile(altitudes, wind_speeds, wind_directions, site_altitude, thermal_ceiling):
    """
    Analyse avancée du profil vertical de vent pour le vol en parapente
    
    Args:
        altitudes: Array des altitudes (m)
        wind_speeds: Array des vitesses de vent (km/h)
        wind_directions: Array des directions de vent (degrés)
        site_altitude: Altitude du site de décollage (m)
        thermal_ceiling: Altitude du plafond thermique (m)
    
    Returns:
        Dict contenant l'analyse détaillée du profil de vent
    """
    # Convertir en arrays numpy si nécessaire
    if not isinstance(altitudes, np.ndarray):
        altitudes = np.array(altitudes)
    if not isinstance(wind_speeds, np.ndarray):
        wind_speeds = np.array(wind_speeds)
    if not isinstance(wind_directions, np.ndarray):
        wind_directions = np.array(wind_directions)
    
    # Vérifier que les données sont suffisantes
    if len(wind_speeds) < 2 or len(wind_directions) < 2:
        return {"valid": False, "message": "Données insuffisantes pour l'analyse du vent"}
    
    # Définir la zone de vol (entre le site et le plafond thermique)
    flight_zone_mask = (altitudes >= site_altitude) & (altitudes <= thermal_ceiling)
    
    # S'assurer qu'il y a des données dans la zone de vol
    if not np.any(flight_zone_mask):
        return {"valid": False, "message": "Pas de données de vent dans la zone de vol"}
    
    # Extraire les données de la zone de vol
    flight_zone_alts = altitudes[flight_zone_mask]
    flight_zone_speeds = wind_speeds[flight_zone_mask]
    flight_zone_dirs = wind_directions[flight_zone_mask]
    
    if len(flight_zone_alts) < 2:
        return {"valid": False, "message": "Données insuffisantes pour l'analyse du vent dans la zone de vol"}
    
    # Calculer les gradients de vent (variation par 100m)
    wind_speed_gradient = np.zeros(len(flight_zone_alts) - 1)
    wind_dir_gradient = np.zeros(len(flight_zone_alts) - 1)
    
    for i in range(len(flight_zone_alts) - 1):
        alt_diff = flight_zone_alts[i+1] - flight_zone_alts[i]
        if alt_diff == 0:  # Éviter la division par zéro
            continue
            
        # Gradient de vitesse
        speed_diff = flight_zone_speeds[i+1] - flight_zone_speeds[i]
        wind_speed_gradient[i] = (speed_diff / alt_diff) * 100  # en km/h par 100m
        
        # Gradient de direction (tenir compte de la nature circulaire)
        dir_diff = flight_zone_dirs[i+1] - flight_zone_dirs[i]
        if dir_diff > 180:
            dir_diff -= 360
        elif dir_diff < -180:
            dir_diff += 360
        wind_dir_gradient[i] = (dir_diff / alt_diff) * 100  # en degrés par 100m
    
    # Statistiques de base
    avg_speed = np.nanmean(flight_zone_speeds)
    max_speed = np.nanmax(flight_zone_speeds)
    avg_speed_gradient = np.nanmean(wind_speed_gradient)
    avg_dir_gradient = np.nanmean(wind_dir_gradient)
    
    # Calculer les cisaillements significatifs
    significant_shear = []
    for i in range(len(wind_speed_gradient)):
        if abs(wind_speed_gradient[i]) > 2 or abs(wind_dir_gradient[i]) > 15:
            shear_altitude = (flight_zone_alts[i] + flight_zone_alts[i+1]) / 2
            significant_shear.append({
                "altitude": shear_altitude,
                "speed_gradient": wind_speed_gradient[i],
                "dir_gradient": wind_dir_gradient[i],
                "lower_alt": flight_zone_alts[i],
                "upper_alt": flight_zone_alts[i+1],
                "severity": "high" if (abs(wind_speed_gradient[i]) > 5 or abs(wind_dir_gradient[i]) > 30) else "medium"
            })
    
    # Analyse de la rotation du vent avec l'altitude (backing/veering)
    # Veering = rotation horaire avec l'altitude (typique dans l'hémisphère nord)
    # Backing = rotation antihoraire avec l'altitude (généralement associé à l'advection d'air froid)
    veering_count = 0
    backing_count = 0
    
    for i in range(len(wind_dir_gradient)):
        if wind_dir_gradient[i] > 0:
            veering_count += 1
        elif wind_dir_gradient[i] < 0:
            backing_count += 1
    
    rotation_type = "neutre"
    if veering_count > backing_count * 2:
        rotation_type = "veering_dominant"
    elif backing_count > veering_count * 2:
        rotation_type = "backing_dominant"
    elif veering_count > backing_count:
        rotation_type = "veering_slight"
    elif backing_count > veering_count:
        rotation_type = "backing_slight"
    
    # Turbulence estimée basée sur les cisaillements
    turbulence_factors = []
    
    # Facteur basé sur le gradient de vitesse
    speed_gradient_factor = min(1.0, max(0.0, abs(avg_speed_gradient) / 10))
    turbulence_factors.append(speed_gradient_factor * 0.4)  # Poids de 40%
    
    # Facteur basé sur le gradient de direction
    dir_gradient_factor = min(1.0, max(0.0, abs(avg_dir_gradient) / 50))
    turbulence_factors.append(dir_gradient_factor * 0.4)  # Poids de 40%
    
    # Facteur basé sur la vitesse du vent
    speed_factor = min(1.0, max(0.0, avg_speed / 30))
    turbulence_factors.append(speed_factor * 0.2)  # Poids de 20%
    
    # Score global de turbulence (0-1)
    turbulence_score = sum(turbulence_factors)
    
    # Interprétation de la turbulence
    if turbulence_score < 0.2:
        turbulence_level = "Très faible"
    elif turbulence_score < 0.4:
        turbulence_level = "Faible"
    elif turbulence_score < 0.6:
        turbulence_level = "Modérée"
    elif turbulence_score < 0.8:
        turbulence_level = "Forte"
    else:
        turbulence_level = "Très forte"
    
    # Déduire les phénomènes particuliers
    wind_phenomena = []
    
    # 1. Détection de jet de basse couche
    for i in range(len(flight_zone_speeds)):
        if i > 0 and i < len(flight_zone_speeds) - 1:
            if (flight_zone_speeds[i] > flight_zone_speeds[i-1] + 5) and \
               (flight_zone_speeds[i] > flight_zone_speeds[i+1] + 5) and \
               flight_zone_speeds[i] > 20:
                wind_phenomena.append({
                    "type": "jet_basse_couche",
                    "altitude": flight_zone_alts[i],
                    "speed": flight_zone_speeds[i],
                    "description": f"Jet de basse couche à {flight_zone_alts[i]:.0f}m, {flight_zone_speeds[i]:.1f} km/h",
                    "impact": "Turbulence notable et risque de rotors sous le jet"
                })
    
    # 2. Détection de couche de vent fort
    for i in range(len(flight_zone_speeds)):
        if flight_zone_speeds[i] > 30:
            # Déterminer l'étendue de la couche
            start_idx = i
            while start_idx > 0 and flight_zone_speeds[start_idx-1] > 25:
                start_idx -= 1
            
            end_idx = i
            while end_idx < len(flight_zone_speeds) - 1 and flight_zone_speeds[end_idx+1] > 25:
                end_idx += 1
                
            if end_idx > start_idx:  # S'assurer que la couche a une épaisseur
                wind_phenomena.append({
                    "type": "couche_vent_fort",
                    "base": flight_zone_alts[start_idx],
                    "top": flight_zone_alts[end_idx],
                    "avg_speed": np.mean(flight_zone_speeds[start_idx:end_idx+1]),
                    "description": f"Couche de vent fort de {flight_zone_alts[start_idx]:.0f}m à {flight_zone_alts[end_idx]:.0f}m",
                    "impact": "Dérive importante et risque de soaring impossible au plafond"
                })
                # Sauter à la fin de cette couche
                i = end_idx
    
    # 3. Détection d'une convergence de vent (rotation importante)
    for i in range(len(wind_dir_gradient)):
        if abs(wind_dir_gradient[i]) > 40:  # Rotation significative sur 100m
            wind_phenomena.append({
                "type": "convergence",
                "altitude": (flight_zone_alts[i] + flight_zone_alts[i+1]) / 2,
                "dir_change": wind_dir_gradient[i],
                "description": f"Convergence à {(flight_zone_alts[i] + flight_zone_alts[i+1]) / 2:.0f}m avec rotation de {abs(wind_dir_gradient[i]):.1f}°/100m",
                "impact": "Possible zone d'ascendance dynamique à exploiter"
            })
    
    return {
        "valid": True,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "avg_speed_gradient": avg_speed_gradient,
        "avg_dir_gradient": avg_dir_gradient,
        "turbulence_score": turbulence_score,
        "turbulence_level": turbulence_level,
        "rotation_type": rotation_type,
        "significant_shear": significant_shear,
        "wind_phenomena": wind_phenomena,
        "flight_safety": "unsafe" if max_speed > 35 or turbulence_score > 0.7 else 
                         "caution" if max_speed > 25 or turbulence_score > 0.5 else "safe"
    }

# 9. Recommandation des sites de décollage
def recommend_best_takeoff_sites(ffvl_sites, wind_direction, wind_speed, thermal_ceiling):
    """
    Recommande les meilleurs sites de décollage en fonction des conditions météo
    
    Args:
        ffvl_sites: Liste des sites FFVL avec leurs caractéristiques
        wind_direction: Direction du vent (degrés)
        wind_speed: Vitesse du vent (km/h)
        thermal_ceiling: Plafond thermique (m)
    
    Returns:
        Dict avec les sites recommandés et leurs scores
    """
    if not ffvl_sites:
        return {"sites": [], "best_score": 0, "wind_too_strong": wind_speed > 25, 
                "thermal_ceiling_adequate": thermal_ceiling > 1500}
                
    recommended_sites = []
    
    for site in ffvl_sites:
        site_score = 0
        orientation = site.get("orientation", "")
        difficulty = site.get("difficulty", "")
        altitude = site.get("altitude", 0)
        
        # Gérer les cas où altitude n'est pas un nombre
        try:
            altitude = float(altitude)
        except (ValueError, TypeError):
            altitude = 0
        
        # Convertir l'orientation textuelle en valeurs numériques
        orientation_ranges = []
        if "N" in orientation:
            orientation_ranges.append((337.5, 22.5))
        if "NE" in orientation:
            orientation_ranges.append((22.5, 67.5))
        if "E" in orientation:
            orientation_ranges.append((67.5, 112.5))
        if "SE" in orientation:
            orientation_ranges.append((112.5, 157.5))
        if "S" in orientation:
            orientation_ranges.append((157.5, 202.5))
        if "SO" in orientation or "SW" in orientation:
            orientation_ranges.append((202.5, 247.5))
        if "O" in orientation or "W" in orientation:
            orientation_ranges.append((247.5, 292.5))
        if "NO" in orientation or "NW" in orientation:
            orientation_ranges.append((292.5, 337.5))
        
        # Pas d'orientation spécifiée = toutes les orientations
        if not orientation_ranges:
            orientation_ranges = [(0, 360)]
        
        # Vérifier si l'orientation du site correspond au vent
        orientation_match = False
        for o_min, o_max in orientation_ranges:
            # Gérer la discontinuité à 360/0
            if o_min > o_max:  # ex: (337.5, 22.5) pour le Nord
                if wind_direction >= o_min or wind_direction <= o_max:
                    orientation_match = True
                    break
            else:
                if o_min <= wind_direction <= o_max:
                    orientation_match = True
                    break
        
        # Attribuer un score d'orientation (max 50 points)
        if orientation_match:
            # Score maximal quand le vent est perpendiculaire à la pente
            # Calculer la moyenne de la plage d'orientation
            for o_min, o_max in orientation_ranges:
                if o_min > o_max:  # Gestion du cas particulier autour de 0°
                    center = (o_min + o_max + 360) / 2
                    if center >= 360:
                        center -= 360
                else:
                    center = (o_min + o_max) / 2
                
                # Calculer la différence angulaire
                diff = abs(center - wind_direction)
                if diff > 180:
                    diff = 360 - diff
                
                # Score maximal à 90° (perpendiculaire)
                if diff > 45:
                    orientation_score = 50 * (90 - min(90, diff)) / 45
                else:
                    orientation_score = 50 * (diff / 45)
                
                site_score += orientation_score
        
        # Attribuer un score de vitesse de vent (max 30 points)
        if wind_speed < 5:
            wind_speed_score = 15  # Trop faible
        elif wind_speed < 15:
            wind_speed_score = 30  # Optimal
        elif wind_speed < 25:
            wind_speed_score = 20  # Acceptable mais fort
        else:
            wind_speed_score = 0   # Trop fort
        
        site_score += wind_speed_score
        
        # Attribuer un score d'altitude (max 20 points)
        # Privilégier les sites plus hauts quand le plafond est élevé
        # et les sites plus bas quand le plafond est bas
        alt_ratio = altitude / thermal_ceiling if thermal_ceiling > 0 else 0
        if alt_ratio < 0.2:
            altitude_score = 10  # Trop bas par rapport au plafond
        elif alt_ratio < 0.5:
            altitude_score = 20  # Optimal
        elif alt_ratio < 0.7:
            altitude_score = 15  # Un peu haut
        else:
            altitude_score = 5   # Trop haut (peu de marge)
        
        site_score += altitude_score
        
        # Normaliser le score sur 100
        normalized_score = min(100, site_score)
        
        # Ajouter un bonus pour les sites faciles par mauvaises conditions
        # ou les sites difficiles par excellentes conditions
        if thermal_ceiling < 1500:  # Conditions médiocres
            if "facile" in str(difficulty).lower() or "école" in str(difficulty).lower():
                normalized_score *= 1.2
            elif "difficile" in str(difficulty).lower():
                normalized_score *= 0.8
        elif thermal_ceiling > 2500:  # Bonnes conditions
            if "difficile" in str(difficulty).lower() or "confirmé" in str(difficulty).lower():
                normalized_score *= 1.1
        
        # Stocker les résultats
        recommended_sites.append({
            "site_name": site.get("name", "Site sans nom"),
            "score": min(100, normalized_score),  # Score sur 100
            "orientation_match": orientation_match,
            "altitude": altitude,
            "difficulty": difficulty,
            "distance": site.get("distance", 0),
            "latitude": site.get("latitude", 0),
            "longitude": site.get("longitude", 0),
            "comment": generate_site_comment(site, orientation_match, wind_speed, thermal_ceiling)
        })
    
    # Trier par score décroissant
    recommended_sites.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "sites": recommended_sites[:5] if recommended_sites else [],  # Top 5 des sites
        "best_score": recommended_sites[0]["score"] if recommended_sites else 0,
        "wind_too_strong": wind_speed > 25,
        "thermal_ceiling_adequate": thermal_ceiling > 1500
    }

def generate_site_comment(site, orientation_match, wind_speed, thermal_ceiling):
    """Génère un commentaire personnalisé pour un site"""
    comments = []
    
    # Commentaire sur l'orientation
    if orientation_match:
        comments.append("Orientation favorable")
    else:
        comments.append("Orientation non optimale")
    
    # Commentaire sur le vent
    if wind_speed < 5:
        comments.append("Vent faible, décollage en courant possible")
    elif wind_speed < 15:
        comments.append("Vent idéal pour le décollage")
    elif wind_speed < 25:
        comments.append("Vent assez fort, technique de décollage à adapter")
    else:
        comments.append("Vent trop fort, décollage déconseillé")
    
    # Commentaire sur le plafond par rapport à l'altitude du site
    try:
        site_altitude = float(site.get("altitude", 0))
    except (ValueError, TypeError):
        site_altitude = 0
        
    usable_height = thermal_ceiling - site_altitude
    
    if usable_height < 500:
        comments.append("Peu de gain d'altitude possible")
    elif usable_height < 1000:
        comments.append("Gain d'altitude modéré possible")
    elif usable_height < 2000:
        comments.append("Bon potentiel de gain d'altitude")
    else:
        comments.append("Excellent potentiel de gain d'altitude")
    
    return ". ".join(comments)

# 10. Prédiction de la durée de vol
def predict_flight_duration(thermal_ceiling, ground_altitude, thermal_strength, 
                         wind_speed, cloud_cover, thermal_gradient):
    """
    Prédit la durée de vol possible en fonction des conditions
    
    Args:
        thermal_ceiling: Plafond thermique en mètres
        ground_altitude: Altitude du site en mètres
        thermal_strength: Force des thermiques (chaîne)
        wind_speed: Vitesse du vent en km/h
        cloud_cover: Couverture nuageuse en %
        thermal_gradient: Gradient thermique en °C/1000m
    
    Returns:
        Dict avec prédictions de durée de vol et d'heure optimale
    """
    # Convertir la force des thermiques en valeur numérique
    thermal_strength_values = {
        "Faible": 1,
        "Modérée": 2,
        "Forte": 3,
        "Très Forte": 4
    }
    thermal_strength_value = thermal_strength_values.get(thermal_strength, 2)
    
    # Calculer le facteur de base basé sur la hauteur exploitable
    height_factor = min(3, max(0.5, (thermal_ceiling - ground_altitude) / 1000))
    
    # Facteur basé sur la force des thermiques
    thermal_factor = thermal_strength_value ** 0.5  # Racine carrée pour atténuer l'effet
    
    # Facteur basé sur le vent
    if wind_speed < 5:
        wind_factor = 0.8  # Vent trop faible = durée limitée
    elif wind_speed < 15:
        wind_factor = 1.0  # Vent optimal
    elif wind_speed < 25:
        wind_factor = 0.7  # Vent fort = vol plus difficile
    else:
        wind_factor = 0.4  # Vent très fort = vol très limité
    
    # Facteur basé sur la couverture nuageuse
    if cloud_cover is None:
        cloud_factor = 1.0  # Valeur par défaut
    else:
        cloud_factor = 1.0 - (cloud_cover / 100) * 0.5  # 50% d'impact max
    
    # Facteur basé sur le gradient thermique
    gradient_factor = min(1.2, max(0.8, thermal_gradient / 6.5))
    
    # Calculer la durée de base en heures
    base_duration = 2.0  # Durée moyenne de vol en conditions standard
    
    # Durée finale ajustée (en heures)
    adjusted_duration = base_duration * height_factor * thermal_factor * wind_factor * cloud_factor * gradient_factor
    
    # Calculer l'heure optimale de début de vol
    # Basé sur la force des thermiques et d'autres facteurs
    if thermal_strength_value >= 3:
        # Thermiques forts: voler en début ou fin de journée
        optimal_start_morning = 10.0  # 10h00
        optimal_start_afternoon = 15.0  # 15h00
    elif thermal_strength_value == 2:
        # Thermiques modérés: voler en milieu de journée
        optimal_start_morning = 11.0  # 11h00
        optimal_start_afternoon = 14.0  # 14h00
    else:
        # Thermiques faibles: voler au moment le plus chaud
        optimal_start_morning = 12.0  # 12h00
        optimal_start_afternoon = 13.0  # 13h00
    
    # Ajuster selon la couverture nuageuse
    if cloud_cover is not None and cloud_cover > 50:
        # Par temps nuageux, décaler vers le milieu de journée
        optimal_start_morning = min(12.0, optimal_start_morning + 1)
        optimal_start_afternoon = max(13.0, optimal_start_afternoon - 1)
    
    # Estimer le nombre de thermiques exploitables par heure
    thermals_per_hour = max(1, min(10, thermal_strength_value * 2 + thermal_gradient / 3))
    
    # Estimer le taux de montée moyen
    if thermal_strength == "Faible":
        climb_rate = 0.8  # m/s
    elif thermal_strength == "Modérée":
        climb_rate = 1.5  # m/s
    elif thermal_strength == "Forte":
        climb_rate = 2.3  # m/s
    else:  # Très Forte
        climb_rate = 3.0  # m/s
    
    # Ajuster en fonction du gradient
    climb_rate *= gradient_factor
    
    # Calculer le temps moyen pour atteindre le plafond
    height_gain = thermal_ceiling - ground_altitude
    time_to_ceiling_minutes = height_gain / (climb_rate * 60)
    
    return {
        "flight_duration_hours": round(adjusted_duration, 1),
        "flight_duration_minutes": round(adjusted_duration * 60, 0),
        "optimal_start_morning": optimal_start_morning,
        "optimal_start_afternoon": optimal_start_afternoon,
        "thermals_per_hour": round(thermals_per_hour, 1),
        "avg_climb_rate": round(climb_rate, 1),
        "time_to_ceiling_minutes": round(time_to_ceiling_minutes, 0),
        "xc_potential": "High" if adjusted_duration > 3 and height_factor > 2 and thermal_factor > 1.5 else 
                       "Medium" if adjusted_duration > 2 else "Low"
    }

# Fonction principale d'analyse pour les pilotes
def analyze_emagramme_for_pilot(analysis):
    """
    Génère une analyse détaillée des conditions de vol destinée aux pilotes de parapente
    Utilisée comme solution de secours quand l'API OpenAI n'est pas disponible
    
    Args:
        analysis: Résultats de l'analyse de l'émagramme
        
    Returns:
        Texte formaté avec l'analyse des conditions de vol
    """
    # Vérifier d'abord si le vol est impossible
    vol_impossible = False
    
    # Vérifier si des précipitations sont prévues
    if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
        vol_impossible = True
        raison = analysis.precipitation_description
    
    # Vérifier si le vent est trop fort
    elif "fort au sol" in analysis.wind_conditions.lower() or "critique" in analysis.wind_conditions.lower():
        vol_impossible = True
        raison = "Vent trop fort"
    else:
        vol_impossible = False
    
    if vol_impossible:
        return f"""## ⚠️ VOL IMPOSSIBLE - {raison}

### Conditions météorologiques dangereuses
{analysis.flight_conditions}

### Conditions de vent
{analysis.wind_conditions}

### Dangers spécifiques
{"- " + chr(10) + "- ".join(analysis.hazards) if analysis.hazards else "Aucun danger spécifique identifié"}

### Recommandation
Il est fortement recommandé de ne pas voler dans ces conditions et d'attendre une amélioration de la météo.
"""
    
    # Évaluation du niveau de difficulté
    if analysis.thermal_strength in ["Faible", "Modérée"] and analysis.stability != "Très Instable":
        difficulty = "débutant à intermédiaire"
    elif analysis.thermal_strength == "Forte" and analysis.stability == "Instable":
        difficulty = "intermédiaire à avancé"
    else:
        difficulty = "avancé"
        
    # Heures optimales de vol
    if analysis.thermal_strength in ["Forte", "Très Forte"]:
        best_hours = "tôt le matin (avant 11h) et en fin d'après-midi (après 16h)"
    else:
        best_hours = "milieu de journée (entre 12h et 15h)"
    
    # Estimer la durée de vol typique
    flight_duration = predict_flight_duration(
        analysis.thermal_ceiling, 
        analysis.ground_altitude, 
        analysis.thermal_strength,
        20,  # Vent estimé moyen de 20 km/h
        analysis.low_cloud_cover if analysis.low_cloud_cover is not None else 0,
        analysis.thermal_gradient
    )
    
    # Analyser les vents
    wind_recommendation = ""
    if "fort" in analysis.wind_conditions.lower() or "critique" in analysis.wind_conditions.lower():
        wind_recommendation = "Vigilance particulière face aux conditions de vent qui pourraient être turbulentes."
    
    # Construire la réponse
    response = f"""## Analyse des conditions pour le vol en parapente

### Résumé
{analysis.flight_conditions}

### Caractéristiques des thermiques
- **Force:** {analysis.thermal_strength}
- **Plafond:** {analysis.thermal_ceiling:.0f}m 
- **Gradient thermique:** {analysis.thermal_gradient:.1f}°C/1000m
- **Type de thermiques:** {"Thermiques bleus (sans marquage nuageux)" if analysis.thermal_type == "Bleu" else f"Cumulus avec base à {analysis.cloud_base:.0f}m et sommet à {analysis.cloud_top:.0f}m"}
- **Taux de montée moyen estimé:** {flight_duration["avg_climb_rate"]}m/s

### Horaires recommandés
Les meilleures conditions devraient se trouver {best_hours}.
Heure optimale de début le matin: {int(flight_duration["optimal_start_morning"])}h{int((flight_duration["optimal_start_morning"] % 1) * 60):02d}
Heure optimale de début l'après-midi: {int(flight_duration["optimal_start_afternoon"])}h{int((flight_duration["optimal_start_afternoon"] % 1) * 60):02d}

### Stratégie de vol
"""

    # Ajouter des conseils de stratégie selon les conditions
    if analysis.thermal_strength in ["Faible", "Modérée"]:
        response += """- Privilégier les faces bien exposées au soleil
- Rester patient dans les thermiques faibles, ne pas abandonner trop vite
- Se concentrer sur les zones de déclenchement connues
- Préférer les zones avec des contrastes de terrain (forêt/champs, roche/végétation)
"""
    else:
        response += """- Prudence dans les thermiques puissants, anticipation des surpuissances
- Possibilité de s'éloigner des reliefs après avoir gagné de l'altitude
- Les zones de convergence peuvent offrir de bonnes ascendances
- Prévoir une marge de sécurité par rapport au relief en raison de la forte activité thermique
"""
    
    # Ajouter une section pour la durée de vol et le potentiel cross
    response += f"""
### Durée de vol et potentiel cross
- **Durée de vol typique:** {flight_duration["flight_duration_hours"]} heures
- **Potentiel cross-country:** {flight_duration["xc_potential"] == "High" and "Élevé" or flight_duration["xc_potential"] == "Medium" and "Moyen" or "Faible"}
- **Temps moyen jusqu'au plafond:** {flight_duration["time_to_ceiling_minutes"]} minutes
"""

    # Ajouter des avertissements si nécessaire
    if analysis.hazards:
        response += "\n### ⚠️ Points de vigilance\n"
        for hazard in analysis.hazards:
            response += f"- {hazard}\n"
    
    # Équipement recommandé
    if analysis.recommended_gear:
        response += "\n### Équipement recommandé\n"
        for gear in analysis.recommended_gear:
            response += f"- {gear}\n"
    
    # Ajouter informations sur les conditions de vent
    response += f"\n### Conditions de vent\n{analysis.wind_conditions}\n"
    if wind_recommendation:
        response += f"\n{wind_recommendation}\n"
    
    # Niveau de difficulté
    response += f"\n### Niveau de difficulté\nConditions adaptées aux pilotes de niveau {difficulty}.\n"
    
    return response
    