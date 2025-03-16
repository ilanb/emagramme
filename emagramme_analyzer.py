#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyseur d'Émagramme pour Parapentistes
----------------------------------------
Ce script permet d'analyser un émagramme météorologique et de fournir des prévisions
pour le vol en parapente, en calculant notamment:
- La stabilité de l'atmosphère
- Le plafond des thermiques
- La formation de nuages (thermiques bleus ou cumulus)
- Le gradient thermique
- Les conditions générales de vol attendues
"""

# 1. Importations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import argparse
import json
import requests
from io import StringIO
import logging
import sys
from datetime import datetime


# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emagramme_debug.log"),  # Enregistre dans un fichier
        logging.StreamHandler()  # Affiche également dans la console
    ]
)
logger = logging.getLogger('EmagrammeAnalyzer')

# Constants
ADIABATIC_DRY_LAPSE_RATE = 1.0  # °C/100m
ADIABATIC_MOIST_LAPSE_RATE = 0.6  # °C/100m
DEW_POINT_LAPSE_RATE = 0.2  # °C/100m
THERMAL_TRIGGER_DELTA = 3.0  # °C (différence de température requise pour déclencher un thermique)

# 2. Classes de données
@dataclass
class AtmosphericLevel:
    """Classe représentant un niveau atmosphérique avec ses propriétés"""
    altitude: float  # en mètres
    pressure: float  # en hPa
    temperature: float  # en °C
    dew_point: float  # en °C
    wind_direction: Optional[int] = None  # en degrés
    wind_speed: Optional[float] = None  # en km/h

@dataclass
class EmagrammeAnalysis:
    """Résultats de l'analyse d'un émagramme pour le vol en parapente"""
    ground_altitude: float  # en mètres
    ground_temperature: float  # en °C
    ground_dew_point: float  # en °C
    thermal_ceiling: float  # en mètres
    thermal_strength: str  # 'Faible', 'Modérée', 'Forte', 'Très Forte'
    stability: str  # 'Stable', 'Neutre', 'Instable', 'Très Instable'
    thermal_type: str  # 'Bleu' ou 'Cumulus'
    thermal_gradient: float  # en °C/1000m
    inversion_layers: List[Tuple[float, float]]  # Liste de tuples (base, sommet) en mètres
    flight_conditions: str  # Description des conditions de vol attendues
    wind_conditions: str  # Description des conditions de vent
    hazards: List[str]  # Liste des dangers potentiels
    recommended_gear: List[str]  # Équipement recommandé
    # Ajout des informations sur les nuages
    low_cloud_cover: Optional[float] = None  # Couverture nuageuse basse (%)
    mid_cloud_cover: Optional[float] = None  # Couverture nuageuse moyenne (%)
    high_cloud_cover: Optional[float] = None  # Couverture nuageuse haute (%)
    # Ajout des informations sur les précipitations
    precipitation_type: Optional[int] = None  # Type de précipitation (0-8)
    precipitation_description: Optional[str] = None  # Description du type de précipitation
    # Nouvel attribut pour l'incohérence
    thermal_inconsistency: Optional[str] = None  # Message d'incohérence entre analyse thermique et nuages
    # Arguments avec valeurs par défaut à la fin
    cloud_base: Optional[float] = None  # en mètres (None si thermiques bleus)
    cloud_top: Optional[float] = None  # en mètres (None si thermiques bleus)
    model_name: Optional[str] = None  # Nom du modèle météo utilisé
    # Ajout des attributs pour le spread
    ground_spread: float = None  # Spread au sol
    spread_levels: Dict[str, float] = None  # Spread à différentes altitudes
    spread_analysis: str = None  # Analyse textuelle du spread

def add_convective_layer_visualization(ax, analysis):
        """
        Ajoute une visualisation claire de la couche convective à l'émagramme
        
        Args:
            ax: Axes matplotlib sur lesquelles tracer
            analysis: Résultats de l'analyse EmagrammeAnalysis
        """
        # Définir les limites de la couche convective
        conv_layer_base = analysis.ground_altitude
        conv_layer_top = analysis.thermal_ceiling
        anabatic_top = min(conv_layer_base + 500, conv_layer_top)
        
        # Ajouter la couche convective avec dégradé de couleur
        ax.axhspan(conv_layer_base, conv_layer_top, alpha=0.05, color='green', 
                label=f'Couche convective ({conv_layer_base:.0f}-{conv_layer_top:.0f}m)')
        
        # Ajouter la couche anabatique
        if anabatic_top > conv_layer_base:
            ax.axhspan(conv_layer_base, anabatic_top, alpha=0.1, color='blue', 
                    label=f'Couche anabatique ({conv_layer_base:.0f}-{anabatic_top:.0f}m)')
        
        # Ajouter des lignes de délimitation plus visibles
        ax.axhline(y=conv_layer_base, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=conv_layer_top, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=anabatic_top, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Ajouter des étiquettes textuelles
        temp_min = ax.get_xlim()[0]
        ax.text(temp_min + 2, conv_layer_base + 50, f"Sol: {conv_layer_base:.0f}m", 
                fontsize=8, color='darkgreen', ha='left', va='bottom', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        ax.text(temp_min + 2, conv_layer_top - 50, f"Plafond: {conv_layer_top:.0f}m", 
                fontsize=8, color='darkgreen', ha='left', va='top', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
        
# 3. Classe d'analyse principale
class EmagrammeAnalyzer:
    """Classe principale pour l'analyse d'un émagramme météorologique pour le vol en parapente"""
    
    def __init__(self, levels: List[AtmosphericLevel], site_altitude: float = None, cloud_info: Dict = None, precip_info: Dict = None, model_name: str = None):
        """
        Initialise l'analyseur avec des niveaux atmosphériques.
        
        Args:
            levels: Liste des niveaux atmosphériques (du sol vers l'altitude)
            site_altitude: Altitude du site de décollage en mètres
            cloud_info: Dictionnaire contenant les informations sur la couverture nuageuse
            precip_info: Dictionnaire contenant les informations sur les précipitations
        """
        self.model_name = model_name

        # Trier les niveaux par altitude croissante
        self.levels = sorted(levels, key=lambda x: x.altitude)
        
        # Déterminer l'altitude du sol si non spécifiée
        self.site_altitude = site_altitude if site_altitude is not None else self.levels[0].altitude
        
        # Stocker les informations sur les nuages
        self.cloud_info = cloud_info or {}
        
        # Stocker les informations sur les précipitations
        self.precip_info = precip_info or {}
        
        # Créer des arrays numpy pour faciliter les calculs
        self.altitudes = np.array([level.altitude for level in self.levels])
        self.pressures = np.array([level.pressure for level in self.levels])
        self.temperatures = np.array([level.temperature for level in self.levels])
        self.dew_points = np.array([level.dew_point for level in self.levels])
        
        if self.levels[0].wind_direction is not None:
            self.wind_directions = np.array([level.wind_direction if level.wind_direction is not None else np.nan 
                                            for level in self.levels])
            self.wind_speeds = np.array([level.wind_speed if level.wind_speed is not None else np.nan 
                                        for level in self.levels])
        else:
            self.wind_directions = None
            self.wind_speeds = None
            
        logger.info(f"Initialisé avec {len(levels)} niveaux atmosphériques. Altitude du site: {self.site_altitude}m")


    def _analyze_spread(self) -> Dict:
        """
        Analyse l'écart entre température et point de rosée à différents niveaux
        
        Returns:
            Dictionnaire contenant les spreads et l'analyse
        """
        # Calculer le spread au sol
        ground_temp, ground_dew = self._get_ground_values()
        ground_spread = ground_temp - ground_dew
        
        # Calculer le spread à différents niveaux standard (bas, moyen, haut)
        levels = {}
        
        # Niveau bas (~900 hPa)
        low_idx = np.searchsorted(self.pressures, 900, side='right')
        if low_idx < len(self.pressures):
            levels["bas"] = self.temperatures[low_idx] - self.dew_points[low_idx]
        
        # Niveau moyen (~700 hPa)
        mid_idx = np.searchsorted(self.pressures, 700, side='right')
        if mid_idx < len(self.pressures):
            levels["moyen"] = self.temperatures[mid_idx] - self.dew_points[mid_idx]
        
        # Niveau haut (~500 hPa)
        high_idx = np.searchsorted(self.pressures, 500, side='right')
        if high_idx < len(self.pressures):
            levels["haut"] = self.temperatures[high_idx] - self.dew_points[high_idx]
        
        # Analyse du spread
        analysis = ""
        if ground_spread < 3:
            analysis += "Spread faible au sol,\nindiquant une humidité élevée\net un risque de brouillard ou nuages bas. "
        elif ground_spread < 8:
            analysis += "Spread modéré au sol, conditions d'humidité normales. "
        else:
            analysis += "Spread important au sol, masse d'air sèche. "
        
        # Analyse de l'évolution du spread avec l'altitude
        spread_values = [v for v in levels.values() if v is not None]
        if spread_values and len(spread_values) > 1:
            if min(spread_values) < 3:
                analysis += "Spread faible en altitude, indiquant des couches nuageuses probables. "
            elif all(x > 8 for x in spread_values):
                analysis += "Spread élevé à tous les niveaux, atmosphère sèche favorable aux thermiques bleus. "
            else:
                analysis += "Spread variable avec l'altitude, indiquant des couches d'humidité différentes. "
        
        return {
            "ground_spread": ground_spread,
            "levels": levels,
            "analysis": analysis
        }

    def _get_ground_values(self) -> Tuple[float, float]:
        """
        Obtient la température et le point de rosée au sol (site de décollage)
        
        Returns:
            Tuple contenant (température, point de rosée) en °C
        """
        # Trouver l'indice du niveau juste au-dessus du sol
        idx = np.searchsorted(self.altitudes, self.site_altitude)
        
        if idx == 0:
            return self.temperatures[0], self.dew_points[0]
        
        # Interpolation linéaire pour obtenir les valeurs au sol
        alt_below, alt_above = self.altitudes[idx-1], self.altitudes[idx]
        temp_below, temp_above = self.temperatures[idx-1], self.temperatures[idx]
        dew_below, dew_above = self.dew_points[idx-1], self.dew_points[idx]
        
        ratio = (self.site_altitude - alt_below) / (alt_above - alt_below)
        
        ground_temp = temp_below + ratio * (temp_above - temp_below)
        ground_dew = dew_below + ratio * (dew_above - dew_below)
        
        return ground_temp, ground_dew

    def _calculate_thermal_path(self, start_temp: float, start_altitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le chemin d'un thermique depuis une température et altitude de départ
        
        Args:
            start_temp: Température de départ en °C
            start_altitude: Altitude de départ en mètres
            
        Returns:
            Tuple contenant (altitudes, températures) du chemin du thermique
        """
        # Créer une grille d'altitude pour tracer le chemin
        thermal_altitudes = np.arange(start_altitude, np.max(self.altitudes) + 100, 100)
        thermal_temps = np.zeros_like(thermal_altitudes, dtype=float)
        
        # Température initiale du thermique
        thermal_temps[0] = start_temp
        
        # Calculer le chemin du thermique en suivant l'adiabatique sèche
        for i in range(1, len(thermal_altitudes)):
            delta_alt = thermal_altitudes[i] - thermal_altitudes[i-1]
            thermal_temps[i] = thermal_temps[i-1] - (ADIABATIC_DRY_LAPSE_RATE * delta_alt / 100)
        
        return thermal_altitudes, thermal_temps

    def _calculate_dew_point_path(self, start_dew: float, start_altitude: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le chemin du point de rosée depuis une valeur et altitude de départ
        
        Args:
            start_dew: Point de rosée de départ en °C
            start_altitude: Altitude de départ en mètres
            
        Returns:
            Tuple contenant (altitudes, points de rosée) du chemin
        """
        # Créer une grille d'altitude pour tracer le chemin
        dew_altitudes = np.arange(start_altitude, np.max(self.altitudes) + 100, 100)
        dew_temps = np.zeros_like(dew_altitudes, dtype=float)
        
        # Point de rosée initial
        dew_temps[0] = start_dew
        
        # Calculer le chemin du point de rosée en suivant le taux iso-R
        for i in range(1, len(dew_altitudes)):
            delta_alt = dew_altitudes[i] - dew_altitudes[i-1]
            dew_temps[i] = dew_temps[i-1] - (DEW_POINT_LAPSE_RATE * delta_alt / 100)
        
        return dew_altitudes, dew_temps

    def _calculate_thermal_ceiling(self, thermal_altitudes: np.ndarray, thermal_temps: np.ndarray) -> float:
        """
        Calcule le plafond du thermique (intersection avec la courbe d'état)
        
        Args:
            thermal_altitudes: Altitudes du chemin du thermique
            thermal_temps: Températures du chemin du thermique
            
        Returns:
            Altitude du plafond thermique en mètres
        """
        # Interpoler la courbe d'état aux mêmes altitudes que le thermique
        interp_temps = np.interp(thermal_altitudes, self.altitudes, self.temperatures)
        
        # Trouver l'intersection (où thermal_temps <= interp_temps)
        intersections = np.where(thermal_temps <= interp_temps)[0]
        
        if len(intersections) > 0:
            return thermal_altitudes[intersections[0]]
        else:
            # Si pas d'intersection, le plafond est au sommet de la grille
            return thermal_altitudes[-1]

    def _find_condensation_level(self, thermal_altitudes: np.ndarray, thermal_temps: np.ndarray,
                            dew_altitudes: np.ndarray, dew_temps: np.ndarray) -> Optional[float]:
        """
        Trouve le niveau de condensation (où le thermique rencontre le point de rosée)
        Version améliorée avec vérification de cohérence
        """
        # Méthode existante pour trouver l'intersection
        common_altitudes = np.arange(self.site_altitude, np.max(self.altitudes) + 100, 100)
        interp_thermal = np.interp(common_altitudes, thermal_altitudes, thermal_temps)
        interp_dew = np.interp(common_altitudes, dew_altitudes, dew_temps)
        
        # Trouver l'intersection (où interp_thermal <= interp_dew)
        intersections = np.where(interp_thermal <= interp_dew)[0]
        
        if len(intersections) > 0:
            condensation_level = common_altitudes[intersections[0]]
            
            # Vérification de cohérence avec les données nuageuses observées
            if hasattr(self, 'cloud_info') and self.cloud_info:
                low_cover = self.cloud_info.get('low_clouds', 0)
                mid_cover = self.cloud_info.get('mid_clouds', 0)
                
                # Si forte couverture nuageuse mais condensation calculée très haute
                if (low_cover is not None and low_cover > 70) and condensation_level > 3000:
                    # Ajuster vers le bas pour refléter la réalité observée
                    adjusted_level = min(condensation_level, 2500)
                    logger.info(f"Niveau de condensation ajusté de {condensation_level}m à {adjusted_level}m")
                    return adjusted_level
            
            return condensation_level
        else:
            # Pas de condensation théorique, mais vérifier la couverture nuageuse observée
            if hasattr(self, 'cloud_info') and self.cloud_info:
                low_cover = self.cloud_info.get('low_clouds', 0)
                if low_cover is not None and low_cover > 50:
                    # Estimer un niveau de condensation pour les nuages observés
                    # (typiquement entre 1500 et 2500m pour les nuages bas)
                    return 2000  # Estimation grossière
            
            # Si pas d'intersection et pas de nuages observés, pas de condensation
            return None

    def _calculate_moist_adiabatic_path(self, condensation_level: float, thermal_temps: np.ndarray, 
                                     thermal_altitudes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule le chemin du thermique après condensation en suivant l'adiabatique humide
        
        Args:
            condensation_level: Altitude du niveau de condensation en mètres
            thermal_temps: Températures du chemin du thermique sec
            thermal_altitudes: Altitudes du chemin du thermique sec
            
        Returns:
            Tuple contenant (altitudes, températures) du chemin humide
        """
        # Trouver la température au niveau de condensation
        idx = np.searchsorted(thermal_altitudes, condensation_level)
        if idx >= len(thermal_altitudes):
            return np.array([]), np.array([])
            
        temp_at_condensation = np.interp(condensation_level, thermal_altitudes, thermal_temps)
        
        # Créer une grille d'altitude pour la continuation du thermique
        moist_altitudes = np.arange(condensation_level, np.max(self.altitudes) + 100, 100)
        moist_temps = np.zeros_like(moist_altitudes, dtype=float)
        
        # Température initiale au niveau de condensation
        moist_temps[0] = temp_at_condensation
        
        # Calculer le chemin en suivant l'adiabatique humide
        for i in range(1, len(moist_altitudes)):
            delta_alt = moist_altitudes[i] - moist_altitudes[i-1]
            moist_temps[i] = moist_temps[i-1] - (ADIABATIC_MOIST_LAPSE_RATE * delta_alt / 100)
        
        return moist_altitudes, moist_temps

    def _calculate_cloud_top(self, moist_altitudes: np.ndarray, moist_temps: np.ndarray) -> float:
        """
        Calcule le sommet du nuage avec une approche plus robuste et réaliste
        """
        # Interpoler la courbe d'état aux mêmes altitudes
        interp_temps = np.interp(moist_altitudes, self.altitudes, self.temperatures)
        
        # Trouver les intersections (où moist_temps <= interp_temps)
        intersections = np.where(moist_temps <= interp_temps)[0]
        
        # Vérifier si nous avons une intersection valide
        if len(intersections) > 0:
            cloud_top = moist_altitudes[intersections[0]]
            
            # Vérifier si le sommet est réaliste (max 5000m pour les cumulus typiques)
            if hasattr(self, 'cloud_base') and self.cloud_base is not None:
                max_reasonable_top = min(self.cloud_base + 3000, 5000)
                
                if cloud_top > max_reasonable_top:
                    logger.info(f"Sommet des nuages ajusté de {cloud_top:.0f}m à {max_reasonable_top:.0f}m (limitation à une hauteur réaliste)")
                    return max_reasonable_top
                    
                # Garantir une épaisseur minimale de 300m pour les cumulus
                min_top = self.cloud_base + 300
                if cloud_top < min_top:
                    logger.info(f"Sommet des nuages ajusté de {cloud_top:.0f}m à {min_top:.0f}m (épaisseur minimale)")
                    return min_top
                    
                return cloud_top
        
        # Si pas d'intersection valide, estimer en fonction de la base
        if hasattr(self, 'cloud_base') and self.cloud_base is not None:
            base_height = self.cloud_base
            
            # Estimation plus conservatrice basée sur la hauteur de la base
            if base_height < 1000:
                thickness = 1000  # Base très basse
            elif base_height < 1500:
                thickness = 800   # Base basse
            elif base_height < 2500:
                thickness = 600   # Base moyenne
            else:
                thickness = 400   # Base haute
                
            estimated_top = base_height + thickness
            logger.info(f"Sommet des nuages estimé à {estimated_top:.0f}m (méthode alternative)")
            return estimated_top
            
        # Fallback si on n'a vraiment aucune information
        return min(max(self.altitudes), 5000)

    def _find_inversions(self) -> List[Tuple[float, float]]:
        """
        Trouve les couches d'inversion thermique dans l'atmosphère
        
        Returns:
            Liste de tuples (base, sommet) des inversions en mètres
        """
        inversions = []
        
        # Calculer le gradient vertical de température (°C/100m)
        gradients = np.zeros(len(self.altitudes)-1)
        for i in range(len(gradients)):
            delta_alt = self.altitudes[i+1] - self.altitudes[i]
            delta_temp = self.temperatures[i+1] - self.temperatures[i]
            gradients[i] = (delta_temp / delta_alt) * 100
        
        # Identifier les couches d'inversion (gradient > 0)
        inversion_start = None
        for i in range(len(gradients)):
            if gradients[i] > 0 and inversion_start is None:
                inversion_start = self.altitudes[i]
            elif (gradients[i] <= 0 or i == len(gradients)-1) and inversion_start is not None:
                inversions.append((inversion_start, self.altitudes[i+1]))
                inversion_start = None
        
        return inversions

    def _evaluate_thermal_strength(self, thermal_gradient: float) -> str:
        """
        Évalue la force des thermiques en fonction du gradient thermique
        
        Args:
            thermal_gradient: Gradient thermique en °C/1000m
            
        Returns:
            Description de la force des thermiques
        """
        if thermal_gradient < 5:
            return "Faible"
        elif thermal_gradient < 7:
            return "Modérée"
        elif thermal_gradient < 9:
            return "Forte"
        else:
            return "Très Forte"

    def _evaluate_stability(self, thermal_gradient: float, inversions: List[Tuple[float, float]]) -> str:
        """
        Évalue la stabilité de l'atmosphère
        
        Args:
            thermal_gradient: Gradient thermique en °C/1000m
            inversions: Liste des inversions thermiques
            
        Returns:
            Description de la stabilité
        """
        if thermal_gradient < 5 or len(inversions) > 2:
            return "Stable"
        elif thermal_gradient < 6.5:
            return "Neutre"
        elif thermal_gradient < 8:
            return "Instable"
        else:
            return "Très Instable"

    def _describe_flight_conditions(self, analysis: EmagrammeAnalysis) -> str:
        """
        Génère une description des conditions de vol attendues
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Description des conditions de vol
        """
        conditions = []

        # Vérifier immédiatement si des précipitations sont prévues
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            conditions.append(f"\n\n⚠️ VOL IMPOSSIBLE - {analysis.precipitation_description} prévue. Le vol en parapente est dangereux et déconseillé en présence de précipitations.")
            return " ".join(conditions)
        
        # Vérifier si le vent est trop fort UNIQUEMENT dans la zone de vol
        if hasattr(self, 'vol_impossible_wind') and self.vol_impossible_wind:
            conditions.append(f"\n\n⚠️ VOL IMPOSSIBLE - Vent trop fort dans la zone de vol ({self.max_wind_in_vol_zone:.1f} km/h)")
            return " ".join(conditions)
        
        # Conditions thermiques
        if analysis.thermal_strength == "Faible":
            conditions.append("\n\nThermiques faibles et peu durables")
        elif analysis.thermal_strength == "Modérée":
            conditions.append("\n\nThermiques modérés et réguliers")
        elif analysis.thermal_strength == "Forte":
            conditions.append("\n\nThermiques forts et actifs")
        else:
            conditions.append("\n\nThermiques très puissants, potentiellement turbulents")
        
        # Ajouter l'incohérence si elle existe - après les conditions thermiques et avant le type de thermiques
        if analysis.thermal_inconsistency:
            conditions.append(f"\n\n⚠️ {analysis.thermal_inconsistency}")
        
        # Type de thermiques
        if analysis.thermal_type == "Bleu":
            if analysis.thermal_inconsistency:
                conditions.append("\n\nThermiques sans condensation mais présence de nuages stratiformes")
            else:
                conditions.append("\n\nThermiques bleus sans marquage nuageux")
        else:
            cloud_thickness = analysis.cloud_top - analysis.cloud_base
            if cloud_thickness < 500:
                conditions.append("\n\nPetits cumulus peu développés")
            elif cloud_thickness < 1000:
                conditions.append("\n\nCumulus bien formés")
            else:
                conditions.append("\n\nCumulus bien développés, risque de surdéveloppement")

        # Ajouter des informations sur la couverture nuageuse
        if analysis.low_cloud_cover is not None or analysis.mid_cloud_cover is not None or analysis.high_cloud_cover is not None:
            cloud_desc = "\n\nCouverture nuageuse: "
            cloud_parts = []
            
            if analysis.low_cloud_cover is not None:
                cover = analysis.low_cloud_cover
                if cover < 25:
                    desc = "peu de nuages bas"
                elif cover < 50:
                    desc = "nuages bas épars"
                elif cover < 75:
                    desc = "nuages bas nombreux"
                else:
                    desc = "couvert de nuages bas"
                cloud_parts.append(desc)
                
            if analysis.mid_cloud_cover is not None:
                cover = analysis.mid_cloud_cover
                if cover < 25:
                    desc = "peu de nuages moyens"
                elif cover < 50:
                    desc = "nuages moyens épars"
                elif cover < 75:
                    desc = "nuages moyens nombreux"
                else:
                    desc = "couvert de nuages moyens"
                cloud_parts.append(desc)
                
            if analysis.high_cloud_cover is not None:
                cover = analysis.high_cloud_cover
                if cover < 25:
                    desc = "peu de nuages hauts"
                elif cover < 50:
                    desc = "nuages hauts épars"
                elif cover < 75:
                    desc = "nuages hauts nombreux"
                else:
                    desc = "couvert de nuages hauts"
                cloud_parts.append(desc)
                
            cloud_desc += ", ".join(cloud_parts)
            conditions.append(cloud_desc)
        
        # Ajouter des informations sur les précipitations
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            conditions.append(f"\n\nPrécipitations: {analysis.precipitation_description}")
        
        # Plafond
        if analysis.thermal_ceiling < 1500:
            conditions.append("\n\nPlafond bas")
        elif analysis.thermal_ceiling < 2500:
            conditions.append("\n\nPlafond moyen")
        else:
            conditions.append("\n\nPlafond élevé")
        
        # Stabilité
        if analysis.stability == "Stable":
            conditions.append("\n\nAtmosphère stable, vol potentiellement technique")
        elif analysis.stability == "Neutre":
            conditions.append("\n\nAtmosphère assez stable, bonnes conditions générales")
        elif analysis.stability == "Instable":
            conditions.append("\n\nAtmosphère instable, attention aux surpuissances")
        else:
            conditions.append("\n\nAtmosphère très instable, conditions potentiellement difficiles à gérer")
        
        # Inversions
        if analysis.inversion_layers:
            conditions.append(f"\n\nPrésence de {len(analysis.inversion_layers)} couche(s) d'inversion qui pourraient limiter les ascendances")
        
        return " ".join(conditions)

    def _evaluate_wind_conditions(self) -> str:
        """
        Évalue les conditions de vent
        
        Returns:
            Description des conditions de vent
        """
        if self.wind_speeds is None or self.wind_directions is None:
            return "Données de vent non disponibles"
        
        # Extraire les vents pertinents (sol, 1000m au-dessus, altitude des thermiques)
        ground_idx = np.searchsorted(self.altitudes, self.site_altitude)
        if ground_idx >= len(self.altitudes):
            ground_idx = len(self.altitudes) - 1
            
        mid_altitude = self.site_altitude + 1000
        mid_idx = np.searchsorted(self.altitudes, mid_altitude)
        if mid_idx >= len(self.altitudes):
            mid_idx = len(self.altitudes) - 1
        
        ground_speed = self.wind_speeds[ground_idx]
        ground_dir = self.wind_directions[ground_idx]
        mid_speed = self.wind_speeds[mid_idx]
        mid_dir = self.wind_directions[mid_idx]
        
        # Calculer le gradient de vent
        wind_gradient = (mid_speed - ground_speed) / 1000  # km/h par 1000m
        
        conditions = []
        
        # Vitesse au sol
        if ground_speed < 5:
            conditions.append(f"Vent faible au sol ({ground_speed:.1f} km/h)")
        elif ground_speed < 15:
            conditions.append(f"Vent modéré au sol ({ground_speed:.1f} km/h)")
        elif ground_speed < 25:
            conditions.append(f"Vent assez fort au sol ({ground_speed:.1f} km/h)")
        else:
            conditions.append(f"Vent fort au sol ({ground_speed:.1f} km/h), potentiellement critique")
        
        # Direction
        dir_names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                    "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
        idx = round(ground_dir / 22.5) % 16
        conditions.append(f"Direction: {dir_names[idx]}")
        
        # Gradient
        if abs(wind_gradient) < 2:
            conditions.append("Peu de gradient de vent en altitude")
        elif abs(wind_gradient) < 5:
            conditions.append("Gradient de vent modéré en altitude")
        else:
            conditions.append("Fort gradient de vent en altitude, risque de cisaillement")
        
        # Variation de direction
        dir_diff = abs(mid_dir - ground_dir)
        if dir_diff > 180:
            dir_diff = 360 - dir_diff
            
        if dir_diff < 30:
            conditions.append("Direction stable en altitude")
        elif dir_diff < 60:
            conditions.append("Légère rotation du vent en altitude")
        else:
            conditions.append("Forte rotation du vent en altitude")
        
        return ", ".join(conditions)

    def _identify_hazards(self, analysis: EmagrammeAnalysis) -> List[str]:
        """
        Identifie les dangers potentiels pour le vol en se concentrant sur la zone de vol
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Liste des dangers identifiés
        """
        hazards = []
        
        # Thermiques trop forts
        if analysis.thermal_strength == "Très Forte":
            hazards.append("Thermiques très puissants, risque de fermetures")
        
        # Dangers liés à la couverture nuageuse
        if analysis.low_cloud_cover is not None and analysis.low_cloud_cover > 75:
            hazards.append("Forte couverture nuageuse basse, risque de plafond bas")
            
        if analysis.mid_cloud_cover is not None and analysis.mid_cloud_cover > 75:
            hazards.append("Forte couverture nuageuse moyenne, risque d'ombrage et d'affaiblissement des thermiques")
            
        if analysis.high_cloud_cover is not None and analysis.high_cloud_cover > 75:
            hazards.append("Forte couverture nuageuse haute, risque de stabilisation de l'atmosphère")

        # Dangers liés aux précipitations
        if analysis.precipitation_type is not None:
            if analysis.precipitation_type == 1:  # Pluie
                hazards.append("Pluie: risque de détrempement de la voile et diminution des performances")
            elif analysis.precipitation_type == 3:  # Pluie verglaçante
                hazards.append("Pluie verglaçante: danger critique, risque de givrage de la voile")
            elif analysis.precipitation_type == 5:  # Neige
                hazards.append("Neige: risque de givrage et de perte de portance")
            elif analysis.precipitation_type == 7:  # Mélange pluie et neige
                hazards.append("Mélange pluie/neige: conditions dangereuses, détrempement et givrage possibles")
            elif analysis.precipitation_type == 8:  # Grésil
                hazards.append("Grésil: danger important, impact sur la voile et turbulences possibles")

        # Surdéveloppement nuageux
        if analysis.thermal_type == "Cumulus" and analysis.cloud_top and analysis.cloud_base:
            cloud_thickness = analysis.cloud_top - analysis.cloud_base
            if cloud_thickness > 1500:
                hazards.append("Risque de surdéveloppement nuageux, surveillance des congestus")
        
        # Température basse
        thermal_ceiling_temp = np.interp(analysis.thermal_ceiling, self.altitudes, self.temperatures)
        if thermal_ceiling_temp < 0:
            hazards.append(f"Températures négatives au plafond ({thermal_ceiling_temp:.1f}°C), risque de froid")
        
        # Utiliser les données de vent de la zone de vol
        if hasattr(self, 'max_wind_in_vol_zone') and self.max_wind_in_vol_zone:
            max_wind = self.max_wind_in_vol_zone
            
            if max_wind > 25 and max_wind <= 35:
                hazards.append(f"Vent soutenu dans la zone de vol ({max_wind:.1f} km/h), vol réservé aux pilotes confirmés")
            elif max_wind > 20:
                hazards.append(f"Vent modéré dans la zone de vol ({max_wind:.1f} km/h), vigilance recommandée")
        
        # Inversions problématiques
        if analysis.inversion_layers:
            for base, top in analysis.inversion_layers:
                if base < 1500 and base > self.site_altitude:
                    hazards.append(f"Inversion basse ({base:.0f}m), risque de thermiques étouffés")
        
        return hazards

    def _recommend_gear(self, analysis: EmagrammeAnalysis) -> List[str]:
        """
        Recommande l'équipement adapté aux conditions prévues
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Liste des équipements recommandés
        """
        # Vérifier si des précipitations sont prévues
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            return ["AUCUN - VOL IMPOSSIBLE EN RAISON DES PRÉCIPITATIONS"]
        
        gear = []
        
        # Température au plafond
        thermal_ceiling_temp = np.interp(analysis.thermal_ceiling, self.altitudes, self.temperatures)
        
        # Recommandations pour le froid
        if thermal_ceiling_temp < -5:
            gear.extend(["Moufles", "Sous-gants", "Chaussettes chauffantes", "Combinaison chaude"])
        elif thermal_ceiling_temp < 0:
            gear.extend(["Gants d'hiver", "Coupe-vent", "Chaussettes épaisses"])
        elif thermal_ceiling_temp < 5:
            gear.extend(["Gants", "Veste chaude"])
        
        # Recommandations basées sur les conditions thermiques
        if analysis.thermal_strength in ["Forte", "Très Forte"]:
            gear.append("Voile adaptée aux conditions fortes")
        
        # Protection solaire pour les vols longs
        if analysis.thermal_ceiling > 2500 or analysis.thermal_strength in ["Modérée", "Forte"]:
            gear.extend(["Crème solaire", "Lunettes de soleil"])
        
        # Eau pour les vols longs
        if analysis.thermal_strength in ["Modérée", "Forte"]:
            gear.append("Eau (potentiel de vol long)")
        
        return gear

    def analyze(self) -> EmagrammeAnalysis:
        """
        Effectue l'analyse complète de l'émagramme
        
        Returns:
            Résultats complets de l'analyse
        """
        # Obtenir les valeurs au sol
        ground_temp, ground_dew = self._get_ground_values()
        
        # Température de déclenchement des thermiques
        trigger_temp = ground_temp + THERMAL_TRIGGER_DELTA
        
        # Calculer le chemin du thermique
        thermal_altitudes, thermal_temps = self._calculate_thermal_path(trigger_temp, self.site_altitude)
        
        # Calculer le chemin du point de rosée
        dew_altitudes, dew_temps = self._calculate_dew_point_path(ground_dew, self.site_altitude)
        
        # Trouver le plafond du thermique
        thermal_ceiling = self._calculate_thermal_ceiling(thermal_altitudes, thermal_temps)
        
        # Trouver le niveau de condensation (base des nuages)
        condensation_level = self._find_condensation_level(thermal_altitudes, thermal_temps, dew_altitudes, dew_temps)
        
        # Variables pour l'analyse des nuages
        cloud_base = None
        cloud_top = None
        thermal_type = "Bleu"

        # Vérifier la cohérence entre analyse thermique et couverture nuageuse
        thermal_inconsistency = None
        if thermal_type == "Bleu" and hasattr(self, 'cloud_info') and self.cloud_info:
            low_cover = self.cloud_info.get('low_clouds')
            mid_cover = self.cloud_info.get('mid_clouds')
            
            if (low_cover is not None and low_cover > 30) or (mid_cover is not None and mid_cover > 30):
                thermal_inconsistency = "Incohérence détectée: l'analyse des courbes indique des thermiques bleus, " \
                                    "mais la prévision indique une couverture nuageuse significative. " \
                                    "\nIl s'agit probablement de nuages stratiformes non liés à l'activité thermique."
        
        
        if condensation_level and condensation_level < thermal_ceiling:
            # Le thermique condense avant d'atteindre son plafond
            cloud_base = condensation_level
            thermal_type = "Cumulus"
            
            # Calculer le chemin après condensation
            moist_altitudes, moist_temps = self._calculate_moist_adiabatic_path(
                condensation_level, thermal_temps, thermal_altitudes)
            
            # Trouver le sommet du nuage
            if len(moist_altitudes) > 0:
                cloud_top = self._calculate_cloud_top(moist_altitudes, moist_temps)
        
        # Calculer le gradient thermique général
        if len(self.altitudes) > 1:
            idx_ceiling = np.searchsorted(self.altitudes, thermal_ceiling)
            if idx_ceiling >= len(self.altitudes):
                idx_ceiling = len(self.altitudes) - 1
                
            delta_alt = self.altitudes[idx_ceiling] - self.altitudes[0]
            delta_temp = self.temperatures[0] - self.temperatures[idx_ceiling]
            thermal_gradient = (delta_temp / delta_alt) * 1000  # °C/1000m
        else:
            thermal_gradient = 6.5  # Valeur standard
        
        # Trouver les inversions
        inversion_layers = self._find_inversions()
        
        # Évaluations des conditions
        thermal_strength = self._evaluate_thermal_strength(thermal_gradient)
        stability = self._evaluate_stability(thermal_gradient, inversion_layers)
        
        # Créer l'analyse
        analysis = EmagrammeAnalysis(
            model_name=self.model_name,
            ground_altitude=self.site_altitude,
            ground_temperature=ground_temp,
            ground_dew_point=ground_dew,
            thermal_ceiling=thermal_ceiling,
            cloud_base=cloud_base,
            cloud_top=cloud_top,
            thermal_strength=thermal_strength,
            stability=stability,
            thermal_type=thermal_type,
            thermal_gradient=thermal_gradient,
            inversion_layers=inversion_layers,
            thermal_inconsistency=thermal_inconsistency,
            flight_conditions="",  # Sera rempli ci-dessous
            wind_conditions="",  # Sera rempli ci-dessous
            hazards=[],  # Sera rempli ci-dessous
            recommended_gear=[]  # Sera rempli ci-dessous
        )
        
        # Ajout des informations sur les nuages si disponibles
        if hasattr(self, 'cloud_info') and self.cloud_info:
            analysis.low_cloud_cover = self.cloud_info.get('low_clouds')
            analysis.mid_cloud_cover = self.cloud_info.get('mid_clouds')
            analysis.high_cloud_cover = self.cloud_info.get('high_clouds')
        
        # Ajout des informations sur les précipitations si disponibles
        if hasattr(self, 'precip_info') and self.precip_info:
            analysis.precipitation_type = self.precip_info.get('type')
            analysis.precipitation_description = self.precip_info.get('description')
        
        # AJOUTEZ CE CODE ICI - Début
        # Définir la zone de vol en parapente
        vol_min_alt = analysis.ground_altitude
        vol_max_alt = min(analysis.thermal_ceiling + 200, cloud_base if cloud_base else 6000)
        
        # Ajouter ces informations à l'objet analyzer pour qu'elles soient accessibles ailleurs
        self.vol_min_alt = vol_min_alt
        self.vol_max_alt = vol_max_alt
        
        # Trouver les indices correspondants dans self.altitudes
        vol_alt_indices = [i for i, alt in enumerate(self.altitudes) 
                        if vol_min_alt <= alt <= vol_max_alt]
        
        # Vérifier le vent seulement dans cette zone
        self.vol_impossible_wind = False
        self.max_wind_in_vol_zone = 0
        
        if self.wind_speeds is not None and len(vol_alt_indices) > 0:
            vol_wind_speeds = self.wind_speeds[vol_alt_indices]
            self.max_wind_in_vol_zone = np.nanmax(vol_wind_speeds)
            
            # Vérifier si le vent est trop fort pour voler
            if self.max_wind_in_vol_zone > 35:  # 35 km/h est souvent un seuil critique
                self.vol_impossible_wind = True

        # Compléter l'analyse
        analysis.flight_conditions = self._describe_flight_conditions(analysis)
        analysis.wind_conditions = self._evaluate_wind_conditions()
        analysis.hazards = self._identify_hazards(analysis)
        analysis.recommended_gear = self._recommend_gear(analysis)
        
        # Calculer l'analyse du spread
        spread_data = self._analyze_spread()
        
        # Ajouter à l'analyse
        analysis.ground_spread = spread_data["ground_spread"]
        analysis.spread_levels = spread_data["levels"]
        analysis.spread_analysis = spread_data["analysis"]

        # Analyser les couches atmosphériques
        layer_analysis = self._analyze_atmospheric_layers(thermal_ceiling)
        
        # Stocker cette analyse dans l'objet d'analyse
        analysis.atmospheric_layers = layer_analysis

        return analysis

    def _analyze_atmospheric_layers(self, thermal_ceiling: float) -> Dict:
        """
        Analyse les différentes couches atmosphériques par rapport à la couche convective
        
        Args:
            thermal_ceiling: Altitude du plafond thermique en mètres
            
        Returns:
            Dictionnaire contenant les informations sur les couches atmosphériques
        """
        # Altitude du sol
        ground_altitude = self.site_altitude
        
        # Définir les limites théoriques des couches
        layers = {
            "sous-convective": (0, ground_altitude),  # Sous le sol
            "convective": (ground_altitude, thermal_ceiling),  # Couche convective
            "supra-convective": (thermal_ceiling, thermal_ceiling + 1000),  # 1000m au-dessus du plafond
            "haute": (thermal_ceiling + 1000, float('inf'))  # Au-delà
        }
        
        # Analyser chaque couche
        layer_analysis = {}
        
        for layer_name, (bottom, top) in layers.items():
            if layer_name == "sous-convective":
                continue  # Pas d'analyse sous le sol
            
            # Trouver les indices des niveaux dans cette couche
            indices = [i for i, alt in enumerate(self.altitudes) 
                    if bottom <= alt < top]
            
            # Si aucune donnée disponible dans cette couche, utiliser les limites théoriques
            # mais sans analyse météorologique
            if not indices:
                layer_analysis[layer_name] = {
                    "indices": [],
                    "altitudes": np.array([]),
                    "theoretical_bottom": bottom,
                    "theoretical_top": top,
                    "mean_temp": None,
                    "mean_dew": None,
                    "mean_spread": None,
                    "stability": "données insuffisantes"
                }
                continue
            
            # Calculer les propriétés de la couche
            temps = self.temperatures[indices]
            dews = self.dew_points[indices]
            
            layer_analysis[layer_name] = {
                "indices": indices,
                "altitudes": self.altitudes[indices],
                "theoretical_bottom": bottom,  # Stocke explicitement les limites théoriques
                "theoretical_top": top,
                "mean_temp": np.mean(temps),
                "mean_dew": np.mean(dews),
                "mean_spread": np.mean(temps - dews),
                "stability": "stable" if np.std(temps) < 1.0 else "instable"
            }
            
            # Ajouter les vents si disponibles
            if self.wind_speeds is not None:
                winds = self.wind_speeds[indices]
                dirs = self.wind_directions[indices]
                layer_analysis[layer_name]["mean_wind"] = np.mean(winds[~np.isnan(winds)])
                
                # Direction dominante (mode)
                if not np.all(np.isnan(dirs)):
                    valid_dirs = dirs[~np.isnan(dirs)]
                    if len(valid_dirs) > 0:
                        # Convertir à la direction cardinale la plus proche
                        dir_names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                                    "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
                        indices = np.round(valid_dirs / 22.5) % 16
                        directions = [dir_names[int(i)] for i in indices]
                        # Trouver la direction la plus fréquente
                        from collections import Counter
                        counts = Counter(directions)
                        layer_analysis[layer_name]["dominant_dir"] = counts.most_common(1)[0][0]
        
        return layer_analysis
    
    def plot_emagramme(self, analysis=None, llm_analysis=None, save_path=None, show=True):
        """Trace l'émagramme avec les résultats de l'analyse"""
        if analysis is None:
            analysis = self.analyze()
        
        # NOUVELLE SOLUTION: Filtrer les données pour ne garder que jusqu'à 6000m
        # Créer des copies filtrées des données atmosphériques
        max_altitude = 6000  # Limite stricte à 6000m
        
        # Filtrer les tableaux numpy
        altitude_mask = self.altitudes <= max_altitude
        filtered_altitudes = self.altitudes[altitude_mask]
        filtered_temperatures = self.temperatures[altitude_mask]
        filtered_dew_points = self.dew_points[altitude_mask]
        filtered_pressures = self.pressures[altitude_mask]
        
        # Filtrer également les données de vent si elles existent
        if self.wind_directions is not None and self.wind_speeds is not None:
            filtered_wind_directions = self.wind_directions[altitude_mask]
            filtered_wind_speeds = self.wind_speeds[altitude_mask]
        
        # Conserver les données originales pour référence
        original_altitudes = self.altitudes
        original_temperatures = self.temperatures
        original_dew_points = self.dew_points
        original_pressures = self.pressures
        original_wind_directions = self.wind_directions if self.wind_directions is not None else None
        original_wind_speeds = self.wind_speeds if self.wind_speeds is not None else None
        
        # Remplacer temporairement les données par les versions filtrées
        self.altitudes = filtered_altitudes
        self.temperatures = filtered_temperatures
        self.dew_points = filtered_dew_points
        self.pressures = filtered_pressures
        if self.wind_directions is not None:
            self.wind_directions = filtered_wind_directions
            self.wind_speeds = filtered_wind_speeds
        
        # Déterminer si l'on doit inclure un panneau pour l'analyse LLM
        include_llm_panel = llm_analysis is not None

        # Créer une figure avec deux sous-graphiques - un pour l'émagramme et un pour le texte
        fig = plt.figure(figsize=(12, 10))  # Augmenter la hauteur pour accommoder le texte en bas
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
        
        # Graphique d'émagramme (3/4 supérieur)
        ax = fig.add_subplot(111)
        
        # Panneau de texte (1/4 inférieur)
        ax_text = fig.add_subplot(gs[1])
        ax_text.axis('off')  # Désactiver les axes pour le panneau de texte
            
        # Définir les limites d'altitude
        alt_min = max(0, self.site_altitude - 500)
        alt_max = min(6000, max(min(self.altitudes) + 2000, self.site_altitude + 3000))  # Plus intelligent
        
        # Définir les limites de température
        temp_min = min(min(self.temperatures) - 10, min(self.dew_points) - 5, -20)
        temp_max = max(max(self.temperatures) + 10, analysis.ground_temperature + THERMAL_TRIGGER_DELTA + 10, 40)
            
        # Tracer les grilles
        for temp in range(-80, 50, 10):
            ax.plot([temp, temp], [alt_min, alt_max], 'gray', alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Niveaux de pression (hPa)
        pressure_levels = [1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300]
        for pressure in pressure_levels:
            # Convertir pression en altitude approximative
            altitude = 44330 * (1 - (pressure / 1013.25) ** 0.1903)
            if alt_min <= altitude <= alt_max:
                ax.plot([temp_min, temp_max], [altitude, altitude], 'gray', alpha=0.3, linestyle='--', linewidth=0.5)
                ax.text(temp_min, altitude, f"{pressure} hPa", fontsize=5, ha='right', va='center')
        
        # Tracer la courbe d'état (température)
        ax.plot(self.temperatures, self.altitudes, 'r-', linewidth=2, label='Courbe d\'état')
        
        # Tracer la courbe du point de rosée
        ax.plot(self.dew_points, self.altitudes, 'b-', linewidth=2, label='Point de rosée')
        
        # Tracer le niveau du sol
        ax.axhline(y=self.site_altitude, color='brown', linestyle='-', linewidth=2, label='Sol')
        
        # Ajouter une représentation visuelle des couches nuageuses
        if analysis.low_cloud_cover is not None or analysis.mid_cloud_cover is not None or analysis.high_cloud_cover is not None:
            # Zones d'altitude pour les différentes couches de nuages
            # Nuages bas: surface à 800 hPa (environ 2000m)
            # Nuages moyens: 800 hPa à 450 hPa (environ 2000m à 6500m)
            # Nuages hauts: au-dessus de 450 hPa (environ 6500m et plus)
            
            # Convertir pression en altitude approximative
            alt_800hpa = 44330 * (1 - (800 / 1013.25) ** 0.1903)  # ~2000m
            alt_450hpa = 44330 * (1 - (450 / 1013.25) ** 0.1903)  # ~6500m
            
            # Ne pas dépasser les limites du graphique
            alt_min = max(0, self.site_altitude - 500)
            alt_max = min(12000, max(self.altitudes) + 500)
            
            # Marge pour les symboles de nuages
            margin = (temp_max - temp_min) * 0.05
            cloud_x_pos = temp_max - margin
            
            # Nuages bas
            if analysis.low_cloud_cover is not None:
                coverage = analysis.low_cloud_cover / 100.0  # Convertir en fraction
                if coverage > 0:
                    # Dessiner un symbole de nuage proportionnel à la couverture
                    y_low = (alt_min + alt_800hpa) / 2
                    cloud_size = coverage * 5  # Taille proportionnelle à la couverture
                    ax.text(cloud_x_pos, y_low, "☁️", fontsize=8+cloud_size*2, ha='center', va='center',
                        color='blue', alpha=0.7)
                    ax.text(cloud_x_pos, y_low - 200, f"{analysis.low_cloud_cover:.0f}%", fontsize=8, 
                        ha='center', va='top', color='blue')
            
            # Nuages moyens
            if analysis.mid_cloud_cover is not None:
                coverage = analysis.mid_cloud_cover / 100.0
                if coverage > 0:
                    y_mid = (alt_800hpa + alt_450hpa) / 2
                    cloud_size = coverage * 5
                    ax.text(cloud_x_pos, y_mid, "☁️", fontsize=8+cloud_size*2, ha='center', va='center',
                        color='darkblue', alpha=0.7)
                    ax.text(cloud_x_pos, y_mid - 200, f"{analysis.mid_cloud_cover:.0f}%", fontsize=8, 
                        ha='center', va='top', color='darkblue')
            
            # Nuages hauts
            if analysis.high_cloud_cover is not None:
                coverage = analysis.high_cloud_cover / 100.0
                if coverage > 0:
                    y_high = (alt_450hpa + alt_max) / 2
                    cloud_size = coverage * 5
                    ax.text(cloud_x_pos, y_high, "☁️", fontsize=8+cloud_size*2, ha='center', va='center',
                        color='purple', alpha=0.7)
                    ax.text(cloud_x_pos, y_high - 200, f"{analysis.high_cloud_cover:.0f}%", fontsize=8, 
                        ha='center', va='top', color='purple')
        
        # Paramètres pour l'analyse
        if analysis:
            # Température de déclenchement
            trigger_temp = analysis.ground_temperature + THERMAL_TRIGGER_DELTA
            
            # Tracer le chemin du thermique
            thermal_altitudes, thermal_temps = self._calculate_thermal_path(trigger_temp, self.site_altitude)
            ax.plot(thermal_temps, thermal_altitudes, 'g--', linewidth=1.5, label='Chemin du thermique')
            
            # Tracer le chemin du point de rosée
            dew_altitudes, dew_temps = self._calculate_dew_point_path(analysis.ground_dew_point, self.site_altitude)
            
            # Tracer le plafond des thermiques
            ax.axhline(y=analysis.thermal_ceiling, color='purple', linestyle='-.', linewidth=1.5, 
                    label=f'Plafond thermique ({analysis.thermal_ceiling:.0f}m)')
            
            # Tracer la base et le sommet des nuages si présents
            if analysis.cloud_base:
                ax.axhline(y=analysis.cloud_base, color='skyblue', linestyle='-.', linewidth=1.5,
                        label=f'Base des cumulus ({analysis.cloud_base:.0f}m)')
                
                # Chemin adiabatique humide
                if analysis.cloud_top:
                    moist_altitudes, moist_temps = self._calculate_moist_adiabatic_path(
                        analysis.cloud_base, thermal_temps, thermal_altitudes)
                    ax.plot(moist_temps, moist_altitudes, 'g-.', linewidth=1.5, label='Chemin après condensation')
                    
                    ax.axhline(y=analysis.cloud_top, color='navy', linestyle='-.', linewidth=1.5,
                            label=f'Sommet des cumulus ({analysis.cloud_top:.0f}m)')
            
            # Tracer les couches d'inversion
            for i, (base, top) in enumerate(analysis.inversion_layers):
                ax.axhspan(base, top, alpha=0.2, color='orange', 
                        label=f'Inversion {i+1} ({base:.0f}-{top:.0f}m)' if i == 0 else "")
            
            # Définir les limites de la couche convective
            conv_layer_base = analysis.ground_altitude
            conv_layer_top = analysis.thermal_ceiling
            anabatic_top = min(conv_layer_base + 500, conv_layer_top)

            # Code de débogage
            print(f"DEBUG - Sol: {conv_layer_base}, Plafond: {conv_layer_top}, Différence: {conv_layer_top - conv_layer_base}")

            # S'assurer que la couche a une épaisseur minimale pour être visible
            if conv_layer_top - conv_layer_base < 10:
                conv_layer_top = conv_layer_base + 10  # Garantir une épaisseur minimale

            # Ajouter la couche convective avec plus de visibilité
            ax.axhspan(conv_layer_base, conv_layer_top, alpha=0.05, color='limegreen', 
                    hatch='////', label='Couche convective', zorder=5)

            # Ajouter la couche anabatique avec plus de visibilité
            if anabatic_top > conv_layer_base:
                ax.axhspan(conv_layer_base, anabatic_top, alpha=0.3, color='skyblue', 
                        hatch='\\\\\\', label='Couche anabatique', zorder=6)

            # Ajouter des lignes de délimitation très visibles
            ax.axhline(y=conv_layer_base, color='darkgreen', linestyle='-', linewidth=2.0, alpha=0.9, zorder=7)
            ax.axhline(y=conv_layer_top, color='darkgreen', linestyle='-', linewidth=2.0, alpha=0.9, zorder=7)
            ax.axhline(y=anabatic_top, color='darkblue', linestyle='--', linewidth=2.0, alpha=0.9, zorder=7)

            # Ajouter des étiquettes textuelles plus visibles
            ax.text(temp_min + 5, conv_layer_base + 50, f"Sol: {conv_layer_base:.0f}m", 
                    fontsize=9, color='darkgreen', ha='left', va='bottom', 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'), zorder=8)
            ax.text(temp_min + 5, conv_layer_top - 50, f"Plafond: {conv_layer_top:.0f}m", 
                    fontsize=9, color='darkgreen', ha='left', va='top', 
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'), zorder=8)

            # Afficher les vents si disponibles
            if self.wind_directions is not None and self.wind_speeds is not None:
                # Sélectionner quelques niveaux pour afficher les vents
                indices = np.linspace(0, len(self.altitudes)-1, min(10, len(self.altitudes))).astype(int)
                
                # Définir la position de la barre avec décalage plus important vers la gauche
                bar_x_start = temp_max - 8  # Position de départ de la barre décalée vers la gauche
                bar_x_end = temp_max - 2     # Position de fin de la barre décalée vers la gauche
                bar_width = bar_x_end - bar_x_start
                
                # Dessiner une grille de référence pour les vents
                # Dessiner un fond rectangulaire pour la zone de vent
                rect_min_y = min(self.altitudes[indices[0]], alt_min)
                rect_max_y = max(self.altitudes[indices[-1]], alt_max)
                rect_height = rect_max_y - rect_min_y
                ax.add_patch(plt.Rectangle((bar_x_start - 0.5, rect_min_y - rect_height*0.05), 
                                        bar_width + 1, rect_height*1.1, 
                                        facecolor='lightskyblue', alpha=0.2, edgecolor='skyblue'))
                
                # Ligne de référence pour 0 km/h
                y_0km = rect_min_y - rect_height*0.02
                ax.plot([bar_x_start, bar_x_end], [y_0km, y_0km], 
                        'b-', linewidth=1, alpha=0.5)
                ax.text(bar_x_start - 0.3, y_0km, "0", fontsize=6, ha='right', va='center')
                
                # Ligne de référence pour 25 km/h
                y_25km = rect_min_y - rect_height*0.02 + rect_height*0.25
                ax.plot([bar_x_start, bar_x_end], [y_25km, y_25km], 
                        'b-', linewidth=1, alpha=0.5)
                ax.text(bar_x_start - 0.3, y_25km, "25", fontsize=6, ha='right', va='center')
                
                # Ligne de référence pour 50 km/h
                y_50km = rect_min_y - rect_height*0.02 + rect_height*0.5
                ax.plot([bar_x_start, bar_x_end], [y_50km, y_50km], 
                        'b-', linewidth=1, alpha=0.5)
                ax.text(bar_x_start - 0.3, y_50km, "50", fontsize=6, ha='right', va='center')
                
                # Titre pour la section du vent
                ax.text((bar_x_start + bar_x_end)/2, rect_max_y + rect_height*0.02, "VENT", 
                        fontsize=8, ha='center', va='bottom', fontweight='bold', color='darkblue')
                
                # Direction du vent (en haut)
                ax.text(bar_x_start, rect_max_y + rect_height*0.01, "Direction", 
                        fontsize=6, ha='left', va='bottom', color='darkblue')
                
                # Vitesse du vent (en bas)
                ax.text(bar_x_start, rect_min_y - rect_height*0.05, "Vitesse (km/h)", 
                        fontsize=6, ha='left', va='top', color='darkblue')
                
                # Dessiner les barres de vent pour chaque niveau
                for i in indices:
                    if not np.isnan(self.wind_directions[i]) and not np.isnan(self.wind_speeds[i]):
                        # Extraire les données de vent
                        wind_dir = self.wind_directions[i]
                        wind_speed = self.wind_speeds[i]
                        altitude = self.altitudes[i]
                        
                        # Dessiner une barre horizontale à la position du niveau d'altitude
                        ax.plot([bar_x_start, bar_x_end], [altitude, altitude], 'k-', linewidth=1.0)
                        
                        # Dessiner plusieurs barres verticales tous les 5 km/h
                        num_bars = int(wind_speed / 5)  # Une barre tous les 5 km/h
                        bar_spacing = bar_width / (num_bars + 1) if num_bars > 0 else bar_width / 2
                        
                        for b in range(num_bars):
                            # Position x de la barre
                            bar_x = bar_x_start + (b + 1) * bar_spacing
                            
                            # Hauteur fixe pour chaque barre (10% de la hauteur du rectangle)
                            bar_height = rect_height * 0.1
                            
                            # Couleur plus intense pour les vitesses plus élevées
                            intensity = min(0.5 + (b / num_bars * 0.5), 1.0) if num_bars > 0 else 0.5
                            bar_color = (intensity, 0, 0)  # Rouge plus intense pour des vitesses plus élevées
                            
                            # Dessiner la barre verticale avec une épaisseur plus fine
                            ax.plot([bar_x, bar_x], [altitude, altitude + bar_height], 
                                color=bar_color, linewidth=1.0)
                        
                        # Ajouter une flèche pour la direction
                        dir_rad = np.radians(wind_dir)
                        arrow_length = bar_width * 0.4
                        dx = arrow_length * np.sin(dir_rad)
                        dy = arrow_length * np.cos(dir_rad)
                        
                        # Dessiner la flèche de direction
                        ax.arrow(bar_x_start + bar_width/2, altitude,
                            dx, dy, head_width=bar_width*0.1, head_length=bar_width*0.15,
                            fc='blue', ec='blue', length_includes_head=True)
                        
                        # Ajouter les valeurs (décalées vers la gauche)
                        ax.text(bar_x_end + 0.2, altitude, f"{wind_speed:.0f}", fontsize=6, ha='left', va='center')
                        
                        # Direction en degrés ou nom (décalée vers la gauche)
                        dir_names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                                    "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
                        idx = round(wind_dir / 22.5) % 16
                        ax.text(bar_x_start - 0.2, altitude, f"{dir_names[idx]}", fontsize=6, ha='right', va='center')

        # Configurer les axes
        ax.set_ylim(alt_min, alt_max)
        ax.set_xlim(temp_min, temp_max)

        # Modifier la couleur et le style de l'axe d'altitude (gauche)
        ax.spines['left'].set_color('darkblue')  # Bordure gauche en bleu foncé
        ax.spines['left'].set_linewidth(1)      # Épaisseur plus importante
        ax.tick_params(axis='y', colors='darkblue', labelsize=10, width=2)  # Graduations bleues et plus grandes
        ax.set_ylabel('Altitude (m)', color='darkblue', fontweight='bold', fontsize=12)  # Étiquette bleue et en gras

        # Ajouter une deuxième échelle pour les pressions (droite)
        ax2 = ax.twinx()
        ax2.spines['right'].set_color('darkred')  # Bordure droite en rouge foncé
        ax2.spines['right'].set_linewidth(1)     # Épaissir la bordure
        ax2.tick_params(axis='y', colors='darkred', labelsize=10, width=2)  # Graduations rouges
        ax2.set_ylabel('Pression (hPa)', color='darkred', fontweight='bold', fontsize=12)  # Étiquette rouge et en gras

        # Définir les limites en pression
        p_min = 1013.25 * (1 - (alt_max/44330)) ** 5.255
        p_max = 1013.25 * (1 - (alt_min/44330)) ** 5.255
        ax2.set_ylim(p_min, p_max)
        
        # Ajouter les informations sur les nuages dans le titre
        model_info = f" - Modèle: {analysis.model_name}" if analysis.model_name else ""
        title = f"Émagramme - Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}{model_info}\n"
        
        if analysis:
            # Créer un texte pour le spread
            spread_text = f"ANALYSE DU SPREAD (T° - Td)\n"
            spread_text += f"Au sol: {analysis.ground_spread:.1f}°C\n"
            
            if analysis.spread_levels:
                for level, value in analysis.spread_levels.items():
                    spread_text += f"Niveau {level}: {value:.1f}°C\n"
            
            spread_text += f"\n{analysis.spread_analysis}"
            
            # Position du texte
            spread_box_x = temp_min + (temp_max - temp_min) * 0.6
            spread_box_y = alt_max - (alt_max - alt_min) * 0.3
            
            # Ajouter le texte avec un fond
            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            ax.text(spread_box_x, spread_box_y, spread_text, fontsize=8, 
                    bbox=props, verticalalignment='bottom', horizontalalignment='left', zorder=100)
            
            # Ajouter une visualisation des couches atmosphériques
            if hasattr(analysis, 'atmospheric_layers'):
                layers_text = "COUCHES ATMOSPHÉRIQUES\n"
                
                for layer_name, layer_data in analysis.atmospheric_layers.items():
                    # Utiliser les limites théoriques au lieu des altitudes calculées
                    bottom = layer_data.get("theoretical_bottom")
                    top = layer_data.get("theoretical_top")
                    
                    # Si l'une des limites est 'inf', utiliser une valeur plus lisible
                    if top == float('inf'):
                        top = 6000  # Ou une autre valeur représentative de la haute atmosphère
                    
                    layers_text += f"{layer_name.capitalize()}: {bottom:.0f}-{top:.0f}m\n"
                    
                    if "mean_wind" in layer_data and layer_data["mean_wind"] is not None:
                        layers_text += f"  Vent moyen: {layer_data['mean_wind']:.0f} km/h"
                        if "dominant_dir" in layer_data:
                            layers_text += f" ({layer_data['dominant_dir']})\n"
                        else:
                            layers_text += "\n"
                    
                    if layer_data["stability"] != "données insuffisantes":
                        layers_text += f"  Stabilité: {layer_data['stability']}\n"
                    else:
                        layers_text += "  Stabilité: données insuffisantes\n"
                
                # Position du texte des couches
                layers_box_x = temp_min + (temp_max - temp_min) * 0.6
                layers_box_y = alt_min + (alt_max - alt_min) * 0.5
                
                # Ajouter le texte des couches
                props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
                ax.text(layers_box_x, layers_box_y, layers_text, fontsize=8, 
                        bbox=props, verticalalignment='bottom', horizontalalignment='left', zorder=100)
                
        if analysis:
            title += f"Site: {analysis.ground_altitude:.0f}m, T°: {analysis.ground_temperature:.1f}°C, "
            title += f"Point de rosée: {analysis.ground_dew_point:.1f}°C\n"
            title += f"Plafond thermique: {analysis.thermal_ceiling:.0f}m, "
            
            if analysis.thermal_type == "Cumulus":
                title += f"Base nuages: {analysis.cloud_base:.0f}m, Sommet: {analysis.cloud_top:.0f}m\n"
            else:
                if analysis.thermal_inconsistency:
                    title += "Thermiques bleus mais présence de nuages stratiformes\n"
                else:
                    title += "Thermiques bleus (pas de condensation)\n"
                
            title += f"Gradient: {analysis.thermal_gradient:.1f}°C/1000m, "
            title += f"Force thermiques: {analysis.thermal_strength}, "
            title += f"Stabilité: {analysis.stability}"
            
            # Ajouter les informations sur la couverture nuageuse
            if analysis.low_cloud_cover is not None or analysis.mid_cloud_cover is not None or analysis.high_cloud_cover is not None:
                title += "\nCouverture nuageuse: "
                cloud_parts = []
                
                # Créer le texte d'information sur les nuages
                cloud_info_text = ""
                
                if analysis.low_cloud_cover is not None:
                    cloud_info_text += f"Nuages bas: {analysis.low_cloud_cover:.0f}%\n"
                    cloud_parts.append(f"bas {analysis.low_cloud_cover:.0f}%")
                
                if analysis.mid_cloud_cover is not None:
                    cloud_info_text += f"Nuages moyens: {analysis.mid_cloud_cover:.0f}%\n"
                    cloud_parts.append(f"moyens {analysis.mid_cloud_cover:.0f}%")
                
                if analysis.high_cloud_cover is not None:
                    cloud_info_text += f"Nuages hauts: {analysis.high_cloud_cover:.0f}%\n"
                    cloud_parts.append(f"hauts {analysis.high_cloud_cover:.0f}%")
                
                # Ajouter le texte au graphique
                props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                cloud_text_y = (alt_min + alt_max)/2 - 2000
                ax.text(temp_max - 15, cloud_text_y, cloud_info_text, fontsize=8, 
                    verticalalignment='center', horizontalalignment='right', bbox=props)
                
                # Optionnel: représenter visuellement les nuages avec des symboles
                x_cloud = temp_max - 3
                if analysis.low_cloud_cover is not None and analysis.low_cloud_cover > 10:
                    y_low = alt_min + 1000
                    ax.text(x_cloud, y_low, "☁️", fontsize=12, ha='center', color='blue', alpha=0.7)
                
                if analysis.mid_cloud_cover is not None and analysis.mid_cloud_cover > 10:
                    y_mid = alt_min + 3000
                    ax.text(x_cloud, y_mid, "☁️", fontsize=12, ha='center', color='royalblue', alpha=0.7)
                
                if analysis.high_cloud_cover is not None and analysis.high_cloud_cover > 10:
                    y_high = alt_min + 5000
                    ax.text(x_cloud, y_high, "☁️", fontsize=12, ha='center', color='purple', alpha=0.7)
                    
                title += ", ".join(cloud_parts)
        
        ax.set_title(title)
        
        # Légende
        # D'abord, récupérez les handles et labels de la légende
        handles, labels = ax.get_legend_handles_labels()

        # Retirez toute légende existante
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # Ajouter un texte avec les résultats d'analyse
        if analysis:
            
            # Ajouter également une légende à l'émagramme
            if len(handles) > 0:
                legend = ax.legend(handles, labels, loc='upper left', fontsize=8)
                legend.set_zorder(100)
        
        # Ajuster l'espacement entre les sous-graphiques
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)
        
        # Sauvegarder l'image
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Afficher le graphique
        if show:
            plt.show()
        
        self.altitudes = original_altitudes
        self.temperatures = original_temperatures
        self.dew_points = original_dew_points
        self.pressures = original_pressures
        self.wind_directions = original_wind_directions
        self.wind_speeds = original_wind_speeds

        return fig
    
# 4. Classe EmagrammeAgent (après EmagrammeAnalysis)
class EmagrammeAgent:
    """
    Agent IA qui utilise un LLM pour analyser et commenter un émagramme
    Fournit des descriptions en langage naturel adaptées aux parapentistes
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
    
    def analyze_conditions(self, analysis: EmagrammeAnalysis) -> str:
        """
        Utilise le LLM pour générer une analyse détaillée des conditions de vol
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Description détaillée en langage naturel
        """
        # Vérifier si le vol est impossible
        vol_impossible = False
        raisons_impossibilite = []
        
        # Vérifier les précipitations
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            vol_impossible = True
            raisons_impossibilite.append(analysis.precipitation_description)
        
        # Vérifier les vents forts - en utilisant les bonnes propriétés
        if "fort au sol" in analysis.wind_conditions.lower() or "critique" in analysis.wind_conditions.lower():
            vol_impossible = True
            raisons_impossibilite.append("Vent trop fort")
        
        if not self.has_openai:
            raw_analysis = self._generate_fallback_analysis(analysis)
            return self.post_process_llm_analysis(raw_analysis, vol_impossible)

        try:
            # Construire le prompt pour l'API
            prompt = self._build_prompt(analysis)
            
            # Appeler l'API
            response = self.openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Ou un autre modèle adapté
                messages=[
                    {"role": "system", "content": """Tu es un expert en parapente et météorologie qui analyse 
                    les émagrammes pour fournir des conseils de vol précis et utiles.
                    Utilise un ton pédagogique mais direct, et concentre-toi sur les informations 
                    pratiques pour les pilotes."""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API LLM: {e}")
            return self._generate_fallback_analysis(analysis)
    
    def post_process_llm_analysis(self, analysis_text: str, vol_impossible: bool) -> str:
        """
        Vérifie et corrige les incohérences dans l'analyse du LLM
        
        Args:
            analysis_text: Texte brut de l'analyse générée par le LLM
            vol_impossible: Booléen indiquant si le vol est impossible
            
        Returns:
            Texte de l'analyse post-traité et corrigé
        """
        import re
        
        if vol_impossible:
            # Supprimer tout contenu mentionnant des "heures optimales"
            analysis_text = re.sub(r"(?i)Heures optimales.*?\.(\n|$)", "", analysis_text)
            
            # Supprimer les stratégies recommandées
            analysis_text = re.sub(r"(?i)Stratégie recommandée.*?(\n\n|\Z)", "", analysis_text)
            
            # Supprimer le niveau de difficulté
            analysis_text = re.sub(r"(?i)Niveau de difficulté.*?\.(\n|$)", "", analysis_text)
            
            # S'assurer que l'avertissement sur l'impossibilité de vol est bien présent
            if "VOL IMPOSSIBLE" not in analysis_text:
                analysis_text = "⚠️ VOL IMPOSSIBLE - Conditions météorologiques dangereuses.\n\n" + analysis_text
        
        return analysis_text
    
    def _build_prompt(self, analysis: EmagrammeAnalysis) -> str:
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

    Ta réponse doit SEULEMENT:
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
    Fournis une analyse détaillée avec:
    1. Une évaluation générale de la journée pour le vol en parapente
    2. Des conseils sur les heures optimales pour voler
    3. Des stratégies pour exploiter au mieux les thermiques
    4. Des mises en garde sur les dangers potentiels
    5. Des suggestions sur les sites de vol adaptés à ces conditions
    6. Une conclusion sur le niveau de difficulté (débutant, intermédiaire, avancé)

    Ton analyse doit être concise, pratique et directement utilisable par un pilote.
    """
        return prompt
    
    def _generate_fallback_analysis(self, analysis: EmagrammeAnalysis) -> str:
        """
        Génère une analyse basique sans LLM en cas d'erreur ou d'absence d'API
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Description basique en langage naturel
        """
        # Vérifier si des précipitations sont prévues ou d'autres conditions qui rendent le vol impossible
        vol_impossible = False
        raisons_impossibilite = []
        
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            vol_impossible = True
            raisons_impossibilite.append(analysis.precipitation_description)
        
        # Vérifier si le vent est trop fort en utilisant la description textuelle
        if "fort au sol" in analysis.wind_conditions.lower() or "critique" in analysis.wind_conditions.lower():
            vol_impossible = True
            raisons_impossibilite.append("Vent trop fort")
        
        if vol_impossible:
            # Construire une réponse d'avertissement
            raisons = ", ".join(raisons_impossibilite)
            response = f"""## ⚠️ VOL IMPOSSIBLE - {raisons}

    ### Conditions météorologiques dangereuses
    {analysis.flight_conditions}

    ### Conditions de vent
    {analysis.wind_conditions}
    """

            if analysis.hazards:
                response += "\n### Dangers spécifiques\n"
                for hazard in analysis.hazards:
                    response += f"- {hazard}\n"
                    
            response += """
    ### Recommandation
    Il est fortement recommandé de ne pas voler dans ces conditions et d'attendre une amélioration de la météo.
    """
            return response
        
        # Si le vol est possible, continuer avec l'analyse normale
        
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
            
        # Construire la réponse
        response = f"""## Analyse de l'émagramme pour le vol en parapente

    ### Évaluation générale
    {analysis.flight_conditions}

    ### Conditions thermiques
    - Plafond: {analysis.thermal_ceiling:.0f}m
    - Force: {analysis.thermal_strength}
    - Gradient: {analysis.thermal_gradient:.1f}°C/1000m
    """

        if analysis.thermal_type == "Cumulus":
            response += f"- Cumulus de {analysis.cloud_base:.0f}m à {analysis.cloud_top:.0f}m\n"
        else:
            response += "- Thermiques bleus (pas de marquage nuageux)\n"
            
        response += f"""
    ### Vent
    {analysis.wind_conditions}

    ### Heures optimales
    Les meilleures conditions devraient se trouver {best_hours}.

    ### Stratégie recommandée
    """
        if analysis.thermal_strength in ["Faible", "Modérée"]:
            response += "- Privilégier les faces bien exposées au soleil\n"
            response += "- Patienter dans les thermiques faibles, ne pas abandonner trop vite\n"
            response += "- Se concentrer sur les zones de déclenchement connues\n"
        else:
            response += "- Prudence dans les thermiques puissants\n"
            response += "- Anticiper les surpuissances et les fermetures possibles\n"
            response += "- Éviter les heures les plus chaudes si vous manquez d'expérience\n"
            
        if analysis.hazards:
            response += "\n### ⚠️ Mises en garde\n"
            for hazard in analysis.hazards:
                response += f"- {hazard}\n"
                
        response += f"""
    ### Équipement recommandé
    {', '.join(analysis.recommended_gear)}

    ### Niveau de difficulté
    Conditions adaptées aux pilotes de niveau {difficulty}.
    """
        return response

# 5. Classe EmagrammeDataFetcher
class EmagrammeDataFetcher:
    """
    Classe pour récupérer les données d'émagramme à partir de différentes sources
    Supporte actuellement: Meteociel
    """
    
    def __init__(self, api_key=None):
        """
        Initialise le récupérateur de données avec une clé API optionnelle
        
        Args:
            api_key: Clé API pour les services météo
        """
        self.api_key = api_key
    

    def fetch_from_openmeteo(self, latitude, longitude, model="meteofrance_arome_france_hd", timestep=0, fetch_evolution=False, evolution_hours=24, evolution_step=3):
        """
        Récupère les données d'émagramme depuis l'API Open-Meteo avec support multi-horaire
        
        Args:
            latitude: Latitude du point d'intérêt
            longitude: Longitude du point d'intérêt
            model: Modèle météo à utiliser ('meteofrance_arome_france_hd', 'meteofrance_arpege_europe')
            timestep: Pas de temps pour la prévision (0 = analyse, 1-36 pour AROME, 1-96 pour ARPEGE)
            fetch_evolution: Si True, récupère les données pour plusieurs heures 
            evolution_hours: Nombre d'heures total pour l'évolution temporelle
            evolution_step: Intervalle (en heures) entre chaque point de données d'évolution
            
        Returns:
            Liste d'objets AtmosphericLevel et informations sur les nuages.
            Si fetch_evolution=True, retourne également un dictionnaire de données d'évolution
        """
        try:
            # Importer les packages nécessaires
            try:
                import openmeteo_requests
                import requests_cache
                from retry_requests import retry
            except ImportError as e:
                logger.error(f"Packages requis non installés: {e}")
                logger.error("Installez les packages avec 'pip install openmeteo-requests requests-cache retry-requests'")
                raise ImportError("Packages requis pour Open-Meteo non installés")
            
            # Déterminer la durée des prévisions selon le modèle
            if model.startswith("meteofrance_arome"):
                max_timestep = 36
                forecast_days = 2  # 36 heures = 1.5 jours, arrondi à 2
            else:  # ARPEGE
                max_timestep = 96
                forecast_days = 4  # 96 heures = 4 jours
            
            # Valider le timestep
            timestep = min(max(0, timestep), max_timestep)
            
            logger.info(f"Récupération des données via Open-Meteo (modèle {model}, heure +{timestep}h)")
            
            # Setup the Open-Meteo API client with cache and retry
            cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
            retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
            openmeteo = openmeteo_requests.Client(session=retry_session)
            
            # Définir les niveaux de pression standard
            pressure_levels = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30]
            
            # Variables combinées pour les différents niveaux de pression
            hourly_vars = [
                "temperature_2m", "dew_point_2m", "relative_humidity_2m",
                "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
                "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
                "precipitation_probability", "rain"
            ]
            
            # Ajouter les variables à différents niveaux de pression
            for level in pressure_levels:
                hourly_vars.extend([
                    f"temperature_{level}hPa",
                    f"relative_humidity_{level}hPa",
                    f"wind_speed_{level}hPa",
                    f"wind_direction_{level}hPa",
                    f"geopotential_height_{level}hPa"
                ])
            
            # Construire les paramètres de requête
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "hourly": hourly_vars,
                "models": [model],
                "forecast_days": forecast_days
            }
            
            # Effectuer la requête API
            try:
                logger.info("Envoi de la requête à Open-Meteo...")
                responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
                response = responses[0]
                logger.info(f"Réponse reçue pour les coordonnées {response.Latitude()}°N {response.Longitude()}°E")
            except Exception as e:
                logger.error(f"Erreur lors de la requête API Open-Meteo: {e}")
                raise
            
            # Récupérer les données horaires
            hourly = response.Hourly()
            
            # Construire un dictionnaire pour accéder facilement aux variables par nom
            hourly_data = {}
            
            # Créer la timeline
            hourly_data["date"] = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
            
            # Extraire chaque variable par index
            for i in range(hourly.VariablesLength()):
                var = hourly.Variables(i)
                # Nous extrayons directement les valeurs sans utiliser Name()
                hourly_data[hourly_vars[i]] = var.ValuesAsNumpy()
            
            # Créer un DataFrame pandas
            hourly_df = pd.DataFrame(data=hourly_data)
            
            # Récupérer les données pour le timestep spécifié
            current_data = hourly_df.iloc[min(timestep, len(hourly_df)-1)]
            
            # Initialiser les niveaux atmosphériques
            levels = []
            
            # Informations pour la surface
            self.cloud_info = {}
            
            # Récupérer les informations sur les nuages si disponibles
            if "cloud_cover_low" in current_data and not pd.isna(current_data["cloud_cover_low"]):
                self.cloud_info["low_clouds"] = float(current_data["cloud_cover_low"])
            else:
                self.cloud_info["low_clouds"] = 0.0
                
            if "cloud_cover_mid" in current_data and not pd.isna(current_data["cloud_cover_mid"]):
                self.cloud_info["mid_clouds"] = float(current_data["cloud_cover_mid"])
            else:
                self.cloud_info["mid_clouds"] = 0.0
                
            if "cloud_cover_high" in current_data and not pd.isna(current_data["cloud_cover_high"]):
                self.cloud_info["high_clouds"] = float(current_data["cloud_cover_high"])
            else:
                self.cloud_info["high_clouds"] = 0.0
            
            # Information sur les précipitations
            self.precip_info = {
                "type": 0,  # 0 = pas de précipitation par défaut
                "description": "Pas de précipitation"
            }
            
            # Analyser les précipitations si les données sont disponibles
            if "rain" in current_data and not pd.isna(current_data["rain"]):
                rain_value = float(current_data["rain"])
                
                if "precipitation_probability" in current_data and not pd.isna(current_data["precipitation_probability"]):
                    precip_prob = float(current_data["precipitation_probability"]) 
                else:
                    precip_prob = 0
                    
                if rain_value > 0.5 or precip_prob > 50:
                    self.precip_info = {
                        "type": 1,  # 1 = pluie
                        "description": "Pluie"
                    }
                    
                    # Si température < 0, considérer comme neige
                    if "temperature_2m" in current_data and not pd.isna(current_data["temperature_2m"]) and float(current_data["temperature_2m"]) < 0:
                        self.precip_info = {
                            "type": 5,  # 5 = neige
                            "description": "Neige"
                        }
                    elif "temperature_2m" in current_data and not pd.isna(current_data["temperature_2m"]) and float(current_data["temperature_2m"]) < 3:
                        self.precip_info = {
                            "type": 7,  # 7 = mélange pluie et neige
                            "description": "Mélange pluie et neige"
                        }
            
            # Créer un niveau pour la surface
            surface_altitude = self.site_altitude if hasattr(self, 'site_altitude') and self.site_altitude is not None else 0
            surface_pressure = 1013.25  # Pression standard au niveau de la mer
            
            # Estimer la pression au niveau du site si l'altitude est connue
            if surface_altitude > 0:
                # Formule barométrique inverse
                surface_pressure = 1013.25 * (1 - (surface_altitude / 44330)) ** 5.255
            
            # Paramètres de surface
            if "temperature_2m" in current_data and not pd.isna(current_data["temperature_2m"]):
                surface_temp = float(current_data["temperature_2m"])
            else:
                surface_temp = 15.0  # Valeur par défaut si non disponible
                
            if "dew_point_2m" in current_data and not pd.isna(current_data["dew_point_2m"]):
                surface_dew_point = float(current_data["dew_point_2m"])
            else:
                surface_dew_point = 10.0  # Valeur par défaut si non disponible
                
            if "wind_direction_10m" in current_data and not pd.isna(current_data["wind_direction_10m"]):
                surface_wind_dir = float(current_data["wind_direction_10m"])
            else:
                surface_wind_dir = 0.0  # Valeur par défaut si non disponible
                
            if "wind_speed_10m" in current_data and not pd.isna(current_data["wind_speed_10m"]):
                surface_wind_speed = float(current_data["wind_speed_10m"])
            else:
                surface_wind_speed = 0.0  # Valeur par défaut si non disponible
            
            # Créer le niveau de surface
            surface_level = AtmosphericLevel(
                altitude=surface_altitude,
                pressure=surface_pressure,
                temperature=surface_temp,
                dew_point=surface_dew_point,
                wind_direction=surface_wind_dir,
                wind_speed=surface_wind_speed
            )
            
            levels.append(surface_level)
            
            # Niveaux de pression disponibles
            available_levels = {}
            
            # Vérifier quels niveaux de pression sont disponibles
            for pressure in pressure_levels:
                temp_key = f"temperature_{pressure}hPa"
                geo_key = f"geopotential_height_{pressure}hPa"
                rh_key = f"relative_humidity_{pressure}hPa"
                wspd_key = f"wind_speed_{pressure}hPa"
                wdir_key = f"wind_direction_{pressure}hPa"
                
                if temp_key in current_data and not pd.isna(current_data[temp_key]):
                    # Récupérer la température
                    temp = float(current_data[temp_key])
                    
                    # Altitude géopotentielle - utiliser la valeur si disponible, sinon estimer
                    if geo_key in current_data and not pd.isna(current_data[geo_key]):
                        altitude = float(current_data[geo_key])
                    else:
                        # Formule barométrique simplifiée si non disponible
                        altitude = 44330 * (1 - (pressure / 1013.25) ** 0.1903)
                    
                    # Ignorer si l'altitude est inférieure à l'altitude de surface
                    if altitude <= surface_altitude:
                        continue
                    
                    # Humidité relative - utiliser la valeur si disponible, sinon valeur par défaut
                    if rh_key in current_data and not pd.isna(current_data[rh_key]):
                        rh = float(current_data[rh_key])
                    else:
                        rh = 50.0  # Valeur par défaut
                    
                    # Calculer le point de rosée à partir de RH et température
                    # Formule de Magnus
                    alpha = 17.27
                    beta = 237.7
                    gamma = (alpha * temp) / (beta + temp) + np.log(rh/100.0)
                    dew_point = (beta * gamma) / (alpha - gamma)
                    
                    # Limiter le point de rosée à la température
                    if dew_point > temp:
                        dew_point = temp
                    
                    # Vent - utiliser les valeurs si disponibles, sinon valeurs par défaut
                    if wspd_key in current_data and not pd.isna(current_data[wspd_key]):
                        wind_speed = float(current_data[wspd_key])
                    else:
                        wind_speed = surface_wind_speed + 5 * (altitude - surface_altitude) / 1000
                    
                    if wdir_key in current_data and not pd.isna(current_data[wdir_key]):
                        wind_direction = float(current_data[wdir_key])
                    else:
                        wind_direction = (surface_wind_dir + 20 * (altitude - surface_altitude) / 1000) % 360
                    
                    # Stocker ce niveau
                    available_levels[altitude] = {
                        "pressure": pressure,
                        "temperature": temp,
                        "dew_point": dew_point,
                        "wind_direction": wind_direction,
                        "wind_speed": wind_speed
                    }
            
            # Si nous n'avons pas assez de niveaux, générer des niveaux supplémentaires
            if len(available_levels) < 3:
                logger.warning(f"Seulement {len(available_levels)} niveaux valides trouvés, génération de niveaux supplémentaires")
                
                # Générer des niveaux tous les 1000m jusqu'à 10000m
                for altitude in range(1000, 11000, 1000):
                    if altitude not in available_levels and altitude > surface_altitude:
                        # Estimer la pression avec la formule barométrique
                        pressure = 1013.25 * (1 - (altitude / 44330)) ** 5.255
                        
                        # Estimer la température avec un gradient standard de -6.5°C/km
                        temp = surface_temp - 6.5 * (altitude - surface_altitude) / 1000
                        
                        # Estimer le point de rosée (plus sec en altitude)
                        dew_point = surface_dew_point - 8 * (altitude - surface_altitude) / 1000
                        
                        # Augmenter la vitesse du vent avec l'altitude
                        wind_speed = surface_wind_speed + 5 * (altitude - surface_altitude) / 1000
                        
                        # Rotation légère du vent avec l'altitude (vers la droite)
                        wind_direction = (surface_wind_dir + 20 * (altitude - surface_altitude) / 1000) % 360
                        
                        # Ajouter ce niveau estimé
                        available_levels[altitude] = {
                            "pressure": pressure,
                            "temperature": temp,
                            "dew_point": dew_point,
                            "wind_direction": wind_direction,
                            "wind_speed": wind_speed
                        }
            
            # Ajouter tous les niveaux à notre liste
            for altitude, data in sorted(available_levels.items()):
                level = AtmosphericLevel(
                    altitude=altitude,
                    pressure=data["pressure"],
                    temperature=data["temperature"],
                    dew_point=data["dew_point"],
                    wind_direction=data["wind_direction"],
                    wind_speed=data["wind_speed"]
                )
                levels.append(level)
            
            # Trier les niveaux par altitude croissante
            levels.sort(key=lambda x: x.altitude)
            
            # Si nous n'avons toujours pas assez de niveaux, lever une exception
            if len(levels) < 4:  # Surface + au moins 3 niveaux
                raise ValueError(f"Données de profil vertical insuffisantes dans la réponse Open-Meteo: seulement {len(levels)} niveaux")
            
            logger.info(f"Données récupérées: {len(levels)} niveaux atmosphériques de Open-Meteo")

            # Si fetch_evolution est activé, extraire les données pour l'évolution temporelle
            evolution_data = None
            if fetch_evolution:
                evolution_data = {
                    "timestamps": [],
                    "thermal_ceilings": [],
                    "thermal_gradients": [],
                    "thermal_strengths": [],
                    "cloud_covers": [],
                    "precipitation": [],
                    "temperatures": [],
                    "wind_speeds": [],
                    "wind_directions": []
                }
                
                # Déterminer les indices pour l'évolution
                start_idx = 0  # Toujours commencer à l'heure actuelle
                end_idx = min(timestep + 1, len(hourly_df))
                
                # Déterminer l'intervalle d'évolution en fonction du modèle
                # ECMWF a une résolution de 3h, donc on doit prendre un pas de 3 au minimum
                evolution_interval = max(3, evolution_step) if model.startswith("ecmwf_ifs") else evolution_step
                
                # Vérifier s'il y a assez de données pour l'évolution
                if end_idx - start_idx < 2:
                    # Pas assez de données pour une évolution significative
                    evolution_data["error"] = "insufficient_data"
                    evolution_data["message"] = f"Le modèle {model} ne fournit pas assez de données pour l'évolution"
                    return levels, evolution_data
                
                # Extraire les données pour chaque pas de temps
                for i in range(start_idx, end_idx, evolution_interval):
                    if i < len(hourly_df):
                        # Ajouter le timestamp
                        evolution_data["timestamps"].append(hourly_df["date"].iloc[i])
                        
                        # Extraire et calculer les données météo pour ce timestamp
                        temp_data = hourly_df.iloc[i]
                        
                        # Température et humidité de surface
                        evolution_data["temperatures"].append(temp_data.get("temperature_2m", 0))
                        
                        # Vent de surface
                        evolution_data["wind_speeds"].append(temp_data.get("wind_speed_10m", 0))
                        evolution_data["wind_directions"].append(temp_data.get("wind_direction_10m", 0))
                        
                        # Couverture nuageuse
                        cloud_cover = {
                            "low": temp_data.get("cloud_cover_low", 0),
                            "mid": temp_data.get("cloud_cover_mid", 0),
                            "high": temp_data.get("cloud_cover_high", 0),
                            "total": temp_data.get("cloud_cover", 0)
                        }
                        evolution_data["cloud_covers"].append(cloud_cover)
                        
                        # Précipitations
                        rain = temp_data.get("rain", 0)
                        precip_prob = temp_data.get("precipitation_probability", 0)
                        evolution_data["precipitation"].append({
                            "rain": rain,
                            "probability": precip_prob
                        })
                        
                        # Pour les autres métriques (plafond thermique, gradient, force), on utilise
                        # une estimation simplifiée basée sur les données disponibles
                        # Cela sera remplacé par une vraie analyse lors de l'affichage
                        
                        # Pour le plafond thermique, on utilise une estimation grossière
                        # basée sur la différence de température entre la surface et 850hPa
                        t_surface = temp_data.get("temperature_2m", 15)
                        t_850 = temp_data.get("temperature_850hPa", t_surface - 15)  # Par défaut ~15°C plus froid
                        if not pd.isna(t_surface) and not pd.isna(t_850):
                            est_gradient = (t_surface - t_850) / 15  # ~1500m différence
                            est_ceiling = 1000 + 3000 * min(1, est_gradient / 10)  # Estimation grossière
                            evolution_data["thermal_ceilings"].append(est_ceiling)
                            evolution_data["thermal_gradients"].append(est_gradient * 10)  # En °C/1000m
                            
                            # Force thermique estimée
                            if est_gradient < 5:
                                evolution_data["thermal_strengths"].append("Faible")
                            elif est_gradient < 7:
                                evolution_data["thermal_strengths"].append("Modérée")
                            elif est_gradient < 9:
                                evolution_data["thermal_strengths"].append("Forte")
                            else:
                                evolution_data["thermal_strengths"].append("Très Forte")
                        else:
                            # Valeurs par défaut
                            evolution_data["thermal_ceilings"].append(2000)
                            evolution_data["thermal_gradients"].append(6.5)
                            evolution_data["thermal_strengths"].append("Modérée")
            

            if fetch_evolution:
                # Vérifier si l'évolution couvre bien toute la période demandée
                expected_points = (timestep // evolution_step) + 1
                actual_points = len(evolution_data["timestamps"])
                
                # Attacher une métadonnée de couverture
                evolution_data["coverage"] = {
                    "expected_points": expected_points,
                    "actual_points": actual_points,
                    "coverage_ratio": actual_points / expected_points if expected_points > 0 else 0,
                    "timestep": timestep,
                    "max_available_time": actual_points * evolution_step if actual_points > 0 else 0
                }
                return levels, evolution_data
            else:
                return levels
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données Open-Meteo: {e}")
            raise
    
    def parse_csv_data(self, csv_text):
        """
        Parse les données d'émagramme au format CSV
        Format attendu: altitude,pressure,temperature,dew_point[,wind_dir,wind_speed]
        
        Args:
            csv_text: Texte CSV contenant les données
            
        Returns:
            Liste d'objets AtmosphericLevel
        """
        levels = []
        
        try:
            # Utiliser pandas pour parser le CSV
            df = pd.read_csv(StringIO(csv_text))
            
            # Vérifier les colonnes requises
            required_cols = ["altitude", "pressure", "temperature", "dew_point"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Colonne requise manquante dans le CSV: {col}")
            
            # Extraire les données
            for _, row in df.iterrows():
                level = AtmosphericLevel(
                    altitude=row["altitude"],
                    pressure=row["pressure"],
                    temperature=row["temperature"],
                    dew_point=row["dew_point"],
                    wind_direction=row.get("wind_dir") if "wind_dir" in df.columns else None,
                    wind_speed=row.get("wind_speed") if "wind_speed" in df.columns else None
                )
                
                levels.append(level)
                
            return levels
            
        except Exception as e:
            logger.error(f"Erreur lors du parsing des données CSV: {e}")
            raise

    def load_from_file(self, filename):
        """
        Charge les données d'émagramme depuis un fichier local
        
        Args:
            filename: Chemin vers le fichier CSV ou JSON
            
        Returns:
            Liste d'objets AtmosphericLevel
        """
        try:
            with open(filename, 'r') as file:
                if filename.endswith('.csv'):
                    return self.parse_csv_data(file.read())
                elif filename.endswith('.json'):
                    data = json.load(file)
                    levels = []
                    
                    for item in data:
                        level = AtmosphericLevel(
                            altitude=item["altitude"],
                            pressure=item["pressure"],
                            temperature=item["temperature"],
                            dew_point=item["dew_point"],
                            wind_direction=item.get("wind_direction"),
                            wind_speed=item.get("wind_speed")
                        )
                        
                        levels.append(level)
                        
                    return levels
                else:
                    raise ValueError("Format de fichier non supporté. Utilisez CSV ou JSON.")
                    
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier {filename}: {e}")
            raise

# Interface pour la ligne de commande
def main():
    """Point d'entrée principal du programme"""
    # Déclaration globale dès le début de la fonction
    global THERMAL_TRIGGER_DELTA

    parser = argparse.ArgumentParser(description="Analyseur d'Émagramme pour Parapentistes")

    # Options principales
    parser.add_argument('-l', '--location', help='Coordonnées (lat,lon) pour récupérer les données', required=True)
    parser.add_argument('-s', '--site-altitude', type=float, help='Altitude du site de décollage en mètres')

    # Options d'affichage
    parser.add_argument('-p', '--plot', action='store_true', help='Afficher l\'émagramme graphiquement')
    parser.add_argument('-o', '--output', help='Chemin pour sauvegarder l\'image de l\'émagramme')
    parser.add_argument('-v', '--verbose', action='store_true', help='Afficher plus d\'informations')

    # Options avancées
    parser.add_argument('--api-key', help='Clé API (open-Meteo, etc.)') # Garder au cas où Meteo磨礪 en aurait besoin un jour
    parser.add_argument('--openai-key', help='Clé API OpenAI pour l\'analyse avancée')
    parser.add_argument('--delta-t', type=float, default=THERMAL_TRIGGER_DELTA,
                        help='Delta de température pour le déclenchement des thermiques (°C)')
    parser.add_argument('--model', default="AROME", choices=["AROME", "ARPEGE"], help='Modèle météo à utiliser (AROME, ARPEGE)')


    args = parser.parse_args()

    # Configuration du niveau de verbosité
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialiser le récupérateur de données
    fetcher = EmagrammeDataFetcher(api_key=args.api_key)

    # Récupérer les données
    levels = []
    cloud_info = None
    precip_info = None

    if args.location:
        try:
            lat, lon = map(float, args.location.split(','))

            # Récupérer les informations sur les nuages et précipitations depuis le fetcher
            cloud_info = None
            precip_info = None

            # Vérifier si on utilise l'API MeteoFetch
            logger.info(f"Récupération des données via MeteoFetch pour la position {lat}, {lon} avec le modèle {args.model}")
            levels = fetcher.fetch_from_meteofrance(lat, lon, model=args.model)


            if hasattr(fetcher, 'cloud_info'):
                cloud_info = fetcher.cloud_info
            if hasattr(fetcher, 'precip_info'):
                precip_info = fetcher.precip_info


        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données: {e}")
            sys.exit(1)
    else:
        logger.error("Veuillez spécifier une localisation")
        parser.print_help()
        sys.exit(1)

    # Vérifier si des niveaux ont été récupérés
    if not levels:
        logger.error("Aucune donnée atmosphérique n'a été récupérée")
        sys.exit(1)

    logger.info(f"Données récupérées: {len(levels)} niveaux atmosphériques")

    # Initialiser l'analyseur avec les informations sur les nuages
    model_name = args.model if args.location else "MeteoFrance" # sera toujours MeteoFrance ici
    analyzer = EmagrammeAnalyzer(levels, site_altitude=args.site_altitude, cloud_info=cloud_info, precip_info=precip_info, model_name=model_name)


    # Modifier le delta de température si spécifié
    if args.delta_t != THERMAL_TRIGGER_DELTA:
        THERMAL_TRIGGER_DELTA = args.delta_t
        logger.info(f"Delta de température pour le déclenchement des thermiques: {THERMAL_TRIGGER_DELTA}°C")

    # Effectuer l'analyse
    analysis = analyzer.analyze()

    # Afficher les résultats
    print("\n=== ANALYSE DE L'ÉMAGRAMME POUR LE VOL EN PARAPENTE ===\n")

    # Informations de base
    print(f"Altitude du site: {analysis.ground_altitude:.0f}m")
    print(f"Température au sol: {analysis.ground_temperature:.1f}°C")
    print(f"Point de rosée au sol: {analysis.ground_dew_point:.1f}°C")
    # Éviter la division par zéro
    if analysis.ground_temperature != 0:
        print(f"Humidité relative: {(analysis.ground_dew_point / analysis.ground_temperature * 100):.1f}%")
    print("")

    # Informations thermiques
    print(f"Plafond thermique: {analysis.thermal_ceiling:.0f}m")
    print(f"Gradient thermique: {analysis.thermal_gradient:.1f}°C/1000m")
    print(f"Force des thermiques: {analysis.thermal_strength}")
    print(f"Stabilité: {analysis.stability}")

    if analysis.thermal_type == "Cumulus":
        print(f"Base des nuages: {analysis.cloud_base:.0f}m")
        print(f"Sommet des nuages: {analysis.cloud_top:.0f}m")
    else:
        print("Thermiques bleus (pas de condensation)")

    print("")
    print("=== PARAMÈTRES DE PRÉVISION ENRICHIS ===")
    print(f"Spread au sol (T° - Td): {analysis.ground_spread:.1f}°C")

    if analysis.spread_levels:
        print("Spread aux différents niveaux:")
        for level, value in analysis.spread_levels.items():
            print(f"  - Niveau {level}: {value:.1f}°C")

    print(f"\nAnalyse du spread: {analysis.spread_analysis}")

    if hasattr(analysis, 'atmospheric_layers'):
        print("\nAnalyse des couches atmosphériques:")
        for layer_name, layer_data in analysis.atmospheric_layers.items():
            bottom = min(layer_data["altitudes"])
            top = max(layer_data["altitudes"])
            print(f"  - Couche {layer_name}: {bottom:.0f}-{top:.0f}m")
            print(f"    Température moyenne: {layer_data['mean_temp']:.1f}°C")
            print(f"    Spread moyen: {layer_data['mean_spread']:.1f}°C")
            print(f"    Stabilité: {layer_data['stability']}")

            if "mean_wind" in layer_data:
                dir_info = f" ({layer_data['dominant_dir']})" if "dominant_dir" in layer_data else ""
                print(f"    Vent moyen: {layer_data['mean_wind']:.0f} km/h{dir_info}")

            print("")

        if analysis.thermal_inconsistency:
                    print("\n⚠️ INCOHÉRENCE DÉTECTÉE:")
                    print(analysis.thermal_inconsistency)

    print("")

    # Ajouter juste après l'affichage des infos thermiques
    if analysis.precipitation_type is not None:
        print(f"Précipitations: {analysis.precipitation_description}")
        print("")

    # Couches d'inversion
    if analysis.inversion_layers:
        print("Couches d'inversion détectées:")
        for i, (base, top) in enumerate(analysis.inversion_layers):
            print(f"  {i+1}: De {base:.0f}m à {top:.0f}m")
        print("")

    # Conditions de vol et vent
    print(f"Conditions de vol: {analysis.flight_conditions}")
    print(f"Conditions de vent: {analysis.wind_conditions}")
    print("")

    # Risques et recommandations
    if analysis.hazards:
        print("⚠️ Risques identifiés:")
        for hazard in analysis.hazards:
            print(f"  - {hazard}")
        print("")

    if analysis.recommended_gear:
        print("Équipement recommandé:")
        for gear in analysis.recommended_gear:
            print(f"  - {gear}")
        print("")

    # Utiliser l'agent LLM si une clé est fournie
    if args.openai_key:
        print("\n=== PRÉPARATION DE L'ANALYSE DÉTAILLÉE PAR L'ASSISTANT IA ===\n")
        print("\n=== ANALYSE DÉTAILLÉE PAR L'ASSISTANT IA ===\n")
        print("Fonctionnalité d'analyse détaillée par IA non disponible dans cette version simplifiée.") # Placeholder pour la fonctionnalité IA


    # Afficher le graphique si demandé
    if args.plot or args.output:
        print("Fonctionnalité graphique non disponible dans cette version simplifiée.") # Placeholder pour la fonctionnalité graphique


if __name__ == "__main__":
    main()