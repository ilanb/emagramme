#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exemple d'utilisation de l'analyseur d'émagramme
------------------------------------------------
Ce script montre comment utiliser la bibliothèque d'analyse d'émagramme
pour les sorties en parapente, en créant des données d'exemple et en visualisant
les résultats.
"""

from emagramme_analyzer import AtmosphericLevel, EmagrammeAnalyzer
import numpy as np
import matplotlib.pyplot as plt

def create_sample_data():
    """Crée un jeu de données d'exemple pour un émagramme"""
    # Définir les niveaux d'altitude
    altitudes = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 10000])
    
    # Définir les pressions correspondantes (approximatives)
    pressures = 1013.25 * (1 - (altitudes/44330)) ** 5.255
    
    # Créer un profil de température avec inversion à 3000m
    temperatures = np.array([25, 22, 18, 14, 10, 6, 4, 0, -4, -12, -20, -36, -52])
    
    # Ajouter une inversion
    temperatures[6] = 5  # Inversion à 3000m
    
    # Définir les points de rosée (avec écart variable)
    dew_points = temperatures - np.array([12, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 12, 13])
    
    # Définir les vents (direction, vitesse)
    wind_directions = np.array([270, 270, 280, 290, 300, 310, 320, 330, 340, 350, 0, 10, 20])
    wind_speeds = np.array([5, 8, 12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45])
    
    # Créer les objets de niveau atmosphérique
    levels = []
    for i in range(len(altitudes)):
        level = AtmosphericLevel(
            altitude=altitudes[i],
            pressure=pressures[i],
            temperature=temperatures[i],
            dew_point=dew_points[i],
            wind_direction=wind_directions[i],
            wind_speed=wind_speeds[i]
        )
        levels.append(level)
    
    return levels

def main():
    """Fonction principale"""
    # Créer des données d'exemple
    levels = create_sample_data()
    
    # Créer l'analyseur pour une altitude de décollage de 1000m
    analyzer = EmagrammeAnalyzer(levels, site_altitude=1000)
    
    # Effectuer l'analyse
    analysis = analyzer.analyze()
    
    # Afficher les résultats
    print("=== RÉSULTATS DE L'ANALYSE ===")
    print(f"Température au sol: {analysis.ground_temperature:.1f}°C")
    print(f"Point de rosée au sol: {analysis.ground_dew_point:.1f}°C")
    print(f"Plafond thermique: {analysis.thermal_ceiling:.0f}m")
    
    if analysis.cloud_base:
        print(f"Base des nuages: {analysis.cloud_base:.0f}m")
        print(f"Sommet des nuages: {analysis.cloud_top:.0f}m")
    else:
        print("Thermiques bleus (pas de condensation)")
    
    print(f"Force des thermiques: {analysis.thermal_strength}")
    print(f"Stabilité: {analysis.stability}")
    print(f"Gradient thermique: {analysis.thermal_gradient:.1f}°C/1000m")
    
    if analysis.inversion_layers:
        print("\nCouches d'inversion:")
        for i, (base, top) in enumerate(analysis.inversion_layers):
            print(f"  Inversion {i+1}: De {base:.0f}m à {top:.0f}m")
    
    print(f"\nConditions de vol: {analysis.flight_conditions}")
    print(f"\nConditions de vent: {analysis.wind_conditions}")
    
    if analysis.hazards:
        print("\nRisques potentiels:")
        for hazard in analysis.hazards:
            print(f"- {hazard}")
    
    if analysis.recommended_gear:
        print("\nÉquipement recommandé:")
        for gear in analysis.recommended_gear:
            print(f"- {gear}")
    
    # Tracer l'émagramme
    print("\nAffichage de l'émagramme...")
    analyzer.plot_emagramme(analysis, save_path="emagramme_example.png", show=True)

if __name__ == "__main__":
    main()
