#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface Streamlit pour l'Analyseur d'Émagramme pour Parapentistes
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re  # Nécessaire pour le nouveau script d'analyse
import logging
from datetime import datetime

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
    
    # Créer le graphique
    fig, ax = plt.subplots(figsize=(4, 6))
    
    # Zone convective complète (en vert clair avec plus de transparence)
    ax.axhspan(ground_alt, thermal_ceiling, alpha=0.1, color='green', 
            label="Couche convective")
    
    # Zone anabatique (premiers 500m au-dessus du sol)
    ax.axhspan(ground_alt, anabatic_zone_top, alpha=0.1, color='blue', 
            label="Zone anabatique")
    
    # Ajout de lignes horizontales pour marquer les altitudes clés
    ax.axhline(y=ground_alt, color='brown', linestyle='-', linewidth=1.5)
    ax.axhline(y=thermal_ceiling, color='purple', linestyle='--', linewidth=1.5)
    ax.axhline(y=anabatic_zone_top, color='blue', linestyle='--', linewidth=1)
    
    # Ajout d'annotations pour les altitudes
    ax.text(0.05, ground_alt + 50, f"Sol: {ground_alt:.0f}m", 
            fontsize=8, ha='left', va='bottom')
    ax.text(0.05, thermal_ceiling - 50, f"Plafond: {thermal_ceiling:.0f}m", 
            fontsize=8, ha='left', va='top')
    ax.text(0.05, anabatic_zone_top - 20, f"Limite anabatique: {anabatic_zone_top:.0f}m", 
            fontsize=8, ha='left', va='top', color='blue')
    
    # Ajouter une information sur l'épaisseur de la couche
    thickness = thermal_ceiling - ground_alt
    ax.text(0.5, (ground_alt + thermal_ceiling) / 2, 
            f"Épaisseur: {thickness:.0f}m", 
            fontsize=9, ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Si des inversions sont présentes, les ajouter au graphique
    if inversions:
        for i, (base, top) in enumerate(inversions):
            if base < thermal_ceiling + 500:  # Ne montrer que les inversions pertinentes
                ax.axhspan(base, top, alpha=0.15, color='red', 
                        label=f"Inversion {i+1}" if i == 0 else "")
                ax.text(0.05, (base + top) / 2, f"Inv. {i+1}: {base:.0f}-{top:.0f}m", 
                        fontsize=8, ha='left', va='center', color='darkred')
    
    # Configuration des axes
    ax.set_ylim(max(0, ground_alt - 200), thermal_ceiling + 500)
    ax.set_xlim(0, 1)
    ax.set_ylabel("Altitude (m)")
    ax.set_xticks([])  # Supprimer les graduations de l'axe X
    
    # Ajouter un titre au graphique
    ax.set_title("Structure verticale de l'atmosphère", fontsize=10)
    
    # Ajouter une légende
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
            fancybox=True, shadow=True, ncol=2, fontsize=8)
    
    return fig

# Fonction pour obtenir et analyser les données
def fetch_and_analyze(lat, lon, model, site_altitude, api_key=None, openai_key=None, delta_t=THERMAL_TRIGGER_DELTA, 
                     data_source="open-meteo"):
    """Récupère les données et effectue l'analyse"""
    
    # Mettre à jour le delta de température si nécessaire
    global THERMAL_TRIGGER_DELTA
    if delta_t != THERMAL_TRIGGER_DELTA:
        THERMAL_TRIGGER_DELTA = delta_t
    
    # Initialiser le récupérateur de données
    fetcher = EmagrammeDataFetcher(api_key=api_key)
    
    # Attribuer l'altitude du site au fetcher
    fetcher.site_altitude = site_altitude
    
    # Récupérer les données - avec gestion des erreurs
    try:
        if data_source == "open-meteo":
            with st.spinner("Récupération des données météo via Open-Meteo..."):
                st.info("Utilisation d'Open-Meteo (sans clé API)")
                try:
                    levels = fetcher.fetch_from_openmeteo(lat, lon, model=model)
                except Exception as e:
                    st.error(f"Erreur avec Open-Meteo: {str(e)}")

        # Récupérer les informations sur les nuages et précipitations si disponibles
        cloud_info = getattr(fetcher, 'cloud_info', None)
        precip_info = getattr(fetcher, 'precip_info', None)
        
        # Initialiser l'analyseur avec toutes les informations
        analyzer = EmagrammeAnalyzer(levels, site_altitude=site_altitude, 
                                   cloud_info=cloud_info, precip_info=precip_info,
                                   model_name=model)
        
        # IMPORTANT : Copier les informations directement dans l'objet analyzer
        # pour qu'elles soient facilement accessibles par l'interface
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
        
        return analyzer, analysis, detailed_analysis
        
    except Exception as e:
        st.error(f"Erreur lors de la récupération ou de l'analyse des données: {str(e)}")
        logger.error(f"Erreur lors de la récupération/analyse: {e}", exc_info=True)
        return None, None, None

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

    if st.sidebar.checkbox("Je découvre la météo aérologique"):
        st.sidebar.info("Tutoriel activé - vous verrez des explications supplémentaires")
        st.session_state.tutorial_mode = True
    else:
        st.session_state.tutorial_mode = False

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
        st.session_state.site_selection = {
            "latitude": 45.5,
            "longitude": 6.0,
            "altitude": 1000,
            "model": "arome"
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
        model_options = ["meteofrance_arome_france_hd", "meteofrance_arpege_europe"]
        model_descriptions = {
            "meteofrance_arome_france_hd": "AROME HD (France ~2km)",
            "meteofrance_arpege_europe": "ARPEGE (Europe ~11km)"
        }
        model_labels = [model_descriptions[m] for m in model_options]
        model_index = st.sidebar.selectbox(
            "Modèle météo",
            options=range(len(model_options)),
            format_func=lambda i: model_labels[i],
            index=0
        )
        model = model_options[model_index]
        
        st.sidebar.info("Open-Meteo fournit un accès aux modèles Météo-France sans clé API")
        
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
    
    # Section pour les sites prédéfinis sur une seule colonne
    st.sidebar.header("Sites prédéfinis")
    st.sidebar.markdown("Cliquez sur un bouton pour charger un site et lancer l'analyse")
    
    # Fonction pour définir le site et déclencher l'analyse
    def set_site_and_analyze(site_data):
        st.session_state.site_selection = {
            "latitude": site_data["lat"],
            "longitude": site_data["lon"],
            "altitude": site_data["altitude"],
            "model": site_data["model"]
        }
        st.session_state.run_analysis = True
    
    # Créer les boutons pour chaque site prédéfini sur une seule colonne
    for i, site in enumerate(PRESET_SITES):
        st.sidebar.button(site["name"], key=f"site_{i}", 
                        on_click=set_site_and_analyze, 
                        args=(site,))
    
    # Section des paramètres de localisation
    st.subheader("Localisation")
    col1, col2, col3 = st.columns(3)

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
                                      min_value=0, max_value=5000, 
                                      value=st.session_state.site_selection["altitude"], 
                                      step=10)
    
    # Bouton pour lancer l'analyse
    analyze_clicked = st.button("Analyser l'émagramme")
    
    # Lancer l'analyse si le bouton est cliqué ou si un site prédéfini a été sélectionné
    should_run_analysis = analyze_clicked or st.session_state.run_analysis

    if should_run_analysis:
        # Réinitialiser le flag pour éviter des analyses en boucle
        st.session_state.run_analysis = False
        
        if use_openmeteo and not api_key:
            # Déterminer la source de données pour la fonction fetch_and_analyze
            data_source_str = "open-meteo"
            
            # Récupérer et analyser les données
            analyzer, analysis, detailed_analysis = fetch_and_analyze(
                latitude, longitude, model, site_altitude, api_key, openai_key, delta_t, data_source=data_source_str
            )
            
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
                cols = st.columns([1, 2])
                with cols[0]:
                    st.metric("Épaisseur de la couche convective", f"{convective_layer['thickness']:.0f} m")
                    st.write(f"**Qualité des ascendances**: {convective_layer['description']}")
                
                with cols[1]:
                    # Créer et afficher le graphique de la couche convective
                    fig = create_convective_layer_plot(analysis, analysis.inversion_layers)
                    st.pyplot(fig)
                
                # Onglets pour le reste des informations
                tab1, tab2, tab3 = st.tabs(["Résultats", "Données brutes", "Aide"])
                
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
                    vol_impossible = (analysis.precipitation_type is not None and analysis.precipitation_type != 0) or \
                                    (hasattr(analyzer, 'wind_speeds') and analyzer.wind_speeds is not None and \
                                    np.nanmax(analyzer.wind_speeds) > 25)
                    
                    if vol_impossible:
                        # Utiliser un style d'alerte visuelle différent
                        st.error("⚠️ VOL IMPOSSIBLE - Conditions météorologiques dangereuses")
                        
                        # Afficher la raison principale
                        raisons = []
                        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
                            raisons.append(analysis.precipitation_description)
                        if hasattr(analyzer, 'wind_speeds') and analyzer.wind_speeds is not None and np.nanmax(analyzer.wind_speeds) > 25:
                            raisons.append(f"Vent trop fort (max {np.nanmax(analyzer.wind_speeds):.1f} km/h)")
                        
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
                        # Ajoutez cette ligne pour afficher le modèle utilisé
                        st.info(f"Modèle météo utilisé: {model.upper()}")

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