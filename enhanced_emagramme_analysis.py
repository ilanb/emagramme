#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module d'analyse améliorée d'émagramme pour parapentistes
Fournit une analyse détaillée et pédagogique adaptée aux besoins des pilotes
"""

import re
import logging
import numpy as np

# Fonctions utilitaires pour éviter les problèmes de f-strings avec backslashes
def format_hazards(hazards):
    result = ""
    for hazard in hazards:
        result += f"- {hazard}\n"
    return result

def format_gear(gear):
    result = ""
    for item in gear:
        result += f"- {item}\n"
    return result

def analyze_emagramme_for_pilot(analysis):
    """
    Génère une analyse détaillée de l'émagramme pour les pilotes de parapente
    en se basant sur les structures thermiques, la stabilité, et les conditions pratiques de vol.
    
    Args:
        analysis: L'objet EmagrammeAnalysis contenant les résultats de l'analyse
        
    Returns:
        Une analyse textuelle détaillée adaptée aux besoins des parapentistes
    """
    # Vérifier si des précipitations sont prévues
    vol_impossible = False
    raisons_impossibilite = []
    
    if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
        vol_impossible = True
        raisons_impossibilite.append(analysis.precipitation_description)
    
    # Vérifier le vent
    if "fort au sol" in analysis.wind_conditions.lower() or "critique" in analysis.wind_conditions.lower():
        vol_impossible = True
        raisons_impossibilite.append("Vent trop fort")
    
    if vol_impossible:
        # Construire un avertissement clair
        raisons = ", ".join(raisons_impossibilite)
        return f"""# ⚠️ VOL IMPOSSIBLE - {raisons}

## Analyse des conditions météorologiques défavorables

Les conditions actuelles ne permettent pas la pratique du parapente en sécurité en raison de {raisons}. 

{analysis.flight_conditions}

### Conditions de vent détaillées
{analysis.wind_conditions}

### Dangers spécifiques identifiés
{format_hazards(analysis.hazards)}

### Recommandation formelle
Il est fortement déconseillé de voler dans ces conditions. Aucun site ni aucune technique ne permet de compenser ces risques.
Attendez une amélioration des conditions météorologiques.
"""
    
    # Si le vol est possible, préparer une analyse complète
    
    # 1. Analyse de la structure thermique
    thermal_structure = _analyze_thermal_structure(analysis)
    
    # 2. Analyse de la stabilité atmosphérique
    stability_analysis = _analyze_atmospheric_stability(analysis)
    
    # 3. Analyse du vent
    wind_analysis = _analyze_wind_conditions(analysis)
    
    # 4. Analyse du spread (écart température/point de rosée)
    spread_analysis = _analyze_spread_patterns(analysis)
    
    # 5. Heures optimales de vol
    optimal_hours = _determine_optimal_hours(analysis)
    
    # 6. Sites recommandés selon les conditions
    sites_advice = _recommend_flying_sites(analysis)
    
    # 7. Niveau de difficulté et public cible
    difficulty = _assess_difficulty_level(analysis)
    
    # 8. Conseils tactiques pour le vol
    tactical_advice = _provide_tactical_advice(analysis)
    
    # 9. Lecture pratique de l'émagramme
    emagramme_interpretation = _interpret_emagramme_for_pilots(analysis)
    
    # Assembler l'analyse complète
    hazards_formatted = format_hazards(analysis.hazards)
    gear_formatted = format_gear(analysis.recommended_gear)
    
    complete_analysis = f"""# Analyse détaillée des conditions de vol en parapente

## Synthèse générale
{analysis.flight_conditions}

## Structure thermique
{thermal_structure}

## Stabilité atmosphérique
{stability_analysis}

## Conditions de vent
{wind_analysis}

## Analyse de l'humidité (Spread)
{spread_analysis}

## Heures optimales de vol
{optimal_hours}

## Sites recommandés
{sites_advice}

## Niveau de difficulté
{difficulty}

## Conseils tactiques
{tactical_advice}

## Lecture pratique de l'émagramme
{emagramme_interpretation}

## Risques potentiels à surveiller
{hazards_formatted}

## Équipement recommandé
{gear_formatted}
"""
    
    return complete_analysis

def _analyze_thermal_structure(analysis):
    """Analyse détaillée de la structure thermique"""
    
    # Épaisseur de la couche convective
    convective_layer_thickness = analysis.thermal_ceiling - analysis.ground_altitude
    
    # Analyse de base sur le type de thermique
    if analysis.thermal_type == "Bleu":
        thermal_type_desc = "Vous allez voler en conditions de **thermiques bleus** (sans marquage nuageux)."
        if analysis.thermal_inconsistency:
            thermal_type_desc += f" Attention : {analysis.thermal_inconsistency}"
    else:
        cloud_thickness = analysis.cloud_top - analysis.cloud_base
        thermal_type_desc = f"Vous allez voler avec des **thermiques à cumulus** bien visibles, base à {analysis.cloud_base:.0f}m et sommet à {analysis.cloud_top:.0f}m."
        if cloud_thickness > 1000:
            thermal_type_desc += " Ces cumulus sont bien développés verticalement, attention aux zones de surdéveloppement."
    
    # Analyser les caractéristiques des thermiques selon leur force
    thermal_characteristics = ""
    if analysis.thermal_strength == "Faible":
        thermal_characteristics = """
Les thermiques seront dispersés et relativement faibles. Attendez-vous à:
- Des ascendances de 0,5 à 1 m/s en moyenne
- Des thermiques étroits et fragmentés
- Un décrochage fréquent des thermiques avant le plafond théorique
- Des transitions délicates nécessitant de la patience"""
    elif analysis.thermal_strength == "Modérée":
        thermal_characteristics = """
Les thermiques seront de force modérée et assez réguliers. Attendez-vous à:
- Des ascendances de 1 à 2 m/s en moyenne
- Des thermiques bien formés mais pouvant être espacés
- Une bonne possibilité d'atteindre le plafond théorique
- Des transitions confortables mais nécessitant une bonne lecture du ciel"""
    elif analysis.thermal_strength == "Forte":
        thermal_characteristics = """
Les thermiques seront puissants et actifs. Attendez-vous à:
- Des ascendances de 2 à 4 m/s en moyenne
- Des thermiques larges et bien structurés
- Des plafonds facilement atteignables
- Un risque de turbulence et fermetures dans les zones les plus actives
- De bonnes possibilités de cross-country"""
    else:  # "Très Forte"
        thermal_characteristics = """
Les thermiques seront très puissants, potentiellement turbulents. Attendez-vous à:
- Des ascendances dépassant 4 m/s
- Des thermiques très larges mais pouvant être turbulents
- Possibilité de surpuissances et fermetures même à mi-course
- Conditions adaptées uniquement aux pilotes expérimentés
- D'excellentes conditions de cross si vous avez le niveau technique"""
    
    # Ajouter des informations sur la couche anabatique (environ 500m au-dessus du sol)
    anabatic_layer = min(analysis.ground_altitude + 500, analysis.thermal_ceiling)
    anabatic_thickness = anabatic_layer - analysis.ground_altitude
    
    anabatic_desc = f"""
## Couche anabatique et déclenchement
La couche anabatique (brises de pente) s'étend jusqu'à environ {anabatic_layer:.0f}m, soit {anabatic_thickness:.0f}m au-dessus du sol.
"""
    
    if analysis.thermal_strength in ["Faible", "Modérée"]:
        anabatic_desc += """
Dans ces conditions, le déclenchement des thermiques pourra être délicat:
- Soyez patient en soaring sur les pentes exposées au soleil
- Cherchez les points de déclenchement classiques (rochers, zones sombres, ruptures de pente)
- Restez près du relief en début de vol pour profiter des brises de pente
"""
    else:
        anabatic_desc += """
Le déclenchement des thermiques devrait être assez franc:
- Les thermiques se déclencheront facilement des points chauds
- Vous pourrez rapidement quitter la brise de pente pour les thermiques
- Attention aux effets venturi et aux accélérations près du relief
"""
    
    # Intégrer les informations sur les inversions si présentes
    inversion_desc = ""
    if analysis.inversion_layers:
        inversion_desc = "\n## Inversions thermiques\n"
        for i, (base, top) in enumerate(analysis.inversion_layers):
            thickness = top - base
            if base < analysis.thermal_ceiling:
                inversion_desc += f"Une inversion de {thickness:.0f}m d'épaisseur est présente entre {base:.0f}m et {top:.0f}m. "
                if base < analysis.ground_altitude + 1000:
                    inversion_desc += "Cette inversion basse pourrait limiter significativement le développement des thermiques et créer un plafond artificiel. "
                elif base < analysis.thermal_ceiling:
                    inversion_desc += f"Cette inversion pourrait ralentir les thermiques vers {base:.0f}m, créant une zone où vous devrez être patient pour continuer à monter. "
            else:
                inversion_desc += f"Une inversion est présente au-dessus du plafond thermique (entre {base:.0f}m et {top:.0f}m) et ne devrait pas affecter votre vol. "
    
    # Assembler l'analyse complète de la structure thermique
    complete_thermal_analysis = f"""
La couche convective s'étend du sol à {analysis.thermal_ceiling:.0f}m, soit une épaisseur utilisable de {convective_layer_thickness:.0f}m.
Le gradient thermique est de {analysis.thermal_gradient:.1f}°C/1000m.

{thermal_type_desc}

{thermal_characteristics}

{anabatic_desc}

{inversion_desc}

Le plafond thermique attendu de {analysis.thermal_ceiling:.0f}m représente généralement la limite maximale à laquelle les meilleurs thermiques pourront vous porter. En pratique, attendez-vous à atteindre environ 80-90% de cette valeur en moyenne.
"""
    
    return complete_thermal_analysis

def _analyze_atmospheric_stability(analysis):
    """Analyse de la stabilité atmosphérique et son impact sur le vol"""
    
    stability_desc = ""
    if analysis.stability == "Stable":
        stability_desc = """
L'atmosphère est stable aujourd'hui, ce qui signifie:
- Les thermiques seront plus difficiles à exploiter et moins puissants
- Les mouvements verticaux seront limités
- L'air sera relativement calme entre les thermiques
- Les ascendances seront plus prévisibles et moins turbulentes
- Privilégiez les zones à fort potentiel de réchauffement
"""
    elif analysis.stability == "Neutre":
        stability_desc = """
L'atmosphère présente une stabilité neutre, ce qui signifie:
- Bonnes conditions générales de vol avec thermiques prévisibles
- Puissance modérée des thermiques
- Bonne organisation des mouvements convectifs
- Transitions relativement confortables
- Conditions adaptées à la majorité des pilotes
"""
    elif analysis.stability == "Instable":
        stability_desc = """
L'atmosphère est instable aujourd'hui, ce qui signifie:
- Thermiques puissants et actifs
- Possibilité de fortes ascendances et de turbulences
- Bonne extension verticale des mouvements convectifs
- Atmosphère "nerveuse" avec possible aérologie irrégulière
- Conditions favorables pour les vols de distance
- Risque de fermetures dans les thermiques puissants
"""
    else:  # "Très Instable"
        stability_desc = """
L'atmosphère est très instable aujourd'hui, ce qui signifie:
- Thermiques très puissants pouvant dépasser 5 m/s
- Forte turbulence dans et autour des ascendances
- Développement vertical important des thermiques
- Risque de surdéveloppement des cumulus et possibles orages
- Conditions exigeantes nécessitant une technique solide
- Possible conditions "ON-OFF" (très fortes ou très faibles)
"""
    
    # Impact de la stabilité sur le développement nuageux
    cloud_analysis = ""
    if analysis.thermal_type == "Cumulus":
        if analysis.stability in ["Instable", "Très Instable"]:
            cloud_analysis = """
Cette instabilité favorisera le développement vertical des cumulus. Surveillez l'évolution des nuages pendant votre vol:
- Les cumulus congestus et les zones d'ombre qu'ils projettent
- L'assombrissement des bases qui indique un renforcement
- Des sommets dépassant significativement le plafond thermique habituel
"""
            if analysis.stability == "Très Instable":
                cloud_analysis += """
⚠️ Attention au risque d'aggravation des conditions et de développement orageux en cours de journée. Planifiez de voler tôt et observez l'évolution des nuages.
"""
        else:
            cloud_analysis = """
Les cumulus devraient rester modérés avec un développement vertical limité. Ils constitueront de bons indicateurs pour localiser les thermiques.
"""
    else:  # Thermiques bleus
        if analysis.stability in ["Instable", "Très Instable"]:
            cloud_analysis = """
Malgré l'absence de cumulus, l'instabilité produira des thermiques puissants. Soyez attentif aux autres indices:
- Oiseaux en vol thermique
- Poussière ou débris soulevés
- Zones d'air tremblotant à l'horizon
- Variations de teinte du ciel (bleu plus foncé au-dessus des zones ascendantes)
"""
        else:
            cloud_analysis = """
Les thermiques bleus seront plus difficiles à repérer. Concentrez-vous sur les zones de déclenchement classiques et les cycles thermiques.
"""
    
    # Assembler l'analyse complète de la stabilité
    complete_stability_analysis = f"""
{stability_desc}

{cloud_analysis}

La stabilité "{analysis.stability}" combinée à un gradient thermique de {analysis.thermal_gradient:.1f}°C/1000m indique le potentiel énergétique de l'atmosphère. Cette valeur détermine à quel point l'air chaud au sol pourra s'élever efficacement, et donc la force de vos thermiques.
"""
    
    return complete_stability_analysis

def _analyze_wind_conditions(analysis):
    """Analyse détaillée des conditions de vent et impact sur le vol"""
    
    # Extraire les informations de vent de la description textuelle
    wind_speed_desc = ""
    wind_direction = ""
    
    if "faible au sol" in analysis.wind_conditions.lower():
        wind_speed_desc = "faible"
    elif "modéré au sol" in analysis.wind_conditions.lower():
        wind_speed_desc = "modéré"
    elif "assez fort au sol" in analysis.wind_conditions.lower():
        wind_speed_desc = "assez fort"
    else:
        wind_speed_desc = "fort"
    
    # Extraire la direction si mentionnée
    direction_search = re.search(r"Direction: (\w+)", analysis.wind_conditions)
    if direction_search:
        wind_direction = direction_search.group(1)
    
    # Analyse du gradient de vent
    gradient_desc = ""
    if "peu de gradient" in analysis.wind_conditions.lower():
        gradient_desc = """
Le gradient de vent est faible, ce qui signifie que la vitesse et la direction du vent changent peu avec l'altitude. Cela vous offrira:
- Des conditions relativement homogènes à toutes les altitudes
- Moins de turbulences dues au cisaillement
- Des thermiques bien définis et peu déformés
- Une dérive constante et prévisible
"""
    elif "gradient de vent modéré" in analysis.wind_conditions.lower():
        gradient_desc = """
Le gradient de vent est modéré, ce qui signifie que le vent se renforce avec l'altitude. Attendez-vous à:
- Une dérive plus importante en altitude
- Des thermiques légèrement inclinés ou déformés
- Une possible turbulence modérée aux transitions entre masses d'air
- Un besoin d'adaptation de votre technique de centrage selon l'altitude
"""
    else:  # Fort gradient
        gradient_desc = """
⚠️ Le gradient de vent est fort, ce qui signifie que le vent augmente significativement avec l'altitude. Cela implique:
- Une dérive beaucoup plus importante en altitude qu'au sol
- Des thermiques fortement inclinés dans le sens du vent
- Des turbulences potentiellement fortes dues au cisaillement
- Un risque accru de fermetures lors des transitions
- Une nécessité de bien anticiper votre plan de vol et vos zones d'atterrissage
"""
    
    # Analyse de la rotation du vent
    rotation_desc = ""
    if "direction stable en altitude" in analysis.wind_conditions.lower():
        rotation_desc = """
La direction du vent reste stable avec l'altitude. Cela facilite:
- La lecture des thermiques qui resteront orientés de façon constante
- La planification de votre parcours 
- L'anticipation de la dérive
"""
    elif "légère rotation" in analysis.wind_conditions.lower():
        rotation_desc = """
Le vent présente une légère rotation avec l'altitude:
- Les thermiques pourront être légèrement torsadés
- Adaptez votre technique de centrage en fonction de l'altitude
- Tenez compte de la variation de direction pour planifier votre route
"""
    else:  # Forte rotation
        rotation_desc = """
⚠️ Le vent présente une forte rotation avec l'altitude:
- Les thermiques seront torsadés et plus difficiles à exploiter
- Votre dérive changera significativement selon l'altitude
- La planification de votre route devra tenir compte de ces variations
- Attendez-vous à une turbulence accrue aux interfaces entre couches d'air
"""
    
    # Interaction vent-relief et formation des thermiques
    terrain_interaction = ""
    if wind_speed_desc in ["faible", "modéré"]:
        if wind_direction:
            terrain_interaction = f"""
Avec un vent {wind_speed_desc} de {wind_direction}, les thermiques seront:
- Bien formés sous le vent des zones de déclenchement
- Légèrement inclinés dans le sens du vent
- Possiblement organisés en "rues" parallèles à la direction du vent
"""
        else:
            terrain_interaction = f"""
Avec un vent {wind_speed_desc}, les thermiques seront:
- Bien définis et généralement verticaux
- Faiblement influencés par la dérive
"""
    else:
        if wind_direction:
            terrain_interaction = f"""
⚠️ Avec un vent {wind_speed_desc} de {wind_direction}:
- Les thermiques seront fortement inclinés et déportés
- Les zones sous le vent des reliefs pourront être turbulentes
- L'effet venturi entre les reliefs pourra créer des accélérations locales
- Privilégiez les décollages et les vols au vent des reliefs
"""
        else:
            terrain_interaction = f"""
⚠️ Avec un vent {wind_speed_desc}:
- Attendez-vous à une forte dérive et à des thermiques inclinés
- Les zones sous le vent des reliefs pourront être turbulentes
- Surveillez particulièrement les conditions à l'atterrissage
"""
    
    # Recommandations tactiques basées sur le vent
    tactical_wind_advice = f"""
## Recommandations par rapport au vent

Avec un vent {wind_speed_desc} {f"de {wind_direction}" if wind_direction else ""}:
"""
    
    if wind_speed_desc in ["faible", "modéré"]:
        tactical_wind_advice += """
- Exploitez normalement les thermiques avec un centrage adapté à la dérive
- Les décollages et atterrissages seront relativement faciles
- Les brises locales pourront jouer un rôle important dans les basses couches
"""
    else:
        tactical_wind_advice += """
- Privilégiez les phases de vol face au vent ou en travers
- Anticipez largement votre approche d'atterrissage
- Soyez conscient de l'accélération du vent autour des reliefs (effet venturi)
- Évitez de vous retrouver sous le vent d'obstacles importants
"""
    
    if "fort gradient" in analysis.wind_conditions.lower():
        tactical_wind_advice += """
- Adaptez votre vitesse de transition en fonction de l'altitude (plus rapide en hauteur)
- Gardez une marge de sécurité accrue par rapport au relief en altitude
- Soyez particulièrement attentif aux signes de turbulence et cisaillement
"""
    
    # Assembler l'analyse complète du vent
    complete_wind_analysis = f"""
## Caractéristiques générales du vent
{analysis.wind_conditions}

{gradient_desc}

{rotation_desc}

{terrain_interaction}

{tactical_wind_advice}
"""
    
    return complete_wind_analysis

def _analyze_spread_patterns(analysis):
    """Analyse détaillée du spread (écart température/point de rosée) et de l'humidité"""
    
    # Récupérer les valeurs de spread
    ground_spread = analysis.ground_spread
    spread_desc = ""
    
    # Analyse basée sur le spread au sol
    if ground_spread < 3:
        spread_desc = f"""
Le spread au sol est très faible ({ground_spread:.1f}°C), indiquant une forte humidité. Cela peut entraîner:
- Un risque de brouillard matinal
- Des nuages bas en début de journée
- Une activation thermique potentiellement retardée
- Des plafonds pouvant être limités par la nébulosité
"""
    elif ground_spread < 8:
        spread_desc = f"""
Le spread au sol est modéré ({ground_spread:.1f}°C), indiquant des conditions d'humidité normales. Cela favorise:
- Une bonne formation des thermiques
- Des cumulus bien définis si le niveau de condensation est atteint
- Des conditions classiques de développement thermique
"""
    else:
        spread_desc = f"""
Le spread au sol est important ({ground_spread:.1f}°C), indiquant une masse d'air sèche. Cela implique:
- Une forte probabilité de thermiques bleus (sans formation de nuages)
- Une atmosphère très transparente
- Une moins bonne visibilité des ascendances
- Potentiellement des thermiques plus "nerveux" et moins organisés
"""
    
    # Analyse de l'évolution du spread avec l'altitude
    evolution_desc = ""
    if analysis.spread_levels:
        spread_values = list(analysis.spread_levels.values())
        if min(spread_values) < 3:
            evolution_desc = """
Le spread diminue fortement avec l'altitude, ce qui indique:
- Une augmentation de l'humidité relative en hauteur
- Une probabilité élevée de formation nuageuse
- Possiblement un plafond limité par une couche nuageuse
"""
        elif all(x > 8 for x in spread_values):
            evolution_desc = """
Le spread reste élevé à tous les niveaux, ce qui confirme:
- Une atmosphère sèche favorable aux thermiques bleus
- Une absence probable de formation nuageuse
- Une difficulté accrue pour repérer visuellement les thermiques
"""
        else:
            evolution_desc = """
Le spread varie avec l'altitude, indiquant:
- Des couches d'humidité différentes
- Une possible formation nuageuse à certains niveaux spécifiques
- Des conditions potentiellement différentes selon l'altitude de vol
"""
    
    # Formation nuageuse et niveau de condensation
    cloud_formation_desc = ""
    if analysis.thermal_type == "Cumulus":
        cloud_formation_desc = f"""
## Formation nuageuse
Le niveau de condensation (base des cumulus) est estimé à {analysis.cloud_base:.0f}m. 
- La condensation se produit lorsque la température de l'air descendant selon l'adiabatique sèche (-1°C/100m) rencontre le point de rosée
- Les cumulus devraient s'étendre verticalement jusqu'à environ {analysis.cloud_top:.0f}m
- L'épaisseur des nuages de {analysis.cloud_top - analysis.cloud_base:.0f}m indique leur développement vertical
"""
        
        if (analysis.cloud_top - analysis.cloud_base) > 1000:
            cloud_formation_desc += """
⚠️ L'importante épaisseur verticale des nuages indique un risque de développement congestus. Surveillez l'évolution des cumulus pendant votre vol.
"""
    else:
        cloud_formation_desc = """
## Absence de condensation (thermiques bleus)
Les thermiques resteront "bleus" (sans condensation visible):
- La température des thermiques ne descendra pas suffisamment pour atteindre le point de rosée
- Vous devrez vous fier à d'autres indices pour repérer les ascendances
- L'air sec peut produire des thermiques plus "rugueux" mais potentiellement plus puissants
"""
        
        if analysis.thermal_inconsistency:
            cloud_formation_desc += f"""
⚠️ Note importante: {analysis.thermal_inconsistency}
"""
    
    # Impact pratique sur le vol
    practical_impact = f"""
## Impact pratique sur votre vol

Avec un spread au sol de {ground_spread:.1f}°C et les variations d'humidité observées:
"""
    
    if analysis.thermal_type == "Cumulus":
        practical_impact += """
- Utilisez les cumulus comme indicateurs fiables des zones ascendantes
- Visez légèrement au vent de la base des nuages pour intercepter le thermique
- Surveillez l'évolution des nuages pour anticiper les cycles thermiques
- Les zones d'ombre projetées par les cumulus peuvent couper temporairement l'activité thermique
"""
    else:
        practical_impact += """
- Sans marquage nuageux, concentrez-vous sur les zones de déclenchement probables
- Observez le comportement des autres ailes, des oiseaux, et les indices au sol
- Les zones de fort contraste (lisières, changements de terrain) sont à privilégier
- Les cycles thermiques pourront être moins évidents à identifier
"""
    
    # Assembler l'analyse complète du spread
    complete_spread_analysis = f"""
{spread_desc}

{evolution_desc}

{cloud_formation_desc}

{practical_impact}
"""
    
    return complete_spread_analysis

def _determine_optimal_hours(analysis):
    """Détermine les heures optimales de vol selon les conditions"""
    
    optimal_hours = ""
    
    # Basé sur la force des thermiques et la stabilité
    if analysis.thermal_strength in ["Faible", "Modérée"]:
        if analysis.ground_spread < 3:
            optimal_hours = """
Les meilleures conditions seront probablement en milieu de journée, une fois que l'humidité matinale se sera dissipée:
- 12h00-15h00: Période optimale quand le sol aura suffisamment chauffé
- Attendez que le soleil ait travaillé plusieurs heures pour dissiper l'humidité
- Le vol du matin risque d'être difficile avec des thermiques faibles ou inexistants
"""
        else:
            optimal_hours = """
Les meilleures conditions se développeront progressivement:
- 11h00-16h00: Fenêtre thermique principale
- La période optimale sera probablement entre 13h00 et 15h00
- L'activité thermique diminuera progressivement en fin d'après-midi
"""
    else:  # Thermiques forts ou très forts
        if analysis.stability in ["Instable", "Très Instable"]:
            optimal_hours = """
⚠️ Avec des thermiques puissants dans une atmosphère instable:
- 10h00-12h00: Conditions s'établissant, généralement les plus confortables
- 12h00-15h00: Pic d'activité thermique, potentiellement très fort et turbulent
- 15h00-17h00: Phase de décroissance, souvent plus agréable mais encore active
- Évitez le milieu de journée si vous êtes un pilote intermédiaire ou si vous préférez le confort de vol
"""
            
            if analysis.thermal_type == "Cumulus" and (analysis.cloud_top - analysis.cloud_base) > 1000:
                optimal_hours += """
- Surveillez attentivement l'évolution des nuages; si vous observez un développement vertical important, envisagez d'atterrir plus tôt
"""
        else:
            optimal_hours = """
Avec des thermiques forts dans une atmosphère relativement stable:
- 11h00-16h00: Bonnes conditions thermiques
- Le pic d'activité sera probablement entre 13h00 et 15h00
- Les conditions devraient rester assez régulières pendant cette fenêtre
"""
    
    # Adaptation selon le spread et l'humidité
    if analysis.ground_spread < 5:
        optimal_hours += """
Note: L'humidité relativement élevée pourrait retarder le démarrage des thermiques le matin. Prévoyez un départ plus tardif.
"""
    elif analysis.ground_spread > 12:
        optimal_hours += """
Note: L'air très sec pourrait produire des thermiques plus rugueux mais potentiellement plus puissants en milieu de journée.
"""
    
    # Considération des conditions de vent
    if "fort gradient" in analysis.wind_conditions.lower() or "fort au sol" in analysis.wind_conditions.lower():
        optimal_hours += """
⚠️ En raison des conditions de vent, privilégiez le début de journée (avant 13h00) quand l'activité thermique est moins forte et les conditions de vent généralement plus modérées.
"""
    
    return optimal_hours


def _recommend_flying_sites(analysis):
    """Recommande les types de sites adaptés aux conditions"""
    
    sites_recommendation = """## Types de sites recommandés aujourd'hui\n\n"""
    
    # Orientation par rapport au vent
    if hasattr(analysis, 'wind_conditions'):
        # Extraire la direction si mentionnée
        direction_search = re.search(r"Direction: (\w+)", analysis.wind_conditions)
        if direction_search:
            wind_direction = direction_search.group(1)
            sites_recommendation += f"""
### Exposition au vent
Favorisez les sites orientés face au vent {wind_direction} ou légèrement de travers. Cette orientation offre:
- Une brise dynamique qui facilite le gonflage et le décollage
- Une portance supplémentaire le long des pentes face au vent
- Une meilleure organisation des thermiques au vent du relief
"""
        else:
            sites_recommendation += """
### Exposition au vent
Privilégiez les sites face au vent dominant, particulièrement si celui-ci est modéré à fort.
"""
    
    # Recommandations basées sur la force des thermiques
    if analysis.thermal_strength in ["Faible", "Modérée"]:
        sites_recommendation += """
### Caractéristiques thermiques
Choisissez des sites avec:
- De grandes surfaces réceptives au soleil (versants sud/sud-est/sud-ouest)
- Des sols sombres ou des rochers qui absorbent bien la chaleur
- Des configurations en cuvette ou en amphithéâtre qui concentrent l'air chaud
- Une pente suffisante pour faciliter le décollage en conditions faibles
"""
    else:  # Thermiques forts ou très forts
        sites_recommendation += """
### Caractéristiques thermiques
En conditions fortes, privilégiez:
- Des sites avec un espace aérien dégagé et suffisant pour gérer les thermiques puissants
- Des zones d'atterrissage amples et multiples
- Évitez les zones encaissées ou à fort potentiel de turbulence
- Considérez des sites légèrement moins exposés au soleil pour modérer la puissance thermique
"""
    
    # Considérations de relief selon la stabilité
    if analysis.stability in ["Stable", "Neutre"]:
        sites_recommendation += """
### Relief et topographie
Dans ces conditions plutôt stables:
- Les grandes faces bien exposées seront efficaces
- Les cols et les convergences peuvent renforcer les ascendances
- Les sites offrant des possibilités de soaring peuvent être intéressants en complément
"""
    else:  # Instable ou très instable
        sites_recommendation += """
### Relief et topographie
Dans ces conditions instables:
- Méfiez-vous des zones de compression et de rotors sous le vent des reliefs
- Les sites offrant de bonnes échappatoires sont à privilégier
- Les zones de plaine peuvent également offrir d'excellentes conditions thermiques
"""
    
    # Recommandations d'altitude
    altitude_rec = """
### Altitude du site
"""
    if analysis.thermal_ceiling < 1500:
        altitude_rec += """
Avec un plafond bas:
- Privilégiez les sites de basse ou moyenne altitude
- Assurez-vous que le dénivelé disponible reste intéressant malgré le plafond limité
"""
    elif analysis.thermal_ceiling < 2500:
        altitude_rec += """
Avec un plafond moyen:
- Les sites de moyenne altitude offriront un bon compromis
- Vérifiez que la hauteur sol-plafond reste suffisante pour exploiter les thermiques
"""
    else:
        altitude_rec += """
Avec un plafond élevé:
- Tous les sites sont potentiellement intéressants, y compris ceux de haute altitude
- Les grands dénivelés pourront être pleinement exploités
"""
    
    sites_recommendation += altitude_rec
    
    # Considérations spécifiques
    if "fort gradient" in analysis.wind_conditions.lower():
        sites_recommendation += """
### Note importante - Gradient de vent
Avec un fort gradient de vent, soyez particulièrement attentifs aux zones d'atterrissage:
- Privilégiez les atterrissages hauts ou à proximité du relief
- Méfiez-vous des atterrissages en vallée encaissée (cisaillement possible)
- Envisagez des sites avec plusieurs options d'atterrissage selon l'évolution des conditions
"""
    
    if analysis.thermal_ceiling - analysis.ground_altitude < 800:
        sites_recommendation += """
### Attention - Couche convective limitée
Avec une fine couche convective, privilégiez:
- Des sites offrant une bonne dynamique ou du soaring
- Des zones où le vol proche du relief reste sécuritaire
- Évitez les sites nécessitant de grandes transitions
"""
    
    return sites_recommendation

def _assess_difficulty_level(analysis):
    """Évalue le niveau de difficulté et le public cible pour ces conditions"""
    
    # Analyse des facteurs influençant la difficulté
    thermal_factor = 0  # 0 = facile, 5 = très difficile
    if analysis.thermal_strength == "Faible":
        thermal_factor = 1  # Délicat mais pas turbulent
    elif analysis.thermal_strength == "Modérée":
        thermal_factor = 2
    elif analysis.thermal_strength == "Forte":
        thermal_factor = 3
    else:  # "Très Forte"
        thermal_factor = 5
    
    # Facteur lié à la stabilité
    stability_factor = 0
    if analysis.stability == "Stable":
        stability_factor = 1
    elif analysis.stability == "Neutre":
        stability_factor = 2
    elif analysis.stability == "Instable":
        stability_factor = 3
    else:  # "Très Instable"
        stability_factor = 5
    
    # Facteur lié au vent
    wind_factor = 0
    if "faible au sol" in analysis.wind_conditions.lower():
        wind_factor = 1
    elif "modéré au sol" in analysis.wind_conditions.lower():
        wind_factor = 2
    elif "assez fort au sol" in analysis.wind_conditions.lower():
        wind_factor = 4
    else:  # Fort
        wind_factor = 5
    
    # Facteur lié au gradient de vent
    gradient_factor = 0
    if "peu de gradient" in analysis.wind_conditions.lower():
        gradient_factor = 1
    elif "gradient de vent modéré" in analysis.wind_conditions.lower():
        gradient_factor = 2
    else:  # Fort gradient
        gradient_factor = 4
    
    # Autres complications
    complexity_factor = 0
    if analysis.thermal_type == "Bleu":
        complexity_factor += 1  # Plus difficile de trouver les thermiques
    
    if analysis.inversion_layers and any(base < analysis.thermal_ceiling for base, _ in analysis.inversion_layers):
        complexity_factor += 1  # Inversions sous le plafond = plus technique
    
    if "forte rotation" in analysis.wind_conditions.lower():
        complexity_factor += 1  # Rotation du vent = plus technique
    
    # Calcul du score global de difficulté (0-20)
    difficulty_score = thermal_factor + stability_factor + wind_factor + gradient_factor + complexity_factor
    
    # Interprétation du score
    if difficulty_score <= 5:
        difficulty_desc = """
## Niveau: Accessible à tous les pilotes

Ces conditions sont favorables pour:
- Les pilotes débutants sous supervision
- Les pilotes brevetés en progression
- Les vols en biplace
- Les reprises d'activité après une pause

Caractéristiques principales:
- Aérologie prévisible et clémente
- Thermiques doux et bien formés
- Peu de turbulence
- Bonnes conditions pour travailler la technique de vol
"""
    elif difficulty_score <= 10:
        difficulty_desc = """
## Niveau: Pilotes autonomes (Brevet initial+)

Ces conditions conviennent aux:
- Pilotes autonomes ayant déjà une expérience en thermique
- Pilotes en progression vers le cross
- Biplaces avec passagers informés

Caractéristiques principales:
- Thermiques modérés nécessitant une bonne technique de centrage
- Conditions permettant de bons vols locaux et petits cross
- Aérologie généralement lisible mais demandant de l'attention
- Bonnes conditions pour perfectionner sa technique et gagner en expérience
"""
    elif difficulty_score <= 15:
        difficulty_desc = """
## Niveau: Pilotes confirmés (Brevet de pilote+)

Ces conditions sont adaptées aux:
- Pilotes expérimentés avec une bonne maîtrise de leur aile
- Crosseurs réguliers
- Pilotes habitués à gérer des conditions variées

Caractéristiques principales:
- Thermiques puissants demandant une bonne gestion de l'aile
- Conditions potentiellement turbulentes par moments
- Nécessité d'anticiper et d'adapter son pilotage
- Vols cross et performances possibles, mais vigilance requise
- À éviter pour les pilotes peu expérimentés ou en manque de pratique
"""
    else:  # > 15
        difficulty_desc = """
## Niveau: Pilotes experts uniquement

Ces conditions exigeantes sont réservées aux:
- Pilotes très expérimentés avec une excellente maîtrise en turbulence
- Compétiteurs et crosseurs chevronnés
- Pilotes volant très régulièrement

⚠️ Caractéristiques principales:
- Thermiques très puissants et potentiellement turbulents
- Conditions exigeant des réactions rapides et précises
- Risque accru de fermetures et incidents de vol
- Nécessité d'une parfaite connaissance de son aile et de ses réactions
- Absolument inadaptées aux pilotes débutants, intermédiaires ou en reprise
"""
    
    # Ajouter des recommandations sur le niveau de l'aile
    wing_advice = """
### Choix de l'aile

"""
    
    if difficulty_score <= 5:
        wing_advice += "Toutes les catégories d'ailes sont adaptées, y compris école (EN-A)."
    elif difficulty_score <= 10:
        wing_advice += "Privilégiez des ailes de catégorie EN-A ou EN-B, offrant un bon compromis entre performance et sécurité passive."
    elif difficulty_score <= 15:
        wing_advice += "Ailes EN-B à EN-C selon votre expérience. Évitez les ailes trop exigeantes si vous n'êtes pas en pratique régulière."
    else:
        wing_advice += "⚠️ Ailes EN-B+ à EN-C pour leur mélange de performance et stabilité. Les ailes EN-D exigent une maîtrise parfaite dans ces conditions fortes."
    
    # Recommandations spécifiques selon les conditions
    specific_advice = """
### Recommandations spécifiques
"""
    
    if wind_factor >= 4:
        specific_advice += "- Maîtrisez parfaitement les techniques de contrôle au sol avant d'envisager le décollage\n"
    
    if thermal_factor >= 4:
        specific_advice += "- Anticipez les réactions de l'aile dans les thermiques puissants (pilotage actif)\n"
    
    if complexity_factor >= 2:
        specific_advice += "- Ces conditions complexes nécessitent une bonne capacité d'analyse et d'adaptation\n"
    
    if gradient_factor >= 3:
        specific_advice += "- Soyez particulièrement vigilant lors des transitions entre différentes masses d'air\n"
    
    return difficulty_desc + wing_advice + specific_advice

def _provide_tactical_advice(analysis):
    """Fournit des conseils tactiques adaptés aux conditions"""
    
    tactical_advice = """## Conseils tactiques pour optimiser votre vol\n\n"""
    
    # Conseils pour le décollage
    takeoff_advice = """
### Au décollage
"""
    
    # Vérifier les conditions de vent
    if "faible au sol" in analysis.wind_conditions.lower():
        takeoff_advice += """
- Privilégiez un décollage face à la pente avec une temporisation minimale
- Anticipez un gonflage dynamique pour compenser le vent faible
- Choisissez idéalement un moment avec une brise thermique établie
"""
    elif "modéré au sol" in analysis.wind_conditions.lower():
        takeoff_advice += """
- Conditions idéales pour un décollage contrôlé face au vent
- Adoptez une technique de gonflage adaptée à la force du vent (temporisation)
- Restez attentif aux variations de la brise thermique
"""
    else:  # Vent assez fort ou fort
        takeoff_advice += """
- ⚠️ Contrôlez parfaitement votre aile au sol avant d'envisager le décollage
- Utilisez des techniques de gonflage adaptées au vent fort (dos à l'aile possible)
- Faites-vous aider si nécessaire pour sécuriser les phases au sol
- Évitez les heures de vent fort ou turbulent
"""
    
    # Conseils pour l'exploitation des thermiques
    thermaling_advice = """
### Exploitation des thermiques
"""
    
    if analysis.thermal_strength == "Faible":
        thermaling_advice += """
- Gardez une vitesse de vol lente pour maximiser le taux de montée
- Acceptez des rayons de virage plus larges pour rester dans l'ascendance
- Soyez patient et exploitez même les ascendances faibles
- Restez près du relief où les thermiques sont plus définis
"""
    elif analysis.thermal_strength == "Modérée":
        thermaling_advice += """
- Utilisez un rayon de virage adapté à la taille du thermique
- Centrez efficacement en utilisant les variations de pression dans la sellette
- Ajustez votre vitesse selon que vous êtes dans ou hors de l'ascendance
- Cherchez à "connecter" les différentes ascendances entre elles
"""
    elif analysis.thermal_strength == "Forte":
        thermaling_advice += """
- Adoptez un pilotage actif pour contrer les mouvements de l'aile
- Utilisez des rayons de virage plus serrés dans les coeurs thermiques puissants
- Soyez prêt à relâcher la pression intérieure en cas de surpilotage
- Gardez une marge de sécurité par rapport au relief (les thermiques peuvent être turbulents)
"""
    else:  # "Très Forte"
        thermaling_advice += """
- ⚠️ Anticipez les réactions violentes de l'aile à l'entrée des thermiques
- Maintenez une pression constante dans l'aile par un pilotage très actif
- N'hésitez pas à voler légèrement accéléré entre les thermiques pour plus de stabilité
- Évitez de surpiloter: des actions douces mais précises sont plus efficaces
"""
    
    # Conseils pour les transitions
    transition_advice = """
### Stratégie en transition
"""
    
    if "peu de gradient" in analysis.wind_conditions.lower():
        transition_advice += """
- Planifiez vos transitions en ligne directe entre les ascendances
- Utilisez un régime de vol efficace (légèrement accéléré)
- Maintenez une hauteur de sécurité constante
"""
    elif "gradient de vent modéré" in analysis.wind_conditions.lower():
        transition_advice += """
- Tenez compte de la dérive plus importante en altitude
- Utilisez les nuages (ou les lignes de terrain) pour identifier les zones ascendantes
- Compensez le vent de face en volant plus vite, mais restez en finesse max
"""
    else:  # Fort gradient
        transition_advice += """
- ⚠️ Anticipez une dérive importante en altitude
- Planifiez vos transitions en tenant compte du vent: au vent des objectifs si possible
- Adaptez votre vitesse: plus rapide face au vent, plus lente vent arrière
- Gardez une marge de sécurité accrue par rapport au relief
"""
    
    # Conseils spécifiques selon le type de thermiques
    if analysis.thermal_type == "Bleu":
        transition_advice += """
- Sans marquage nuageux, utilisez d'autres indices: ailes en montée, oiseaux, zones favorables au sol
- Restez attentif aux sensations de montée même légères
- Les transitions peuvent nécessiter plus d'exploration
"""
    else:  # Cumulus
        transition_advice += """
- Utilisez les cumulus comme indicateurs fiables des zones ascendantes
- Visez légèrement au vent de la base des nuages
- Observez le cycle de vie des cumulus pour anticiper les meilleures zones
"""
    
    # Conseils pour l'approche et l'atterrissage
    landing_advice = """
### Approche et atterrissage
"""
    
    if "faible au sol" in analysis.wind_conditions.lower():
        landing_advice += """
- Prévoyez une approche longue avec suffisamment de hauteur
- Anticipez des conditions potentiellement thermiques à l'atterrissage
- Soyez prêt à gérer les effets de gradient inversé (vent plus faible au sol)
"""
    elif "modéré au sol" in analysis.wind_conditions.lower():
        landing_advice += """
- Réalisez une approche standard face au vent
- Soyez attentif aux effets de compression près du sol
- Gardez une vitesse adéquate en finale
"""
    else:  # Vent assez fort ou fort
        landing_advice += """
- ⚠️ Anticipez largement votre approche et restez face au vent
- Préparez-vous à rencontrer des turbulences à l'approche du sol
- Utilisez les oreilles si nécessaire pour augmenter votre taux de chute
- Choisissez si possible un atterrissage dégagé d'obstacles sous le vent
"""
    
    # Conseil sur la gestion des inversions si présentes
    inversion_advice = ""
    if analysis.inversion_layers:
        inversion_layers_under_ceiling = [layer for layer in analysis.inversion_layers 
                                          if layer[0] < analysis.thermal_ceiling]
        if inversion_layers_under_ceiling:
            inversion_advice = """
### Gestion des inversions
"""
            for base, top in inversion_layers_under_ceiling:
                inversion_advice += f"""
- Une inversion est présente entre {base:.0f}m et {top:.0f}m
- Attendez-vous à rencontrer une zone de ralentissement des thermiques
- Restez patient et continuez à travailler l'ascendance, même faible
- Si vous perdez l'ascendance, cherchez latéralement où le thermique pourrait avoir percé l'inversion
"""
    
    # Compilation des conseils
    complete_tactical_advice = tactical_advice + takeoff_advice + thermaling_advice + transition_advice + landing_advice + inversion_advice
    
    return complete_tactical_advice

def _interpret_emagramme_for_pilots(analysis):
    """Fournit une explication pratique de l'émagramme pour les pilotes"""
    
    emagramme_explanation = """## Lecture pratique de l'émagramme pour pilotes\n\n"""
    
    # Explication des courbes principales
    curves_explanation = """
### Les courbes essentielles
- **Courbe d'état (rouge)**: Elle montre la température réelle de l'atmosphère à différentes altitudes. C'est le "profil thermique" actuel.
- **Courbe du point de rosée (bleue)**: Elle indique la température à laquelle l'air se condense à chaque altitude.
- **Chemin du thermique (vert pointillé)**: Il représente l'évolution d'une particule d'air qui s'élève depuis le sol.
"""

    # Explication du gradient thermique
    gradient_explanation = f"""
### Le gradient thermique
Votre émagramme montre un gradient de {analysis.thermal_gradient:.1f}°C/1000m dans la couche convective.

- Un gradient proche de 0°C/1000m indique une atmosphère très stable
- Un gradient de 6,5°C/1000m correspond à l'atmosphère standard
- Un gradient de 7-10°C/1000m favorise de bons thermiques
- Au-delà de 10°C/1000m, l'atmosphère devient très instable

Pour le vol, ce gradient de {analysis.thermal_gradient:.1f}°C/1000m indique {"une atmosphère stable avec des thermiques limités" if analysis.thermal_gradient < 5 else "de bonnes conditions thermiques" if analysis.thermal_gradient < 8 else "une forte instabilité favorable aux thermiques puissants"}.
"""

    # Explication sur les inversions si présentes
    inversion_explanation = ""
    if analysis.inversion_layers:
        inversion_explanation = """
### Les inversions thermiques
Sur l'émagramme, les inversions sont visibles lorsque la courbe d'état (rouge) penche vers la droite au lieu de pencher vers la gauche.

Ces inversions agissent comme des "couvercles" qui peuvent:
- Bloquer ou ralentir la progression des thermiques
- Créer des plafonds artificiels plus bas que le plafond théorique
- Concentrer la pollution et l'humidité sous leur niveau
"""

    # Explication de la formation des nuages
    cloud_formation = ""
    if analysis.thermal_type == "Cumulus":
        cloud_formation = f"""
### Formation des cumulus
Sur l'émagramme, la base des nuages ({analysis.cloud_base:.0f}m) correspond au point où:
- La courbe du thermique (vert pointillé) croise la courbe du point de rosée (bleue)
- C'est le "niveau de condensation" où l'air devient saturé
- L'adiabatique sèche (-1°C/100m) devient humide (-0,6°C/100m)

La différence entre la courbe d'état (rouge) et la courbe du point de rosée (bleue) vous indique l'humidité relative à chaque altitude. Plus elles sont proches, plus l'air est humide, et plus la formation de nuages est probable.
"""
    else:
        cloud_formation = """
### Absence de cumulus (thermiques bleus)
Sur votre émagramme, on peut observer que:
- La courbe du thermique (vert pointillé) ne croise pas la courbe du point de rosée (bleue)
- L'air reste sec même en s'élevant et en se refroidissant
- Sans condensation, les thermiques restent invisibles
- Le spread (écart entre température et point de rosée) reste important en altitude
"""
    
    # Explication du plafond thermique
    ceiling_explanation = f"""
### Le plafond thermique
Le plafond thermique estimé à {analysis.thermal_ceiling:.0f}m correspond au point où:
- La courbe du thermique (vert pointillé) croise la courbe d'état (rouge)
- À cette altitude, la particule d'air n'est plus plus chaude que l'air environnant
- La poussée d'Archimède cesse, et l'ascendance s'arrête

Ce plafond représente la limite théorique maximale. En pratique, les thermiques peuvent s'arrêter plus bas à cause:
- Des pertes d'énergie par mélange avec l'air environnant
- Des inversions ou couches stables qui freinent l'ascendance
- D'un étalement horizontal du thermique avant d'atteindre le plafond théorique
"""

    # Implications pratiques
    practical_implications = """
### Ce que cela signifie pour votre vol

En termes pratiques, cet émagramme vous indique que:
"""

    if analysis.thermal_strength in ["Faible", "Modérée"]:
        practical_implications += """
- Vous devrez être patient et précis dans votre technique de centrage
- Les meilleures ascendances se trouveront près des zones de déclenchement optimales
- Privilégiez la régularité plutôt que la recherche de forts taux de montée
"""
    else:
        practical_implications += """
- Vous pourrez trouver des ascendances puissantes et bien définies
- Préparez-vous à un pilotage actif, surtout à l'entrée des thermiques
- Exploitez pleinement la couche convective pour réaliser des transitions efficaces
"""

    if analysis.thermal_type == "Cumulus":
        practical_implications += """
- Les cumulus vous serviront de repères fiables pour localiser les ascendances
- La hauteur de la base des nuages vous donnera une indication visuelle claire du plafond exploitable
"""
    else:
        practical_implications += """
- Sans marqueurs nuageux, fiez-vous à votre expérience et aux indices de terrain
- Observez les autres ailes et les oiseaux pour identifier les zones actives
"""

    # Assembler l'explication complète
    complete_emagramme_explanation = emagramme_explanation + curves_explanation + gradient_explanation + inversion_explanation + cloud_formation + ceiling_explanation + practical_implications
    
    return complete_emagramme_explanation

# Mettre à jour la classe EmagrammeAgent pour utiliser notre nouvelle analyse
class EnhancedEmagrammeAgent:
    """
    Version améliorée de l'agent IA qui fournit une analyse détaillée 
    adaptée aux besoins des parapentistes
    """
    
    def __init__(self, openai_api_key=None):
        """
        Initialise l'agent avec une clé API pour le LLM (optionnelle)
        
        Args:
            openai_api_key: Clé API pour OpenAI (facultative)
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
                logging.info("Module OpenAI initialisé avec succès")
            except ImportError:
                logging.warning("Module OpenAI non disponible. Installez-le avec: pip install openai")
    
    def analyze_conditions(self, analysis):
        """
        Génère une analyse détaillée des conditions de vol adaptée aux parapentistes
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme (EmagrammeAnalysis)
            
        Returns:
            Description détaillée en langage naturel
        """
        # Vérifier si l'API OpenAI est disponible et si oui, l'utiliser
        if self.has_openai:
            try:
                return self._get_openai_analysis(analysis)
            except Exception as e:
                logging.error(f"Erreur lors de l'analyse via OpenAI: {e}")
                # En cas d'erreur, utiliser notre analyse interne
                return analyze_emagramme_for_pilot(analysis)
        else:
            # Utiliser notre analyse interne avancée
            return analyze_emagramme_for_pilot(analysis)
            
    def _get_openai_analysis(self, analysis):
        """
        Obtient une analyse via l'API OpenAI
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Description détaillée générée par OpenAI
        """
        # Construire le prompt pour l'API
        prompt = self._build_prompt(analysis)
        
        # Appeler l'API
        response = self.openai.ChatCompletion.create(
            model="gpt-4",  # ou un autre modèle adapté
            messages=[
                {"role": "system", "content": """Tu es un expert en parapente et météorologie qui analyse 
                les émagrammes pour fournir des conseils de vol précis et utiles.
                Utilise un ton pédagogique mais direct, et concentre-toi sur les informations 
                pratiques pour les pilotes."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    def _build_prompt(self, analysis):
        """
        Construit un prompt détaillé pour l'API OpenAI
        
        Args:
            analysis: Résultats de l'analyse de l'émagramme
            
        Returns:
            Prompt formaté pour l'API
        """
        # Vérifier si des conditions dangereuses sont présentes
        vol_impossible = False
        raisons_impossibilite = []
        
        if analysis.precipitation_type is not None and analysis.precipitation_type != 0:
            vol_impossible = True
            raisons_impossibilite.append(analysis.precipitation_description)
        
        if "fort au sol" in analysis.wind_conditions.lower() or "critique" in analysis.wind_conditions.lower():
            vol_impossible = True
            raisons_impossibilite.append("Vent trop fort")
        
        # Construction du prompt détaillé
        prompt = f"""Analyse cet émagramme pour le vol en parapente de manière détaillée et pédagogique:

Site d'altitude: {analysis.ground_altitude}m
Température au sol: {analysis.ground_temperature:.1f}°C
Point de rosée au sol: {analysis.ground_dew_point:.1f}°C (Spread: {analysis.ground_spread:.1f}°C)
Plafond thermique: {analysis.thermal_ceiling:.0f}m
Gradient thermique: {analysis.thermal_gradient:.1f}°C/1000m
Force des thermiques: {analysis.thermal_strength}
Stabilité de l'atmosphère: {analysis.stability}
Type de thermiques: {analysis.thermal_type}
"""
        
        # Ajouter des informations sur les cumulus si présents
        if analysis.thermal_type == "Cumulus":
            prompt += f"Base des nuages: {analysis.cloud_base:.0f}m\n"
            prompt += f"Sommet des nuages: {analysis.cloud_top:.0f}m\n"
        elif analysis.thermal_inconsistency:
            prompt += f"Note importante: {analysis.thermal_inconsistency}\n"
        
        # Ajouter des informations sur les inversions
        if analysis.inversion_layers:
            prompt += "\nCouches d'inversion:\n"
            for i, (base, top) in enumerate(analysis.inversion_layers):
                prompt += f"- De {base:.0f}m à {top:.0f}m\n"
        
        # Ajouter des informations sur le vent
        prompt += f"\nConditions de vent: {analysis.wind_conditions}\n"
        
        # Ajouter des informations sur la couverture nuageuse
        cloud_info = "Informations sur la couverture nuageuse:\n"
        cloud_present = False
        
        if hasattr(analysis, 'low_cloud_cover') and analysis.low_cloud_cover is not None:
            cloud_info += f"- Nuages bas: {analysis.low_cloud_cover:.0f}%\n"
            cloud_present = True
            
        if hasattr(analysis, 'mid_cloud_cover') and analysis.mid_cloud_cover is not None:
            cloud_info += f"- Nuages moyens: {analysis.mid_cloud_cover:.0f}%\n"
            cloud_present = True
            
        if hasattr(analysis, 'high_cloud_cover') and analysis.high_cloud_cover is not None:
            cloud_info += f"- Nuages hauts: {analysis.high_cloud_cover:.0f}%\n"
            cloud_present = True
            
        if cloud_present:
            prompt += f"\n{cloud_info}\n"
        
        # Ajouter des informations sur les risques
        if analysis.hazards:
            prompt += "\nRisques identifiés:\n"
            for hazard in analysis.hazards:
                prompt += f"- {hazard}\n"
        
        # Instructions pour l'analyse
        if vol_impossible:
            raisons = ", ".join(raisons_impossibilite)
            prompt += f"""
IMPORTANT: Le vol est IMPOSSIBLE aujourd'hui en raison de: {raisons}.

Ton analyse doit:
1. Commencer par un avertissement clair expliquant pourquoi le vol est impossible et dangereux
2. Expliquer les risques spécifiques de ces conditions
3. Recommander explicitement de ne PAS voler
4. Ne pas suggérer d'heures optimales ou de stratégies de vol
5. Fournir une explication pédagogique de l'émagramme pour que le pilote comprenne la situation météorologique

Format ta réponse comme un article pédagogique organisé en sections claires.
"""
        else:
            prompt += f"""
Ta mission est de fournir une analyse complète et pédagogique avec:

1. Une description détaillée de la structure thermique (force, organisation, plafond)
2. Une explication de la stabilité atmosphérique et ses implications
3. Une analyse des conditions de vent à différentes altitudes
4. Des recommandations sur les heures optimales de vol
5. Des conseils tactiques pour exploiter au mieux les thermiques
6. Des suggestions sur les types de sites adaptés à ces conditions
7. Une évaluation du niveau de difficulté (débutant, intermédiaire, avancé)
8. Une explication claire et pédagogique de la lecture de l'émagramme pour aider le pilote

Format ta réponse comme un article pédagogique organisé en sections claires avec des titres.
Utilise un ton expert mais accessible aux pilotes de tous niveaux.
"""
        
        return prompt