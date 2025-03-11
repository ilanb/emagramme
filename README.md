
# Emagramme - Analyse Atmosphérique pour le Vol Libre

## Description

Emagramme est une application interactive permettant d'analyser les profils atmosphériques en lien avec l'aérologie et la météorologie. Conçue pour les pilotes de parapente, planeur et autres pratiquants du vol libre, elle aide à interpréter les conditions météo et à prendre des décisions informées avant le vol.

L'application utilise **Streamlit** pour l'interface utilisateur et **matplotlib** pour la visualisation graphique des données atmosphériques.

## Fonctionnalités

* Analyse des couches atmosphériques
* Identification des inversions thermiques
* Visualisation des mouvements d'air verticaux (thermiques, vents anabatiques)
* Interprétation des conditions météo pour le vol libre
* Recommandations basées sur les conditions prévues
* Interface interactive via **Streamlit**

## Installation

### Prérequis

* Python 3.8+
* pip (gestionnaire de paquets Python)

### Installation des dépendances

Clonez ce dépôt et installez les dépendances requises :

```
git clone https://github.com/ilanb/emagramme.git
cd emagramme
pip install -r requirements.txt
```

## Utilisation

Pour lancer l'application **Streamlit**, exécutez :

```
streamlit run streamlit-app.py
```

Cela ouvrira l'interface de l'application dans votre navigateur.

## Fichiers principaux

* `<span>emagramme_analyzer.py</span>` : Module d'analyse des données atmosphériques
* `<span>enhanced_emagramme_analysis.py</span>` : Version améliorée de l'analyse des données
* `<span>streamlit-app.py</span>` : Interface utilisateur avec **Streamlit**

## Contribution

Les contributions sont les bienvenues ! Pour proposer des modifications :

1. Forkez le dépôt
2. Créez une branche (`<span>feature-nouvelle-fonction</span>`)
3. Effectuez vos modifications et commitez-les
4. Soumettez une Pull Request

## Licence

Ce projet est distribué sous la licence MIT. Voir le fichier [LICENSE]() pour plus de détails.

## Auteurs

* [Ilan B.](https://github.com/ilanb)
