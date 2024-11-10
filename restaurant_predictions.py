import pandas as pd
import numpy as np


def predict_performance(panier_moyen, surface, nb_places, type_service,
                        revenu_moyen, taux_chomage, densite_pop,
                        type_quartier, nb_concurrents, distance_transport):
    """
    Calcule les prédictions de performance basées sur les paramètres
    """
    # Base calculation for monthly revenue
    capacite_theorique = nb_places * 4  # 4 rotations par jour en moyenne
    taux_occupation = calculate_occupation_rate(
        type_quartier, distance_transport, nb_concurrents
    )
    multiplicateur_service = len(type_service) * 0.3 + 0.7  # Plus de services = plus de revenus

    # Calculer le nombre de clients par jour
    clients_base = capacite_theorique * taux_occupation
    clients_ajustes = clients_base * multiplicateur_service

    # Ajuster selon les facteurs sociodémographiques
    facteur_socio = (revenu_moyen / 60000) * (1 - taux_chomage) * (densite_pop / 3000)
    clients_jour = clients_ajustes * facteur_socio

    # Calculer le CA mensuel
    ca_mensuel = clients_jour * panier_moyen * 30

    # Calculer la marge
    cout_fixe = calculate_fixed_costs(surface, nb_places)
    cout_variable = ca_mensuel * 0.4  # 40% de coûts variables
    marge = ((ca_mensuel - cout_fixe - cout_variable) / ca_mensuel) * 100

    return {
        'ca_mensuel': ca_mensuel,
        'ca_trend': 5.5,  # Exemple de tendance
        'clients_jour': clients_jour,
        'clients_trend': 3.2,
        'marge': round(marge, 1),
        'marge_trend': 0.8,
        'taux_occupation': taux_occupation
    }


def calculate_occupation_rate(type_quartier, distance_transport, nb_concurrents):
    """Calcule le taux d'occupation en fonction des paramètres"""
    base_rate = {
        "Commercial": 0.7,
        "Mixte": 0.6,
        "Résidentiel": 0.5
    }

    rate = base_rate[type_quartier]
    rate *= (1 - distance_transport * 0.1)  # Impact distance transport
    rate *= (1 - nb_concurrents * 0.02)  # Impact concurrence

    return max(0.2, min(0.9, rate))  # Borner entre 20% et 90%


def calculate_fixed_costs(surface, nb_places):
    """Calcule les coûts fixes mensuels"""
    cout_loyer = surface * 2  # $2 par pied carré
    cout_personnel = nb_places * 200  # $200 par place
    autres_couts = 5000  # Coûts fixes divers

    return cout_loyer + cout_personnel + autres_couts


def generate_monthly_projection(predictions):
    """Génère une projection mensuelle sur 12 mois"""
    base_ca = predictions['ca_mensuel']
    months = range(1, 13)

    data = {
        'Mois': [f'Mois {m}' for m in months],
        "Chiffre d'Affaires": [
            base_ca * (1 + (m - 1) * 0.02 + np.random.normal(0, 0.01))
            for m in months
        ]
    }

    return pd.DataFrame(data)


def analyze_impact_factors(predictions):
    """Analyse l'impact des différents facteurs sur la performance"""
    data = {
        'Facteur': [
            'Localisation',
            'Type de Service',
            'Capacité',
            'Prix',
            'Concurrence'
        ],
        'Impact': [0.8, 0.6, 0.7, 0.5, -0.4]
    }

    return pd.DataFrame(data)


def generate_strategic_recommendations(predictions):
    """Génère des recommandations stratégiques basées sur les prédictions"""
    return {
        "Marketing": [
            "Concentrer la publicité aux heures de pointe",
            "Développer un programme de fidélité",
            "Cibler les entreprises locales pour le déjeuner"
        ],
        "Opérations": [
            f"Optimiser les rotations pour {predictions['clients_jour']:.0f} clients/jour",
            "Ajuster le personnel selon les pics d'affluence",
            "Maintenir un stock pour {predictions['clients_jour']*1.2:.0f} clients"
        ],
        "Finance": [
            f"Prévoir un CA mensuel de ${predictions['ca_mensuel']:,.2f}",
            f"Viser une marge de {predictions['marge']}%",
            "Négocier les contrats fournisseurs sur 12 mois"
        ]
    }
