import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class RevenuePredictionModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.features = None

    def train(self, data):
        try:
            self.features = [
                'population_density', 'avg_household_income', 'competitors_nearby',
                'accessibility_score', 'commercial_zone_score', 'loyalty_rate',
                'customer_satisfaction', 'avg_basket', 'daily_transactions',
                'seating_capacity', 'square_footage', 'operating_cost',
                'unemployment_rate', 'families_with_children', 'crime_rate'
            ]

            X = data[self.features]
            y = data['monthly_revenue']

            X_scaled = self.scaler.fit_transform(X)

            self.model.fit(X_scaled, y)
            return True
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")
            return False

    def predict(self, input_data):
        try:
            missing_cols = set(self.features) - set(input_data.columns)
            if missing_cols:
                raise ValueError(f"Colonnes manquantes: {missing_cols}")

            input_data = input_data[self.features]

            input_scaled = self.scaler.transform(input_data)

            prediction = self.model.predict(input_scaled)
            return prediction
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")
            return None

@st.cache_data
def load_and_prepare_data():
    """Charge et prépare les données"""
    try:
        np.random.seed(42)
        n_samples = 1000

        data = pd.DataFrame({
            'monthly_revenue': np.random.normal(50000, 15000, n_samples),
            'customer_satisfaction': np.random.exponential(1, n_samples).clip(1, 5),
            'loyalty_rate': np.random.beta(5, 2, n_samples),
            'operating_cost': np.random.normal(30000, 8000, n_samples),
            'avg_basket': np.random.normal(35, 8, n_samples),
            'daily_transactions': np.random.normal(200, 50, n_samples),
            'population_density': np.random.normal(5000, 1500, n_samples),
            'avg_household_income': np.random.normal(75000, 15000, n_samples),
            'competitors_nearby': np.random.randint(0, 15, n_samples),
            'accessibility_score': np.random.randint(1, 11, n_samples),
            'commercial_zone_score': np.random.randint(1, 11, n_samples),
            'seating_capacity': np.random.randint(30, 150, n_samples),
            'square_footage': np.random.normal(2500, 500, n_samples),
            'unemployment_rate': np.random.beta(2, 8, n_samples),
            'families_with_children': np.random.beta(5, 5, n_samples),
            'crime_rate': np.random.beta(2, 8, n_samples),
            'neighborhood_type': np.random.choice(['Urbain', 'Suburbain', 'Rural'], n_samples),
            'region': np.random.choice(['Montréal', 'Québec', 'Laval', 'Gatineau', 'Sherbrooke'], n_samples),
            'dine_in_sales': np.random.beta(5, 5, n_samples),
            'delivery_sales': np.random.beta(5, 5, n_samples),
            'drive_thru_sales': np.random.beta(5, 5, n_samples)
        })

        sales_sum = data[['dine_in_sales', 'delivery_sales', 'drive_thru_sales']].sum(axis=1)
        data['dine_in_sales'] = data['dine_in_sales'] / sales_sum
        data['delivery_sales'] = data['delivery_sales'] / sales_sum
        data['drive_thru_sales'] = data['drive_thru_sales'] / sales_sum

        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None
