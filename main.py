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
            'customer_satisfaction': np.random.normal(4.2, 0.3, n_samples).clip(1, 5),
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

import plotly.graph_objects as go
import plotly.express as px

def create_revenue_distribution_plot(data):
    """Crée un graphique de distribution des revenus"""
    try:
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=data['monthly_revenue'],
            name='Distribution',
            nbinsx=30,
            marker_color='#2ecc71',
            opacity=0.7
        ))

        fig.add_trace(go.Violin(
            x=data['monthly_revenue'],
            name='Densité',
            side='positive',
            line_color='#e74c3c',
            fillcolor='rgba(0,0,0,0)'
        ))

        fig.update_layout(
            title={
                'text': 'Distribution des Revenus Mensuels',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Revenu Mensuel (CAD)",
            yaxis_title="Fréquence",
            showlegend=True,
            template='plotly_white',
            height=500
        )

        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création du graphique de distribution: {str(e)}")
        return None

def create_sales_channel_analysis(data):
    """Crée une analyse des canaux de vente"""
    try:
        avg_channels = {
            'Sur place': data['dine_in_sales'].mean(),
            'Livraison': data['delivery_sales'].mean(),
            'Service au volant': data['drive_thru_sales'].mean()
        }

        fig = go.Figure()

        fig.add_trace(go.Pie(
            values=list(avg_channels.values()),
            labels=list(avg_channels.keys()),
            hole=.7,
            marker_colors=['#3498db', '#e74c3c', '#2ecc71'],
            textinfo='label+percent',
            textposition='outside',
            textfont_size=14
        ))

        fig.update_layout(
            title={
                'text': 'Répartition des Ventes par Canal',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            annotations=[{
                'text': 'Canaux de<br>Distribution',
                'showarrow': False,
                'font_size': 20
            }],
            showlegend=False,
            height=500,
            template='plotly_white'
        )

        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création de l'analyse des canaux: {str(e)}")
        return None

def create_performance_heatmap(data):
    """Crée une carte de chaleur des corrélations"""
    try:
        metrics = [
            'monthly_revenue', 'customer_satisfaction', 'loyalty_rate',
            'operating_cost', 'avg_basket', 'daily_transactions'
        ]

        corr_matrix = data[metrics].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=metrics,
            y=metrics,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig.update_layout(
            title={
                'text': 'Corrélations entre Métriques Clés',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=500,
            template='plotly_white'
        )

        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création de la heatmap: {str(e)}")
        return None

def create_geographical_analysis(data):
    """Crée une analyse géographique des performances"""
    try:
        geo_data = data.groupby('region').agg({
            'monthly_revenue': 'mean',
            'customer_satisfaction': 'mean',
            'loyalty_rate': 'mean',
            'operating_cost': 'mean'
        }).round(2)

        fig = go.Figure()

        metrics = {
            'monthly_revenue': 'Revenu Mensuel (CAD)',
            'customer_satisfaction': 'Satisfaction Client',
            'loyalty_rate': 'Taux de Fidélité',
            'operating_cost': 'Coût d\'Exploitation (CAD)'
        }

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

        for i, (metric, label) in enumerate(metrics.items()):
            fig.add_trace(go.Bar(
                name=label,
                x=geo_data.index,
                y=geo_data[metric],
                marker_color=colors[i]
            ))

        fig.update_layout(
            title={
                'text': 'Performance par Région',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            barmode='group',
            xaxis_title="Région",
            yaxis_title="Valeur",
            height=500,
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig, geo_data
    except Exception as e:
        st.error(f"Erreur lors de la création de l'analyse géographique: {str(e)}")
        return None, None

def create_seasonal_analysis(data):
    """Crée une analyse des tendances saisonnières"""
    try:
        months = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
        seasonal_data = pd.DataFrame({
            'Month': months,
            'Revenue': [
                1.1, 0.9, 1.0, 1.2, 1.3, 1.4,
                1.5, 1.4, 1.2, 1.1, 1.0, 1.3
            ]
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=seasonal_data['Month'],
            y=seasonal_data['Revenue'],
            mode='lines+markers',
            name='Coefficient Saisonnier',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))

        fig.update_layout(
            title={
                'text': 'Tendances Saisonnières des Revenus',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Mois",
            yaxis_title="Coefficient Saisonnier",
            height=500,
            template='plotly_white',
            showlegend=True
        )

        return fig
    except Exception as e:
        st.error(f"Erreur lors de la création de l'analyse saisonnière: {str(e)}")
        return None

def set_custom_style():
    """Configure le style personnalisé de l'application"""
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
            background-color: #f8f9fa;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2C3E50;
        }
        .metric-label {
            font-size: 14px;
            color: #7F8C8D;
        }
        .stPlotlyChart {
            background-color: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        h1 {
            color: #2C3E50;
            font-weight: 700;
            padding-bottom: 1rem;
            border-bottom: 2px solid #3498db;
        }
        h2 {
            color: #34495E;
            font-weight: 600;
            margin-top: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: white;
            padding: 0.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .stTabs [data-baseweb="tab"] {
            padding: 1rem 2rem;
            color: #2C3E50;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

def create_filters_sidebar(data):
    """Crée les filtres dans la barre latérale"""
    st.sidebar.title("🔍 Filtres d'Analyse")

    regions = ['Toutes'] + sorted(data['region'].unique().tolist())
    selected_region = st.sidebar.selectbox('Région', regions)

    neighborhoods = ['Tous'] + sorted(data['neighborhood_type'].unique().tolist())
    selected_neighborhood = st.sidebar.selectbox('Type de quartier', neighborhoods)

    min_revenue = int(data['monthly_revenue'].min())
    max_revenue = int(data['monthly_revenue'].max())
    revenue_range = st.sidebar.slider(
        'Plage de revenus (CAD)',
        min_revenue, max_revenue,
        (min_revenue, max_revenue)
    )

    satisfaction_range = st.sidebar.slider(
        'Satisfaction client',
        float(data['customer_satisfaction'].min()),
        float(data['customer_satisfaction'].max()),
        (3.5, 5.0)
    )

    filtered_data = data.copy()
    if selected_region != 'Toutes':
        filtered_data = filtered_data[filtered_data['region'] == selected_region]
    if selected_neighborhood != 'Tous':
        filtered_data = filtered_data[filtered_data['neighborhood_type'] == selected_neighborhood]

    filtered_data = filtered_data[
        (filtered_data['monthly_revenue'].between(revenue_range[0], revenue_range[1])) &
        (filtered_data['customer_satisfaction'].between(satisfaction_range[0], satisfaction_range[1]))
        ]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Synthèse")
    st.sidebar.markdown(f"**Nombre de restaurants:** {len(filtered_data)}")
    st.sidebar.markdown(f"**Revenu moyen:** {filtered_data['monthly_revenue'].mean():,.0f} CAD")
    st.sidebar.markdown(f"**Satisfaction moyenne:** {filtered_data['customer_satisfaction'].mean():.2f}/5")

    return filtered_data

def display_kpi_metrics(data):
    """Affiche les métriques KPI principales"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,.0f} CAD</div>
                <div class="metric-label">Revenu Moyen Mensuel</div>
            </div>
        """.format(data['monthly_revenue'].mean()), unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}/5</div>
                <div class="metric-label">Satisfaction Client</div>
            </div>
        """.format(data['customer_satisfaction'].mean()), unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1%}</div>
                <div class="metric-label">Taux de Fidélité</div>
            </div>
        """.format(data['loyalty_rate'].mean()), unsafe_allow_html=True)

    with col4:
        margin = (data['monthly_revenue'].mean() - data['operating_cost'].mean()) / data['monthly_revenue'].mean()
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1%}</div>
                <div class="metric-label">Marge Opérationnelle</div>
            </div>
        """.format(margin), unsafe_allow_html=True)

def display_performance_analysis(data):
    """Affiche l'analyse des performances par région"""
    st.header("📈 Analyse des Performances par Région")

    col1, col2 = st.columns(2)

    with col1:
        region_revenue = data.groupby('region')['monthly_revenue'].mean().sort_values(ascending=False)
        fig_revenue = px.bar(
            region_revenue,
            title="Revenu Moyen par Région",
            labels={'value': 'Revenu Moyen (CAD)', 'region': 'Région'},
            color=region_revenue.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_revenue)

    with col2:
        region_satisfaction = data.groupby('region')['customer_satisfaction'].mean().sort_values(ascending=False)
        fig_satisfaction = px.bar(
            region_satisfaction,
            title="Satisfaction Client par Région",
            labels={'value': 'Satisfaction Moyenne', 'region': 'Région'},
            color=region_satisfaction.values,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_satisfaction)

    st.subheader("📊 Tableau Détaillé des Performances")
    performance_table = data.groupby('region').agg({
        'monthly_revenue': ['mean', 'std'],
        'customer_satisfaction': 'mean',
        'loyalty_rate': 'mean',
        'operating_cost': 'mean'
    }).round(2)

    performance_table.columns = [
        'Revenu Moyen', 'Écart-type Revenu',
        'Satisfaction Moyenne', 'Taux de Fidélité',
        'Coût Opérationnel Moyen'
    ]

    st.dataframe(performance_table)

def main():
    """Fonction principale de l'application"""
    st.set_page_config(
        page_title="Analyse de Marché - Restaurants de Poulet",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_custom_style()

    st.title("🍗 Analyse de Marché - Chaîne de Restaurants de Poulet")
    st.markdown("""
    ### Système d'Analyse Prédictive pour l'Expansion au Québec
    Cette application fournit une analyse détaillée du marché de la restauration rapide
    spécialisée dans le poulet, avec un focus sur les opportunités d'expansion au Québec.
    """)

    data = load_and_prepare_data()
    if data is None:
        st.error("❌ Erreur lors du chargement des données.")
        return

    filtered_data = create_filters_sidebar(data)

    display_kpi_metrics(filtered_data)

    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "🔍 Analyse détaillée",
        "🎯 Prédictions",
        "📈 Performance par région"
    ])

    with tabs[0]:
        st.header("📊 Vue d'ensemble du Marché")

        revenue_fig = create_revenue_distribution_plot(filtered_data)
        if revenue_fig:
            st.plotly_chart(revenue_fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            sales_fig = create_sales_channel_analysis(filtered_data)
            if sales_fig:
                st.plotly_chart(sales_fig)

        with col2:
            seasonal_fig = create_seasonal_analysis(filtered_data)
            if seasonal_fig:
                st.plotly_chart(seasonal_fig)

    with tabs[1]:
        st.header("🔍 Analyse Détaillée des Métriques")

        heatmap_fig = create_performance_heatmap(filtered_data)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)

        st.subheader("📍 Analyse par Type de Quartier")
        neighborhood_metrics = filtered_data.groupby('neighborhood_type').agg({
            'monthly_revenue': 'mean',
            'customer_satisfaction': 'mean',
            'loyalty_rate': 'mean',
            'competitors_nearby': 'mean'
        }).round(2)

        col1, col2 = st.columns(2)
        with col1:
            fig_neighborhood = px.bar(
                neighborhood_metrics,
                title="Métriques par Type de Quartier",
                barmode='group'
            )
            st.plotly_chart(fig_neighborhood)

        with col2:
            st.dataframe(neighborhood_metrics)

    with tabs[2]:
        st.header("🎯 Prédictions de Revenus")

        st.subheader("Entrez les caractéristiques du nouveau restaurant")

        col1, col2, col3 = st.columns(3)
        with col1:
            population = st.number_input("Densité de population", 1000, 10000, 5000)
            income = st.number_input("Revenu moyen des ménages", 30000, 150000, 75000)
            competitors = st.number_input("Nombre de concurrents", 0, 20, 5)

        with col2:
            satisfaction = st.slider("Satisfaction client prévue", 1.0, 5.0, 4.2)
            loyalty = st.slider("Taux de fidélité prévu", 0.0, 1.0, 0.7)
            basket = st.number_input("Panier moyen prévu", 10, 100, 35)

        with col3:
            capacity = st.number_input("Capacité d'accueil", 20, 200, 80)
            cost = st.number_input("Coût d'exploitation prévu", 10000, 100000, 30000)
            transactions = st.number_input("Transactions quotidiennes prévues", 50, 500, 200)

        if st.button("Calculer la Prédiction"):
            model = RevenuePredictionModel()
            success = model.train(filtered_data)

            if success:
                prediction_data = pd.DataFrame({
                    'population_density': [population],
                    'avg_household_income': [income],
                    'competitors_nearby': [competitors],
                    'accessibility_score': [8],  # Valeurs par défaut pour les autres métriques
                    'commercial_zone_score': [7],
                    'loyalty_rate': [loyalty],
                    'customer_satisfaction': [satisfaction],
                    'avg_basket': [basket],
                    'daily_transactions': [transactions],
                    'seating_capacity': [capacity],
                    'square_footage': [capacity * 20],  # Estimation basée sur la capacité
                    'operating_cost': [cost],
                    'unemployment_rate': [0.06],
                    'families_with_children': [0.4],
                    'crime_rate': [0.02]
                })

                prediction = model.predict(prediction_data)

                if prediction is not None:
                    st.success(f"Revenu mensuel prévu: {prediction[0]:,.2f} CAD")

                    lower_bound = prediction[0] * 0.85
                    upper_bound = prediction[0] * 1.15
                    st.info(f"Intervalle de confiance (±15%): {lower_bound:,.2f} CAD - {upper_bound:,.2f} CAD")
            else:
                st.error("Erreur lors de l'entraînement du modèle")

        with tabs[3]:
            st.header("📈 Analyse des Performances Régionales")

            geo_fig, geo_data = create_geographical_analysis(filtered_data)
            if geo_fig and geo_data is not None:
                st.plotly_chart(geo_fig, use_container_width=True)

                st.subheader("📊 Métriques Détaillées par Région")
                st.dataframe(geo_data)

                st.subheader("🔑 Facteurs de Succès par Région")
                success_factors = filtered_data.groupby('region').agg({
                    'competitors_nearby': 'mean',
                    'accessibility_score': 'mean',
                    'commercial_zone_score': 'mean',
                    'families_with_children': 'mean'
                }).round(2)

                success_factors.columns = [
                    'Concurrents à proximité',
                    'Score d\'accessibilité',
                    'Score zone commerciale',
                    'Proportion familles'
                ]

                st.dataframe(success_factors)

if __name__ == "__main__":
    main()


