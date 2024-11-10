import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu

def set_custom_style():
    """Configure le style moderne et dynamique de l'application"""
    st.markdown("""
        <style>
        /* Variables globales */
        :root {
            --primary-color: #6366F1;
            --secondary-color: #4F46E5;
            --background-color: #F9FAFB;
            --card-background: #FFFFFF;
            --text-primary: #1F2937;
            --text-secondary: #6B7280;
            --success-color: #10B981;
            --warning-color: #F59E0B;
            --error-color: #EF4444;
        }

        /* Style général */
        .main {
            padding: 2rem;
            background-color: var(--background-color);
        }

        /* Animation pour les cartes */
        @keyframes slideIn {
            from {
                transform: translateY(20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        /* Cartes métriques modernes */
        .metric-card {
            background: var(--card-background);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            margin: 1rem 0;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            animation: slideIn 0.5s ease-out forwards;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 0.5rem;
            transition: color 0.2s ease;
        }

        .metric-label {
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Graphiques modernisés */
        .stPlotlyChart {
            background: var(--card-background);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin: 1rem 0;
            transition: transform 0.2s ease;
        }

        .stPlotlyChart:hover {
            transform: scale(1.01);
        }

        /* Headers stylisés */
        h1 {
            color: var(--text-primary);
            font-weight: 800;
            font-size: 2.25rem;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 3px solid var(--primary-color);
            transition: color 0.2s ease;
        }

        h2 {
            color: var(--text-primary);
            font-weight: 700;
            font-size: 1.75rem;
            margin-top: 2.5rem;
            margin-bottom: 1.5rem;
        }

        /* Tabs modernisés */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: var(--card-background);
            padding: 0.75rem;
            border-radius: 12px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            color: var(--text-primary);
            border-radius: 8px;
            transition: all 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: rgba(99, 102, 241, 0.1);
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--primary-color);
            color: white;
        }

        /* Sidebar moderne */
        .sidebar .sidebar-content {
            background-color: var(--background-color);
            padding: 2rem 1rem;
        }

        /* DataFrames stylisés */
        .dataframe {
            border: none !important;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .dataframe th {
            background-color: var(--primary-color) !important;
            color: white !important;
            font-weight: 600;
            padding: 1rem !important;
        }

        .dataframe td {
            padding: 0.75rem !important;
            border-bottom: 1px solid #E5E7EB;
        }

        /* Boutons modernisés */
        .stButton button {
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
            border: none;
            background-color: var(--primary-color);
            color: white;
        }

        .stButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)


def create_filters_sidebar(data):
    """Crée un menu d'options avec des filtres dans la barre latérale"""

    # Menu de navigation dans la barre latérale
    with st.sidebar:
        selected_option = option_menu(
            menu_title="🔍 Menu",  # Titre du menu
            options=["Accueil", "Filtres", "Synthèse"],  # Options du menu
            icons=["house", "filter", "bar-chart"],  # Icônes pour chaque option
            menu_icon="menu-app",  # Icône pour le menu principal
            default_index=1,
            styles={
                "container": {"padding": "1rem", "background-color": "#F9FAFB"},
                "icon": {"color": "#6366F1", "font-size": "1.2rem"},
                "nav-link": {
                    "font-size": "1rem",
                    "text-align": "left",
                    "margin": "0.5rem",
                    "--hover-color": "#E5E7EB"
                },
                "nav-link-selected": {"background-color": "#6366F1", "color": "white"},
            }
        )

        # Si l'utilisateur sélectionne "Filtres", afficher les sélecteurs dans la barre latérale
        if selected_option == "Filtres":
            st.markdown("## 🛠️ Filtres de données")

            regions = ['Toutes'] + sorted(data['region'].unique().tolist())
            selected_region = st.selectbox('Région', regions)

            neighborhoods = ['Tous'] + sorted(data['neighborhood_type'].unique().tolist())
            selected_neighborhood = st.selectbox('Type de quartier', neighborhoods)

            min_revenue = int(data['monthly_revenue'].min())
            max_revenue = int(data['monthly_revenue'].max())
            revenue_range = st.slider(
                'Plage de revenus (CAD)',
                min_revenue, max_revenue,
                (min_revenue, max_revenue)
            )

            satisfaction_range = st.slider(
                'Satisfaction client',
                float(data['customer_satisfaction'].min()),
                float(data['customer_satisfaction'].max()),
                (3.5, 5.0)
            )

            # Filtrer les données
            filtered_data = data.copy()
            if selected_region != 'Toutes':
                filtered_data = filtered_data[filtered_data['region'] == selected_region]
            if selected_neighborhood != 'Tous':
                filtered_data = filtered_data[filtered_data['neighborhood_type'] == selected_neighborhood]

            filtered_data = filtered_data[
                (filtered_data['monthly_revenue'].between(revenue_range[0], revenue_range[1])) &
                (filtered_data['customer_satisfaction'].between(satisfaction_range[0], satisfaction_range[1]))
                ]

            st.markdown(f"### 📊 Résultats : {len(filtered_data)} restaurants trouvés")
            return filtered_data

        elif selected_option == "Synthèse":
            # Si l'utilisateur sélectionne "Synthèse", afficher les statistiques
            st.markdown("## 📊 Synthèse des données")
            st.markdown(f"""
            <div style='padding: 1rem; background: #F9FAFB; border-radius: 8px;'>
                <ul>
                    <li>Restaurants analysés: {len(data)}</li>
                    <li>Revenu moyen: {data['monthly_revenue'].mean():,.0f} CAD</li>
                    <li>Satisfaction moyenne: {data['customer_satisfaction'].mean():.2f}/5</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            return data

        else:
            st.markdown("# Bienvenue dans l'Analyse des Données 📊")
            st.markdown("Utilisez le menu pour explorer les filtres et les synthèses.")
            return data

def display_kpi_metrics(data):
    """Affiche les métriques KPI avec un design moderne"""
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        {
            'value': f"{data['monthly_revenue'].mean():,.0f} CAD",
            'label': "Revenu Moyen Mensuel",
            'icon': "💰",
            'color': "#6366F1"
        },
        {
            'value': f"{data['customer_satisfaction'].mean():.2f}/5",
            'label': "Satisfaction Client",
            'icon': "⭐",
            'color': "#10B981"
        },
        {
            'value': f"{data['loyalty_rate'].mean():.1%}",
            'label': "Taux de Fidélité",
            'icon': "🎯",
            'color': "#F59E0B"
        },
        {
            'value': f"{(data['monthly_revenue'].mean() - data['operating_cost'].mean()) / data['monthly_revenue'].mean():.1%}",
            'label': "Marge Opérationnelle",
            'icon': "📈",
            'color': "#EF4444"
        }
    ]

    for col, metric in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 24px; margin-bottom: 0.5rem;">{metric['icon']}</div>
                    <div class="metric-value" style="color: {metric['color']};">
                        {metric['value']}
                    </div>
                    <div class="metric-label">
                        {metric['label']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def display_performance_analysis(data):
    """Affiche l'analyse des performances avec un design moderne"""
    st.markdown("""
        <h2 style='
            color: #1F2937;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 2rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #6366F1;
        '>
            📈 Analyse des Performances par Région
        </h2>
    """, unsafe_allow_html=True)

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