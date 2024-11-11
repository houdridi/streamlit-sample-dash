import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from model import RevenuePredictionModel, load_and_prepare_data
from visualizations import (
    create_revenue_distribution_plot, create_sales_channel_analysis,
    create_performance_heatmap, create_geographical_analysis, create_neighborhood_analysis,
    create_seasonal_analysis
)
from utils import set_custom_style, create_filters_sidebar, display_kpi_metrics, display_performance_analysis


def generate_fake_data(n=100):
    np.random.seed(42)
    data = pd.DataFrame({
        'restaurant_id': range(1, n + 1),
        'revenue': np.random.randint(50000, 200000, n),
        'satisfaction': np.random.uniform(3.5, 5, n),
        'location': np.random.choice(['Urbain', 'Suburbain', 'Rural'], n),
        'menu_items': np.random.randint(10, 30, n),
        'avg_price': np.random.uniform(8, 25, n),
        'delivery_percentage': np.random.uniform(0.2, 0.8, n),
        'month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                  n),
        'region': np.random.choice(['Montréal', 'Québec', 'Laval', 'Gatineau', 'Sherbrooke'], n),
        'capacity': np.random.randint(30, 150, n)
    })
    # Dictionary of coordinates for Québec cities
    QUEBEC_CITIES = {
        'Montréal': {'lat': 45.5017, 'lon': -73.5673},
        'Québec': {'lat': 46.8139, 'lon': -71.2080},
        'Laval': {'lat': 45.5867, 'lon': -73.7242},
        'Gatineau': {'lat': 45.4765, 'lon': -75.7013},
        'Sherbrooke': {'lat': 45.4040, 'lon': -71.8929}
    }

    # Map the coordinates based on the 'region' column
    data['latitude'] = data['region'].map(lambda x: QUEBEC_CITIES.get(x, {}).get('lat'))
    data['longitude'] = data['region'].map(lambda x: QUEBEC_CITIES.get(x, {}).get('lon'))

    return data


def create_charts_page(data):
    st.header("📊 Analyses et Visualisations Avancées")

    # Nouvelle section : Carte interactive du Québec
    st.subheader("📍 Carte des Restaurants par Région")

    # Coordonnées des villes québécoises
    if data is None:
        st.error("No data provided!")
        return

    if 'region' not in data.columns:
        st.error("Error: 'region' column is missing!")
        return

    # Dictionary of coordinates for Québec cities
    QUEBEC_CITIES = {
        'Montréal': {'lat': 45.5017, 'lon': -73.5673},
        'Québec': {'lat': 46.8139, 'lon': -71.2080},
        'Laval': {'lat': 45.5867, 'lon': -73.7242},
        'Gatineau': {'lat': 45.4765, 'lon': -75.7013},
        'Sherbrooke': {'lat': 45.4040, 'lon': -71.8929}
    }

    data['latitude'] = data['region'].map(lambda x: QUEBEC_CITIES.get(x, {}).get('lat'))
    data['longitude'] = data['region'].map(lambda x: QUEBEC_CITIES.get(x, {}).get('lon'))

    # Calculer les statistiques par région
    region_stats = data.groupby('region').agg({
        'revenue': ['mean', 'count'],
        'satisfaction': 'mean',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

    region_stats.columns = ['region', 'avg_revenue', 'restaurant_count', 'avg_satisfaction', 'latitude', 'longitude']

    # Créer la carte
    fig_map = go.Figure()

    # Ajouter les marqueurs pour chaque ville
    for _, row in region_stats.iterrows():
        fig_map.add_trace(go.Scattermapbox(
            lat=[row['latitude']],
            lon=[row['longitude']],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=row['restaurant_count']*2,
                color=row['avg_revenue'],
                colorscale='Viridis',
                showscale=False,
                colorbar=dict(title="Revenu moyen")
            ),
            text=f"{row['region']}<br>" +
                 f"Nombre de restaurants: {row['restaurant_count']}<br>" +
                 f"Revenu moyen: ${row['avg_revenue']:,.2f}<br>" +
                 f"Satisfaction: {row['avg_satisfaction']:.2f}/5",
            hoverinfo='text',
            name=row['region']
        ))

    # Configurer la mise en page de la carte
    fig_map.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            center=dict(lat=46.0, lon=-73.0),  # Centre du Québec
            zoom=6
        ),
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # Afficher la carte
    st.plotly_chart(fig_map, use_container_width=False)


    # with col1:
    #     st.metric("Région la plus rentable",
    #               region_stats.loc[region_stats['avg_revenue'].idxmax(), 'region'],
    #               f"${region_stats['avg_revenue'].max():,.2f}")
    #
    # with col2:
    #     st.metric("Plus grand nombre de restaurants",
    #               region_stats.loc[region_stats['restaurant_count'].idxmax(), 'region'],
    #               region_stats['restaurant_count'].max())
    #
    # with col3:
    #     st.metric("Meilleure satisfaction client",
    #               region_stats.loc[region_stats['avg_satisfaction'].idxmax(), 'region'],
    #               f"{region_stats['avg_satisfaction'].max():.2f}/5")

    # 1. Scatter plot: Revenue vs Satisfaction
    st.subheader("Relation entre le revenu et la satisfaction client")
    fig_scatter = px.scatter(data, x='satisfaction', y='revenue', color='location',
                             size='capacity', hover_data=['restaurant_id'],
                             title="Revenu vs Satisfaction par emplacement")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 2. Bar chart: Average revenue by location
    st.subheader("Revenu moyen par type d'emplacement")
    avg_revenue = data.groupby('location')['revenue'].mean().reset_index()
    fig_bar = px.bar(avg_revenue, x='location', y='revenue', color='location',
                     title="Revenu moyen par type d'emplacement")
    st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Pie chart: Delivery vs Dine-in
    st.subheader("Répartition des ventes : Livraison vs Sur place")
    delivery_data = pd.DataFrame({
        'Type': ['Livraison', 'Sur place'],
        'Percentage': [data['delivery_percentage'].mean(), 1 - data['delivery_percentage'].mean()]
    })
    fig_pie = px.pie(delivery_data, values='Percentage', names='Type',
                     title="Répartition des ventes : Livraison vs Sur place")
    st.plotly_chart(fig_pie, use_container_width=True)

    # 4. Heatmap: Correlation matrix
    st.subheader("Matrice de corrélation des variables clés")
    corr_matrix = data[['revenue', 'satisfaction', 'menu_items', 'avg_price', 'delivery_percentage', 'capacity']].corr()
    fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                            title="Matrice de corrélation des variables clés")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # 5. Line chart: Monthly revenue trend
    st.subheader("Tendance des revenus mensuels")
    monthly_revenue = data.groupby('month')['revenue'].mean().reset_index()
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_revenue['month'] = pd.Categorical(monthly_revenue['month'], categories=month_order, ordered=True)
    monthly_revenue = monthly_revenue.sort_values('month')
    fig_line = px.line(monthly_revenue, x='month', y='revenue', markers=True,
                       title="Tendance des revenus mensuels")
    st.plotly_chart(fig_line, use_container_width=True)
    # 6. Box plot: Revenue distribution by location
    st.subheader("Distribution des revenus par emplacement")
    fig_box = px.box(data, x='location', y='revenue', color='location',
                     title="Distribution des revenus par type d'emplacement")
    st.plotly_chart(fig_box, use_container_width=True)

    # 7. Scatter plot: Average price vs Menu items
    st.subheader("Relation entre le prix moyen et le nombre d'articles au menu")
    fig_scatter_menu = px.scatter(data, x='menu_items', y='avg_price', color='location',
                                  size='revenue', hover_data=['restaurant_id'],
                                  title="Prix moyen vs Nombre d'articles au menu")
    st.plotly_chart(fig_scatter_menu, use_container_width=True)

    # 10. Stacked bar chart: Revenue composition by location
    st.subheader("Composition des revenus par emplacement")
    data['delivery_revenue'] = data['revenue'] * data['delivery_percentage']
    data['dine_in_revenue'] = data['revenue'] - data['delivery_revenue']
    revenue_composition = data.groupby('location')[['delivery_revenue', 'dine_in_revenue']].sum().reset_index()
    fig_stacked = px.bar(revenue_composition, x='location', y=['delivery_revenue', 'dine_in_revenue'],
                         title="Composition des revenus par emplacement",
                         labels={'value': 'Revenu', 'variable': 'Type de service'})
    st.plotly_chart(fig_stacked, use_container_width=True)

    # 11. Sunburst chart: Hierarchical view of revenue by location and satisfaction
    st.subheader("Vue hiérarchique des revenus par emplacement et satisfaction")
    data['satisfaction_group'] = pd.cut(data['satisfaction'], bins=[0, 3, 4, 5], labels=['Faible', 'Moyen', 'Élevé'])
    fig_sunburst = px.sunburst(data, path=['location', 'satisfaction_group'], values='revenue',
                               title="Répartition des revenus par emplacement et niveau de satisfaction")
    st.plotly_chart(fig_sunburst, use_container_width=True)

    # 12. Parallel coordinates plot: Multi-dimensional analysis
    st.subheader("Analyse multidimensionnelle des restaurants")
    fig_parallel = px.parallel_coordinates(data, color="revenue",
                                           dimensions=['revenue', 'satisfaction', 'capacity', 'menu_items',
                                                       'avg_price'],
                                           title="Analyse multidimensionnelle des caractéristiques des restaurants")
    st.plotly_chart(fig_parallel, use_container_width=True)

    # 13. Treemap: Revenue by location and delivery percentage
    st.subheader("Treemap des revenus par emplacement et pourcentage de livraison")
    fig_treemap = px.treemap(data, path=[px.Constant("Tous"), 'location', 'delivery_percentage'], values='revenue',
                             color='delivery_percentage', hover_data=['revenue'],
                             title="Répartition des revenus par emplacement et pourcentage de livraison")
    st.plotly_chart(fig_treemap, use_container_width=True)

    # 14. Violin plot: Distribution of average price by location
    st.subheader("Distribution du prix moyen par emplacement")
    fig_violin = px.violin(data, x='location', y='avg_price', color='location', box=True, points="all",
                           title="Distribution du prix moyen par emplacement")
    st.plotly_chart(fig_violin, use_container_width=True)

    # 15. Radar chart: Restaurant profiles by location
    st.subheader("Profils des restaurants par emplacement")
    avg_metrics = data.groupby('location')[['revenue', 'satisfaction', 'capacity', 'menu_items', 'avg_price']].mean()
    fig_radar = go.Figure()
    for location in avg_metrics.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_metrics.loc[location],
            theta=avg_metrics.columns,
            fill='toself',
            name=location
        ))
    fig_radar.update_layout(title="Profils moyens des restaurants par emplacement")
    st.plotly_chart(fig_radar, use_container_width=True)

def main():
    """Fonction principale de l'application"""
    st.set_page_config(
        page_title="Analyse de Marché - Restaurants de Poulet",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Charger le style personnalisé
    set_custom_style()

    # Titre principal
    st.title("🍗 Analyse de Marché - Chaîne de Restaurants de Poulet")
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:10px; border-radius:10px;">
        <h3 style="color:#333333;">Système d'Analyse Prédictive pour l'Expansion au Québec</h3>
        <p style="color:#666666;">
        Cette application fournit une analyse détaillée du marché de la restauration rapide spécialisée dans le poulet,
        avec un focus sur les opportunités d'expansion au Québec.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Chargement des données
    data = load_and_prepare_data()
    print(data)
    if data is None:
        st.error("❌ Erreur lors du chargement des données.")
        return

    # Barre latérale avec filtres
    with st.sidebar:
        filtered_data = create_filters_sidebar(data)

    # Affichage des KPI
    display_kpi_metrics(filtered_data)

    # Ajout d'onglets pour l'analyse
    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "🔍 Analyse détaillée",
        "🎯 Prédictions",
        "📈 Performance par région",
        "📊 Analyses Avancées"
    ])

    # ---- Vue d'ensemble ----
    with tabs[0]:
        st.header("📊 Vue d'ensemble du Marché")
        st.markdown("---")

        # Distribution des revenus
        st.subheader("Distribution des Revenus")
        revenue_fig = create_revenue_distribution_plot(filtered_data)
        if revenue_fig:
            st.plotly_chart(revenue_fig, use_container_width=True)

        # Analyse des canaux de vente et saisonnalité
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Analyse des Canaux de Vente")
            sales_fig = create_sales_channel_analysis(filtered_data)
            if sales_fig:
                st.plotly_chart(sales_fig, use_container_width=True)

        with col2:
            st.subheader("Analyse de la Saison")
            seasonal_fig = create_seasonal_analysis(filtered_data)
            if seasonal_fig:
                st.plotly_chart(seasonal_fig, use_container_width=True)

    # ---- Analyse détaillée ----
        with tabs[1]:
            st.header("🔍 Analyse Détaillée des Métriques")
            st.markdown("---")

            # Carte de performance
            st.subheader("Carte de Performance")
            heatmap_fig = create_performance_heatmap(filtered_data)
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)

            st.subheader("📍 Analyse par Type de Quartier")
            radar_fig, bar_fig, neighborhood_metrics = create_neighborhood_analysis(filtered_data)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.plotly_chart(radar_fig, use_container_width=True)
                st.markdown("### Interprétation des résultats")
            with col2:
                st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("### Valeurs détaillées par type de quartier")
            st.dataframe(neighborhood_metrics)

        # ---- Prédictions ----
        with tabs[2]:
            st.header("🎯 Prédictions de Revenus")
            st.markdown("---")
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
                    # Préparer les données pour la prédiction
                    prediction_data = pd.DataFrame({
                        'population_density': [population],
                        'avg_household_income': [income],
                        'competitors_nearby': [competitors],
                        'loyalty_rate': [loyalty],
                        'customer_satisfaction': [satisfaction],
                        'avg_basket': [basket],
                        'daily_transactions': [transactions],
                        'seating_capacity': [capacity],
                        'operating_cost': [cost],
                        'commercial_zone_score': [7],
                        'accessibility_score': [8],
                        'unemployment_rate': [0.06],
                        'crime_rate': [0.02],
                        'families_with_children': [0.4],
                        'square_footage': [capacity * 20]
                    })

                    prediction = model.predict(prediction_data)

                    if prediction is not None:
                        predicted_revenue = prediction[0]
                        st.success(f"Revenu mensuel prévu: {predicted_revenue:,.2f} CAD")

                        # Calcul des intervalles de confiance
                        lower_bound = predicted_revenue * 0.85
                        upper_bound = predicted_revenue * 1.15
                        st.info(f"Intervalle de confiance (±15%): {lower_bound:,.2f} CAD - {upper_bound:,.2f} CAD")

                        # Définir le seuil de rentabilité
                        profit_threshold = cost * 1.2

                        # Définir la couleur de la recommandation
                        gauge_color = "green" if predicted_revenue > profit_threshold else "red"
                        recommendation = "✅ Ouvrir la succursale" if predicted_revenue > profit_threshold else "❌ Ne pas ouvrir la succursale"

                        st.markdown(f"**Recommandation : {recommendation}**")

                        # Création du Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=predicted_revenue,
                            delta={'reference': profit_threshold, 'increasing': {'color': "green"},
                                   'decreasing': {'color': "red"}},
                            gauge={
                                'axis': {'range': [0, max(predicted_revenue, profit_threshold) * 1.5]},
                                'bar': {'color': gauge_color},
                                'steps': [
                                    {'range': [0, profit_threshold], 'color': 'lightcoral'},
                                    {'range': [profit_threshold, max(predicted_revenue, profit_threshold) * 1.5],
                                     'color': 'lightgreen'}
                                ],
                                'threshold': {
                                    'line': {'color': "blue", 'width': 4},
                                    'thickness': 0.75,
                                    'value': profit_threshold
                                }
                            },
                            title={'text': "Revenu Prévu vs Seuil de Rentabilité"}
                        ))

                        fig.update_layout(height=400, margin=dict(l=50, r=50, t=50, b=50))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Erreur lors de la prédiction.")

        # ---- Analyse par région ----
        with tabs[3]:
            st.header("📈 Performance par région")
            display_performance_analysis(filtered_data)

        # ---- Analyses Avancées ----
        with tabs[4]:
            fake_data = generate_fake_data()
            create_charts_page(fake_data)

if __name__ == "__main__":
    main()
