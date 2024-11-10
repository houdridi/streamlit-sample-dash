import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

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
        months = pd.date_range(start='2023-01-01', end='2023-12-31', freq='ME')
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

def create_neighborhood_analysis(data):
    # Sélectionner les métriques clés
    key_metrics = ['monthly_revenue', 'customer_satisfaction', 'loyalty_rate', 'competitors_nearby', 'avg_basket',
                   'daily_transactions']

    neighborhood_metrics = data.groupby('neighborhood_type')[key_metrics].mean().round(2)

    # Normaliser les données pour le graphique radar
    normalized_metrics = (neighborhood_metrics - neighborhood_metrics.min()) / (
                neighborhood_metrics.max() - neighborhood_metrics.min())

    # Formater les valeurs pour l'affichage
    display_metrics = neighborhood_metrics.copy()
    display_metrics['monthly_revenue'] = display_metrics['monthly_revenue'].map('${:,.0f}'.format)
    display_metrics['customer_satisfaction'] = display_metrics['customer_satisfaction'].map('{:.2f}'.format)
    display_metrics['loyalty_rate'] = display_metrics['loyalty_rate'].map('{:.1%}'.format)
    display_metrics['competitors_nearby'] = display_metrics['competitors_nearby'].map('{:.1f}'.format)
    display_metrics['avg_basket'] = display_metrics['avg_basket'].map('${:.2f}'.format)
    display_metrics['daily_transactions'] = display_metrics['daily_transactions'].map('{:.0f}'.format)

    # Renommer les colonnes
    new_column_names = ['Revenu Mensuel', 'Satisfaction Client', 'Taux de Fidélité', 'Concurrents à Proximité',
                        'Panier Moyen', 'Transactions Quotidiennes']
    display_metrics.columns = new_column_names
    normalized_metrics.columns = new_column_names

    # Graphique en radar
    fig_radar = go.Figure()

    for neighborhood in normalized_metrics.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=normalized_metrics.loc[neighborhood],
            theta=new_column_names,
            fill='toself',
            name=neighborhood
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Comparaison des Métriques par Type de Quartier",
        height=500
    )

    # Graphique à barres pour les revenus
    fig_bar = px.bar(
        neighborhood_metrics,
        x='monthly_revenue',
        y=neighborhood_metrics.index,
        orientation='h',
        title="Revenu Mensuel Moyen par Type de Quartier",
        labels={'monthly_revenue': 'Revenu Mensuel Moyen', 'neighborhood_type': 'Type de Quartier'},
        height=300
    )
    fig_bar.update_traces(texttemplate='${:,.0f}', textposition='outside')

    return fig_radar, fig_bar, display_metrics