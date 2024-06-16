import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def calculate_mean_score(scores):
    if scores:
        return sum(scores) / len(scores)
    else:
        return 0

def analyze_sentiment(note):
    analysis = TextBlob(note)
    return analysis.sentiment.polarity

def unify_tag_category(tag):
    if isinstance(tag, dict):
        tag_type = tag.get('type', '').lower()
        if tag_type in ['emociones', 'emotions']:
            return 'Emociones'
        elif tag_type in ['actividades', 'activities']:
            return 'Actividades'
        else:
            return tag.get('type', None)
    return None

def main():
    st.title("Análisis de Datos de Pixels")

    uploaded_file = st.file_uploader("Sube un archivo JSON", type="json")

    if uploaded_file is not None:
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['score'] = df['scores'].apply(calculate_mean_score)
        main_df = df[['date', 'score', 'notes']]

        main_df['rolling_avg_score'] = main_df['score'].rolling(window=10, min_periods=1).mean()
        main_df['sentiment'] = main_df['notes'].apply(analyze_sentiment)

        st.write("Datos procesados:")
        st.write(main_df)

        # Gráfico de Puntajes y Promedio Móvil
        st.subheader("Gráfico de Puntajes y Promedio Móvil")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=main_df['date'], y=main_df['score'], mode='lines', name='Puntaje Diario', line=dict(color='#736836', width=1)))
        fig.add_trace(go.Scatter(x=main_df['date'], y=main_df['rolling_avg_score'], mode='lines', name='Promedio Móvil (10 días)', line=dict(color='goldenrod', width=3)))
        fig.update_layout(template='plotly_dark', title='Puntaje Diario y Promedio Móvil', xaxis_title='Fecha', yaxis_title='Puntaje', legend_title='Leyenda')
        st.plotly_chart(fig, use_container_width=True)

        # Distribución de Puntajes de Estado de Ánimo
        st.subheader("Distribución de Puntajes de Estado de Ánimo")
        fig_hist = px.histogram(main_df, x='score', nbins=5, range_x=[0.5, 5.5], title='Distribución de Puntajes de Estado de Ánimo')
        fig_hist.update_layout(template='plotly_dark', xaxis_title='Puntaje', yaxis_title='Frecuencia', xaxis=dict(tickmode='linear', tick0=1, dtick=1), bargap=0.05)
        fig_hist.update_traces(marker_color='khaki')
        st.plotly_chart(fig_hist, use_container_width=True)

        # Análisis de tags para todas las categorías
        st.subheader("Frecuencia de Tags")
        tags_df = df.explode('tags')
        tags_df.dropna(subset=['tags'], inplace=True)
        tags_df['tag_entries'] = tags_df['tags'].apply(lambda x: x['entries'] if isinstance(x, dict) else [])
        tags_exploded = tags_df.explode('tag_entries')

        # Unificar categorías de tags en español e inglés
        tags_exploded['tags'] = tags_exploded['tags'].apply(lambda x: {'type': unify_tag_category(x), 'entries': x['entries']} if isinstance(x, dict) else x)
        tag_categories = tags_exploded['tags'].apply(lambda x: x['type'] if isinstance(x, dict) else None).unique()

        combined_tags_df = main_df.copy()

        for category in tag_categories:
            category_df = tags_exploded[tags_exploded['tags'].apply(lambda x: x['type'] if isinstance(x, dict) else None) == category]
            category_count = category_df['tag_entries'].value_counts().reset_index()
            category_count.columns = ['Tag', 'Frecuencia']

            st.write(f"Frecuencia de {category}")
            fig_category = px.bar(category_count, x='Tag', y='Frecuencia', title=f'Frecuencia de {category}')
            fig_category.update_layout(template='plotly_dark', xaxis_title=category, yaxis_title='Frecuencia')
            fig_category.update_traces(marker_color='khaki')
            st.plotly_chart(fig_category, use_container_width=True)

            category_df['flag'] = 1
            pivot_df = category_df.pivot_table(index='date', columns='tag_entries', values='flag', fill_value=0)
            combined_tags_df = pd.merge(combined_tags_df, pivot_df, on='date', how='left').fillna(0)

        # Correlación entre todas las categorías de tags y el puntaje
        st.subheader("Correlación entre Tags y Puntaje")
        numeric_df = combined_tags_df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        fig_corr = px.imshow(correlation_matrix, title='Matriz de Correlación')
        fig_corr.update_layout(coloraxis_colorscale='Viridis', template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)

        # Sentimiento de las notas
        st.subheader("Sentimiento de las Notas Diarias")
        fig_sentiment = go.Figure()

        # Promedio móvil del sentimiento (línea khaki)
        main_df['rolling_avg_sentiment'] = main_df['sentiment'].rolling(window=10, min_periods=1).mean()
        fig_sentiment.add_trace(go.Scatter(
            x=main_df['date'], y=main_df['rolling_avg_sentiment'],
            mode='lines',
            name='Promedio Móvil (10 días)',
            line=dict(color='khaki')
        ))

        # Actualizar layout del gráfico de sentimiento
        fig_sentiment.update_layout(
            template='plotly_dark',
            title='Sentimiento suavizado de las Notas Diarias',
            xaxis_title='Fecha',
            yaxis_title='Sentimiento',
            legend_title='Leyenda',
            hovermode='x unified'
        )

        # Mostrar el gráfico de sentimiento en Streamlit
        fig_sentiment.update_traces(marker_color='khaki')  # Cambia el color de las líneas del gráfico de sentimiento a 'khaki'
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # Modelo predictivo para puntaje de estado de ánimo
        st.subheader("Modelo Predictivo para Puntaje de Estado de Ánimo")
        
        # Preparar datos para el modelo
        X = combined_tags_df.drop(columns=['date', 'score', 'notes', 'rolling_avg_score', 'sentiment', 'rolling_avg_sentiment'])
        y = combined_tags_df['score']

        # Dividir los datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear y entrenar el modelo de regresión lineal
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Realizar predicciones
        y_pred = model.predict(X_test)

        # Evaluar el modelo
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error del modelo: {mse:.2f}")

        # Visualizar predicciones vs valores reales
        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fig_pred = px.scatter(results_df, x='Actual', y='Predicted', title='Predicciones vs Valores Reales')
        fig_pred.add_trace(go.Scatter(x=[0, 5], y=[0, 5], mode='lines', line=dict(color='red', dash='dash')))
        fig_pred.update_layout(template='plotly_dark', xaxis_title='Valores Reales', yaxis_title='Predicciones')
        st.plotly_chart(fig_pred, use_container_width=True)

        # Correlaciones más importantes entre actividades y emociones
        st.subheader("Correlaciones entre Actividades y Emociones")
        
        # Filtrar actividades y emociones
        activities_df = tags_exploded[tags_exploded['tags'].apply(lambda x: unify_tag_category(x) == 'Actividades')]
        emotions_df = tags_exploded[tags_exploded['tags'].apply(lambda x: unify_tag_category(x) == 'Emociones')]
        
        activities_df['activity_flag'] = 1
        emotions_df['emotion_flag'] = 1

        pivot_activities_df = activities_df.pivot_table(index='date', columns='tag_entries', values='activity_flag', fill_value=0)
        pivot_emotions_df = emotions_df.pivot_table(index='date', columns='tag_entries', values='emotion_flag', fill_value=0)

        combined_activities_emotions_df = pd.merge(pivot_activities_df, pivot_emotions_df, on='date', how='left').fillna(0)

        # Calcular las correlaciones entre actividades y emociones
        corr_activities_emotions = combined_activities_emotions_df.corr()

        # Preparar estructuras para almacenar los resultados
        activities = activities_df['tag_entries'].unique()
        correlations = {}

        # Calcular umbral dinámico para selección de correlaciones significativas
        threshold = corr_activities_emotions.abs().mean().mean()

        # Filtrar y ordenar las correlaciones significativas para cada actividad
        for activity in activities:
            correlations[activity] = corr_activities_emotions.loc[activity].drop(activity).abs().nlargest(5)

        # Graficar las correlaciones más altas para cada actividad
        st.subheader("Correlaciones más Altas entre Actividades y Emociones")

        for activity, top_correlations in correlations.items():
            if len(top_correlations) > 0:
                fig_corr_activity = go.Figure()

                # Preparar datos para las barras
                x_values = top_correlations.index
                y_values = top_correlations.values
                colors = ['green' if corr > 0 else 'red' for corr in y_values]

                fig_corr_activity.add_trace(go.Bar(
                    x=x_values,
                    y=y_values,
                    marker_color=colors,
                    orientation='v',
                    text=[f'{corr:.2f}' for corr in y_values],
                    textposition='outside'
                ))

                fig_corr_activity.update_layout(
                    title=f'Correlaciones para la Actividad: {activity}',
                    xaxis_title='Emoción',
                    yaxis_title='Correlación',
                    template='plotly_dark',
                    yaxis=dict(tickmode='linear', tick0=0, dtick=0.1),
                    bargap=0.05
                )

                st.plotly_chart(fig_corr_activity, use_container_width=True)

if __name__ == "__main__":
    main()