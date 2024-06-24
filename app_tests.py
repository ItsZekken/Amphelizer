import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def calculate_mean_score(scores):
    return sum(scores) / len(scores) if scores else 0 

def main():
    st.title("Análisis de Datos de Pixels")

    uploaded_file = st.file_uploader("Sube un archivo JSON", type="json")
    api_key = "AIzaSyBeRubHVgurpgUP-PMkW56St_rweW02tqQ"

    if uploaded_file is not None:
        data = json.load(uploaded_file)
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['score'] = df['scores'].apply(calculate_mean_score)
        main_df = df[['date', 'score', 'notes']]
        main_df['rolling_avg_score'] = main_df['score'].rolling(window=10, min_periods=1).mean()

        st.write("Datos procesados:")
        st.write(main_df)

        # Gráfico de Puntajes y Promedio Móvil
        st.subheader("Gráfico de Puntajes y Promedio Móvil")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=main_df['date'], y=main_df['score'], mode='lines', name='Puntaje Diario', line=dict(color='#736836', width=1)))
        fig.add_trace(go.Scatter(x=main_df['date'], y=main_df['rolling_avg_score'], mode='lines', name='Promedio Móvil (10 días)', line=dict(color='khaki', width=3)))
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
        
        tag_categories = tags_df['tags'].apply(lambda x: x['type'] if isinstance(x, dict) else None).unique()
        combined_tags_df = main_df.copy()

        category_mapping = {
            'Actividades': ['Actividades', 'Activities'],
            'Emociones': ['Emociones', 'Emotions']
        }

        for category in tag_categories:
            category_df = tags_exploded[tags_exploded['tags'].apply(lambda x: x['type'] if isinstance(x, dict) else None) == category]
            category_df['flag'] = 1
            pivot_df = category_df.pivot_table(index='date', columns='tag_entries', values='flag', fill_value=0)
            combined_tags_df = pd.merge(combined_tags_df, pivot_df, on='date', how='left').fillna(0)

            # Calcular correlaciones para la categoría
            correlations = combined_tags_df[pivot_df.columns].corrwith(combined_tags_df['score'])
            correlation_df = pd.DataFrame({'Tag': correlations.index, 'Correlación': correlations.values})

            # Calcular frecuencia de los tags
            category_count = category_df['tag_entries'].value_counts().reset_index()
            category_count.columns = ['Tag', 'Frecuencia']

            # Normalizar la correlación con base en la escala de la frecuencia
            max_freq = category_count['Frecuencia'].max()
            correlation_df['Correlación_Normalizada'] = correlation_df['Correlación'] * max_freq

            # Traducción opcional para "Actividades" y "Emociones"
            category_name = category  
            if category in ['Actividades', 'Activities']:
                category_name = 'Actividades' 
            elif category in ['Emociones', 'Emotions']:
                category_name = 'Emociones' 

            # Mostrar gráfico de barras combinado
            st.write(f"Frecuencia y Correlación de {category_name} con Puntaje")
            fig_combined = go.Figure()
            fig_combined.add_trace(go.Bar(x=category_count['Tag'], y=category_count['Frecuencia'], name='Frecuencia', marker_color='khaki'))
            fig_combined.add_trace(go.Bar(x=correlation_df['Tag'], y=correlation_df['Correlación_Normalizada'], name='Correlación Normalizada', marker_color='lightblue'))
            fig_combined.update_layout(
                template='plotly_dark', 
                barmode='group', 
                title=f'Frecuencia y Correlación de {category_name} con Puntaje',
                xaxis_title=category_name, 
                yaxis_title='Valor'
            )
            st.plotly_chart(fig_combined, use_container_width=True)

        # MATRIZ DE CORRELACIÓN
        st.subheader("Correlación entre Tags y Puntaje")
        numeric_df = combined_tags_df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        fig_corr = px.imshow(correlation_matrix, title='Matriz de Correlación')
        fig_corr.update_layout(coloraxis_colorscale='Viridis', template='plotly_dark')
        st.plotly_chart(fig_corr, use_container_width=True)

        # GRAFICO INTERACTIVO DE CORRELACIONES SEGÚN TAG
        st.subheader("Gráfico Interactivo de Correlaciones")
        selected_category_1 = st.selectbox("Categoría 1", tag_categories, key="first_category", index=1)
        selected_category_2 = st.selectbox("Categoría 2", tag_categories, key="second_category", index=0)

        tags_df['tag_entries'] = tags_df['tags'].apply(lambda x: x['entries'] if isinstance(x, dict) else [])
        tags_exploded = tags_df.explode('tag_entries')
        category_1_tags = tags_exploded[tags_exploded['tags'].apply(lambda x: x['type'] if isinstance(x, dict) else None) == selected_category_1]['tag_entries'].unique()
        category_2_tags = tags_exploded[tags_exploded['tags'].apply(lambda x: x['type'] if isinstance(x, dict) else None) == selected_category_2]['tag_entries'].unique()

        selected_tag_1 = st.selectbox(f"Selecciona una etiqueta de {selected_category_1}", category_1_tags, key="selected_tag_1")

        if selected_tag_1 in correlation_matrix.index and category_2_tags.size > 0:
            correlations = correlation_matrix.loc[selected_tag_1, category_2_tags]
            threshold = correlations.abs().quantile(0.6)
            significant_corr = correlations[abs(correlations) >= threshold]

            if not significant_corr.empty:
                significant_corr_sorted = significant_corr.sort_values(ascending=False)
                st.write(f"Correlaciones significativas para {selected_tag_1} con Tags de {selected_category_2}:")
                fig_corr_bars = go.Figure()
                fig_corr_bars.add_trace(go.Bar(
                    x=list(significant_corr_sorted.values),
                    y=list(significant_corr_sorted.index),
                    orientation='h',
                    marker=dict(color=['#f0e68c' if corr > 0 else '#54382a' for corr in significant_corr_sorted.values])
                ))
                fig_corr_bars.update_layout(
                    template='plotly_dark',
                    title=f'Correlaciones de {selected_tag_1} con Tags de {selected_category_2}',
                    xaxis_title='Correlación',
                    yaxis_title='Tags',
                    xaxis=dict(range=[-1, 1])
                )
                st.plotly_chart(fig_corr_bars, use_container_width=True)
            else:
                st.write(f"No hay correlaciones significativas para {selected_tag_1} con Tags de {selected_category_2}.")
        else:
            st.write(f"No se encontraron datos de correlación para los tags seleccionados.")

if __name__ == "__main__":
    main()
