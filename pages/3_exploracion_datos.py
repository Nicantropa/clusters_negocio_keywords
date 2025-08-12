import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

sns.set_theme(style="whitegrid")

# --- Utilidades de descarga ---
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# --- Carga de datos ---
def cargar_dataset_streamlit(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile | None):
    if not uploaded_file:
        st.info("Sube un archivo CSV para comenzar.")
        return None
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Dataset cargado: {uploaded_file.name}")
        st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
        return df
    except Exception as e:
        st.error(f"Error al cargar el CSV: {e}")
        return None


# --- Exploración básica ---
def explorar_dataset_streamlit(df: pd.DataFrame):
    st.subheader("Exploración inicial del dataset")
    st.write("Primeras filas:")
    head_df = df.head()
    st.dataframe(head_df)
    st.download_button(
        "Descargar primeras filas (CSV)",
        data=df_to_csv_bytes(head_df),
        file_name="head.csv",
        mime="text/csv",
        key="dl_head",
    )

    st.write("Tipos de datos:")
    st.dataframe(pd.DataFrame({"columna": df.columns, "dtype": df.dtypes.astype(str)}))

    st.write("Estadísticas descriptivas (incluye categóricas):")
    with st.expander("Ver describe(include='all')"):
        try:
            desc = df.describe(include='all', datetime_is_numeric=True)
        except TypeError:
            # Compatibilidad con versiones antiguas de pandas que no soportan datetime_is_numeric
            desc = df.describe(include='all')
        st.dataframe(desc)
        st.download_button(
            "Descargar describe (CSV)",
            data=df_to_csv_bytes(desc.reset_index()),
            file_name="describe.csv",
            mime="text/csv",
            key="dl_desc",
        )


def columnas_numericas(df: pd.DataFrame):
    return df.select_dtypes(include='number').columns.tolist()


def seleccionar_columnas_numericas_streamlit(df: pd.DataFrame, label: str, min_cols: int = 2):
    cols_num = columnas_numericas(df)
    if len(cols_num) < min_cols:
        st.warning(f"Se requieren al menos {min_cols} columnas numéricas para esta operación.")
        return []
    seleccion = st.multiselect(label, cols_num, default=cols_num)
    if len(seleccion) < min_cols:
        st.warning(f"Selecciona al menos {min_cols} columnas.")
        return []
    return seleccion


# --- Correlación ---
def mostrar_correlacion_streamlit(df: pd.DataFrame, columnas_corr: list[str], metodo: str = "pearson"):
    matriz_corr = df[columnas_corr].corr(method=metodo)
    st.write(f"Matriz de Correlación ({metodo.capitalize()}):")
    st.dataframe(matriz_corr)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title(f'Correlación de {metodo.capitalize()} entre Variables')
    st.pyplot(fig)
    st.download_button(
        "Descargar matriz (CSV)",
        data=df_to_csv_bytes(matriz_corr.reset_index()),
        file_name=f"correlacion_{metodo}.csv",
        mime="text/csv",
        key=f"dl_corr_csv_{metodo}",
    )
    st.download_button(
        "Descargar heatmap (PNG)",
        data=fig_to_png_bytes(fig),
        file_name=f"heatmap_{metodo}.png",
        mime="image/png",
        key=f"dl_corr_png_{metodo}",
    )


# --- Distribuciones ---
def visualizar_distribuciones_avanzado_streamlit(df: pd.DataFrame, columnas_seleccionadas: list[str]):
    st.info("Para cada variable: Box Plot y Histograma; se detecta asimetría y se sugiere escala logarítmica.")
    bins_default = st.slider("Número de bins para histogramas", 10, 150, 40)
    for col in columnas_seleccionadas:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f'Análisis de Distribución para: {col}', fontsize=16)
        # Box Plot
        sns.boxplot(x=df[col], ax=axes[0])
        axes[0].set_title('Box Plot (Resumen y Outliers)')
        # Histograma
        sns.histplot(data=df, x=col, ax=axes[1], kde=True, bins=bins_default)
        axes[1].set_title('Histograma (Forma de la Distribución)')
        # Asimetría
        datos = df[col].dropna()
        if not datos.empty and datos.min() >= 0 and datos.quantile(0.95) < (datos.max() / 5 if datos.max() else np.inf):
            axes[0].set_xscale('log')
            axes[1].set_xscale('log')
            st.caption(f"Escala logarítmica aplicada a '{col}' por alta asimetría.")
        st.pyplot(fig)
        st.download_button(
            f"Descargar distribución '{col}' (PNG)",
            data=fig_to_png_bytes(fig),
            file_name=f"distribucion_{col}.png",
            mime="image/png",
            key=f"dl_dist_{col}",
        )
        # Gráfico extra sin ceros si > 30% de ceros
        porcentaje_ceros = (df[col] == 0).mean() if col in df.columns else 0
        if porcentaje_ceros > 0.30:
            st.caption(f"La columna '{col}' tiene {porcentaje_ceros:.1%} de ceros. Gráfico adicional sin ellos.")
            df_filtrado = df[df[col] != 0]
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            sns.histplot(data=df_filtrado, x=col, kde=True, bins=min(50, max(10, bins_default)))
            ax2.set_title(f'Distribución de "{col}" (Excluyendo Ceros)')
            st.pyplot(fig2)
            st.download_button(
                f"Descargar distribución sin ceros '{col}' (PNG)",
                data=fig_to_png_bytes(fig2),
                file_name=f"distribucion_sin_ceros_{col}.png",
                mime="image/png",
                key=f"dl_dist0_{col}",
            )


def graficar_histogramas_streamlit(df: pd.DataFrame, columnas_corr: list[str]):
    hue_parameter = 'cluster' if 'cluster' in df.columns else None
    for col in columnas_corr:
        datos = df[col].dropna()
        if col.endswith("_num"):
            fig, ax = plt.subplots(figsize=(8, 5))
            valores, conteos = np.unique(datos, return_counts=True)
            ax.bar(valores, conteos, color="skyblue")
            ax.set_title(f'Bar chart de {col}')
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)
            st.download_button(
                f"Descargar bar chart '{col}' (PNG)",
                data=fig_to_png_bytes(fig),
                file_name=f"barchart_{col}.png",
                mime="image/png",
                key=f"dl_bar_{col}",
            )
        else:
            min_val = datos.min() if not datos.empty else 0
            max_val = datos.max() if not datos.empty else 0
            bins = 100 if col == "avg_monthly_searches" else 40
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(
                data=df,
                x=col,
                bins=bins,
                binrange=(min_val, max_val) if min_val != max_val else None,
                ax=ax,
                kde=True,
                hue=hue_parameter,
                palette='viridis',
            )
            ax.set_title(f'Distribución de {col} ({int(min_val)} a {int(max_val)})')
            if not datos.empty and datos.quantile(0.95) < datos.max() / 5 and datos.min() >= 0:
                ax.set_xscale('log')
            st.pyplot(fig)
            st.download_button(
                f"Descargar histograma '{col}' (PNG)",
                data=fig_to_png_bytes(fig),
                file_name=f"hist_{col}.png",
                mime="image/png",
                key=f"dl_hist_{col}",
            )


# --- Relaciones bivariadas ---
def graficar_relaciones_bivariadas_streamlit(df: pd.DataFrame, columnas_numericas_sel: list[str]):
    pares = [
        (columnas_numericas_sel[i], columnas_numericas_sel[j])
        for i in range(len(columnas_numericas_sel))
        for j in range(i + 1, len(columnas_numericas_sel))
    ]
    opciones = [f"{a} vs {b}" for a, b in pares]
    sel = st.multiselect("Selecciona pares a graficar (scatter)", opciones, default=opciones[: min(5, len(opciones))])
    hue_parameter = 'cluster' if 'cluster' in df.columns else None
    for etiqueta in sel:
        i = opciones.index(etiqueta)
        x, y = pares[i]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            x=df[x],
            y=df[y],
            hue=df[hue_parameter] if hue_parameter else None,
            palette='viridis',
            alpha=0.7,
            ax=ax,
        )
        datos_x = df[x].dropna()
        datos_y = df[y].dropna()
        if not datos_x.empty and datos_x.quantile(0.95) < datos_x.max() / 5 and datos_x.min() >= 0:
            ax.set_xscale('log')
        if not datos_y.empty and datos_y.quantile(0.95) < datos_y.max() / 5 and datos_y.min() >= 0:
            ax.set_yscale('log')
        ax.set_title(f'Relación bivariada: {x} vs {y}')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        st.pyplot(fig)
        st.download_button(
            f"Descargar scatter {x}_vs_{y} (PNG)",
            data=fig_to_png_bytes(fig),
            file_name=f"scatter_{x}_vs_{y}.png",
            mime="image/png",
            key=f"dl_scatter_{x}_{y}",
        )


# --- Comparar métricas por categoría ---
def comparar_metricas_por_categoria_streamlit(df: pd.DataFrame):
    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numericas = columnas_numericas(df)
    if not categoricas or not numericas:
        st.warning("No hay suficientes variables categóricas o numéricas para comparar.")
        return
    cat = st.selectbox("Variable categórica", categoricas)
    metricas = st.multiselect("Métricas numéricas a comparar", numericas, default=numericas[: min(3, len(numericas))])
    if not metricas:
        st.info("Selecciona al menos una métrica numérica.")
        return
    for num in metricas:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=df[cat], y=df[num], ax=ax)
        ax.set_title(f'Boxplot de {num} por {cat}')
        ax.set_xlabel(cat)
        ax.set_ylabel(num)
        st.pyplot(fig)
        st.download_button(
            f"Descargar boxplot {num}_por_{cat} (PNG)",
            data=fig_to_png_bytes(fig),
            file_name=f"boxplot_{num}_por_{cat}.png",
            mime="image/png",
            key=f"dl_box_{num}_{cat}",
        )


# --- Frecuencias de categóricas ---
def mostrar_frecuencias_categoricas_streamlit(df: pd.DataFrame):
    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not categoricas:
        st.info("No hay variables categóricas.")
        return
    for col in categoricas:
        st.write(f"Frecuencia de '{col}':")
        freq_df = df[col].value_counts().rename_axis(col).reset_index(name="Frecuencia")
        st.dataframe(freq_df)
        st.download_button(
            f"Descargar frecuencias '{col}' (CSV)",
            data=df_to_csv_bytes(freq_df),
            file_name=f"frecuencias_{col}.csv",
            mime="text/csv",
            key=f"dl_freq_{col}",
        )


# --- App principal ---
def main():
    st.title("Análisis Exploratorio de Datos")
    uploaded_file = st.file_uploader("Sube el dataset limpio (CSV)", type=["csv"])
    df = cargar_dataset_streamlit(uploaded_file)
    if df is None:
        return

    tabs = st.tabs(
        [
            "Resumen",
            "Correlación",
            "Distribuciones",
            "Pairplot",
            "Bivariadas",
            "Métricas por categoría",
            "Frecuencias categóricas",
        ]
    )

    with tabs[0]:
        explorar_dataset_streamlit(df)

    with tabs[1]:
        cols_sel = seleccionar_columnas_numericas_streamlit(df, "Selecciona columnas para correlación")
        if cols_sel:
            metodo = st.selectbox("Método de correlación", ["pearson", "spearman"], index=1)
            mostrar_correlacion_streamlit(df, cols_sel, metodo)

    with tabs[2]:
        cols_sel = seleccionar_columnas_numericas_streamlit(df, "Selecciona columnas para distribuciones", min_cols=1)
        if cols_sel:
            visualizar_distribuciones_avanzado_streamlit(df, cols_sel)

    with tabs[3]:
        cols_sel = seleccionar_columnas_numericas_streamlit(df, "Selecciona columnas para pairplot")
        if cols_sel:
            if len(cols_sel) > 5:
                st.warning("El pairplot puede ser lento con más de 5 variables.")
            if st.button("Generar pairplot"):
                hue_parameter = 'cluster' if 'cluster' in df.columns else None
                g = sns.pairplot(data=df, vars=cols_sel, hue=hue_parameter, palette='viridis', plot_kws={'alpha': 0.6})
                g.fig.suptitle("Matriz de Gráficos de Dispersión", y=1.02)
                st.pyplot(g.fig)
                st.download_button(
                    "Descargar pairplot (PNG)",
                    data=fig_to_png_bytes(g.fig),
                    file_name="pairplot.png",
                    mime="image/png",
                    key="dl_pairplot",
                )

    with tabs[4]:
        cols_sel = seleccionar_columnas_numericas_streamlit(df, "Selecciona columnas numéricas para pares bivariados")
        if cols_sel and len(cols_sel) >= 2:
            graficar_relaciones_bivariadas_streamlit(df, cols_sel)

    with tabs[5]:
        comparar_metricas_por_categoria_streamlit(df)

    with tabs[6]:
        mostrar_frecuencias_categoricas_streamlit(df)


if __name__ == "__main__":
    main()
