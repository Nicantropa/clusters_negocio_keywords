import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import io

# --- Utilidades de descarga ---
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def plotly_fig_to_png_bytes(fig) -> bytes | None:
    """Convierte una figura Plotly a PNG. Requiere `kaleido`.
    Devuelve None si kaleido no está disponible.
    """
    try:
        img_bytes = pio.to_image(fig, format="png", scale=2)
        return img_bytes
    except Exception:
        return None


def plotly_fig_to_html_bytes(fig) -> bytes:
    html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    return html.encode("utf-8")


def offer_plotly_downloads(fig, base_name: str, key_prefix: str):
    """Ofrece descarga en PNG (si kaleido disponible) y HTML interactivo."""
    png = plotly_fig_to_png_bytes(fig)
    if png is not None:
        st.download_button(
            "Descargar (PNG)",
            data=png,
            file_name=f"{base_name}.png",
            mime="image/png",
            key=f"{key_prefix}_png",
        )
    st.download_button(
        "Descargar (HTML interactivo)",
        data=plotly_fig_to_html_bytes(fig),
        file_name=f"{base_name}.html",
        mime="text/html",
        key=f"{key_prefix}_html",
    )


# --- Configuración de tema/estilo para gráficos ---
def init_plotly_theme_controls():
    """Renderiza una única vez el expander de opciones y guarda preferencia en session_state."""
    base = st.get_option("theme.base") or "light"
    if "plotly_transparent_bg" not in st.session_state:
        st.session_state["plotly_transparent_bg"] = (base == "dark")
    with st.expander("Opciones de visualización", expanded=False):
        st.session_state["plotly_transparent_bg"] = st.checkbox(
            "Usar fondo transparente en gráficos (Plotly)",
            value=st.session_state["plotly_transparent_bg"],
            help="Integra los gráficos con el tema (ideal en modo oscuro).",
            key="opt_plotly_transparent_bg",
        )


def get_plotly_theme():
    """Devuelve (template, transparent) sin renderizar widgets (seguro de llamar muchas veces)."""
    base = st.get_option("theme.base") or "light"
    if "plotly_transparent_bg" not in st.session_state:
        st.session_state["plotly_transparent_bg"] = (base == "dark")
    template = "plotly_dark" if base == "dark" else "plotly"
    transparent = bool(st.session_state["plotly_transparent_bg"])
    return template, transparent


def style_plotly(fig, template: str, transparent: bool):
    fig.update_layout(template=template)
    if transparent:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_bgcolor="rgba(0,0,0,0)",
        )
    return fig


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
    template, transparent = get_plotly_theme()
    fig = px.imshow(
        matriz_corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        title=f"Correlación de {metodo.capitalize()} entre Variables",
        aspect="auto",
        color_continuous_scale="RdBu",
    )
    fig = style_plotly(fig, template, transparent)
    st.plotly_chart(fig, use_container_width=True)
    st.download_button(
        "Descargar matriz (CSV)",
        data=df_to_csv_bytes(matriz_corr.reset_index()),
        file_name=f"correlacion_{metodo}.csv",
        mime="text/csv",
        key=f"dl_corr_csv_{metodo}",
    )
    offer_plotly_downloads(fig, f"heatmap_{metodo}", f"dl_corr_plot_{metodo}")


# --- Distribuciones ---
def visualizar_distribuciones_avanzado_streamlit(df: pd.DataFrame, columnas_seleccionadas: list[str]):
    st.info("Para cada variable: Box Plot y Histograma; se detecta asimetría y se sugiere escala logarítmica.")
    bins_default = st.slider("Número de bins para histogramas", 10, 150, 40)
    for col in columnas_seleccionadas:
        template, transparent = get_plotly_theme()
        c1, c2 = st.columns(2)
        with c1:
            fig_box = px.box(df, y=col, points=False, title=f"Box Plot — {col}")
            fig_box = style_plotly(fig_box, template, transparent)
            st.plotly_chart(fig_box, use_container_width=True)
            offer_plotly_downloads(fig_box, f"boxplot_{col}", f"dl_box_adv_{col}")
        with c2:
            datos = df[col].dropna()
            skew_log = (not datos.empty and datos.min() >= 0 and datos.quantile(0.95) < (datos.max() / 5 if datos.max() else np.inf))
            fig_hist = px.histogram(df, x=col, nbins=bins_default, marginal="rug", title=f"Histograma — {col}")
            if skew_log:
                fig_hist.update_xaxes(type="log")
                st.caption(f"Escala log aplicada a '{col}' por alta asimetría.")
            fig_hist = style_plotly(fig_hist, template, transparent)
            st.plotly_chart(fig_hist, use_container_width=True)
            offer_plotly_downloads(fig_hist, f"hist_{col}", f"dl_hist_adv_{col}")
        # Gráfico extra sin ceros si > 30% de ceros
        porcentaje_ceros = (df[col] == 0).mean() if col in df.columns else 0
        if porcentaje_ceros > 0.30:
            st.caption(f"La columna '{col}' tiene {porcentaje_ceros:.1%} de ceros. Gráfico adicional sin ellos.")
            df_filtrado = df[df[col] != 0]
            fig2 = px.histogram(df_filtrado, x=col, nbins=min(50, max(10, bins_default)), title=f"Distribución de {col} (sin ceros)")
            fig2 = style_plotly(fig2, template, transparent)
            st.plotly_chart(fig2, use_container_width=True)
            offer_plotly_downloads(fig2, f"distribucion_sin_ceros_{col}", f"dl_dist0_{col}")


def graficar_histogramas_streamlit(df: pd.DataFrame, columnas_corr: list[str]):
    hue_parameter = 'cluster' if 'cluster' in df.columns else None
    for col in columnas_corr:
        datos = df[col].dropna()
        if col.endswith("_num"):
            template, transparent = get_plotly_theme()
            valores, conteos = np.unique(datos, return_counts=True)
            df_bar = pd.DataFrame({col: valores, "Frecuencia": conteos})
            fig = px.bar(df_bar, x=col, y="Frecuencia", title=f"Bar chart de {col}")
            fig = style_plotly(fig, template, transparent)
            st.plotly_chart(fig, use_container_width=True)
            offer_plotly_downloads(fig, f"barchart_{col}", f"dl_bar_{col}")
        else:
            min_val = datos.min() if not datos.empty else 0
            max_val = datos.max() if not datos.empty else 0
            bins = 100 if col == "avg_monthly_searches" else 40
            template, transparent = get_plotly_theme()
            fig = px.histogram(
                df,
                x=col,
                nbins=bins,
                color=hue_parameter,
                title=f"Distribución de {col} ({int(min_val)} a {int(max_val)})",
            )
            if not datos.empty and datos.quantile(0.95) < datos.max() / 5 and datos.min() >= 0:
                fig.update_xaxes(type="log")
            fig = style_plotly(fig, template, transparent)
            st.plotly_chart(fig, use_container_width=True)
            offer_plotly_downloads(fig, f"hist_{col}", f"dl_hist_{col}")


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
        template, transparent = get_plotly_theme()
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=hue_parameter,
            opacity=0.7,
            title=f"Relación bivariada: {x} vs {y}",
        )
        datos_x = df[x].dropna()
        datos_y = df[y].dropna()
        if not datos_x.empty and datos_x.quantile(0.95) < datos_x.max() / 5 and datos_x.min() >= 0:
            fig.update_xaxes(type="log")
        if not datos_y.empty and datos_y.quantile(0.95) < datos_y.max() / 5 and datos_y.min() >= 0:
            fig.update_yaxes(type="log")
        fig = style_plotly(fig, template, transparent)
        st.plotly_chart(fig, use_container_width=True)
        offer_plotly_downloads(fig, f"scatter_{x}_vs_{y}", f"dl_scatter_{x}_{y}")


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
        template, transparent = get_plotly_theme()
        fig = px.box(df, x=cat, y=num, points="outliers", title=f"Boxplot de {num} por {cat}")
        fig = style_plotly(fig, template, transparent)
        st.plotly_chart(fig, use_container_width=True)
        offer_plotly_downloads(fig, f"boxplot_{num}_por_{cat}", f"dl_box_{num}_{cat}")


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
    st.title("Análisis Exploratorio de Datos (EDA)")
    st.markdown("<small style='color:#6c757d;'>Clustering para SEO y SEM V.1 - agosto, 2025<br>Verónica Angarita @nicantropa</small>", unsafe_allow_html=True)
    # Configura tema/estilo Plotly antes de crear figuras (muestra expander de opciones)
    init_plotly_theme_controls()
    uploaded = st.file_uploader("Carga un CSV para EDA", type=["csv"])
    if uploaded is None:
        st.info("Sube un archivo CSV para comenzar.")
        return
    df = pd.read_csv(uploaded)
    st.success(f"Dataset cargado: {uploaded.name} — {df.shape[0]} filas x {df.shape[1]} columnas")

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
                template, transparent = get_plotly_theme()
                fig = px.scatter_matrix(df, dimensions=cols_sel, color=hue_parameter, title="Matriz de Gráficos de Dispersión")
                fig.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(opacity=0.6))
                fig = style_plotly(fig, template, transparent)
                st.plotly_chart(fig, use_container_width=True)
                offer_plotly_downloads(fig, "pairplot", "dl_pairplot")

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
