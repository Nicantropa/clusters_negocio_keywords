"""
modelo_clustering3_streamlit.py
Aplicación Streamlit para clustering K-Means con opciones de selección manual o PCA,
gráfico del codo, métricas (Silhouette y Calinski-Harabasz), y visualizaciones
de scatter y radar. Incluye descargas de resultados (CSV/PNG).
"""

from io import BytesIO
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import streamlit as st


# ------------------------- Utilidades -------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def clasificar_variables(df: pd.DataFrame, umbral_categorica: float = 0.1):
    n_filas = len(df)
    string_vars, categoricas = [], []
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        n_unicos = df[col].nunique(dropna=True)
        (categoricas if (n_unicos / max(n_filas, 1) < umbral_categorica) else string_vars).append(col)
    numericas = df.select_dtypes(include="number").columns.tolist()

    with st.expander("Variables detectadas", expanded=True):
        st.write("Primeras filas:")
        st.dataframe(df.head())
        st.write("Tipos de columna:")
        st.write({c: str(t) for c, t in df.dtypes.items()})
        st.write({
            "categóricas": categoricas,
            "texto_libre": string_vars,
            "numéricas": numericas,
        })
    return categoricas, string_vars, numericas


def seleccionar_numericas_streamlit(numericas: list[str]) -> list[str]:
    enfoque = st.radio(
        "¿Cómo quieres seleccionar variables numéricas?",
        ("Incluir manualmente", "Excluir algunas"),
        index=1,
        help="Puedes incluir manualmente un subconjunto o excluir columnas y usar el resto.",
    )
    if enfoque == "Incluir manualmente":
        seleccion = st.multiselect("Selecciona variables numéricas a incluir", options=numericas, default=numericas)
        return seleccion
    else:
        excluir = st.multiselect("Selecciona variables numéricas a excluir", options=numericas, default=[])
        return [c for c in numericas if c not in set(excluir)]


def calcular_codo(X_scaled: np.ndarray, k_min: int, k_max: int) -> tuple[list[int], list[float], plt.Figure]:
    inercia = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercia.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inercia, "bo-")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Inercia")
    ax.set_title("Método del codo")
    ax.grid(True, alpha=0.3)
    return ks, inercia, fig


def scatter_clusters(df: pd.DataFrame, x: str, y: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    hue = "cluster" if "cluster" in df.columns else None
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="viridis", alpha=0.7, ax=ax)
    # Escala log si es positivo y muy asimétrico
    for axis, col in [(ax.set_xscale, x), (ax.set_yscale, y)]:
        datos = df[col].dropna()
        if len(datos) > 0 and (datos >= 0).all() and (datos.quantile(0.95) < (datos.max() / 5 if datos.max() else 1)):
            axis("log")
    ax.set_title(f"Clusters por {x} y {y}")
    ax.legend(loc="best")
    return fig


def radar_clusters(df: pd.DataFrame, cols: list[str]) -> plt.Figure:
    agg = df.groupby("cluster")[cols].mean()
    # Normalización min-max por columna para comparabilidad
    min_vals = agg.min()
    max_vals = agg.max()
    span = (max_vals - min_vals).replace(0, 1e-9)
    norm = (agg - min_vals) / span

    labels = cols
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    for idx, (row_name, row) in enumerate(norm.iterrows()):
        values = row.tolist() + [row.tolist()[0]]
        plt.polar(angles, values, label=f"Cluster {row_name}", linewidth=2)
    plt.xticks(angles[:-1], labels, color="grey", size=9)
    plt.title("Gráfico radar de perfiles de cluster", size=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05))
    return fig


# ------------------------- App -------------------------
st.set_page_config(page_title="Clustering K-Means", layout="wide")
st.title("Aplicación de Modelo de Clustering (K-Means)")

uploaded = st.file_uploader("Carga un CSV limpio", type=["csv"])
if uploaded is None:
    st.info("Sube un archivo CSV para comenzar.")
    st.stop()

df = pd.read_csv(uploaded)
st.success(f"Dataset cargado: {uploaded.name} — {df.shape[0]} filas x {df.shape[1]} columnas")

with st.expander("Vista rápida del dataset"):
    st.dataframe(df.head(20))

# 1) Clasificación de variables
categoricas, string_vars, numericas = clasificar_variables(df)
if not numericas:
    st.error("No se detectaron variables numéricas para el clustering.")
    st.stop()

# 2) Selección de dimensionalidad
st.subheader("Selección de características")
usar_pca = st.toggle("Usar PCA (Análisis de Componentes Principales)", value=False)

if usar_pca:
    num_cols = st.multiselect("Variables numéricas para PCA", options=numericas, default=numericas)
    if len(num_cols) < 2:
        st.warning("Selecciona al menos 2 variables numéricas para PCA.")
        st.stop()
    n_comp = st.slider("Número de componentes", min_value=1, max_value=len(num_cols), value=min(3, len(num_cols)))
    scaler_pca = StandardScaler()
    X_scaled_for_pca = scaler_pca.fit_transform(df[num_cols])
    pca = PCA(n_components=n_comp, random_state=42)
    componentes = pca.fit_transform(X_scaled_for_pca)
    pca_cols = [f"PCA_{i+1}" for i in range(n_comp)]
    X = pd.DataFrame(componentes, columns=pca_cols, index=df.index)
    st.info({"Explained variance ratio": [round(v, 4) for v in pca.explained_variance_ratio_.tolist()]})
else:
    seleccion_cols = seleccionar_numericas_streamlit(numericas)
    if len(seleccion_cols) < 2:
        st.warning("Selecciona al menos 2 variables para entrenar K-Means.")
        st.stop()
    X = df[seleccion_cols].copy()

st.write("Características seleccionadas:", list(X.columns))

# 3) Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4) Método del codo
st.subheader("Método del codo")
col_c1, col_c2 = st.columns(2)
with col_c1:
    k_min = st.number_input("K mínimo", min_value=2, max_value=50, value=2, step=1)
with col_c2:
    k_max = st.number_input("K máximo", min_value=max(3, k_min+1), max_value=50, value=max(k_min+8, k_min+1), step=1)

if st.button("Calcular codo"):
    ks, inercia, fig_codo = calcular_codo(X_scaled, int(k_min), int(k_max))
    st.pyplot(fig_codo)
    st.download_button(
        "Descargar gráfico del codo (PNG)",
        data=fig_to_png_bytes(fig_codo),
        file_name="metodo_del_codo.png",
        mime="image/png",
        key="dl_codo",
    )

# 5) Entrenamiento K-Means
st.subheader("Entrenamiento del modelo")
sel_k = st.slider("Selecciona K (número de clusters)", min_value=2, max_value=20, value=4)
entrenar = st.button("Entrenar K-Means")

if entrenar:
    km = KMeans(n_clusters=int(sel_k), random_state=42, n_init=10)
    km.fit(X_scaled)
    df_clusters = df.copy()
    df_clusters["cluster"] = km.labels_
    st.session_state["df_clusters"] = df_clusters
    st.session_state["X_scaled_shape"] = X_scaled.shape
    st.session_state["X_scaled"] = X_scaled  # para métricas
    st.success("Modelo entrenado y clusters asignados al DataFrame.")


# 6) Resultados y métricas
if "df_clusters" in st.session_state:
    dfc = st.session_state["df_clusters"]
    st.subheader("Resultados")
    st.dataframe(dfc.head(30))
    st.download_button(
        "Descargar dataset con clusters (CSV)",
        data=df_to_csv_bytes(dfc),
        file_name="dataset_con_clusters.csv",
        mime="text/csv",
        key="dl_df_clusters",
    )

    # Métricas
    st.markdown("---")
    st.subheader("Métricas del modelo")
    try:
        sil = silhouette_score(st.session_state["X_scaled"], dfc["cluster"])
        ch = calinski_harabasz_score(st.session_state["X_scaled"], dfc["cluster"])
        st.write({"silhouette": round(sil, 4), "calinski_harabasz": round(ch, 2)})
        if sil < 0.25:
            st.warning("Silhouette bajo: los clusters pueden no estar bien definidos.")
        elif sil < 0.5:
            st.info("Silhouette moderado: clusters razonables.")
        else:
            st.success("Silhouette alto: clusters bien definidos.")
    except Exception as e:
        st.error(f"No se pudieron calcular las métricas: {e}")

    # Descripción agregada
    st.markdown("---")
    st.subheader("Características promedio por cluster")
    numericas_actuales = dfc.select_dtypes(include="number").columns.tolist()
    if "cluster" in numericas_actuales:
        numericas_actuales.remove("cluster")
    if numericas_actuales:
        desc = dfc.groupby("cluster")[numericas_actuales].mean().round(2)
        desc["n_keywords"] = dfc["cluster"].value_counts().sort_index()
        st.dataframe(desc)
        st.download_button(
            "Descargar agregados por cluster (CSV)",
            data=df_to_csv_bytes(desc.reset_index()),
            file_name="caracteristicas_promedio_por_cluster.csv",
            mime="text/csv",
            key="dl_desc",
        )

    # 7) Visualización: Scatter
    st.markdown("---")
    st.subheader("Visualización de clusters (Scatter)")
    num_cols_all = [c for c in dfc.select_dtypes(include="number").columns if c != "cluster"]
    if len(num_cols_all) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("Eje X", options=num_cols_all, index=0)
        with c2:
            y_col = st.selectbox("Eje Y", options=num_cols_all, index=min(1, len(num_cols_all)-1))
        fig_sc = scatter_clusters(dfc, x_col, y_col)
        st.pyplot(fig_sc)
        st.download_button(
            "Descargar scatter (PNG)",
            data=fig_to_png_bytes(fig_sc),
            file_name="scatter_clusters.png",
            mime="image/png",
            key="dl_scatter",
        )
    else:
        st.info("Se requieren al menos 2 columnas numéricas para el scatter.")

    # 8) Visualización: Radar
    st.markdown("---")
    st.subheader("Gráfico radar por cluster")
    if numericas_actuales and len(dfc["cluster"].unique()) > 1:
        cols_radar = st.multiselect("Selecciona variables para el radar", options=numericas_actuales, default=numericas_actuales[:min(6, len(numericas_actuales))])
        if cols_radar:
            fig_rd = radar_clusters(dfc, cols_radar)
            st.pyplot(fig_rd)
            st.download_button(
                "Descargar radar (PNG)",
                data=fig_to_png_bytes(fig_rd),
                file_name="radar_clusters.png",
                mime="image/png",
                key="dl_radar",
            )
        else:
            st.info("Selecciona al menos una variable para el radar.")
    else:
        st.info("No es posible generar radar: se necesitan columnas numéricas y al menos 2 clusters.")
