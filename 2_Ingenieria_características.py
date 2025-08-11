"""
Ingeniería de Características - Versión Streamlit (limpia)

Este módulo expone únicamente una aplicación Streamlit para la capa de ingeniería de características.
Se eliminaron completamente las rutas de ejecución por consola y cualquier dependencia de Tkinter.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# Utilidades
# ---------------------------

def cargar_csv_streamlit(uploaded_file):
    if not uploaded_file:
        st.info("Sube un CSV para continuar.")
        return None
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"Cargado: {uploaded_file.name}")
        return df
    except Exception as e:
        st.error(f"Error al leer el CSV: {e}")
        return None


def clasificar_columnas(df: pd.DataFrame, umbral_categorica: float = 0.1):
    nums = df.select_dtypes(include=np.number).columns.tolist()
    cat_pot = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cats: list[str] = []
    texts: list[str] = []
    if "keyword" in cat_pot:
        texts.append("keyword")
        cat_pot.remove("keyword")
    for c in cat_pot:
        ratio = df[c].nunique(dropna=True) / max(len(df), 1)
        (cats if ratio < umbral_categorica else texts).append(c)
    return nums, cats, texts


def seleccionar_columnas(label: str, columnas: list[str]):
    st.write(f"--- {label} ---")
    return st.multiselect(label, columnas, default=columnas)


def transformar_categoricas(df: pd.DataFrame, cols: list[str]):
    for col in cols:
        metodo = st.radio(
            f"Transformación para '{col}'",
            ["Codificación ordinal manual", "One-hot encoding", "Frequency encoding"],
            key=f"met_{col}",
        )
        if metodo == "One-hot encoding":
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            st.success(f"One-hot aplicado. Nuevas columnas: {list(dummies.columns)}")
        elif metodo == "Frequency encoding":
            freq = df[col].value_counts()
            df[f"{col}_freq_enc"] = df[col].map(freq)
            st.success("Frequency encoding aplicado.")
        else:
            valores = list(df[col].dropna().unique())
            st.caption(f"Valores únicos en '{col}': {valores}")
            mapeo = {}
            usados = set()
            for v in valores:
                n = st.number_input(
                    f"Número para '{v}'",
                    min_value=0,
                    step=1,
                    key=f"map_{col}_{v}",
                )
                if n in usados:
                    st.warning("Número repetido; considera otro.")
                mapeo[v] = n
                usados.add(n)
            df[f"{col}_num"] = df[col].map(mapeo)
            st.success("Codificación ordinal aplicada.")
    return df


def transformar_texto(df: pd.DataFrame, cols: list[str]):
    for col in cols:
        if st.checkbox(f"Crear longitud de '{col}'", value=True, key=f"len_{col}"):
            df[f"{col}_length"] = df[col].astype(str).apply(len)
        if st.checkbox(f"Crear #palabras de '{col}'", value=True, key=f"words_{col}"):
            df[f"{col}_words"] = df[col].astype(str).apply(lambda x: len(x.split()))
    return df


def columna_compuesta(df: pd.DataFrame):
    st.subheader("Columna compuesta (opcional)")
    if not st.checkbox("¿Crear nueva columna compuesta?", value=False):
        return df
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    metodo = st.selectbox("Método", ["Media", "Mediana", "Moda", "Volatilidad"])
    sel = st.multiselect("Columnas numéricas", nums)
    nombre = st.text_input("Nombre de la nueva columna")
    if not nombre:
        return df
    if metodo == "Media" and sel:
        df[nombre] = df[sel].mean(axis=1)
    elif metodo == "Mediana" and sel:
        df[nombre] = df[sel].median(axis=1)
    elif metodo == "Moda" and sel:
        df[nombre] = df[sel].mode(axis=1)[0]
    elif metodo == "Volatilidad" and len(sel) == 2:
        df[nombre] = df[sel[0]] - df[sel[1]]
    if nombre in df.columns:
        st.success(f"Columna '{nombre}' creada.")
    return df


def configurar_score(columnas: list[str], tipo: str):
    st.write(f"--- Score {tipo} ---")
    sel = st.multiselect(f"Columnas para score {tipo}", columnas)
    pesos: dict[str, float] = {}
    invertir: set[str] = set()
    total = 0.0
    for c in sel:
        p = st.number_input(
            f"Peso de '{c}' (1-100)", min_value=1.0, max_value=100.0, value=10.0, key=f"p_{tipo}_{c}"
        )
        pesos[c] = p
        total += p
        if st.checkbox(f"Invertir '{c}' (valores bajos mejores)", key=f"inv_{tipo}_{c}"):
            invertir.add(c)
    if sel and total != 100.0:
        st.warning(f"La suma de pesos es {total} y debe ser 100.")
    return {k: v / 100.0 for k, v in pesos.items()}, invertir


def aplicar_score(df: pd.DataFrame, df_scaled: pd.DataFrame, pesos: dict[str, float], tipo: str, invertir: set[str]):
    score = pd.Series(0.0, index=df.index)
    for c, p in pesos.items():
        comp = (1 - df_scaled[c]) if c in invertir else df_scaled[c]
        score += comp * p
    df[f"score_{tipo.lower()}"] = score


# ---------------------------
# App principal
# ---------------------------

def main():
    st.title("Ingeniería de Características - Streamlit")
    up = st.file_uploader("Sube tu CSV limpio", type=["csv"])
    if not up:
        return
    df = cargar_csv_streamlit(up)
    if df is None:
        return

    nums, cats, texts = clasificar_columnas(df)
    st.write(f"Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
    st.write({"numéricas": nums, "categóricas": cats, "texto": texts})

    sel_nums = seleccionar_columnas("Selecciona columnas numéricas", nums)
    sel_cats = seleccionar_columnas("Selecciona columnas categóricas", cats)
    sel_texts = seleccionar_columnas("Selecciona columnas de texto", texts)

    if sel_cats and st.checkbox("¿Transformar categóricas?"):
        df = transformar_categoricas(df, sel_cats)
    if sel_texts and st.checkbox("¿Transformar texto?"):
        df = transformar_texto(df, sel_texts)

    df = columna_compuesta(df)

    # Scores
    num_finales = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_finales:
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[num_finales]), columns=num_finales, index=df.index)
        if st.checkbox("¿Calcular score SEO?"):
            pesos, inv = configurar_score(num_finales, "SEO")
            if pesos and abs(sum(v * 100 for v in pesos.values()) - 100.0) < 1e-6:
                aplicar_score(df, df_scaled, pesos, "SEO", inv)
        if st.checkbox("¿Calcular score SEM?"):
            pesos, inv = configurar_score(num_finales, "SEM")
            if pesos and abs(sum(v * 100 for v in pesos.values()) - 100.0) < 1e-6:
                aplicar_score(df, df_scaled, pesos, "SEM", inv)

    if "score_seo" in df.columns:
        st.subheader("Top 5 por Score SEO")
        cols = [c for c in ["keyword", "score_seo"] if c in df.columns]
        st.dataframe(df[cols].sort_values("score_seo", ascending=False).head())
    if "score_sem" in df.columns:
        st.subheader("Top 5 por Score SEM")
        cols = [c for c in ["keyword", "score_sem"] if c in df.columns]
        st.dataframe(df[cols].sort_values("score_sem", ascending=False).head())

    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("Descargar dataset enriquecido", csv_bytes, "dataset_enriquecido.csv", "text/csv")

    st.info("Módulo 100% Streamlit. Sin rutas de consola.")


if __name__ == "__main__":
    main()