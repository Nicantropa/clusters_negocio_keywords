import streamlit as st
import pandas as pd
import csv
import os
import openpyxl
import re
import numpy as np

# --- 1. CLASES DE TRANSFORMACIÓN (Se mantienen intactas) ---
# Todas tus clases (UtilidadesColumnas, PreAgregacion, TransformacionCategorica, etc.)
# van aquí sin ningún cambio. Son las "herramientas" que usará nuestro pipeline.

class UtilidadesColumnas:
    @staticmethod
    def encontrar_columna(df, nombre_objetivo):
        for col in df.columns:
            if col.strip().lower() == nombre_objetivo.lower():
                return col
        return None
    @staticmethod
    def normalizar_columnas(df):
        df.columns = [UtilidadesColumnas.normalize_column(c) for c in df.columns]
    @staticmethod
    def normalize_column(col):
        col = col.strip().lower()
        col = re.sub(r"[^\w\s]", "", col)
        col = col.replace(" ", "_")
        return col
    @staticmethod
    def aplicar_moda(serie):
        moda = serie.mode()
        return moda.iloc[0] if not moda.empty else np.nan
    @staticmethod
    def convertir_arrays_a_string(df):
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)

class PreAgregacion:
    def __init__(self, df):
        self.df = df
    def eliminar_columnas_muy_nulas(self, threshold=0.9):
        nulls = self.df.isnull().mean()
        cols_to_drop = nulls[nulls > threshold].index.tolist()
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
        return cols_to_drop
        

class TransformacionCategorica:
    def __init__(self, df):
        self.df = df
    def limpiar_texto(self, columnas):
        for col in columnas:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.lower()
    def convertir_a_categoria(self, columnas):
        for col in columnas:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("category")

class TransformacionMetrica:
    def __init__(self, df):
        self.df = df
    def convertir_moneda(self, columna, tasa, nuevo_nombre):
        if columna in self.df.columns:
            self.df[nuevo_nombre] = self.df[columna] / tasa

class TransformacionPorcentaje:
    def __init__(self, df):
        self.df = df
    @staticmethod
    def convertir_porcentaje_robusto(valor):
        if pd.isna(valor): return None
        valor_str = str(valor)
        if '∞' in valor_str: return None
        try:
            return float(valor_str.replace("%", "").strip()) / 100
        except (ValueError, TypeError):
            return None
    def transformar_porcentaje(self, columnas, nuevos_nombres):
        for col, new_col in zip(columnas, nuevos_nombres):
            if col in self.df.columns:
                self.df[new_col] = self.df[col].apply(self.convertir_porcentaje_robusto)

class PostAgregacion:
    def __init__(self, df):
        self.df = df
    def imputar_mediana_numericas(self):
        rellenadas = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            mediana = self.df[col].median()
            if not pd.isna(mediana):
                self.df[col].fillna(mediana, inplace=True)
                rellenadas[col] = mediana
        return rellenadas


# --- 2. FUNCIÓN PRINCIPAL DEL PIPELINE ---
def ejecutar_pipeline_de_limpieza_streamlit(uploaded_files, tasa=None, sufijo_moneda=None):
    st.write("Librerías importadas correctamente.")
    if not uploaded_files:
        st.warning("No se han subido archivos CSV.")
        return None
    files = {f"df_{os.path.splitext(f.name)[0].replace(' ', '_').lower()}": f for f in uploaded_files}
    dataframes = {}
    for key, file in files.items():
        try:
            df = pd.read_csv(file, encoding="utf-16", sep="\t", engine="python", quotechar='"', quoting=csv.QUOTE_MINIMAL, header=2)
            unnamed_cols = [col for col in df.columns if "Unnamed" in col]
            if unnamed_cols: df = df.drop(columns=unnamed_cols)
            df["marca"] = key
            dataframes[key] = df
        except Exception as e:
            st.error(f"Error al cargar {file.name}: {e}")
    if not dataframes:
        st.error("No se pudieron cargar archivos. Abortando.")
        return None
    df_final = pd.concat(dataframes.values(), ignore_index=True)
    UtilidadesColumnas.normalizar_columnas(df_final)
    st.write(f"Registros: {df_final.shape[0]} | Columnas: {df_final.shape[1]}")

    # Umbral interactivo para eliminar columnas con muchos nulos
    st.subheader("Limpieza: eliminación de columnas con muchos nulos")
    threshold = st.slider("Umbral de nulos (elimina columnas por encima de este porcentaje)", 0.0, 1.0, 0.9, 0.05)
    pre_agg = PreAgregacion(df_final)
    eliminadas = pre_agg.eliminar_columnas_muy_nulas(threshold=threshold)
    if eliminadas:
        st.info(f"Columnas eliminadas (> {int(threshold*100)}% nulos): {eliminadas}")
    # Conversión de porcentajes (selección interactiva)
    st.subheader("Transformación: columnas de porcentaje")
    posibles_pct = [col for col in ['three_month_change', 'yoy_change'] if col in df_final.columns]
    sel_pct = st.multiselect("Selecciona columnas de porcentaje a convertir (0-100 a 0-1)", posibles_pct, default=posibles_pct)
    if sel_pct:
        nuevos_nombres = [f"{c}_proportion" for c in sel_pct]
        porc = TransformacionPorcentaje(df_final)
        porc.transformar_porcentaje(sel_pct, nuevos_nombres)
        st.success(f"Columnas convertidas a proporción: {list(zip(sel_pct, nuevos_nombres))}")
    # Limpieza y tipificación de categóricas
    st.subheader("Transformación: columnas categóricas")
    posibles_cat = [c for c in ['competition', 'marca'] if c in df_final.columns]
    sel_cat = st.multiselect("Selecciona columnas categóricas a limpiar y tipar", posibles_cat, default=posibles_cat)
    if sel_cat:
        cat = TransformacionCategorica(df_final)
        cat.limpiar_texto(sel_cat)
        cat.convertir_a_categoria(sel_cat)
        st.success(f"Columnas categóricas procesadas: {sel_cat}")
    # Conversión de moneda (opcional)
    if tasa and sufijo_moneda:
        st.subheader("Transformación: conversión de moneda")
        aplicar_moneda = st.checkbox("Aplicar conversión de moneda", value=True)
        if aplicar_moneda:
            try:
                met = TransformacionMetrica(df_final)
                if 'top_of_page_bid_low_range' in df_final.columns:
                    met.convertir_moneda('top_of_page_bid_low_range', tasa, f'top_of_page_bid_low_{sufijo_moneda}')
                if 'top_of_page_bid_high_range' in df_final.columns:
                    met.convertir_moneda('top_of_page_bid_high_range', tasa, f'top_of_page_bid_high_{sufijo_moneda}')
                st.success("Conversión de moneda aplicada.")
            except Exception as e:
                st.error(f"Error en la conversión de moneda: {e}")
    # Agregación por keyword
    if 'keyword' not in df_final.columns:
        st.error("No se encontró la columna 'keyword' para la agregación.")
        return None
    columnas_numericas = df_final.select_dtypes(include=np.number).columns.tolist()
    columnas_categoricas = [c for c in ['competition', 'marca'] if c in df_final.columns]
    agg_dict = {col: 'mean' for col in columnas_numericas}
    for col in columnas_categoricas:
        agg_dict[col] = 'first'
    df_agregado = df_final.groupby('keyword').agg(agg_dict).reset_index()
    st.success(f"Dataset agregado. Número de keywords únicas: {len(df_agregado)}")
    post_agg = PostAgregacion(df_agregado)
    med_fills = post_agg.imputar_mediana_numericas()
    if med_fills:
        st.info(f"NaN rellenados con mediana en {len(med_fills)} columnas.")
    nulos_finales = df_agregado.isnull().sum().sum()
    if nulos_finales == 0:
        st.success("¡Perfecto! Tu dataset está 100% limpio y sin valores nulos.")
    else:
        st.warning(f"Aún quedan {nulos_finales} nulos.")
    st.write("Proceso de limpieza completado.")
    return df_agregado


if __name__ == "__main__":
    st.title("Pipeline de Preparación y Limpieza de Datos")
    st.write("Sube tus archivos CSV para iniciar el proceso de limpieza.")
    uploaded_files = st.file_uploader("Selecciona los archivos CSV", type=["csv"], accept_multiple_files=True)
    tasa = st.number_input("Ingresa la tasa de cambio (ej. 4000)", min_value=0.0, value=0.0)
    sufijo_moneda = st.text_input("Ingresa el sufijo para la moneda (ej. usd)")
    if st.button("Ejecutar limpieza"):
        df_limpio = ejecutar_pipeline_de_limpieza_streamlit(uploaded_files, tasa if tasa > 0 else None, sufijo_moneda if sufijo_moneda else None)
        if df_limpio is not None:
            st.write("Vista previa del dataset limpio:")
            st.dataframe(df_limpio.head())
            csv = df_limpio.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("Descargar dataset limpio CSV", csv, "dataset_limpio.csv", "text/csv")
