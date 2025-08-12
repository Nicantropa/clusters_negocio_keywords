import streamlit as st
import pandas as pd
import csv
import os
import openpyxl
import re
import numpy as np

# --- 1. CLASES DE TRANSFORMACIÓN  ---


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


# --- 2. CARGA Y PIPELINE (separados UI vs lógica) ---
def cargar_concat_df(uploaded_files):
    """Carga múltiples CSV del uploader, concatena y normaliza columnas."""
    if not uploaded_files:
        return None
    files = {f"df_{os.path.splitext(f.name)[0].replace(' ', '_').lower()}": f for f in uploaded_files}
    dataframes = {}
    for key, file in files.items():
        try:
            # Reiniciar el puntero del archivo por si ya fue leído en un rerun
            try:
                file.seek(0)
            except Exception:
                pass
            df = pd.read_csv(
                file,
                encoding="utf-16",
                sep="\t",
                engine="python",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                header=2,
            )
            unnamed_cols = [col for col in df.columns if "Unnamed" in col]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
            df["marca"] = key
            dataframes[key] = df
        except Exception as e:
            st.error(f"Error al cargar {getattr(file, 'name', 'archivo')}: {e}")
    if not dataframes:
        return None
    df_final = pd.concat(dataframes.values(), ignore_index=True)
    UtilidadesColumnas.normalizar_columnas(df_final)
    return df_final


def aplicar_pipeline(df_final: pd.DataFrame, *, threshold: float, sel_pct: list[str], sel_cat: list[str], aplicar_moneda: bool, tasa: float | None, sufijo_moneda: str | None):
    """Aplica las transformaciones al DataFrame ya cargado según parámetros."""
    if df_final is None or df_final.empty:
        st.error("No hay datos para procesar.")
        return None

    st.write(f"Registros: {df_final.shape[0]} | Columnas: {df_final.shape[1]}")

    # 1) Eliminar columnas con muchos nulos
    st.subheader("Limpieza: eliminación de columnas con muchos nulos")
    pre_agg = PreAgregacion(df_final)
    eliminadas = pre_agg.eliminar_columnas_muy_nulas(threshold=threshold)
    if eliminadas:
        st.info(f"Columnas eliminadas (> {int(threshold*100)}% nulos): {eliminadas}")

    # 2) Conversión de porcentajes
    if sel_pct:
        st.subheader("Transformación: columnas de porcentaje")
        nuevos_nombres = [f"{c}_proportion" for c in sel_pct]
        porc = TransformacionPorcentaje(df_final)
        porc.transformar_porcentaje(sel_pct, nuevos_nombres)
        st.success(f"Columnas convertidas a proporción: {list(zip(sel_pct, nuevos_nombres))}")

    # 3) Categóricas
    if sel_cat:
        st.subheader("Transformación: columnas categóricas")
        cat = TransformacionCategorica(df_final)
        cat.limpiar_texto(sel_cat)
        cat.convertir_a_categoria(sel_cat)
        st.success(f"Columnas categóricas procesadas: {sel_cat}")

    # 4) Conversión de moneda (opcional)
    if aplicar_moneda and tasa and sufijo_moneda:
        st.subheader("Transformación: conversión de moneda")
        try:
            met = TransformacionMetrica(df_final)
            if 'top_of_page_bid_low_range' in df_final.columns:
                met.convertir_moneda('top_of_page_bid_low_range', tasa, f'top_of_page_bid_low_{sufijo_moneda}')
            if 'top_of_page_bid_high_range' in df_final.columns:
                met.convertir_moneda('top_of_page_bid_high_range', tasa, f'top_of_page_bid_high_{sufijo_moneda}')
            st.success("Conversión de moneda aplicada.")
        except Exception as e:
            st.error(f"Error en la conversión de moneda: {e}")

    # 5) Agregación por keyword
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

    # 6) Imputación mediana
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


st.title("Pipeline de Preparación y Limpieza de Datos")
st.write("Sube tus archivos CSV, configura parámetros y ejecuta la limpieza.")

# Estado de la página para evitar que los cambios de widgets oculten el pipeline
if "run_cleanup" not in st.session_state:
    st.session_state.run_cleanup = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None
if "tasa" not in st.session_state:
    st.session_state.tasa = 0.0
if "sufijo_moneda" not in st.session_state:
    st.session_state.sufijo_moneda = ""

with st.form("form_limpieza"):
    uploaded_files = st.file_uploader(
        "Selecciona los archivos CSV",
        type=["csv"],
        accept_multiple_files=True,
        key="uploader_csvs",
        help="Puedes seleccionar múltiples archivos a la vez.",
    )
    tasa = st.number_input("Ingresa la tasa de cambio (ej. 4000)", min_value=0.0, value=st.session_state.tasa)
    sufijo_moneda = st.text_input("Ingresa el sufijo para la moneda (ej. usd)", value=st.session_state.sufijo_moneda)
    submitted = st.form_submit_button("Ejecutar limpieza")

if submitted:
    # Persistir selección y marcar ejecución
    st.session_state.uploaded_files = uploaded_files
    st.session_state.tasa = tasa
    st.session_state.sufijo_moneda = sufijo_moneda
    st.session_state.run_cleanup = True
    st.session_state.just_submitted = True

# Ejecutar pipeline cuando haya archivos y el usuario lo haya pedido (y mantenerlo visible en reruns)
if st.session_state.run_cleanup:
    if not st.session_state.uploaded_files:
        st.warning("No se han subido archivos CSV.")
    else:
        # 1) Cargar y mostrar info general
        df_base = cargar_concat_df(st.session_state.uploaded_files)
        if df_base is None:
            st.error("No se pudieron cargar los archivos.")
        else:
            st.info(f"Dataset cargado: {df_base.shape[0]} filas x {df_base.shape[1]} columnas")

            # 2) Parámetros de limpieza (persisten entre ejecuciones)
            st.subheader("Parámetros de limpieza")
            default_threshold = st.session_state.get("param_threshold", 0.9)
            threshold = st.slider("Umbral de nulos (elimina columnas por encima de este porcentaje)", 0.0, 1.0, float(default_threshold), 0.05, key="threshold_slider")
            st.session_state.param_threshold = threshold

            posibles_pct = [c for c in ['three_month_change', 'yoy_change'] if c in df_base.columns]
            default_pct = st.session_state.get("param_sel_pct", posibles_pct)
            sel_pct = st.multiselect("Columnas de porcentaje a convertir (0-100 a 0-1)", posibles_pct, default=default_pct, key="sel_pct_ms")
            st.session_state.param_sel_pct = sel_pct

            posibles_cat = [c for c in ['competition', 'marca'] if c in df_base.columns]
            default_cat = st.session_state.get("param_sel_cat", posibles_cat)
            sel_cat = st.multiselect("Columnas categóricas a limpiar y tipar", posibles_cat, default=default_cat, key="sel_cat_ms")
            st.session_state.param_sel_cat = sel_cat

            aplicar_moneda_default = st.session_state.get("param_aplicar_moneda", True)
            aplicar_moneda = st.checkbox("Aplicar conversión de moneda", value=aplicar_moneda_default, key="chk_moneda") if (st.session_state.tasa and st.session_state.sufijo_moneda) else False
            st.session_state.param_aplicar_moneda = aplicar_moneda

            # 3) Ejecutar limpieza: auto-una-vez tras el submit, luego solo con botón
            ejecutar_ahora = st.session_state.get("just_submitted", False)
            btn_re_ejecutar = st.button("Ejecutar limpieza", key="btn_ejecutar_params")
            do_process = ejecutar_ahora or btn_re_ejecutar
            st.session_state.just_submitted = False  # consumir el auto-run

            if do_process:
                df_limpio = aplicar_pipeline(
                    df_base.copy(),
                    threshold=threshold,
                    sel_pct=sel_pct,
                    sel_cat=sel_cat,
                    aplicar_moneda=aplicar_moneda,
                    tasa=st.session_state.tasa if st.session_state.tasa and st.session_state.tasa > 0 else None,
                    sufijo_moneda=st.session_state.sufijo_moneda if st.session_state.sufijo_moneda else None,
                )
                if df_limpio is not None:
                    st.write("Vista previa del dataset limpio:")
                    st.dataframe(df_limpio.head())
                    csv = df_limpio.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button("Descargar dataset limpio CSV", csv, "dataset_limpio.csv", "text/csv", key="dl_dataset_limpio")
