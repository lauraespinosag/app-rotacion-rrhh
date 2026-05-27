import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Rotación voluntaria - RRHH",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "resultados modelo rotacion"

MODEL_PATH = RESULTS_DIR / "modelo_rotacion.pkl"
SCALER_PATH = RESULTS_DIR / "scaler_rotacion.pkl"
COLUMNS_PATH = RESULTS_DIR / "columnas_modelo.pkl"
METADATA_PATH = RESULTS_DIR / "metadata_modelo.json"
DATASET_PATH = RESULTS_DIR / "dataset_app_streamlit.csv"


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columnas_modelo = joblib.load(COLUMNS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return modelo, scaler, columnas_modelo, metadata


@st.cache_data
def cargar_dataset():
    return pd.read_csv(DATASET_PATH)


modelo, scaler, columnas_modelo, metadata = cargar_modelo()
df = cargar_dataset()

categorias = metadata["categorias_disponibles"]

variables_modelo = [
    "antiguedad_anios",
    "posicionamiento_salarial",
    "evaluacion_global",
    "movilidad_funcional_sn",
    "tipo_contrato"
]


# ============================================================
# FUNCIONES
# ============================================================

def preparar_para_modelo(df_input):
    X = pd.get_dummies(df_input[variables_modelo], drop_first=True)
    X = X.reindex(columns=columnas_modelo, fill_value=0)
    X_scaled = scaler.transform(X)
    return X_scaled


def predecir_probabilidad(df_input):
    X_scaled = preparar_para_modelo(df_input)
    return modelo.predict_proba(X_scaled)[:, 1]


def clasificar_riesgo(prob):
    if prob < 0.35:
        return "Bajo"
    elif prob < 0.55:
        return "Medio"
    else:
        return "Alto"


def calcular_riesgo_dataset(df_base):
    df_pred = df_base.copy()
    df_pred["probabilidad_rotacion"] = predecir_probabilidad(df_pred)
    df_pred["probabilidad_%"] = df_pred["probabilidad_rotacion"] * 100
    df_pred["nivel_riesgo"] = df_pred["probabilidad_rotacion"].apply(clasificar_riesgo)
    return df_pred


def resumen_por_colectivo(df_pred, variable):
    resumen = (
        df_pred
        .groupby(variable)
        .agg(
            empleados=(variable, "count"),
            riesgo_medio=("probabilidad_%", "mean"),
            riesgo_alto=("nivel_riesgo", lambda x: (x == "Alto").sum()),
            riesgo_medio_n=("nivel_riesgo", lambda x: (x == "Medio").sum()),
            riesgo_bajo=("nivel_riesgo", lambda x: (x == "Bajo").sum())
        )
        .reset_index()
    )

    resumen["pct_medio_alto"] = (
        (resumen["riesgo_alto"] + resumen["riesgo_medio_n"]) /
        resumen["empleados"] * 100
    )

    resumen = resumen.sort_values("riesgo_medio", ascending=False)

    return resumen


def mejor_posicionamiento_salarial():
    opciones = categorias["posicionamiento_salarial"]

    for op in opciones:
        if "Por encima" in op:
            return op

    for op in opciones:
        if "máximo" in op or "maximo" in op:
            return op

    return opciones[-1]


# ============================================================
# CABECERA
# ============================================================

st.title("App de apoyo a RRHH: rotación voluntaria")

st.write(
    "Esta aplicación permite identificar colectivos con mayor riesgo estimado de rotación "
    "y simular cómo cambiaría el riesgo si se modifican algunas variables organizacionales."
)

st.warning(
    "La app no predice con certeza quién se va a ir. Sirve como apoyo para priorizar análisis de RRHH."
)


# Calcular predicción para todo el dataset
df_pred = calcular_riesgo_dataset(df)


# ============================================================
# PESTAÑAS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "1. Colectivos de riesgo",
    "2. Simulador de acciones",
    "3. Predicción individual"
])


# ============================================================
# TAB 1 - COLECTIVOS DE RIESGO
# ============================================================

with tab1:
    st.subheader("Colectivos de riesgo")

    st.write(
        "Esta pestaña permite ver qué colectivos presentan mayor riesgo medio estimado. "
        "Es útil para priorizar acciones de RRHH por áreas o segmentos."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Empleados analizados", len(df_pred))
    col2.metric("Riesgo medio estimado", f"{df_pred['probabilidad_%'].mean():.2f} %")
    col3.metric(
        "% riesgo medio/alto",
        f"{df_pred['nivel_riesgo'].isin(['Medio', 'Alto']).mean() * 100:.2f} %"
    )

    opciones_agrupacion = [
        col for col in [
            "empresa",
            "area_departamento",
            "posicionamiento_salarial",
            "evaluacion_global",
            "movilidad_funcional_sn",
            "tipo_contrato",
            "tramo_antiguedad"
        ] if col in df_pred.columns
    ]

    variable = st.selectbox("Agrupar por:", opciones_agrupacion)

    resumen = resumen_por_colectivo(df_pred, variable)

    fig = px.bar(
        resumen.sort_values("riesgo_medio"),
        x="riesgo_medio",
        y=variable,
        orientation="h",
        text=resumen.sort_values("riesgo_medio")["riesgo_medio"].round(1),
        title=f"Riesgo medio estimado por {variable}"
    )

    fig.update_layout(
        xaxis_title="Probabilidad media estimada de rotación (%)",
        yaxis_title="",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(resumen.round(2), use_container_width=True)

    st.info(
        "Interpretación: los colectivos con mayor riesgo medio estimado deberían revisarse "
        "desde una perspectiva organizativa, no individual."
    )


# ============================================================
# TAB 2 - SIMULADOR DE ACCIONES
# ============================================================

with tab2:
    st.subheader("Simulador de acciones")

    st.write(
        "Este apartado permite simular cómo cambiaría el riesgo estimado si se modifican "
        "algunas variables relacionadas con las hipótesis del TFM."
    )

    opciones_agrupacion = [
        col for col in [
            "empresa",
            "area_departamento",
            "posicionamiento_salarial",
            "evaluacion_global",
            "movilidad_funcional_sn",
            "tipo_contrato",
            "tramo_antiguedad"
        ] if col in df.columns
    ]

    col_a, col_b = st.columns(2)

    with col_a:
        grupo = st.selectbox("Selecciona colectivo:", opciones_agrupacion)

    with col_b:
        valor = st.selectbox(
            "Selecciona segmento:",
            sorted(df[grupo].dropna().astype(str).unique().tolist())
        )

    df_segmento = df[df[grupo].astype(str) == valor].copy()

    if len(df_segmento) == 0:
        st.warning("No hay empleados en el segmento seleccionado.")
    else:
        escenarios = []

        # Escenario actual
        actual = calcular_riesgo_dataset(df_segmento)

        escenarios.append({
            "Escenario": "Situación actual",
            "Riesgo medio estimado": actual["probabilidad_%"].mean()
        })

        # Escenario movilidad funcional
        sim_mov = df_segmento.copy()
        sim_mov["movilidad_funcional_sn"] = "Sí"
        sim_mov = calcular_riesgo_dataset(sim_mov)

        escenarios.append({
            "Escenario": "Con movilidad funcional",
            "Riesgo medio estimado": sim_mov["probabilidad_%"].mean()
        })

        # Escenario mejora salarial
        sim_sal = df_segmento.copy()
        sim_sal["posicionamiento_salarial"] = mejor_posicionamiento_salarial()
        sim_sal = calcular_riesgo_dataset(sim_sal)

        escenarios.append({
            "Escenario": "Mejor posicionamiento salarial",
            "Riesgo medio estimado": sim_sal["probabilidad_%"].mean()
        })

        # Escenario combinado
        sim_comb = df_segmento.copy()
        sim_comb["movilidad_funcional_sn"] = "Sí"
        sim_comb["posicionamiento_salarial"] = mejor_posicionamiento_salarial()
        sim_comb = calcular_riesgo_dataset(sim_comb)

        escenarios.append({
            "Escenario": "Movilidad + mejora salarial",
            "Riesgo medio estimado": sim_comb["probabilidad_%"].mean()
        })

        escenarios_df = pd.DataFrame(escenarios)

        riesgo_actual = escenarios_df.loc[
            escenarios_df["Escenario"] == "Situación actual",
            "Riesgo medio estimado"
        ].iloc[0]

        escenarios_df["Cambio frente a situación actual"] = (
            escenarios_df["Riesgo medio estimado"] - riesgo_actual
        )

        fig = px.bar(
            escenarios_df.sort_values("Riesgo medio estimado"),
            x="Riesgo medio estimado",
            y="Escenario",
            orientation="h",
            text=escenarios_df.sort_values("Riesgo medio estimado")["Riesgo medio estimado"].round(1),
            title="Comparación de escenarios"
        )

        fig.update_layout(
            xaxis_title="Riesgo medio estimado (%)",
            yaxis_title="",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(escenarios_df.round(2), use_container_width=True)

        st.warning(
            "Importante: esta simulación no demuestra causalidad. Solo muestra cómo cambiaría "
            "la probabilidad estimada según el modelo si se modifican ciertas variables."
        )


# ============================================================
# TAB 3 - PREDICCIÓN INDIVIDUAL
# ============================================================

with tab3:
    st.subheader("Predicción individual orientativa")

    st.write(
        "Este apartado permite introducir un perfil concreto. Debe usarse solo como simulación orientativa."
    )

    col1, col2 = st.columns(2)

    with col1:
        antiguedad = st.slider("Antigüedad en años", 0.0, 40.0, 5.0, 0.5)

        posicionamiento = st.selectbox(
            "Posicionamiento salarial",
            categorias["posicionamiento_salarial"]
        )

        evaluacion = st.selectbox(
            "Evaluación global",
            categorias["evaluacion_global"]
        )

        movilidad = st.selectbox(
            "Movilidad funcional",
            categorias["movilidad_funcional_sn"]
        )

        contrato = st.selectbox(
            "Tipo de contrato",
            categorias["tipo_contrato"]
        )

    caso = pd.DataFrame([{
        "antiguedad_anios": antiguedad,
        "posicionamiento_salarial": posicionamiento,
        "evaluacion_global": evaluacion,
        "movilidad_funcional_sn": movilidad,
        "tipo_contrato": contrato
    }])

    prob = float(predecir_probabilidad(caso)[0])
    nivel = clasificar_riesgo(prob)

    with col2:
        st.metric("Probabilidad estimada", f"{prob * 100:.2f} %")
        st.metric("Nivel orientativo", nivel)

        if nivel == "Bajo":
            st.success("Riesgo bajo: seguimiento ordinario.")
        elif nivel == "Medio":
            st.info("Riesgo medio: revisar contexto del perfil.")
        else:
            st.warning("Riesgo alto: revisar con mayor detalle, sin tomar decisiones automáticas.")

    st.dataframe(caso, use_container_width=True)
