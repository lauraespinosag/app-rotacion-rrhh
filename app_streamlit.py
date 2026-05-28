import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Rotación voluntaria - RRHH",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# RUTAS
# ============================================================

BASE_DIR = Path(__file__).parent

# Detecta automáticamente si la carpeta está con guiones bajos o espacios
if (BASE_DIR / "resultados_modelo_rotacion").exists():
    RESULTS_DIR = BASE_DIR / "resultados_modelo_rotacion"
else:
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
# FUNCIONES DEL MODELO
# ============================================================

def preparar_para_modelo(df_input):
    X = pd.get_dummies(df_input[variables_modelo], drop_first=True)
    X = X.reindex(columns=columnas_modelo, fill_value=0)
    X_scaled = scaler.transform(X)
    return X_scaled


def predecir_probabilidad(df_input):
    X_scaled = preparar_para_modelo(df_input)
    return modelo.predict_proba(X_scaled)[:, 1]


def clasificar_estado(probabilidad_pct):
    if probabilidad_pct >= 75:
        return "🔴 Prioridad alta"
    elif probabilidad_pct >= 55:
        return "🟡 Requiere seguimiento"
    else:
        return "🟢 Sin alerta"


def calcular_riesgo_dataset(df_base):
    df_pred = df_base.copy()
    df_pred["probabilidad_rotacion"] = predecir_probabilidad(df_pred)
    df_pred["probabilidad_%"] = df_pred["probabilidad_rotacion"] * 100
    df_pred["estado"] = df_pred["probabilidad_%"].apply(clasificar_estado)
    return df_pred


def resumen_estados(df_pred):
    total = len(df_pred)

    extremos = int((df_pred["estado"] == "🔴 Riesgo extremo").sum())
    seguimiento = int((df_pred["estado"] == "🟡 Requiere seguimiento").sum())
    sin_alerta = int((df_pred["estado"] == "🟢 Sin alerta").sum())

    return {
        "total": total,
        "extremos": extremos,
        "seguimiento": seguimiento,
        "sin_alerta": sin_alerta
    }


def mejor_posicionamiento_salarial():
    opciones = categorias["posicionamiento_salarial"]

    for op in opciones:
        if "Por encima" in op:
            return op

    for op in opciones:
        if "máximo" in op or "maximo" in op:
            return op

    return opciones[-1]


def resumen_por_colectivo(df_pred, variable):
    resumen = (
        df_pred
        .groupby(variable)
        .agg(
            empleados=(variable, "count"),
            riesgo_extremo=("estado", lambda x: (x == "🔴 Riesgo extremo").sum()),
            seguimiento=("estado", lambda x: (x == "🟡 Requiere seguimiento").sum()),
            sin_alerta=("estado", lambda x: (x == "🟢 Sin alerta").sum())
        )
        .reset_index()
    )

    resumen["casos_a_revisar"] = resumen["riesgo_extremo"] + resumen["seguimiento"]

    resumen["% a revisar"] = (
        resumen["casos_a_revisar"] / resumen["empleados"] * 100
    )

    resumen = resumen.sort_values(
        ["riesgo_extremo", "seguimiento", "% a revisar"],
        ascending=False
    )

    return resumen


def aplicar_simulacion(df_segmento, simulacion):
    df_sim = df_segmento.copy()

    if simulacion == "Movilidad funcional = Sí":
        df_sim["movilidad_funcional_sn"] = "Sí"

    elif simulacion == "Posicionamiento salarial más favorable":
        df_sim["posicionamiento_salarial"] = mejor_posicionamiento_salarial()

    elif simulacion == "Movilidad + salario":
        df_sim["movilidad_funcional_sn"] = "Sí"
        df_sim["posicionamiento_salarial"] = mejor_posicionamiento_salarial()

    return df_sim


def tabla_comparacion(actual, simulado):
    actual_res = resumen_estados(actual)
    sim_res = resumen_estados(simulado)

    datos = pd.DataFrame([
        {
            "Estado": "🔴 Riesgo extremo",
            "Situación actual": actual_res["extremos"],
            "Tras modificar la variable": sim_res["extremos"],
            "Diferencia": sim_res["extremos"] - actual_res["extremos"]
        },
        {
            "Estado": "🟡 Requiere seguimiento",
            "Situación actual": actual_res["seguimiento"],
            "Tras modificar la variable": sim_res["seguimiento"],
            "Diferencia": sim_res["seguimiento"] - actual_res["seguimiento"]
        },
        {
            "Estado": "🟢 Sin alerta",
            "Situación actual": actual_res["sin_alerta"],
            "Tras modificar la variable": sim_res["sin_alerta"],
            "Diferencia": sim_res["sin_alerta"] - actual_res["sin_alerta"]
        }
    ])

    return datos


# ============================================================
# DATOS PREDICHOS
# ============================================================

df_pred = calcular_riesgo_dataset(df)
resumen_global = resumen_estados(df_pred)


# ============================================================
# CABECERA
# ============================================================

st.title("App de apoyo a RRHH: rotación voluntaria")

st.write(
    "Herramienta sencilla para ver cuántos perfiles están en riesgo extremo, "
    "cuántos requieren seguimiento y cómo cambiaría la estimación del modelo "
    "si se modificaran algunas variables organizacionales."
)

st.warning(
    "La app no demuestra causalidad ni toma decisiones automáticas. "
    "Solo muestra cómo cambia la estimación del modelo."
)


# ============================================================
# PESTAÑAS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "1. Panel general",
    "2. Colectivos de riesgo",
    "3. Simulador de variables"
])


# ============================================================
# TAB 1 - PANEL GENERAL
# ============================================================

with tab1:
    st.subheader("Panel general")

    st.write(
        "Resumen de la situación estimada por el modelo para toda la muestra."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Empleados analizados", resumen_global["total"])
    col2.metric("🔴 Riesgo extremo", resumen_global["extremos"])
    col3.metric("🟡 Requiere seguimiento", resumen_global["seguimiento"])
    col4.metric("🟢 Sin alerta", resumen_global["sin_alerta"])

    st.markdown("### Distribución de estados")

    distribucion = pd.DataFrame([
        {"Estado": "🔴 Riesgo extremo", "Casos": resumen_global["extremos"]},
        {"Estado": "🟡 Requiere seguimiento", "Casos": resumen_global["seguimiento"]},
        {"Estado": "🟢 Sin alerta", "Casos": resumen_global["sin_alerta"]}
    ])

    fig = px.pie(
        distribucion,
        names="Estado",
        values="Casos",
        hole=0.45,
        color="Estado",
        color_discrete_map={
            "🔴 Riesgo extremo": "#D9534F",
            "🟡 Requiere seguimiento": "#F2B705",
            "🟢 Sin alerta": "#4CAF50"
        },
        title="Distribución global de perfiles"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Lectura sencilla")

    st.info(
        f"El modelo clasifica {resumen_global['extremos']} casos como riesgo extremo "
        f"y {resumen_global['seguimiento']} como casos que requieren seguimiento. "
        f"Estos grupos son los que conviene observar con mayor detalle en las siguientes pestañas."
    )


# ============================================================
# TAB 2 - COLECTIVOS DE RIESGO
# ============================================================

with tab2:
    st.subheader("Colectivos de riesgo")

    st.write(
        "Esta pestaña permite ver en qué colectivos se concentran más casos en riesgo extremo "
        "o que requieren seguimiento."
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

    nombres = {
        "empresa": "Empresa",
        "area_departamento": "Área / departamento",
        "posicionamiento_salarial": "Posicionamiento salarial",
        "evaluacion_global": "Evaluación global",
        "movilidad_funcional_sn": "Movilidad funcional",
        "tipo_contrato": "Tipo de contrato",
        "tramo_antiguedad": "Tramo de antigüedad"
    }

    variable = st.selectbox(
        "Selecciona qué quieres revisar",
        opciones_agrupacion,
        format_func=lambda x: nombres.get(x, x)
    )

    resumen = resumen_por_colectivo(df_pred, variable)

    st.markdown("### Colectivos ordenados por prioridad")

    top = resumen.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Colectivo con más casos extremos", str(top[variable]))
    col2.metric("Casos en riesgo extremo", int(top["riesgo_extremo"]))
    col3.metric("Casos a revisar", int(top["casos_a_revisar"]))

    st.markdown("### Visualización por colectivo")

    resumen_grafico = resumen.melt(
        id_vars=[variable],
        value_vars=["riesgo_extremo", "seguimiento", "sin_alerta"],
        var_name="Estado",
        value_name="Casos"
    )

    resumen_grafico["Estado"] = resumen_grafico["Estado"].replace({
        "riesgo_extremo": "🔴 Riesgo extremo",
        "seguimiento": "🟡 Requiere seguimiento",
        "sin_alerta": "🟢 Sin alerta"
    })

    fig = px.bar(
        resumen_grafico,
        x="Casos",
        y=variable,
        color="Estado",
        orientation="h",
        color_discrete_map={
            "🔴 Riesgo extremo": "#D9534F",
            "🟡 Requiere seguimiento": "#F2B705",
            "🟢 Sin alerta": "#4CAF50"
        },
        title=f"Casos por estado según {nombres.get(variable, variable)}"
    )

    fig.update_layout(
        yaxis_title="",
        xaxis_title="Número de casos",
        height=550,
        legend_title="Estado"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Tabla sencilla")

    tabla = resumen[[
        variable,
        "empleados",
        "riesgo_extremo",
        "seguimiento",
        "sin_alerta",
        "casos_a_revisar"
    ]].copy()

    tabla = tabla.rename(columns={
        variable: "Colectivo",
        "empleados": "Total empleados",
        "riesgo_extremo": "🔴 Riesgo extremo",
        "seguimiento": "🟡 Seguimiento",
        "sin_alerta": "🟢 Sin alerta",
        "casos_a_revisar": "Total a revisar"
    })

    st.dataframe(tabla, use_container_width=True)


# ============================================================
# TAB 3 - SIMULADOR DE VARIABLES
# ============================================================

with tab3:
    st.subheader("Simulador de variables")

    st.write(
        "Selecciona un colectivo y una variable a modificar. "
        "La app mostrará cuántos casos cambiarían de estado según la estimación del modelo."
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
        grupo = st.selectbox(
            "Selecciona colectivo",
            opciones_agrupacion,
            format_func=lambda x: nombres.get(x, x),
            key="grupo_simulador"
        )

    with col_b:
        segmento = st.selectbox(
            "Selecciona segmento",
            sorted(df[grupo].dropna().astype(str).unique().tolist()),
            key="segmento_simulador"
        )

    simulacion = st.selectbox(
        "Selecciona qué variable quieres modificar",
        [
            "Movilidad funcional = Sí",
            "Posicionamiento salarial más favorable",
            "Movilidad + salario"
        ]
    )

    df_segmento = df[df[grupo].astype(str) == segmento].copy()

    actual = calcular_riesgo_dataset(df_segmento)

    df_modificado = aplicar_simulacion(df_segmento, simulacion)
    simulado = calcular_riesgo_dataset(df_modificado)

    actual_res = resumen_estados(actual)
    sim_res = resumen_estados(simulado)

    st.markdown("### Resultado de la simulación")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🔴 Riesgo extremo",
        sim_res["extremos"],
        delta=sim_res["extremos"] - actual_res["extremos"]
    )

    col2.metric(
        "🟡 Requiere seguimiento",
        sim_res["seguimiento"],
        delta=sim_res["seguimiento"] - actual_res["seguimiento"]
    )

    col3.metric(
        "🟢 Sin alerta",
        sim_res["sin_alerta"],
        delta=sim_res["sin_alerta"] - actual_res["sin_alerta"]
    )

    diferencia_extremos = actual_res["extremos"] - sim_res["extremos"]
    diferencia_revision = (
        (actual_res["extremos"] + actual_res["seguimiento"]) -
        (sim_res["extremos"] + sim_res["seguimiento"])
    )

    if diferencia_extremos > 0:
        st.success(
            f"Según el modelo, al modificar esta variable habría {diferencia_extremos} casos menos en riesgo extremo."
        )
    elif diferencia_extremos == 0:
        st.info(
            "Según el modelo, esta modificación no reduce los casos en riesgo extremo."
        )
    else:
        st.warning(
            "Según el modelo, esta modificación aumentaría los casos en riesgo extremo."
        )

    if diferencia_revision > 0:
        st.info(
            f"En total, habría {diferencia_revision} casos menos que requerirían revisión."
        )

    st.markdown("### Comparación antes / después")

    comparacion = tabla_comparacion(actual, simulado)

    fig = px.bar(
        comparacion.melt(
            id_vars="Estado",
            value_vars=["Situación actual", "Tras modificar la variable"],
            var_name="Escenario",
            value_name="Casos"
        ),
        x="Estado",
        y="Casos",
        color="Escenario",
        barmode="group",
        title="Comparación de casos antes y después de modificar la variable"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(comparacion, use_container_width=True)

    st.warning(
        "Nota: esta simulación no demuestra que la acción cause una reducción real de la rotación. "
        "Solo muestra cómo cambiaría la clasificación estimada por el modelo."
    )
