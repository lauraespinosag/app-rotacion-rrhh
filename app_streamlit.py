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
RESULTS_DIR = BASE_DIR / "resultados modelo rotacion"

MODEL_PATH = RESULTS_DIR / "modelo_rotacion.pkl"
SCALER_PATH = RESULTS_DIR / "scaler_rotacion.pkl"
COLUMNS_PATH = RESULTS_DIR / "columnas_modelo.pkl"
METADATA_PATH = RESULTS_DIR / "metadata_modelo.json"
DATASET_PATH = RESULTS_DIR / "dataset_app_streamlit.csv"

APP_VERSION = "modelo_sintetico_umbral_060_v2"


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

APP_VERSION = "modelo_sintetico_umbral_060_v4"

def cargar_modelo():
    modelo = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columnas_modelo = joblib.load(COLUMNS_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return modelo, scaler, columnas_modelo, metadata


def cargar_dataset():
    return pd.read_csv(DATASET_PATH)


modelo, scaler, columnas_modelo, metadata = cargar_modelo()

st.caption(f"Versión cargada: {APP_VERSION}")
st.caption(f"Carpeta de modelo cargada: {RESULTS_DIR}")
st.caption(f"Umbral principal cargado: {metadata.get('umbral_modelo_principal')}")

categorias = metadata.get("categorias_disponibles", {})

variables_modelo = metadata.get(
    "variables_modelo",
    [
        "antiguedad_anios",
        "posicionamiento_salarial",
        "evaluacion_global",
        "movilidad_funcional_sn",
        "tipo_contrato"
    ]
)

st.sidebar.header("Carga de datos")

archivo_subido = st.sidebar.file_uploader(
    "Sube un dataset para evaluarlo",
    type=["csv", "xlsx"]
)


def leer_dataset_subido(archivo):
    if archivo.name.endswith(".csv"):
        return pd.read_csv(archivo)
    elif archivo.name.endswith(".xlsx"):
        return pd.read_excel(archivo)
    else:
        st.error("Formato no compatible. Sube un archivo CSV o Excel.")
        st.stop()


def preparar_dataset_usuario(df_usuario):
    df_usuario = df_usuario.copy()

    columnas_faltantes = [
        col for col in variables_modelo
        if col not in df_usuario.columns
    ]

    if columnas_faltantes:
        st.error(
            "El dataset subido no tiene todas las columnas necesarias para aplicar el modelo."
        )
        st.write("Columnas que faltan:")
        st.write(columnas_faltantes)

        st.write("Columnas obligatorias:")
        st.write(variables_modelo)

        st.stop()

    df_usuario["antiguedad_anios"] = pd.to_numeric(
        df_usuario["antiguedad_anios"],
        errors="coerce"
    )

    for col in [
        "posicionamiento_salarial",
        "evaluacion_global",
        "movilidad_funcional_sn",
        "tipo_contrato"
    ]:
        df_usuario[col] = df_usuario[col].astype(str)

    if "tramo_antiguedad" not in df_usuario.columns:
        df_usuario["tramo_antiguedad"] = pd.cut(
            df_usuario["antiguedad_anios"],
            bins=[-1, 1, 3, 7, 15, 100],
            labels=[
                "0-1 años",
                "1-3 años",
                "4-7 años",
                "8-15 años",
                "Más de 15 años"
            ]
        ).astype(str)

    return df_usuario


if archivo_subido is not None:
    df = leer_dataset_subido(archivo_subido)
    df = preparar_dataset_usuario(df)

    st.sidebar.success("Dataset subido correctamente.")
    st.sidebar.write(f"Registros cargados: {len(df)}")

else:
    df = cargar_dataset()
    st.sidebar.info("Usando dataset de ejemplo guardado en la app.")
# ============================================================
# UMBRALES USADOS EN COLAB
# ============================================================

# Estos umbrales son los mismos que se definieron en Google Colab:
# Bajo: < 40 %
# Medio: 40 % - 60 %
# Alto: >= 60 %

UMBRAL_SEGUIMIENTO = 40
UMBRAL_PRIORIDAD_ALTA = 60

ESTADO_ALTA = "🔴 Prioridad alta"
ESTADO_SEGUIMIENTO = "🟡 Requiere seguimiento"
ESTADO_SIN_ALERTA = "🟢 Sin alerta"


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
    if probabilidad_pct >= UMBRAL_PRIORIDAD_ALTA:
        return ESTADO_ALTA
    elif probabilidad_pct >= UMBRAL_SEGUIMIENTO:
        return ESTADO_SEGUIMIENTO
    else:
        return ESTADO_SIN_ALERTA


def calcular_riesgo_dataset(df_base):
    df_pred = df_base.copy()
    df_pred["probabilidad_rotacion"] = predecir_probabilidad(df_pred)
    df_pred["probabilidad_%"] = df_pred["probabilidad_rotacion"] * 100
    df_pred["estado"] = df_pred["probabilidad_%"].apply(clasificar_estado)
    return df_pred


def resumen_estados(df_pred):
    total = len(df_pred)

    prioridad_alta = int((df_pred["estado"] == ESTADO_ALTA).sum())
    seguimiento = int((df_pred["estado"] == ESTADO_SEGUIMIENTO).sum())
    sin_alerta = int((df_pred["estado"] == ESTADO_SIN_ALERTA).sum())

    return {
        "total": total,
        "prioridad_alta": prioridad_alta,
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
            prioridad_alta=("estado", lambda x: (x == ESTADO_ALTA).sum()),
            seguimiento=("estado", lambda x: (x == ESTADO_SEGUIMIENTO).sum()),
            sin_alerta=("estado", lambda x: (x == ESTADO_SIN_ALERTA).sum())
        )
        .reset_index()
    )

    resumen["casos_a_revisar"] = resumen["prioridad_alta"] + resumen["seguimiento"]

    resumen["% a revisar"] = (
        resumen["casos_a_revisar"] / resumen["empleados"] * 100
    )

    resumen = resumen.sort_values(
        ["prioridad_alta", "seguimiento", "% a revisar"],
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
            "Estado": ESTADO_ALTA,
            "Situación actual": actual_res["prioridad_alta"],
            "Tras modificar la variable": sim_res["prioridad_alta"],
            "Diferencia": sim_res["prioridad_alta"] - actual_res["prioridad_alta"]
        },
        {
            "Estado": ESTADO_SEGUIMIENTO,
            "Situación actual": actual_res["seguimiento"],
            "Tras modificar la variable": sim_res["seguimiento"],
            "Diferencia": sim_res["seguimiento"] - actual_res["seguimiento"]
        },
        {
            "Estado": ESTADO_SIN_ALERTA,
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
    "Herramienta sencilla para ver cuántos perfiles aparecen como prioridad alta, "
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
    col2.metric("🔴 Prioridad alta", resumen_global["prioridad_alta"])
    col3.metric("🟡 Requiere seguimiento", resumen_global["seguimiento"])
    col4.metric("🟢 Sin alerta", resumen_global["sin_alerta"])

    st.markdown("### Distribución de estados")

    distribucion = pd.DataFrame([
        {"Estado": ESTADO_ALTA, "Casos": resumen_global["prioridad_alta"]},
        {"Estado": ESTADO_SEGUIMIENTO, "Casos": resumen_global["seguimiento"]},
        {"Estado": ESTADO_SIN_ALERTA, "Casos": resumen_global["sin_alerta"]}
    ])

    fig = px.pie(
        distribucion,
        names="Estado",
        values="Casos",
        hole=0.45,
        color="Estado",
        color_discrete_map={
            ESTADO_ALTA: "#D9534F",
            ESTADO_SEGUIMIENTO: "#F2B705",
            ESTADO_SIN_ALERTA: "#4CAF50"
        },
        title="Distribución global de perfiles"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Lectura sencilla")

    st.info(
        f"El modelo clasifica {resumen_global['prioridad_alta']} casos como prioridad alta "
        f"y {resumen_global['seguimiento']} como casos que requieren seguimiento. "
        f"Estos grupos son los que conviene observar con mayor detalle en las siguientes pestañas."
    )


# ============================================================
# TAB 2 - COLECTIVOS DE RIESGO
# ============================================================

with tab2:
    st.subheader("Colectivos de riesgo")

    st.write(
        "Esta pestaña permite ver en qué colectivos se concentran más casos en prioridad alta "
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

    col1.metric("Colectivo con más casos en prioridad alta", str(top[variable]))
    col2.metric("Casos en prioridad alta", int(top["prioridad_alta"]))
    col3.metric("Casos a revisar", int(top["casos_a_revisar"]))

    st.markdown("### Visualización por colectivo")

    resumen_grafico = resumen.melt(
        id_vars=[variable],
        value_vars=["prioridad_alta", "seguimiento", "sin_alerta"],
        var_name="Estado",
        value_name="Casos"
    )

    resumen_grafico["Estado"] = resumen_grafico["Estado"].replace({
        "prioridad_alta": ESTADO_ALTA,
        "seguimiento": ESTADO_SEGUIMIENTO,
        "sin_alerta": ESTADO_SIN_ALERTA
    })

    fig = px.bar(
        resumen_grafico,
        x="Casos",
        y=variable,
        color="Estado",
        orientation="h",
        color_discrete_map={
            ESTADO_ALTA: "#D9534F",
            ESTADO_SEGUIMIENTO: "#F2B705",
            ESTADO_SIN_ALERTA: "#4CAF50"
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
        "prioridad_alta",
        "seguimiento",
        "sin_alerta",
        "casos_a_revisar"
    ]].copy()

    tabla = tabla.rename(columns={
        variable: "Colectivo",
        "empleados": "Total empleados",
        "prioridad_alta": "🔴 Prioridad alta",
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
        "🔴 Prioridad alta",
        sim_res["prioridad_alta"],
        delta=sim_res["prioridad_alta"] - actual_res["prioridad_alta"],
        delta_color="inverse"
    )

    col2.metric(
        "🟡 Requiere seguimiento",
        sim_res["seguimiento"],
        delta=sim_res["seguimiento"] - actual_res["seguimiento"],
        delta_color="inverse"
    )

    col3.metric(
        "🟢 Sin alerta",
        sim_res["sin_alerta"],
        delta=sim_res["sin_alerta"] - actual_res["sin_alerta"]
    )

    diferencia_prioridad_alta = actual_res["prioridad_alta"] - sim_res["prioridad_alta"]

    diferencia_revision = (
        (actual_res["prioridad_alta"] + actual_res["seguimiento"]) -
        (sim_res["prioridad_alta"] + sim_res["seguimiento"])
    )

    if diferencia_prioridad_alta > 0:
        st.success(
            f"Según el modelo, al modificar esta variable habría {diferencia_prioridad_alta} casos menos en prioridad alta."
        )
    elif diferencia_prioridad_alta == 0:
        st.info(
            "Según el modelo, esta modificación no reduce los casos en prioridad alta."
        )
    else:
        st.warning(
            "Según el modelo, esta modificación aumentaría los casos en prioridad alta."
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
