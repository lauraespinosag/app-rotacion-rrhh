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

# Detecta automáticamente si la carpeta está con guiones bajos o con espacios
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


def calcular_riesgo_dataset(df_base):
    df_pred = df_base.copy()
    df_pred["probabilidad_rotacion"] = predecir_probabilidad(df_pred)
    df_pred["probabilidad_%"] = df_pred["probabilidad_rotacion"] * 100
    df_pred["prioridad"] = df_pred["probabilidad_%"].apply(clasificar_prioridad)
    return df_pred


# ============================================================
# FUNCIONES DE LECTURA PARA RRHH
# ============================================================

def clasificar_prioridad(probabilidad_pct):
    if probabilidad_pct >= 55:
        return "🔴 Revisar primero"
    elif probabilidad_pct >= 35:
        return "🟡 Revisar contexto"
    else:
        return "🟢 Seguimiento ordinario"


def color_prioridad(prioridad):
    if "🔴" in prioridad:
        return "#D9534F"
    elif "🟡" in prioridad:
        return "#F2B705"
    else:
        return "#4CAF50"


def lectura_prioridad(prioridad):
    if "🔴" in prioridad:
        return "Este colectivo debería revisarse de forma prioritaria."
    elif "🟡" in prioridad:
        return "Este colectivo no requiere alarma, pero sí seguimiento y revisión de contexto."
    else:
        return "Este colectivo no muestra una señal agregada alta según el modelo."


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
            riesgo_medio=("probabilidad_%", "mean"),
            riesgo_alto=("prioridad", lambda x: x.astype(str).str.contains("🔴").sum()),
            riesgo_medio_n=("prioridad", lambda x: x.astype(str).str.contains("🟡").sum()),
            riesgo_bajo=("prioridad", lambda x: x.astype(str).str.contains("🟢").sum())
        )
        .reset_index()
    )

    resumen["pct_revisar"] = (
        (resumen["riesgo_alto"] + resumen["riesgo_medio_n"]) /
        resumen["empleados"] * 100
    )

    resumen["prioridad"] = resumen["riesgo_medio"].apply(clasificar_prioridad)
    resumen["lectura_RRHH"] = resumen["prioridad"].apply(lectura_prioridad)

    resumen = resumen.sort_values("riesgo_medio", ascending=False)

    return resumen


def explicar_puntos_dolor(df_segmento, df_total):
    puntos = []

    # Movilidad funcional
    if "movilidad_funcional_sn" in df_segmento.columns:
        pct_sin_mov = (df_segmento["movilidad_funcional_sn"].astype(str) == "No").mean() * 100

        if pct_sin_mov >= 50:
            puntos.append({
                "Variable": "Movilidad funcional",
                "Lectura": "Gran parte del colectivo no tiene movilidad funcional registrada.",
                "Qué revisar desde RRHH": "Oportunidades de movilidad interna, cambio de proyecto, promoción o desarrollo profesional."
            })

    # Posicionamiento salarial
    if "posicionamiento_salarial" in df_segmento.columns:
        texto_salario = df_segmento["posicionamiento_salarial"].astype(str)

        pct_salario_bajo = (
            texto_salario.str.contains("Por debajo", case=False, na=False) |
            texto_salario.str.contains("mínimo", case=False, na=False) |
            texto_salario.str.contains("minimo", case=False, na=False)
        ).mean() * 100

        if pct_salario_bajo >= 40:
            puntos.append({
                "Variable": "Posicionamiento salarial",
                "Lectura": "Una parte relevante del colectivo está en posiciones salariales bajas o medio-bajas.",
                "Qué revisar desde RRHH": "Equidad interna, competitividad salarial y evolución dentro de la banda."
            })

    # Antigüedad
    if "antiguedad_anios" in df_segmento.columns:
        antig_segmento = df_segmento["antiguedad_anios"].mean()
        antig_total = df_total["antiguedad_anios"].mean()

        if antig_segmento < antig_total:
            puntos.append({
                "Variable": "Antigüedad",
                "Lectura": "El colectivo tiene una antigüedad media inferior a la media de la muestra.",
                "Qué revisar desde RRHH": "Onboarding extendido, seguimiento de expectativas, carrera temprana y vinculación con el proyecto."
            })

    # Evaluación global
    if "evaluacion_global" in df_segmento.columns:
        evaluacion_mas_frecuente = df_segmento["evaluacion_global"].astype(str).mode()

        if len(evaluacion_mas_frecuente) > 0:
            puntos.append({
                "Variable": "Evaluación global",
                "Lectura": f"La evaluación más frecuente en el colectivo es: {evaluacion_mas_frecuente.iloc[0]}.",
                "Qué revisar desde RRHH": "Usar la evaluación como información complementaria, no como explicación única del riesgo."
            })

    if not puntos:
        puntos.append({
            "Variable": "Contexto organizativo",
            "Lectura": "No aparece un único punto de dolor claramente dominante con las variables disponibles.",
            "Qué revisar desde RRHH": "Complementar el análisis con información cualitativa del área, responsables y entrevistas internas."
        })

    return pd.DataFrame(puntos)


def simular_escenarios(df_segmento):
    escenarios = []

    # Situación actual
    actual = calcular_riesgo_dataset(df_segmento)
    riesgo_actual = actual["probabilidad_%"].mean()
    prioridad_actual = clasificar_prioridad(riesgo_actual)

    escenarios.append({
        "Escenario": "Situación actual",
        "Prioridad estimada": prioridad_actual,
        "Riesgo medio": riesgo_actual,
        "Lectura para RRHH": "Punto de partida"
    })

    # Movilidad funcional
    sim_mov = df_segmento.copy()
    sim_mov["movilidad_funcional_sn"] = "Sí"
    sim_mov = calcular_riesgo_dataset(sim_mov)
    riesgo_mov = sim_mov["probabilidad_%"].mean()

    escenarios.append({
        "Escenario": "Ofrecer movilidad interna",
        "Prioridad estimada": clasificar_prioridad(riesgo_mov),
        "Riesgo medio": riesgo_mov,
        "Lectura para RRHH": "Revisar movilidad, cambio de proyecto o desarrollo interno"
    })

    # Revisión salarial
    sim_sal = df_segmento.copy()
    sim_sal["posicionamiento_salarial"] = mejor_posicionamiento_salarial()
    sim_sal = calcular_riesgo_dataset(sim_sal)
    riesgo_sal = sim_sal["probabilidad_%"].mean()

    escenarios.append({
        "Escenario": "Revisar posicionamiento salarial",
        "Prioridad estimada": clasificar_prioridad(riesgo_sal),
        "Riesgo medio": riesgo_sal,
        "Lectura para RRHH": "Revisar equidad interna y competitividad salarial"
    })

    # Combinación
    sim_comb = df_segmento.copy()
    sim_comb["movilidad_funcional_sn"] = "Sí"
    sim_comb["posicionamiento_salarial"] = mejor_posicionamiento_salarial()
    sim_comb = calcular_riesgo_dataset(sim_comb)
    riesgo_comb = sim_comb["probabilidad_%"].mean()

    escenarios.append({
        "Escenario": "Movilidad + revisión salarial",
        "Prioridad estimada": clasificar_prioridad(riesgo_comb),
        "Riesgo medio": riesgo_comb,
        "Lectura para RRHH": "Combinar desarrollo interno y revisión salarial"
    })

    escenarios_df = pd.DataFrame(escenarios)
    escenarios_df["Cambio respecto a situación actual"] = (
        escenarios_df["Riesgo medio"] - riesgo_actual
    )

    return escenarios_df


# ============================================================
# DATOS PREDICHOS
# ============================================================

df_pred = calcular_riesgo_dataset(df)


# ============================================================
# CABECERA
# ============================================================

st.title("App de apoyo a RRHH: rotación voluntaria")

st.write(
    "Esta herramienta ayuda a identificar colectivos con mayor prioridad de revisión, "
    "entender posibles puntos de dolor y simular acciones de RRHH."
)

st.warning(
    "Uso responsable: el modelo no predice con certeza quién se irá. "
    "Debe utilizarse como apoyo para priorizar análisis, no para tomar decisiones automáticas."
)


# ============================================================
# PESTAÑAS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Mapa de prioridades",
    "2. Explicación del riesgo",
    "3. Simulador de acciones",
    "4. Consulta individual"
])


# ============================================================
# TAB 1 - MAPA DE PRIORIDADES
# ============================================================

with tab1:
    st.subheader("Mapa de prioridades")

    st.write(
        "Esta pantalla muestra dónde debería mirar primero RRHH. "
        "La prioridad se presenta con una escala sencilla de colores."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Empleados analizados", len(df_pred))
    col2.metric("Riesgo medio estimado", f"{df_pred['probabilidad_%'].mean():.1f} %")
    col3.metric(
        "Perfiles a revisar",
        f"{df_pred['prioridad'].astype(str).str.contains('🔴|🟡').mean() * 100:.1f} %"
    )

    st.markdown(
        """
        **Leyenda**
        - 🟢 **Seguimiento ordinario**: no se observa señal agregada alta.
        - 🟡 **Revisar contexto**: conviene observar el colectivo y contrastar información.
        - 🔴 **Revisar primero**: colectivo prioritario para análisis de RRHH.
        """
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
        "¿Qué quieres revisar?",
        opciones_agrupacion,
        format_func=lambda x: nombres.get(x, x)
    )

    resumen = resumen_por_colectivo(df_pred, variable)

    top = resumen.iloc[0]

    st.markdown("### Colectivo prioritario")

    c1, c2, c3 = st.columns(3)
    c1.metric("Colectivo", str(top[variable]))
    c2.metric("Prioridad", top["prioridad"])
    c3.metric("Personas en el colectivo", int(top["empleados"]))

    st.info(top["lectura_RRHH"])

    fig = px.bar(
        resumen.sort_values("riesgo_medio"),
        x="riesgo_medio",
        y=variable,
        orientation="h",
        color="prioridad",
        color_discrete_map={
            "🟢 Seguimiento ordinario": "#4CAF50",
            "🟡 Revisar contexto": "#F2B705",
            "🔴 Revisar primero": "#D9534F"
        },
        text="prioridad",
        title=f"Prioridad por {nombres.get(variable, variable)}"
    )

    fig.update_layout(
        xaxis_title="Riesgo medio estimado (%)",
        yaxis_title="",
        height=520,
        legend_title="Prioridad"
    )

    st.plotly_chart(fig, use_container_width=True)

    tabla = resumen[[variable, "empleados", "prioridad", "lectura_RRHH"]].copy()
    tabla = tabla.rename(columns={
        variable: "Colectivo",
        "empleados": "Nº empleados",
        "prioridad": "Prioridad",
        "lectura_RRHH": "Lectura para RRHH"
    })

    st.dataframe(tabla, use_container_width=True)


# ============================================================
# TAB 2 - EXPLICACIÓN DEL RIESGO
# ============================================================

with tab2:
    st.subheader("Explicación del riesgo")

    st.write(
        "Selecciona un colectivo para ver qué variables pueden estar detrás de la prioridad detectada."
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

    col_a, col_b = st.columns(2)

    with col_a:
        grupo = st.selectbox(
            "Variable de análisis",
            opciones_agrupacion,
            format_func=lambda x: nombres.get(x, x),
            key="grupo_explicacion"
        )

    with col_b:
        segmento = st.selectbox(
            "Colectivo",
            sorted(df_pred[grupo].dropna().astype(str).unique().tolist()),
            key="segmento_explicacion"
        )

    df_segmento = df_pred[df_pred[grupo].astype(str) == segmento].copy()

    riesgo_segmento = df_segmento["probabilidad_%"].mean()
    prioridad_segmento = clasificar_prioridad(riesgo_segmento)

    st.markdown("### Lectura rápida")

    col1, col2, col3 = st.columns(3)

    col1.metric("Colectivo seleccionado", segmento)
    col2.metric("Prioridad", prioridad_segmento)
    col3.metric("Nº empleados", len(df_segmento))

    st.info(lectura_prioridad(prioridad_segmento))

    st.markdown("### Posibles puntos de dolor")

    puntos_dolor = explicar_puntos_dolor(df_segmento, df_pred)

    st.dataframe(puntos_dolor, use_container_width=True)


# ============================================================
# TAB 3 - SIMULADOR DE ACCIONES
# ============================================================

with tab3:
    st.subheader("Simulador de acciones")

    st.write(
        "El simulador muestra cómo cambiaría la prioridad estimada si se modificaran algunas variables. "
        "No demuestra causalidad; ayuda a explorar posibles líneas de actuación."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        grupo_sim = st.selectbox(
            "Variable de análisis",
            opciones_agrupacion,
            format_func=lambda x: nombres.get(x, x),
            key="grupo_simulador"
        )

    with col_b:
        segmento_sim = st.selectbox(
            "Colectivo",
            sorted(df_pred[grupo_sim].dropna().astype(str).unique().tolist()),
            key="segmento_simulador"
        )

    df_segmento_sim = df[df[grupo_sim].astype(str) == segmento_sim].copy()

    escenarios_df = simular_escenarios(df_segmento_sim)

    actual = escenarios_df[escenarios_df["Escenario"] == "Situación actual"].iloc[0]
    mejor = escenarios_df.sort_values("Riesgo medio").iloc[0]

    st.markdown("### Resultado para RRHH")

    col1, col2 = st.columns(2)

    col1.metric("Situación actual", actual["Prioridad estimada"])
    col2.metric("Mejor escenario estimado", mejor["Prioridad estimada"])

    if mejor["Escenario"] == "Situación actual":
        st.info(
            "Según el modelo, ninguna acción simulada reduce claramente la prioridad actual. "
            "Conviene revisar el caso con información cualitativa."
        )
    else:
        st.success(
            f"La acción más favorable sería: **{mejor['Escenario']}**."
        )

        st.write(
            f"Lectura para RRHH: {mejor['Lectura para RRHH']}."
        )

    st.markdown("### Comparación sencilla de escenarios")

    fig = px.bar(
        escenarios_df.sort_values("Riesgo medio"),
        x="Riesgo medio",
        y="Escenario",
        orientation="h",
        color="Prioridad estimada",
        color_discrete_map={
            "🟢 Seguimiento ordinario": "#4CAF50",
            "🟡 Revisar contexto": "#F2B705",
            "🔴 Revisar primero": "#D9534F"
        },
        text="Prioridad estimada",
        title="Cómo cambiaría la prioridad según la acción simulada"
    )

    fig.update_layout(
        xaxis_title="Riesgo medio estimado (%)",
        yaxis_title="",
        height=420,
        legend_title="Prioridad"
    )

    st.plotly_chart(fig, use_container_width=True)

    tabla_escenarios = escenarios_df[[
        "Escenario",
        "Prioridad estimada",
        "Lectura para RRHH"
    ]].copy()

    st.dataframe(tabla_escenarios, use_container_width=True)

    st.warning(
        "Nota: la simulación no prueba que una acción reduzca realmente la rotación. "
        "Solo muestra cómo cambiaría la estimación del modelo."
    )


# ============================================================
# TAB 4 - CONSULTA INDIVIDUAL
# ============================================================

with tab4:
    st.subheader("Consulta individual orientativa")

    st.write(
        "Esta consulta permite probar un perfil concreto. Debe usarse como apoyo, no como etiqueta individual."
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

    prob = float(predecir_probabilidad(caso)[0]) * 100
    prioridad = clasificar_prioridad(prob)

    with col2:
        st.metric("Prioridad estimada", prioridad)

        if "🔴" in prioridad:
            st.warning("Revisar primero: conviene analizar contexto, movilidad y salario.")
        elif "🟡" in prioridad:
            st.info("Revisar contexto: conviene realizar seguimiento.")
        else:
            st.success("Seguimiento ordinario.")

    st.dataframe(caso, use_container_width=True)
