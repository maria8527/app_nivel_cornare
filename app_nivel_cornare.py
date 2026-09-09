"""
App de Streamlit — Nivel de río, Estación 49 (CORNARE / MARCO)
------------------------------------------------------------------
Estación: 49 — Hidrológica
Ubicación: San Rafael, Río Guatapé, Vereda El Bizcocho
Coordenadas: 6.2942, -75.06251

Para correrla:
    streamlit run app_nivel_cornare.py
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import urllib3
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ------------------------------------------------------------------
# Configuración fija de MI estación (a diferencia de la app base,
# aquí no se pide el código por texto libre: la app está hecha
# a la medida de la estación 49).
# ------------------------------------------------------------------
CODIGO_ESTACION = "49"
NOMBRE_ESTACION = "San Rafael, Río Guatapé, Vereda El Bizcocho"
TIPO_ESTACION = "Hidrológica"
LAT_ESTACION = 6.2942
LON_ESTACION = -75.06251

API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"
LLAVE_FECHA = "level_date"
LLAVE_VALOR = "level"

st.set_page_config(
    page_title=f"Nivel — Estación {CODIGO_ESTACION} (San Rafael)",
    page_icon="🌊",
    layout="wide",
)


# ------------------------------------------------------------------
# Funciones de consulta (con cache para no golpear la API de más)
# ------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def obtener_serie_nivel(codigo_estacion, desde, hasta, calidad=1, timeout=30):
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"
    params = {"desde": desde, "hasta": hasta, "calidad": calidad}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"HTTP {resp.status_code}"
    except requests.exceptions.RequestException as e:
        return None, f"Error de red: {e}"


@st.cache_data(ttl=300, show_spinner=False)
def obtener_todas_las_paginas(datos_json, timeout=30):
    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")
    while siguiente_url:
        try:
            resp = requests.get(siguiente_url, timeout=timeout, verify=False)
        except requests.exceptions.RequestException:
            break
        if resp.status_code != 200:
            break
        pagina = resp.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")
    return registros


def calcular_indice_calidad(df):
    """Índice simple (0-100) combinando completitud de la serie y proporción de outliers."""
    if df.empty or len(df) < 2:
        return 0.0, 0, 0

    df_idx = df.set_index("fecha")
    frecuencia_tipica = df["fecha"].diff().dropna().mode()
    if len(frecuencia_tipica) == 0:
        return 0.0, 0, 0
    frecuencia_tipica = frecuencia_tipica[0]

    rango_completo = pd.date_range(start=df_idx.index.min(), end=df_idx.index.max(), freq=frecuencia_tipica)
    esperados = len(rango_completo)
    huecos = esperados - len(df_idx)
    completitud = max(0.0, 1 - (huecos / esperados)) if esperados > 0 else 0.0

    Q1, Q3 = df["nivel"].quantile(0.25), df["nivel"].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf, lim_sup = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    es_outlier = (df["nivel"] < lim_inf) | (df["nivel"] > lim_sup) | (df["nivel"] < 0)
    proporcion_outliers = es_outlier.mean()

    indice = (completitud * 0.7 + (1 - proporcion_outliers) * 0.3) * 100
    return round(indice, 1), int(huecos), int(es_outlier.sum())


def clasificar_estado(nivel_actual, nivel_alerta, nivel_alarma):
    """Clasifica el nivel actual contra dos umbrales configurables por el usuario."""
    if nivel_actual is None or np.isnan(nivel_actual):
        return "Sin dato", "gray"
    if nivel_actual >= nivel_alarma:
        return "Alarma", "red"
    if nivel_actual >= nivel_alerta:
        return "Alerta", "orange"
    return "Seguro", "green"


# ------------------------------------------------------------------
# Sidebar — parámetros de la consulta y de la app
# ------------------------------------------------------------------
st.sidebar.header(f"Estación {CODIGO_ESTACION} — {NOMBRE_ESTACION}")
nombre_estudiante = st.sidebar.text_input("Nombre del estudiante", "")

st.sidebar.subheader("Rango de fechas")
fecha_desde = st.sidebar.date_input("Desde", pd.to_datetime("2026-08-23")).strftime("%Y-%m-%d")
fecha_hasta = st.sidebar.date_input("Hasta", pd.to_datetime("2026-09-09")).strftime("%Y-%m-%d")
calidad = st.sidebar.selectbox("Calidad", [1, 0], index=0, help="1 = solo datos validados")

st.sidebar.subheader("Umbrales de nivel (editables)")
st.sidebar.caption("Ajusta estos valores según los boletines oficiales de CORNARE para tu estación.")
nivel_alerta = st.sidebar.number_input("Umbral de Alerta (cm)", value=350.0, step=5.0)
nivel_alarma = st.sidebar.number_input("Umbral de Alarma (cm)", value=450.0, step=5.0)

ventana_promedio = st.sidebar.slider("Ventana de promedio móvil (horas)", 1, 48, 6)

consultar = st.sidebar.button("🔍 Consultar", type="primary")

# ------------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------------
st.title("🌊 Nivel del Río Guatapé — Estación 49")
col_a, col_b, col_c = st.columns(3)
col_a.markdown(f"**Estación:** {TIPO_ESTACION}")
col_b.markdown(f"**Ubicación:** {NOMBRE_ESTACION}")
col_c.markdown(f"**Código:** {CODIGO_ESTACION}")
if nombre_estudiante:
    st.caption(f"Consulta preparada por: **{nombre_estudiante}**")

# ------------------------------------------------------------------
# Consulta y procesamiento
# ------------------------------------------------------------------
if consultar:
    with st.spinner("Consultando la API de MARCO / CORNARE..."):
        datos_crudos, error = obtener_serie_nivel(CODIGO_ESTACION, fecha_desde, fecha_hasta, calidad)

    if error:
        st.error(f"❌ {error}")
    else:
        registros = obtener_todas_las_paginas(datos_crudos)

        if not registros:
            st.warning("No hay registros para este rango de fechas. Prueba otro rango.")
        else:
            df = pd.DataFrame(registros)
            df = df.rename(columns={LLAVE_FECHA: "fecha", LLAVE_VALOR: "nivel"})
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce")
            df = df.dropna(subset=["fecha", "nivel"]).sort_values("fecha").reset_index(drop=True)

            df["promedio_movil"] = df["nivel"].rolling(ventana_promedio, min_periods=1).mean()

            indice_calidad, huecos, n_outliers = calcular_indice_calidad(df)

            nivel_actual = df["nivel"].iloc[-1] if not df.empty else np.nan
            fecha_ultimo_registro = df["fecha"].iloc[-1] if not df.empty else None
            estado, color_estado = clasificar_estado(nivel_actual, nivel_alerta, nivel_alarma)

            nivel_anterior = df["nivel"].iloc[-2] if len(df) > 1 else nivel_actual
            delta = nivel_actual - nivel_anterior if len(df) > 1 else 0.0

            # --- Estado actual destacado ---
            st.subheader("Estado actual del cauce")
            box = st.container(border=True)
            with box:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nivel actual", f"{nivel_actual:.1f} cm", delta=f"{delta:+.1f} cm")
                c2.markdown(
                    f"<div style='padding-top:10px'><b>Estado:</b> "
                    f"<span style='color:{color_estado}; font-weight:bold'>{estado}</span></div>",
                    unsafe_allow_html=True,
                )
                c3.metric("Últ. registro", fecha_ultimo_registro.strftime("%d/%m/%Y %H:%M") if fecha_ultimo_registro is not None else "—")
                c4.metric("Índice de calidad de datos", f"{indice_calidad} / 100")

            # --- Métricas de la serie ---
            st.subheader("Resumen del período consultado")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Lecturas", len(df))
            col2.metric("Nivel promedio", f"{df['nivel'].mean():.1f} cm")
            col3.metric("Nivel mínimo", f"{df['nivel'].min():.1f} cm")
            col4.metric("Nivel máximo", f"{df['nivel'].max():.1f} cm")
            col5.metric("Outliers detectados", n_outliers)

            # --- Gráfico de la serie con promedio móvil ---
            st.subheader("Serie de nivel")
            st.line_chart(df.set_index("fecha")[["nivel", "promedio_movil"]])

            # --- Mapa de la estación ---
            st.subheader("Ubicación de la estación")
            st.map(pd.DataFrame({"lat": [LAT_ESTACION], "lon": [LON_ESTACION]}), zoom=12)
            st.caption(f"Lat/Lon: {LAT_ESTACION}, {LON_ESTACION}")

            # --- Detalle de calidad ---
            with st.expander("Detalle del índice de calidad"):
                st.write(f"- Huecos de reporte detectados: **{huecos}**")
                st.write(f"- Outliers (IQR + nivel negativo): **{n_outliers}** de {len(df)} lecturas")
                st.write("El índice combina completitud de la serie (70%) y proporción de datos sin outliers (30%).")

            # --- Tabla y descarga ---
            with st.expander("Ver datos crudos"):
                st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Descargar CSV",
                csv,
                file_name=f"nivel_estacion_{CODIGO_ESTACION}_{fecha_desde}_a_{fecha_hasta}.csv",
                mime="text/csv",
            )
else:
    st.info("Ajusta el rango de fechas y los umbrales en el sidebar, luego presiona **Consultar**.")
    st.caption(
        "Esta app está configurada específicamente para la Estación 49 "
        "(San Rafael, Río Guatapé, Vereda El Bizcocho)."
    )
