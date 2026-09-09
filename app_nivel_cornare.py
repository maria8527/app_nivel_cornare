"""
Portafolio 1 — Análisis de variables ambientales, Proyecto MARCO
==================================================================
Estación 49 — San Rafael, Río Guatapé, Vereda El Bizcocho (Hidrológica)

Requisitos (Python 3.11):
    pip install pandas numpy requests urllib3 matplotlib scikit-learn

Uso:
    python portafolio1_analisis_ambiental.py

Los gráficos se guardan como archivos .png en la carpeta ./salidas
(no se muestran en pantalla, para poder correrlo también en servidores
sin interfaz gráfica). Cambia MOSTRAR_GRAFICOS a True si estás en un
entorno local con ventana y quieres verlos con plt.show().
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla; cambia si corres localmente con GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import urllib3
from sklearn.preprocessing import MinMaxScaler, StandardScaler

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# 1. Parámetros de tu consulta (edítalos)
# ------------------------------------------------------------------
NOMBRE_ESTUDIANTE = "Escribe tu nombre aquí"
CODIGO_ESTACION = "49"
NOMBRE_ESTACION = "San Rafael, Río Guatapé, Vereda El Bizcocho"
TIPO_ESTACION = "Hidrológica"
LAT_ESTACION = 6.2942
LON_ESTACION = -75.06251

FECHA_DESDE = "2026-08-23"
FECHA_HASTA = "2026-09-09"
CALIDAD = 1  # 1 = solo datos validados

API_BASE_URL = "https://marco.cornare.gov.co/api/v1/estaciones"
LLAVE_FECHA = "level_date"
LLAVE_VALOR = "level"

# Umbrales de nivel (ajústalos según el boletín oficial de tu estación)
NIVEL_ALERTA = 350.0
NIVEL_ALARMA = 450.0

# Split cronológico
PROP_TRAIN = 0.70
PROP_VAL = 0.15  # el resto (0.15) queda para test

# Límites físicos para outliers
LIMITE_FISICO_MIN = 0.0
LIMITE_FISICO_MAX = 1000.0

CARPETA_SALIDAS = "salidas"
MOSTRAR_GRAFICOS = False  # ponlo en True si corres localmente y quieres plt.show()


@dataclass
class ResultadoPortafolio:
    """Contenedor con los DataFrames intermedios, útil si luego quieres
    importar este script y reutilizar los resultados sin recalcular todo."""

    df: pd.DataFrame
    df_regular: pd.DataFrame
    serie_limpia: pd.Series
    df_escalado: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


# ------------------------------------------------------------------
# 2. Consultar la API real y traer todas las páginas de resultados
# ------------------------------------------------------------------
def obtener_serie_nivel(codigo_estacion: str, desde: str, hasta: str, calidad: int = 1, timeout: int = 30) -> dict:
    url = f"{API_BASE_URL}/{codigo_estacion}/nivel"
    params = {"desde": desde, "hasta": hasta, "calidad": calidad}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=timeout, verify=False)
    resp.raise_for_status()
    return resp.json()


def obtener_todas_las_paginas(datos_json: dict, timeout: int = 30) -> list[dict]:
    registros = list(datos_json.get("values", []))
    siguiente_url = datos_json.get("next")
    while siguiente_url:
        resp = requests.get(siguiente_url, timeout=timeout, verify=False)
        resp.raise_for_status()
        pagina = resp.json()
        registros.extend(pagina.get("values", []))
        siguiente_url = pagina.get("next")
    return registros


# ------------------------------------------------------------------
# 3-4. Construir el DataFrame, tipos de datos y orden temporal
# ------------------------------------------------------------------
def construir_dataframe(registros: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(registros)
    df = df.rename(columns={LLAVE_FECHA: "fecha", LLAVE_VALOR: "nivel"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce")
    df = df.dropna(subset=["fecha", "nivel"]).sort_values("fecha").reset_index(drop=True)
    return df


def revisar_tipos_y_orden(df: pd.DataFrame) -> pd.Timedelta:
    print("\n--- 4. Tipos de datos y orden temporal ---")
    print(df.dtypes)
    print("¿Ordenado ascendente?:", df["fecha"].is_monotonic_increasing)
    print("Fechas duplicadas:", df["fecha"].duplicated().sum())
    print("Rango temporal:", df["fecha"].min(), "→", df["fecha"].max())

    frecuencia_tipica = df["fecha"].diff().dropna().mode()[0]
    print("Frecuencia típica de reporte:", frecuencia_tipica)
    return frecuencia_tipica


# ------------------------------------------------------------------
# 5. Missing values reales (huecos en la serie)
# ------------------------------------------------------------------
def detectar_huecos_y_reindexar(df: pd.DataFrame, frecuencia_tipica: pd.Timedelta) -> pd.DataFrame:
    print("\n--- 5. Missing values reales ---")
    df_idx = df.set_index("fecha")
    rango_completo = pd.date_range(start=df_idx.index.min(), end=df_idx.index.max(), freq=frecuencia_tipica)
    df_regular = df_idx.reindex(rango_completo)
    df_regular.index.name = "fecha"

    n_esperados = len(rango_completo)
    n_observados = int(df_idx["nivel"].notna().sum())
    n_huecos = int(df_regular["nivel"].isna().sum())

    print(f"Instantes esperados: {n_esperados}")
    print(f"Instantes observados: {n_observados}")
    print(f"Huecos reales detectados: {n_huecos} ({n_huecos / n_esperados:.1%} de la serie)")

    df_regular["nivel_interpolado"] = df_regular["nivel"].interpolate(method="time")

    fig, ax = plt.subplots(figsize=(12, 4))
    df_regular["nivel_interpolado"].plot(ax=ax, label="Interpolado", alpha=0.7)
    df_regular["nivel"].plot(ax=ax, label="Original (con huecos)", marker="o", linestyle="None", markersize=3)
    ax.set_title("Nivel original vs. interpolado")
    ax.set_ylabel("Nivel (cm)")
    ax.legend()
    _guardar_o_mostrar(fig, "05_missing_values.png")

    return df_regular


# ------------------------------------------------------------------
# 6. Outliers con IQR + límites físicos
# ------------------------------------------------------------------
def detectar_outliers(df_regular: pd.DataFrame) -> pd.Series:
    print("\n--- 6. Outliers (IQR + límites físicos) ---")
    serie = df_regular["nivel_interpolado"].dropna()

    Q1, Q3 = serie.quantile(0.25), serie.quantile(0.75)
    IQR = Q3 - Q1
    lim_inf_iqr, lim_sup_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    es_outlier_iqr = (serie < lim_inf_iqr) | (serie > lim_sup_iqr)
    es_outlier_fisico = (serie < LIMITE_FISICO_MIN) | (serie > LIMITE_FISICO_MAX)
    es_outlier = es_outlier_iqr | es_outlier_fisico

    print(f"Límites IQR: [{lim_inf_iqr:.1f}, {lim_sup_iqr:.1f}] cm")
    print(f"Outliers por IQR: {int(es_outlier_iqr.sum())}")
    print(f"Outliers por límite físico: {int(es_outlier_fisico.sum())}")
    print(f"Total outliers: {int(es_outlier.sum())} de {len(serie)} lecturas")

    fig, ax = plt.subplots(figsize=(12, 4))
    serie.plot(ax=ax, label="Nivel", alpha=0.7)
    serie[es_outlier].plot(ax=ax, label="Outlier", marker="x", linestyle="None", color="red")
    ax.axhline(lim_sup_iqr, color="orange", linestyle="--", linewidth=1, label="Límite IQR sup.")
    ax.axhline(lim_inf_iqr, color="orange", linestyle="--", linewidth=1, label="Límite IQR inf.")
    ax.set_title("Detección de outliers (IQR + límites físicos)")
    ax.set_ylabel("Nivel (cm)")
    ax.legend()
    _guardar_o_mostrar(fig, "06_outliers.png")

    serie_limpia = serie[~es_outlier].copy()
    return serie_limpia


# ------------------------------------------------------------------
# 7. Normalización y estandarización
# ------------------------------------------------------------------
def normalizar_y_estandarizar(serie_limpia: pd.Series) -> pd.DataFrame:
    print("\n--- 7. Normalización y estandarización ---")
    valores = serie_limpia.values.reshape(-1, 1)

    minmax_scaler = MinMaxScaler()
    valores_normalizados = minmax_scaler.fit_transform(valores).flatten()

    std_scaler = StandardScaler()
    valores_estandarizados = std_scaler.fit_transform(valores).flatten()

    df_escalado = pd.DataFrame(
        {
            "fecha": serie_limpia.index,
            "nivel": serie_limpia.values,
            "nivel_normalizado": valores_normalizados,
            "nivel_estandarizado": valores_estandarizados,
        }
    ).set_index("fecha")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    df_escalado["nivel"].plot(ax=axes[0], title="Original (cm)")
    df_escalado["nivel_normalizado"].plot(ax=axes[1], title="Normalizado (Min-Max, 0–1)", color="green")
    df_escalado["nivel_estandarizado"].plot(ax=axes[2], title="Estandarizado (Z-score)", color="purple")
    _guardar_o_mostrar(fig, "07_normalizacion.png")

    print(df_escalado.describe())
    return df_escalado


# ------------------------------------------------------------------
# 8. Train / validation / test — split cronológico
# ------------------------------------------------------------------
def split_cronologico(df_escalado: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n--- 8. Split cronológico train/val/test ---")
    n = len(df_escalado)
    corte_train = int(n * PROP_TRAIN)
    corte_val = int(n * (PROP_TRAIN + PROP_VAL))

    train = df_escalado.iloc[:corte_train]
    val = df_escalado.iloc[corte_train:corte_val]
    test = df_escalado.iloc[corte_val:]

    print(f"Train: {len(train)} lecturas  ({train.index.min()} → {train.index.max()})")
    print(f"Val:   {len(val)} lecturas  ({val.index.min()} → {val.index.max()})")
    print(f"Test:  {len(test)} lecturas  ({test.index.min()} → {test.index.max()})")

    fig, ax = plt.subplots(figsize=(12, 4))
    train["nivel"].plot(ax=ax, label="Train")
    val["nivel"].plot(ax=ax, label="Validation")
    test["nivel"].plot(ax=ax, label="Test")
    ax.set_title("Split cronológico train / validation / test")
    ax.set_ylabel("Nivel (cm)")
    ax.legend()
    _guardar_o_mostrar(fig, "08_split.png")

    return train, val, test


# ------------------------------------------------------------------
# 9. Estadística descriptiva
# ------------------------------------------------------------------
def estadistica_descriptiva(
    serie_limpia: pd.Series, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    print("\n--- 9. Estadística descriptiva ---")
    print(serie_limpia.describe())
    print(f"Asimetría (skewness): {serie_limpia.skew():.3f}")
    print(f"Curtosis: {serie_limpia.kurt():.3f}")

    resumen = pd.DataFrame(
        {
            "train": train["nivel"].describe(),
            "val": val["nivel"].describe(),
            "test": test["nivel"].describe(),
        }
    )
    print("\nComparación por subconjunto:")
    print(resumen)

    fig, ax = plt.subplots(figsize=(8, 4))
    serie_limpia.plot(kind="hist", bins=30, ax=ax, edgecolor="black", alpha=0.75)
    ax.set_title(f"Distribución del nivel — Estación {CODIGO_ESTACION} ({NOMBRE_ESTACION})")
    ax.set_xlabel("Nivel (cm)")
    _guardar_o_mostrar(fig, "09_estadistica_descriptiva.png")


# ------------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------------
def _guardar_o_mostrar(fig: plt.Figure, nombre_archivo: str) -> None:
    os.makedirs(CARPETA_SALIDAS, exist_ok=True)
    ruta = os.path.join(CARPETA_SALIDAS, nombre_archivo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    print(f"[gráfico guardado] {ruta}")
    if MOSTRAR_GRAFICOS:
        plt.show()
    plt.close(fig)


# ------------------------------------------------------------------
# Orquestación principal
# ------------------------------------------------------------------
def ejecutar_portafolio() -> ResultadoPortafolio:
    print(f"Estudiante: {NOMBRE_ESTUDIANTE}")
    print(f"Estación {CODIGO_ESTACION} — {NOMBRE_ESTACION} ({TIPO_ESTACION})")
    print(f"Rango: {FECHA_DESDE} a {FECHA_HASTA} | Calidad={CALIDAD}")

    print("\n--- 2. Consultando la API de MARCO/CORNARE ---")
    datos_crudos = obtener_serie_nivel(CODIGO_ESTACION, FECHA_DESDE, FECHA_HASTA, CALIDAD)
    registros = obtener_todas_las_paginas(datos_crudos)
    print(f"Registros descargados: {len(registros)}")

    print("\n--- 3. Construyendo el DataFrame ---")
    df = construir_dataframe(registros)
    print(df.head())

    frecuencia_tipica = revisar_tipos_y_orden(df)
    df_regular = detectar_huecos_y_reindexar(df, frecuencia_tipica)
    serie_limpia = detectar_outliers(df_regular)
    df_escalado = normalizar_y_estandarizar(serie_limpia)
    train, val, test = split_cronologico(df_escalado)
    estadistica_descriptiva(serie_limpia, train, val, test)

    print("\nListo. Revisa la carpeta 'salidas/' para ver todos los gráficos generados.")

    return ResultadoPortafolio(
        df=df,
        df_regular=df_regular,
        serie_limpia=serie_limpia,
        df_escalado=df_escalado,
        train=train,
        val=val,
        test=test,
    )


if __name__ == "__main__":
    ejecutar_portafolio()
