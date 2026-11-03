"""OBSERVATORIO ECONÓMICO Y SOCIAL | Nariño y Colombia.
Ajustado para tipografía en negro de alto contraste en toda la interfaz.
"""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from contextlib import suppress

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.api import VAR, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. FUENTES DE DATOS Y NORMALIZACIÓN DIVIPOLA
# ============================================================
SOURCES = {
    'economia': {
        'nombre': 'Actividad económica por actividades (precios constantes 2015)',
        'organismo': 'Gobernación de Caldas · portal Datos Abiertos Colombia',
        'id': 'kgyi-qc7j',
        'url': 'https://www.datos.gov.co/Econom-a-y-Finanzas/PIB-Departamental-con-proyecci-n/kgyi-qc7j',
        'frecuencia': 'Anual',
        'advertencia': 'Suma de actividades en la fuente publicada; puede no equivaler al PIB oficial DANE.'
    },
    'homicidios': {
        'nombre': 'Homicidio · víctimas',
        'organismo': 'Ministerio de Defensa Nacional',
        'id': 'm8fd-ahd9',
        'url': 'https://www.datos.gov.co/Seguridad-y-Defensa/HOMICIDIO/m8fd-ahd9',
        'frecuencia': 'Registro de hecho; agregado anual',
        'advertencia': 'Conteo absoluto; no es tasa por 100.000 habitantes.'
    },
    'transito': {
        'nombre': 'Homicidio en accidente de tránsito · víctimas',
        'organismo': 'Ministerio de Defensa Nacional',
        'id': 'uav5-b85g',
        'url': 'https://www.datos.gov.co/Seguridad-y-Defensa/HOMICIDIO-EN-ACCIDENTE-DE-TR-NSITO/uav5-b85g',
        'frecuencia': 'Registro de hecho; agregado anual',
        'advertencia': 'Conteo de muertes en siniestros viales, NO homicidio intencional.'
    },
}
GEO_SOURCE = 'https://sig.sispro.gov.co/arcgis_msp/rest/services/AGOL/DOTACION/MapServer/0/query'
GEO_INFO = '
