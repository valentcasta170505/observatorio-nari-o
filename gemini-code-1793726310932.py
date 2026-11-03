"""OBSERVATORIO ECONÓMICO Y SOCIAL | Nariño y Colombia.
Edición ultra-optimizada para bajo consumo de memoria RAM en Streamlit Cloud.
"""
from __future__ import annotations
import hashlib
import json
import re
import unicodedata
import gc
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# 1. FUENTES DE DATOS Y DIVIPOLA
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

GEO_SOURCES = [
    'https://sig.sispro.gov.co/arcgis_msp/rest/services/AGOL/DOTACION/MapServer/0/query?where=1%3D1&outFields=CODDPTO,NOMDPTO&returnGeometry=true&f=geojson&outSR=4326',
    'https://raw.githubusercontent.com/mledoze/countries/master/data/col.geo.json',
    'https://raw.githubusercontent.com/marcospassos/colombia-geojson/master/colombia.geojson'
]

DEP_NAMES = [
    'Amazonas','Antioquia','Arauca','Atlántico','Bogotá D.C.','Bolívar','Boyacá',
    'Caldas','Caquetá','Casanare','Cauca','Cesar','Chocó','Córdoba',
    'Cundinamarca','Guainía','Guaviare','Huila','La Guajira','Magdalena',
    'Meta','Nariño','Norte de Santander','Putumayo','Quindío','Risaralda',
    'San Andrés y Providencia','Santander','Sucre','Tolima','Valle del Cauca',
    'Vaupés','Vichada'
]
DEP_CODES = ['91','05','81','08','11','13','15','17','18','85','19','20',
             '27','23','25','94','95','41','44','47','50','52','54','86',
             '63','66','88','68','70','73','76','97','99']
CODE_TO_NAME = dict(zip(DEP_CODES, DEP_NAMES))
NACIONAL = 'Todos los departamentos (33 territorios)'
TOTAL_TERRITORIOS = len(DEP_CODES)

def _text(x):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(x or '').lower())
                   if not unicodedata.combining(c)).replace('.', '').replace(',', '').strip()

NAME_TO_CODE = {_text(v): k for k,v in CODE_TO_NAME.items()}
NAME_TO_CODE.update({'bogota dc':'11','bogota d c':'11','distrito capital':'11',
                     'san andres':'88','san andres providencia y santa catalina':'88',
                     'archipielago de san andres providencia y santa catalina':'88',
                     'valle':'76'})

def normalizar_codigo(x):
    if pd.isna(x): return None
    s = re.sub(r'\D', '', str(x).split('.')[0])
    if not s: return None
    if len(s) == 5: s = s[:2]
    if len(s) > 2: return None
    return s.zfill(2) if s.zfill(2) in CODE_TO_NAME else None

def nombre_a_codigo(nombre):
    return NAME_TO_CODE.get(_text(nombre))

def session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.6, status_forcelist=[429,500,502,503,504], allowed_methods=['GET'])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers.update({'User-Agent':'ObservatorioEconomicoSocial/1.0 (academic use)','Accept':'application/json'})
    return s

def socrata_pages(dataset_id, fields, page_size=20000, max_pages=50):
    parts=[]
    with session() as s:
        for page in range(max_pages):
            params={'$select':','.join(fields), '$order':':id ASC', '$limit':page_size, '$offset':page*page_size}
            response=s.get(f'https://www.datos.gov.co/resource/{dataset_id}.json', params=params, timeout=(8,50))
            response.raise_for_status()
            values=response.json()
            if not isinstance(values,list): raise ValueError('Respuesta no válida.')
            if not values: break
            parts.append(pd.DataFrame(values))
            if len(values) < page_size: break
    if not parts: raise ValueError('Dataset vacío.')
    df_res = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return df_res

def _normalize_base(raw, code_field, date_field, value_field, indicator, source, unit):
    v=raw.copy()
    v['codigo_departamento']=v[code_field].map(normalizar_codigo)
    if date_field=='fecha_hecho':
        v['anio']=pd.to_datetime(v[date_field],errors='coerce').dt.year
    else:
        v['anio']=pd.to_numeric(v[date_field], errors='coerce')
    v['valor']=pd.to_numeric(v[value_field],errors='coerce')
    v=v.dropna(subset=['codigo_departamento','anio','valor'])
    v=v[(v['anio']>=1990)&(v['anio']<datetime.now(timezone.utc).year)]
    v['anio']=v['anio'].astype('int16')
    v['valor']=v['valor'].astype('float32')
    v=