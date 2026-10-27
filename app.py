"""OBSERVATORIO ECONÓMICO Y SOCIAL | Nariño y Colombia.
Despliegue autocontenido: la interfaz y el motor comparten app.py.
No requiere motor.py por separado.
"""
from __future__ import annotations
"""Motor verificable de datos y modelos. Sin generación de observaciones ficticias."""
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

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

SOURCES = {
    'economia': {
        'nombre': 'Actividad económica por actividades (precios constantes 2015)',
        'organismo': 'Gobernación de Caldas · portal Datos Abiertos Colombia',
        'id': 'kgyi-qc7j',
        'url': 'https://www.datos.gov.co/Econom-a-y-Finanzas/PIB-Departamental-con-proyecci-n/kgyi-qc7j',
        'frecuencia': 'Anual',
        'advertencia': 'Suma de actividades en la fuente publicada; puede no equivaler al PIB oficial DANE por valoración, impuestos o metodología de proyección.'
    },
    'homicidios': {
        'nombre': 'Homicidio · víctimas',
        'organismo': 'Ministerio de Defensa Nacional',
        'id': 'm8fd-ahd9',
        'url': 'https://www.datos.gov.co/Seguridad-y-Defensa/HOMICIDIO/m8fd-ahd9',
        'frecuencia': 'Registro de hecho; agregado anual',
        'advertencia': 'Conteo absoluto; no es tasa por 100.000 habitantes. Años incompletos excluidos.'
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
GEO_INFO = 'https://sig.sispro.gov.co/arcgis_msp/rest/services/AGOL/DOTACION/MapServer/0'
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

def _text(x):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(x or '').lower())
                   if not unicodedata.combining(c)).replace('.', '').replace(',', '').strip()
NAME_TO_CODE = {_text(v): k for k,v in CODE_TO_NAME.items()}
NAME_TO_CODE.update({'bogota dc':'11','bogota d c':'11','distrito capital':'11',
                     'san andres':'88','san andres providencia y santa catalina':'88',
                     'archipielago de san andres providencia y santa catalina':'88',
                     'valle':'76'})

def normalizar_codigo(x):
    if pd.isna(x):
        return None
    s = re.sub(r'\D', '', str(x).split('.')[0])
    if not s:
        return None
    if len(s) == 5: s = s[:2]
    if len(s) > 2: return None
    return s.zfill(2) if s.zfill(2) in CODE_TO_NAME else None

def nombre_a_codigo(nombre):
    return NAME_TO_CODE.get(_text(nombre))

def session():
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.6, status_forcelist=[429,500,502,503,504],
                  allowed_methods=['GET'])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers.update({'User-Agent':'ObservatorioEconomicoSocial/1.0 (academic use)',
                      'Accept':'application/json'})
    return s

def socrata_pages(dataset_id, fields, page_size=20000, max_pages=50):
    """Paginación reproducible, ordenada, sin truncar silenciosamente la fuente."""
    parts=[]
    with session() as s:
        for page in range(max_pages):
            params={'$select':','.join(fields), '$order':':id ASC',
                    '$limit':page_size,'$offset':page*page_size}
            response=s.get(f'https://www.datos.gov.co/resource/{dataset_id}.json',
                           params=params,timeout=(8,50))
            response.raise_for_status()
            values=response.json()
            if not isinstance(values,list): raise ValueError('La respuesta no es una lista JSON.')
            if not values: break
            parts.append(pd.DataFrame(values))
            if len(values) < page_size: break
        else:
            raise RuntimeError('La paginación alcanzó su límite de seguridad y se detuvo; no se publicaron datos parciales.')
    if not parts: raise ValueError('El conjunto de datos no devolvió registros.')
    return pd.concat(parts,ignore_index=True)

def _normalize_base(raw, code_field, date_field, value_field, indicator, source, unit):
    for field in (code_field,date_field,value_field):
        if field not in raw: raise ValueError(f'Falta columna esperada {field}; revisar esquema original.')
    v=raw.copy()
    v['codigo_departamento']=v[code_field].map(normalizar_codigo)
    if date_field=='fecha_hecho':
        v['anio']=pd.to_datetime(v[date_field],errors='coerce').dt.year
    else:
        v['anio']=pd.to_numeric(v[date_field], errors='coerce')
    v['valor']=pd.to_numeric(v[value_field],errors='coerce')
    v=v.dropna(subset=['codigo_departamento','anio','valor'])
    # Excluye el año todavía en curso para evitar falsa comparación de períodos parciales.
    v=v[(v['anio']>=1990)&(v['anio']<datetime.now(timezone.utc).year)]
    v['anio']=v['anio'].astype(int)
    v=v.groupby(['codigo_departamento','anio'],as_index=False)['valor'].sum()
    v['departamento']=v['codigo_departamento'].map(CODE_TO_NAME)
    v['indicador']=indicator;v['unidad']=unit;v['fuente']=source
    return v[['codigo_departamento','departamento','anio','indicador','valor','unidad','fuente']]

def load_source(key):
    spec=SOURCES[key]
    timestamp=datetime.now(timezone.utc).isoformat(timespec='seconds')
    status={'id':spec['id'],'organismo':spec['organismo'],'fuente':spec['url'],
            'consulta_utc':timestamp,'indicador':spec['nombre'],'estado':'Error',
            'registros_origen':0,'registros_finales':0,'observaciones':'',
            'advertencia':spec['advertencia']}
    try:
        if key=='economia':
            raw=socrata_pages(spec['id'], ['a_o','actividad','sector','tipo_de_precios',
                                            'c_digo_departamento_divipola','valor_miles_de_millones_de'])
            status['registros_origen']=len(raw)
            cols=raw.columns
            # Evitar sumas múltiples del total si la tabla incluye una fila de agregado.
            const=raw['tipo_de_precios'].fillna('').str.contains('constantes',case=False)
            d=raw.loc[const].copy()
            if d.empty: raise ValueError('No se encontró la categoría de precios constantes.')
            # Los registros se deben identificar por año, departamento, actividad y tipo de precio.
            unique=['a_o','c_digo_departamento_divipola','actividad','tipo_de_precios']
            if d.duplicated(unique).any():
                raise ValueError('Claves actividad/año/departamento repetidas; no es seguro sumar sin investigar la fuente.')
            totals=d['actividad'].fillna('').str.fullmatch(
                r'(?i)\s*(pib|producto interno bruto|total|total actividades|valor agregado total)\s*')
            if totals.any():
                d=d.loc[totals]
                status['observaciones']='Se utilizaron las filas explícitas de total, evitando doble conteo.'
            else:
                status['observaciones']='Suma de actividades económicas; verificar si el agregado equivale al PIB oficial.'
            out=_normalize_base(d,'c_digo_departamento_divipola','a_o',
                                'valor_miles_de_millones_de','Actividad económica (constantes 2015)',
                                spec['url'],'Miles de millones COP, precios 2015')
        else:
            raw=socrata_pages(spec['id'],['fecha_hecho','cod_depto','cantidad'],page_size=25000)
            status['registros_origen']=len(raw)
            out=_normalize_base(raw,'cod_depto','fecha_hecho','cantidad',
                                'Homicidios (víctimas)' if key=='homicidios' else 'Muertes en tránsito (víctimas)',
                                spec['url'],'Víctimas, conteo absoluto')
        if out.empty: raise ValueError('No quedaron filas válidas tras la validación.')
        status['estado']='OK';status['registros_finales']=len(out)
        status['periodo']=f"{out.anio.min()}–{out.anio.max()}"
        status['departamentos']=int(out.codigo_departamento.nunique())
        status['hash_sha256']=hashlib.sha256(out.sort_values(['codigo_departamento','anio'])
                                            .to_csv(index=False).encode()).hexdigest()[:16]
        return out,status
    except Exception as exc:
        status['observaciones']=str(exc)
        return pd.DataFrame(),status

def load_official():
    with ThreadPoolExecutor(max_workers=3) as executor:
        results=list(executor.map(load_source,SOURCES))
    rows=[r[0] for r in results if not r[0].empty]
    result=(pd.concat(rows,ignore_index=True) if rows else
            pd.DataFrame(columns=['codigo_departamento','departamento','anio',
                                  'indicador','valor','unidad','fuente']))
    return result, pd.DataFrame([r[1] for r in results])

def validate_upload(content,name):
    if name.lower().endswith('.xlsx'):
        df=pd.read_excel(BytesIO(content))
    elif name.lower().endswith('.csv'):
        try: df=pd.read_csv(BytesIO(content))
        except UnicodeDecodeError: df=pd.read_csv(BytesIO(content),encoding='latin-1')
    else:
        raise ValueError('Archivo no admitido. Use CSV o XLSX.')
    df.columns=[_text(c).replace(' ','_') for c in df.columns]
    needed={'anio','indicador','valor'}
    if not needed.issubset(df.columns) or not ({'departamento','codigo_departamento'}&set(df.columns)):
        raise ValueError('Se requiere: año (anio), indicador, valor y departamento o codigo_departamento.')
    if 'codigo_departamento' in df:
        df['codigo_departamento']=df['codigo_departamento'].map(normalizar_codigo)
    else: df['codigo_departamento']=df['departamento'].map(nombre_a_codigo)
    if df.codigo_departamento.isna().any():
        raise ValueError(f'Hay {df.codigo_departamento.isna().sum()} departamentos sin identificación válida.')
    df['departamento']=df['codigo_departamento'].map(CODE_TO_NAME)
    df['anio']=pd.to_numeric(df['anio'],errors='coerce')
    df['valor']=pd.to_numeric(df['valor'].astype(str).str.replace(',','.',regex=False),errors='coerce')
    if df[['anio','valor']].isna().any().any(): raise ValueError('Existen años o valores no numéricos.')
    if (~df['anio'].between(1990,2100)).any() or (df['anio']%1 != 0).any():
        raise ValueError('Años fuera de rango o no enteros.')
    df['anio']=df['anio'].astype(int)
    df['indicador']=df['indicador'].astype(str).str.strip()
    if (df.indicador=='').any(): raise ValueError('Indicador vacío.')
    if df.duplicated(['codigo_departamento','anio','indicador']).any():
        raise ValueError('Filas duplicadas por departamento, año e indicador. Depure antes de importar.')
    if 'unidad' not in df: df['unidad']='No especificada'
    if 'fuente' not in df: df['fuente']='Archivo suministrado por usuario (sin fuente externa verificada)'
    df['fuente']=df['fuente'].fillna('Archivo suministrado por usuario (no verificado)')
    # Un mismo indicador debe mantener unidades consistentes en todo el panel.
    if (df.groupby('indicador')['unidad'].nunique()>1).any():
        raise ValueError('Un indicador contiene unidades inconsistentes.')
    return df[['codigo_departamento','departamento','anio','indicador','valor','unidad','fuente']]

def join_data(official,uploaded):
    if uploaded is None or uploaded.empty: return official.copy()
    joint=pd.concat([official,uploaded],ignore_index=True)
    keys=['codigo_departamento','anio','indicador']
    if joint.duplicated(keys).any():
        raise ValueError('Hay indicadores superpuestos entre datos oficiales y archivo. Renombre o elimine los duplicados.')
    if (joint.groupby('indicador')['unidad'].nunique()>1).any():
        raise ValueError('Las unidades de un indicador no coinciden entre orígenes.')
    return joint

def panel_wide(df):
    if df.empty: return pd.DataFrame(columns=['departamento','anio'])
    v=df.pivot(index=['departamento','anio'],columns='indicador',values='valor').reset_index()
    v.columns.name=None
    return v.sort_values(['departamento','anio']).reset_index(drop=True)

def coverage(df):
    if df.empty: return pd.DataFrame()
    return df.groupby(['indicador','unidad','fuente'],dropna=False).agg(
        observaciones=('valor','size'),departamentos=('departamento','nunique'),
        anio_inicial=('anio','min'),anio_final=('anio','max'),
        faltantes=('valor',lambda x: x.isna().sum())).reset_index()

def numeric_panel(panel,vars,year_min=None,year_max=None,deptos=None):
    if not vars or any(x not in panel.columns for x in vars):
        return pd.DataFrame()
    d=panel.copy()
    if year_min is not None: d=d[d.anio>=year_min]
    if year_max is not None: d=d[d.anio<=year_max]
    if deptos: d=d[d.departamento.isin(deptos)]
    return d.dropna(subset=vars).sort_values(['anio','departamento'])

def eda_stats(df):
    if df.empty: return pd.DataFrame()
    return df.groupby('departamento')['valor'].agg(['count','mean','median','std','min','max']).round(3)

def compute_outliers(df,method):
    if df.empty: return df.copy()
    vals=df.valor.astype(float)
    if method=='IQR':
        q1,q3=vals.quantile([.25,.75]); lo=q1-1.5*(q3-q1);hi=q3+1.5*(q3-q1)
        return df[(vals<lo)|(vals>hi)]
    std=vals.std(ddof=0)
    return df[np.abs((vals-vals.mean())/(std if std else 1))>3]

def cluster(panel,variables,year,k,method):
    d=numeric_panel(panel,variables,year,year).copy()
    if len(variables)<2 or len(d)<max(6,k+2): raise ValueError('Requiere 2 variables y al menos 6 departamentos completos.')
    xs=StandardScaler().fit_transform(d[variables]);k=int(k)
    if method=='KMeans':labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(xs)
    elif method=='Jerárquico':labels=AgglomerativeClustering(n_clusters=k).fit_predict(xs)
    else:labels=DBSCAN(eps=1.2,min_samples=2).fit_predict(xs)
    n=len(set(labels))-(1 if -1 in labels else 0)
    score=silhouette_score(xs,labels) if 1<len(set(labels))<len(labels) else float('nan')
    pc=PCA(n_components=2).fit_transform(xs)
    d['cluster']=labels.astype(str);d['PC1']=pc[:,0];d['PC2']=pc[:,1]
    return d,n,score

def pca(panel,variables,year_min,year_max):
    d=numeric_panel(panel,variables,year_min,year_max).copy()
    if len(variables)<2 or len(d)<5:raise ValueError('PCA requiere al menos 2 indicadores y 5 registros completos.')
    xs=StandardScaler().fit_transform(d[variables]);model=PCA(n_components=min(len(variables),len(d)))
    scores=model.fit_transform(xs)
    d['PC1']=scores[:,0];d['PC2']=scores[:,1]
    loadings=pd.DataFrame(model.components_.T,index=variables,columns=[f'PC{i+1}' for i in range(len(model.components_))])
    return d,pd.Series(model.explained_variance_ratio_,index=loadings.columns),loadings

def regressor(name,alpha=1.0):
    options={'Ridge':Ridge(alpha=alpha),'Lasso':Lasso(alpha=alpha,max_iter=12000),
             'ElasticNet':ElasticNet(alpha=alpha,l1_ratio=.5,max_iter=12000),
             'SVR':SVR(kernel='rbf',C=max(.01,alpha)),
             'Bosque aleatorio':RandomForestRegressor(n_estimators=150,max_depth=7,random_state=42,n_jobs=-1),
             'Gradient boosting':GradientBoostingRegressor(n_estimators=100,max_depth=2,random_state=42),
             'MCO':LinearRegression()}
    return options[name]

def supervised(panel,target,features,algorithm='Ridge',alpha=1.0):
    if not features or target in features or len(set(features))!=len(features):
        raise ValueError('Seleccione predictores distintos de la variable objetivo.')
    d=numeric_panel(panel,[target]+features)
    years=sorted(d.anio.unique())
    if len(d)<30 or len(years)<5: raise ValueError('Se requieren ≥30 registros completos y ≥5 años.')
    split=max(1,int(len(years)*.75));split=min(split,len(years)-1)
    train=d[d.anio.isin(years[:split])];test=d[d.anio.isin(years[split:])]
    if len(train)<15 or len(test)<5: raise ValueError('Partición temporal con observaciones insuficientes.')
    pipeline=Pipeline([('imputer',SimpleImputer(strategy='median')),
                       ('scaler',StandardScaler()),('model',regressor(algorithm,alpha))])
    pipeline.fit(train[features],train[target]);pred=pipeline.predict(test[features])
    actual=test[target].to_numpy()
    metrics={'MAE':float(mean_absolute_error(actual,pred)),
             'RMSE':float(np.sqrt(mean_squared_error(actual,pred))),
             'R²':float(r2_score(actual,pred)),
             'Entrenamiento':len(train),'Prueba':len(test),
             'Años entrenamiento':f'{min(years[:split])}–{max(years[:split])}',
             'Años prueba':f'{min(years[split:])}–{max(years[split:])}'}
    predictions=test[['anio','departamento']].copy();predictions['observado']=actual;predictions['predicho']=pred
    imp=permutation_importance(pipeline,test[features],test[target],n_repeats=8,random_state=42,
                               scoring='neg_mean_absolute_error')
    importance=pd.DataFrame({'variable':features,'aumento_MAE':imp.importances_mean,
                             'desviacion':imp.importances_std}).sort_values('aumento_MAE',ascending=False)
    return metrics,predictions,importance,pipeline

def series_from_panel(panel,depto,variables):
    if depto=='Promedio departamental (no ponderado)':
        ts=panel.groupby('anio')[variables].mean()
    else:
        ts=panel[panel.departamento==depto].set_index('anio')[variables]
    ts=ts.sort_index().dropna()
    if ts.empty or (np.diff(ts.index.to_numpy())!=1).any():
        raise ValueError('La serie seleccionada tiene años faltantes; no se interpola automáticamente.')
    return ts

def is_stationary(x):
    if len(x)<9 or x.nunique()<4: return None
    try: return float(adfuller(x,autolag='AIC')[1])<.05
    except Exception: return None

def var_vec(panel,depto,variables,model_type,maxlags,horizon):
    if len(variables)<2 or len(variables)>4:raise ValueError('Seleccione 2–4 variables distintas.')
    ts=series_from_panel(panel,depto,variables)
    k=len(variables)
    if len(ts)<max(16,5*k+5): raise ValueError('Serie demasiado corta para estimación multivariada estable.')
    tests=pd.DataFrame({'variable':variables,
        'ADF_niveles_estacionaria_5pct':[is_stationary(ts[v]) for v in variables],
        'ADF_diferencias_estacionaria_5pct':[is_stationary(ts[v].diff().dropna()) for v in variables]})
    if model_type=='VAR':
        diff=not tests.ADF_niveles_estacionaria_5pct.eq(True).all()
        data=ts.diff().dropna() if diff else ts
        lagcap=max(1,min(int(maxlags), (len(data)-k-3)//(k+1)))
        result=VAR(data).fit(maxlags=lagcap,ic='aic')
        if result.k_ar==0:result=VAR(data).fit(1)
        irf=result.irf(min(8,horizon+3)).irfs
        fc=result.forecast(data.values[-result.k_ar:],steps=int(horizon))
        if diff:
            fc=ts.iloc[-1].to_numpy()+np.cumsum(fc,axis=0)
        forecast=pd.DataFrame(fc,columns=variables,index=range(int(ts.index.max())+1,int(ts.index.max())+int(horizon)+1))
        return {'tipo':'VAR','n_obs':len(data),'rezagos':result.k_ar,
                'estable':bool(result.is_stable()),'transformacion':'Primera diferencia para todas las series' if diff else 'Niveles',
                'AIC':float(result.aic)},tests,forecast,irf
    if not (tests.ADF_niveles_estacionaria_5pct.eq(False) & tests.ADF_diferencias_estacionaria_5pct.eq(True)).all():
        raise ValueError('VECM requiere evidencia de series I(1): ADF no rechaza raíz unitaria en niveles y la rechaza en diferencias.')
    order=max(1,min(int(maxlags),2))
    rank_result=select_coint_rank(ts,det_order=0,k_ar_diff=order,method='trace',signif=.05)
    rank=int(rank_result.rank)
    if rank<=0 or rank>=k: raise ValueError(f'Johansen estimó rango {rank}; un VECM cointegrado requiere 0 < rango < {k}.')
    result=VECM(ts,k_ar_diff=order,coint_rank=rank,deterministic='ci').fit()
    fc=result.predict(steps=int(horizon))
    forecast=pd.DataFrame(fc,columns=variables,index=range(int(ts.index.max())+1,int(ts.index.max())+int(horizon)+1))
    return {'tipo':'VECM','n_obs':len(ts),'rango_cointegracion':rank,'k_ar_diff':order,
            'alpha_ajuste':np.round(result.alpha,4).tolist()},tests,forecast,None

def forecast_univariate(panel,depto,indicator,horizon):
    ts=series_from_panel(panel,depto,[indicator])[indicator]
    if len(ts)<10: raise ValueError('Se necesitan 10 o más años consecutivos.')
    ntest=max(2,int(len(ts)*.2));train,test=ts.iloc[:-ntest],ts.iloc[-ntest:]
    def fit_model(x,steps):
        model=ExponentialSmoothing(x,trend='add',damped_trend=True,
                                   seasonal=None,initialization_method='estimated').fit(optimized=True)
        return np.asarray(model.forecast(steps)),model
    try:
        pred,_=fit_model(train,ntest)
        model_mae=mean_absolute_error(test,pred)
    except Exception:
        model_mae=float('inf')
    naive_mae=mean_absolute_error(test,np.repeat(train.iloc[-1],ntest))
    use_naive=naive_mae<=model_mae
    if use_naive:future=np.repeat(ts.iloc[-1],horizon)
    else:
        try:future,_=fit_model(ts,horizon)
        except Exception:future=np.repeat(ts.iloc[-1],horizon);use_naive=True
    fc=pd.DataFrame({'anio':range(int(ts.index.max())+1,int(ts.index.max())+horizon+1),
                     'pronostico':future})
    return ts,fc,{'modelo_seleccionado':'Persistencia (naive)' if use_naive else 'Suavizamiento exponencial, tendencia amortiguada',
                  'MAE_validacion':float(min(model_mae,naive_mae)),
                  'MAE_naive':float(naive_mae),'MAE_exponencial':float(model_mae),
                  'años_prueba':f'{test.index.min()}–{test.index.max()}'}

def deep_mlp(panel,depto,indicator,lookback,horizon):
    ts=series_from_panel(panel,depto,[indicator])[indicator]
    if len(ts)<max(15,lookback+9):raise ValueError('La red neuronal requiere al menos 15 años consecutivos.')
    values=ts.to_numpy(dtype=float);lookback=int(lookback)
    X=np.array([values[t-lookback:t] for t in range(lookback,len(values))]);y=values[lookback:]
    cut=max(6,int(len(X)*.75))
    if len(X)-cut<3:raise ValueError('Quedan pocas observaciones de validación.')
    xsc=StandardScaler().fit(X[:cut]);ysc=StandardScaler().fit(y[:cut].reshape(-1,1))
    model=MLPRegressor(hidden_layer_sizes=(48,24,12),alpha=1.,max_iter=1200,
                       early_stopping=False,random_state=42,learning_rate_init=.002)
    model.fit(xsc.transform(X[:cut]),ysc.transform(y[:cut].reshape(-1,1)).ravel())
    pred=ysc.inverse_transform(model.predict(xsc.transform(X[cut:])).reshape(-1,1)).ravel()
    naive=X[cut:,-1];mae=mean_absolute_error(y[cut:],pred)
    history=list(values[-lookback:]);future=[]
    for _ in range(horizon):
        next_value=float(ysc.inverse_transform(model.predict(xsc.transform(
            np.array(history[-lookback:]).reshape(1,-1))).reshape(-1,1))[0,0])
        future.append(next_value);history.append(next_value)
    return ts,pd.DataFrame({'anio':range(int(ts.index[-1])+1,int(ts.index[-1])+1+horizon),
                             'pronostico':future}),{
        'MAE_MLP':float(mae),'MAE_persistencia':float(mean_absolute_error(y[cut:],naive)),
        'observaciones_entrenamiento':int(cut),'observaciones_validacion':int(len(X)-cut),
        'arquitectura':'Red neuronal feedforward: 48–24–12 neuronas, rezagos como entradas'}

def policy_scenario(panel,depto,target,features,shock_var,shock_pct,horizon,method='Ridge'):
    if not features or target in features or shock_var not in features:
        raise ValueError('Seleccione predictores diferentes de la meta y aplique el shock a uno de ellos.')
    d=numeric_panel(panel,[target]+features).sort_values(['departamento','anio']).copy()
    d['target_futuro']=d.groupby('departamento')[target].shift(-1)
    d['anio_futuro']=d.groupby('departamento')['anio'].shift(-1)
    training=d[(d.anio_futuro==d.anio+1)].dropna(subset=['target_futuro'])
    if len(training)<35 or training.anio.nunique()<5:raise ValueError('Se requieren ≥35 pares consecutivos t→t+1 y ≥5 años.')
    years=sorted(training.anio.unique());split=max(1,int(.75*len(years)))
    tr=training[training.anio.isin(years[:split])];te=training[training.anio.isin(years[split:])]
    cols=[target]+features
    estimator=GradientBoostingRegressor(n_estimators=80,max_depth=2,random_state=42) if method=='Gradient boosting' else Ridge(alpha=10.)
    model=Pipeline([('scale',StandardScaler()),('reg',estimator)])
    model.fit(tr[cols],tr.target_futuro)
    mae=mean_absolute_error(te.target_futuro,model.predict(te[cols])) if len(te) else float('nan')
    baseline_start=numeric_panel(panel,[target]+features,deptos=[depto]).sort_values('anio')
    if baseline_start.empty:raise ValueError('No hay última observación completa del departamento.')
    last=baseline_start.iloc[-1]
    base={v:float(last[v]) for v in cols};with_shock=base.copy()
    with_shock[shock_var]*=(1+shock_pct/100.)
    trajectory=[]
    for h in range(1,int(horizon)+1):
        def advance(state):
            x=pd.DataFrame([{v:state[v] for v in cols}])
            result=float(model.predict(x)[0]);state[target]=result
            return result
        b=advance(base);s=advance(with_shock)
        trajectory.append({'anio':int(last.anio)+h,'base':b,'escenario':s,'diferencia':s-b})
    return pd.DataFrame(trajectory),{'MAE_temporal_1_año':float(mae),
        'año_base':int(last.anio),'train_n':len(tr),'test_n':len(te),
        'shock':f'{shock_var}: {shock_pct:+.1f}%',
        'supuesto':'Se mantienen constantes los predictores exógenos y el shock se sostiene en el tiempo. Simulación asociativa, NO estimación causal.'}

def geojson_official():
    params={'where':'1=1','outFields':'CODDPTO,NOMDPTO','returnGeometry':'true',
            'f':'geojson','outSR':'4326'}
    with session() as s:
        r=s.get(GEO_SOURCE,params=params,timeout=(10,45));r.raise_for_status()
        geo=r.json()
    if geo.get('type')!='FeatureCollection' or not geo.get('features'):
        raise ValueError('Servicio geográfico no entregó polígonos GeoJSON.')
    for feature in geo['features']:
        feature.setdefault('properties',{})['CODDPTO']=normalizar_codigo(feature['properties'].get('CODDPTO'))
    geo['features']=[x for x in geo['features'] if x['properties']['CODDPTO']]
    return geo

def spatial_w(geodata,method='KNN',k=4):
    import geopandas as gpd
    from libpysal.weights import Queen,Rook,KNN
    gdf=gpd.GeoDataFrame.from_features(geodata['features'],crs='EPSG:4326')
    gdf['codigo_departamento']=gdf['CODDPTO'].map(normalizar_codigo)
    gdf=gdf.dropna(subset=['codigo_departamento']).drop_duplicates('codigo_departamento')
    gdf=gdf.sort_values('codigo_departamento').reset_index(drop=True)
    if method=='Queen':w=Queen.from_dataframe(gdf,use_index=False)
    elif method=='Rook':w=Rook.from_dataframe(gdf,use_index=False)
    else:w=KNN.from_dataframe(gdf.to_crs(3116),k=min(k,len(gdf)-1))
    w.transform='R'
    return gdf,w

def spatial_analysis(geo,df,indicator,year,method='KNN',explanatory=None):
    import geopandas as gpd
    import esda
    import spreg
    from libpysal.weights import Queen,Rook,KNN
    base=gpd.GeoDataFrame.from_features(geo['features'],crs='EPSG:4326')
    base['codigo_departamento']=base['CODDPTO'].map(normalizar_codigo)
    vals=df[df.anio==year].pivot(index='codigo_departamento',columns='indicador',values='valor')
    required=[indicator]+([explanatory] if explanatory and explanatory!=indicator else [])
    if any(x not in vals.columns for x in required):raise ValueError('Indicador dependiente o explicativo sin datos ese año.')
    geo_df=base.merge(vals[required].dropna().reset_index(),on='codigo_departamento',how='inner')
    geo_df=geo_df.sort_values('codigo_departamento').reset_index(drop=True)
    if len(geo_df)<12:raise ValueError('Se necesitan ≥12 departamentos completos en el mismo año.')
    if method=='Queen':w=Queen.from_dataframe(geo_df,use_index=False)
    elif method=='Rook':w=Rook.from_dataframe(geo_df,use_index=False)
    else:w=KNN.from_dataframe(geo_df.to_crs(3116),k=min(4,len(geo_df)-1))
    if method in ['Queen','Rook'] and len(w.islands)>0:
        raise ValueError(f'La contigüidad {method} tiene departamentos-isla ({len(w.islands)}). Use KNN o estudie un subconjunto conectado.')
    w.transform='R'
    y=geo_df[indicator].to_numpy(dtype=float)
    moran=esda.Moran(y,w,permutations=499)
    local=esda.Moran_Local(y,w,permutations=199)
    geo_df['LISA_categoria']=np.where(local.p_sim<.05,
            np.select([(local.q==1),(local.q==2),(local.q==3),(local.q==4)],
                      ['Alto-Alto','Bajo-Alto','Bajo-Bajo','Alto-Bajo'],default='Otro'),
            'No significativo')
    result={'Moran I':float(moran.I),'p_permutacion':float(moran.p_sim),
            'N_departamentos':len(geo_df),'W':method,'vecinos_aislados':len(w.islands)}
    models=pd.DataFrame()
    if explanatory and explanatory!=indicator:
        x=geo_df[[explanatory]].to_numpy(dtype=float);y2=y.reshape(-1,1)
        try:
            lag=spreg.ML_Lag(y2,x,w=w,method='full',name_y=indicator,name_x=[explanatory])
            err=spreg.ML_Error(y2,x,w=w,method='full',name_y=indicator,name_x=[explanatory])
            models=pd.DataFrame([{'modelo':'SAR / rezago espacial','AIC':float(lag.aic),
                                  'log_L':float(lag.logll),'coef_espacial':float(lag.rho)},
                                 {'modelo':'SEM / error espacial','AIC':float(err.aic),
                                  'log_L':float(err.logll),'coef_espacial':float(err.lam)}])
        except Exception as exc:
            result['aviso_modelos']=f'No fue posible estimar SAR/SEM: {exc}'
    return geo_df,result,models


# ===== INTERFAZ STREAMLIT (INTEGRADA) =====
"""Observatorio Económico y Social · Nariño y Colombia. Streamlit Cloud entrypoint."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Observatorio Económico y Social',page_icon='🌐',layout='wide',
                   initial_sidebar_state='expanded')
COLORS=['#17B9A5','#4966E7','#FFB65C','#B882FF','#FA6E93','#4398D7']
PLOT_LAYOUT=dict(template='plotly_white',font=dict(family='Inter, Arial',color='#23304E'),
                 paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                 margin=dict(l=24,r=24,b=35,t=60),hovermode='closest')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stMarkdown{font-family:'DM Sans',Arial,sans-serif}
[data-testid="stAppViewContainer"] {background:linear-gradient(140deg,#F4F8FE 10%,#EBF4FA 60%,#F8F9FD 100%)}
[data-testid="stSidebar"] {background:#10223D}
[data-testid="stSidebar"] *{color:#E9F2FF!important}
[data-testid="stSidebar"] input{color:#16223F!important}
.block-container{padding-top:1.65rem;max-width:1460px}
h1,h2,h3 {color:#122C51!important;letter-spacing:-.04em}
.hero{background:linear-gradient(112deg,#102E55,#1A487A 55%,#0D817E);border-radius:22px;
  padding:30px 36px;color:white;box-shadow:0 16px 50px rgba(22,49,92,.15);margin-bottom:18px}
.hero h1{color:white!important;font-size:2.25rem;margin:3px 0 10px}
.hero p{color:#E1F1FC;font-size:1.04rem;margin:0;max-width:920px}
.hero-label{text-transform:uppercase;letter-spacing:.2em;color:#89E4D1;font-weight:700;font-size:.76rem}
.card{border-radius:16px;padding:18px;background:white;border:1px solid #E4EBF6;
 box-shadow:0 4px 18px rgba(22,49,92,.045)}
.badge{display:inline-block;background:#DAF4EC;color:#157D6E;border-radius:40px;
 padding:5px 12px;font-weight:700;font-size:.82rem}
section[data-testid="stSidebar"] hr{border-color:#3C5170}
div[data-testid="stMetric"]{background:#FFF;padding:14px 18px;border-radius:14px;
 border:1px solid #E0E9F5;box-shadow:0 4px 18px rgba(22,49,92,.04)}
div[data-testid="stMetricLabel"]{color:#62738B}
.stTabs [data-baseweb="tab-list"]{gap:4px;overflow-x:auto;scrollbar-width:thin}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:9px 15px;font-weight:600}
.stButton>button[kind="primary"]{border:0;border-radius:10px;background:#148A92}
.stButton>button[kind="primary"]:hover{background:#116B75}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=12*3600,show_spinner=False)
def get_official():
    return load_official()
@st.cache_data(ttl=12*3600,show_spinner=False)
def get_geodata():
    return geojson_official()

def plot(fig,height=None):
    fig.update_layout(**PLOT_LAYOUT,colorway=COLORS)
    if height:fig.update_layout(height=height)
    st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False,'toImageButtonOptions':{'format':'png','scale':2}})

def d_button(df,name,label='Descargar CSV'):
    st.download_button(label,df.to_csv(index=False).encode('utf-8-sig'),file_name=name,
                       mime='text/csv',key=f'dl_{name}_{len(df)}')

def error_box(fn,*args,**kwargs):
    try:return fn(*args,**kwargs)
    except Exception as exc:
        st.error(f'Análisis no disponible: {exc}')
        return None

with st.sidebar:
    st.markdown('### ◉ OES | INVESTIGACIÓN')
    st.caption('Plataforma territorial · Edición académica')
    st.markdown('---')
    if st.button('↻ Actualizar fuentes oficiales',use_container_width=True):
        get_official.clear();get_geodata.clear();st.rerun()
    st.caption('Fuentes oficiales actualizadas bajo demanda. Caché de 12 horas.')
    st.markdown('**Anexar indicadores propios**')
    file=st.file_uploader('CSV o Excel (.xlsx)',type=['csv','xlsx'],help='Columnas: departamento/anio/indicador/valor/unidad/fuente')
    st.download_button('⬇ Plantilla CSV',
        'departamento,anio,indicador,valor,unidad,fuente\n'.encode('utf-8'),
        'plantilla_indicadores.csv','text/csv',use_container_width=True)

with st.spinner('Verificando datos públicos y preparando el panel...'):
    raw,logs=get_official()
uploaded=None
if file is not None:
    try:
        uploaded=validate_upload(file.getvalue(),file.name)
        st.sidebar.success(f'Archivo validado: {len(uploaded):,} observaciones.')
    except Exception as exc:
        st.sidebar.error(str(exc))
try: df=join_data(raw,uploaded)
except Exception as exc:
    st.sidebar.error(f'No se integró el archivo: {exc}');df=raw.copy()

st.markdown('''<div class="hero"><div class="hero-label">Colombia · Nariño en el centro del análisis</div>
<h1>Observatorio Económico y Social</h1><p>Inteligencia territorial · Estadística exploratoria · Econometría
espacial · Machine learning · Pronósticos y escenarios de política</p></div>''',unsafe_allow_html=True)
if df.empty:
    st.error('No se pudieron obtener series de las fuentes oficiales. Revisa la pestaña «Fuentes y datos» y sus errores; también puedes adjuntar un CSV real.')
    st.dataframe(logs,use_container_width=True,hide_index=True)
    st.stop()

panel=panel_wide(df)
indicators=sorted(df.indicador.unique().tolist())
deptos=sorted(df.departamento.unique().tolist())
years=sorted(df.anio.unique().tolist())
with st.sidebar:
    st.markdown('---')
    depto=st.selectbox('Departamento principal',deptos,
                        index=deptos.index('Nariño') if 'Nariño' in deptos else 0)
    comparisons=st.multiselect('Comparar con', [x for x in deptos if x!=depto],
                                default=[x for x in ['Cauca','Putumayo','Valle del Cauca'] if x in deptos and x!=depto])
    indic=st.selectbox('Indicador principal',indicators)
    min_year,max_year=st.select_slider('Ventana de observación',options=years,value=(years[0],years[-1]))
    st.caption('Si no hay datos para una combinación, se informará sin inventar valores.')

selection=df[(df.anio>=min_year)&(df.anio<=max_year)]
ind_df=selection[selection.indicador==indic]
main=ind_df[ind_df.departamento==depto].sort_values('anio')
latest=main.iloc[-1] if not main.empty else None
cols=st.columns(4)
cols[0].metric('Departamento central',depto)
cols[1].metric('Último valor publicado',f"{latest.valor:,.2f}" if latest is not None else 'Sin dato')
cols[2].metric('Año última observación',str(latest.anio) if latest is not None else '—')
cols[3].metric('Registros disponibles',f'{len(selection):,}')
st.caption(f'**Unidad:** {latest.unidad if latest is not None else "No disponible"} · **Origen:** {latest.fuente if latest is not None else "No disponible"} · Los años incompletos de registros oficiales se excluyen del agregado.')

tabs=st.tabs(['⌂ Panorama','▥ EDA','⬡ Minería','◉ PCA','♜ Machine learning',
              '↗ VAR / VECM','▧ Mapas + Espacial','◈ Deep learning',
              '⌁ Predicción','⚑ Simulación','⊞ Interpretabilidad','☷ Fuentes y datos'])

with tabs[0]:
    st.subheader(f'Pulso territorial: {depto}')
    subset=ind_df[ind_df.departamento.isin([depto]+comparisons)]
    if subset.empty:st.warning('No hay registros para los filtros globales.')
    else:
        fig=px.line(subset,x='anio',y='valor',color='departamento',markers=True,
                    title=f'Evolución · {indic}',color_discrete_sequence=COLORS)
        plot(fig,390)
    left,right=st.columns([1.3,1])
    with left:
        current=ind_df[ind_df.anio==max_year]
        if current.empty:
            available=sorted(ind_df.anio.unique())
            if available:current=ind_df[ind_df.anio==available[-1]]
        if not current.empty:
            fig=px.bar(current.sort_values('valor').tail(16),x='valor',y='departamento',orientation='h',
                title=f'Comparación territorial ({int(current.anio.iloc[0])})',color='valor',
                color_continuous_scale=['#CDE9F1','#0F857D'])
            plot(fig,485)
    with right:
        st.markdown('#### Ficha del indicador')
        st.write(f'**Indicador:** {indic}')
        st.write(f'**Departamento:** {depto}')
        st.write(f'**Unidad:** {latest.unidad if latest is not None else "—"}')
        st.write(f'**Serie observada:** {int(main.anio.min())}–{int(main.anio.max())}' if not main.empty else 'Sin serie local')
        st.info('Las comparaciones de conteos absolutos dependen del tamaño poblacional. No deben interpretarse como tasas de riesgo.')
    d_button(subset,'panorama_territorial.csv')

with tabs[1]:
    st.subheader('Análisis exploratorio de datos (EDA)')
    ed=ind_df[ind_df.departamento.isin([depto]+comparisons)] if comparisons else ind_df[ind_df.departamento==depto]
    if ed.empty: st.warning('No hay serie para esta selección.')
    else:
        st.dataframe(eda_stats(ed),use_container_width=True)
        c1,c2=st.columns(2)
        with c1:plot(px.histogram(ed,x='valor',color='departamento',marginal='box',
                                   title='Distribución y dispersión'),360)
        with c2:plot(px.box(ed,x='departamento',y='valor',points='outliers',
                              title='Valores atípicos por territorio'),360)
        method=st.radio('Detección de valores extremos',['IQR','Z-score'],horizontal=True)
        outliers=compute_outliers(ed,method)
        st.metric('Valores atípicos detectados',len(outliers))
        if not outliers.empty:st.dataframe(outliers,hide_index=True,use_container_width=True)
    st.markdown('#### Correlación entre indicadores (muestra completa)')
    cv=panel[(panel.anio>=min_year)&(panel.anio<=max_year)]
    corr=cv[indicators].corr(min_periods=8)
    if corr.notna().sum().sum()>=4:
        plot(px.imshow(corr,text_auto='.2f',color_continuous_scale='RdBu_r',zmin=-1,zmax=1,
                       title='Pearson · pares observados, sin imputación'),430)
    else:st.info('No hay suficientes observaciones conjuntas para correlación.')
    st.caption('Las correlaciones no representan causalidad. Indicadores con unidades diferentes no deben agregarse.')

with tabs[2]:
    st.subheader('Minería de datos y segmentación departamental')
    a,b,c=st.columns(3)
    with a:cyear=st.selectbox('Año de corte',list(reversed(years)),key='cy')
    with b:method=st.selectbox('Método',['KMeans','Jerárquico','DBSCAN'])
    with c:k=st.slider('Núm. grupos',2,8,4)
    vars_cl=st.multiselect('Variables de segmentación',indicators,default=indicators[:min(3,len(indicators))],key='clvars')
    if st.button('Ejecutar segmentación',type='primary'):
        result=error_box(cluster,panel,vars_cl,cyear,k,method)
        if result:
            d,n,score=result
            st.write(f'**Grupos detectados:** {n} · **Silhouette:** {score:.3f}' if np.isfinite(score) else f'**Grupos detectados:** {n} · Silhouette no definido')
            plot(px.scatter(d,x='PC1',y='PC2',color='cluster',text='departamento',
                            title='Proyección 2D PCA de las agrupaciones'),480)
            st.dataframe(d[['departamento','cluster']+vars_cl],use_container_width=True,hide_index=True)
            d_button(d,'clusters_departamentos.csv')
    st.caption('La segmentación se estima únicamente con territorios con todas las variables observadas para el año escogido.')

with tabs[3]:
    st.subheader('Análisis de componentes principales (PCA)')
    vars_pca=st.multiselect('Variables numéricas',indicators,default=indicators[:min(3,len(indicators))],key='pcavars')
    if st.button('Calcular componentes principales',type='primary'):
        result=error_box(pca,panel,vars_pca,min_year,max_year)
        if result:
            d,explained,loadings=result
            a,b=st.columns(2)
            with a:plot(px.bar(x=explained.index,y=explained.values,
                                labels={'x':'Componente','y':'Varianza explicada'},title='Scree plot'),390)
            with b:plot(px.scatter(d,x='PC1',y='PC2',hover_name='departamento',color='anio',
                                    title='Proyección PC1 / PC2'),390)
            st.write(f'**Varianza explicada por PC1 + PC2:** {explained.iloc[:2].sum():.1%}')
            st.dataframe(loadings.round(4),use_container_width=True)
            d_button(loadings.reset_index().rename(columns={'index':'variable'}),'pca_cargas.csv')
    st.caption('Se estandarizan las variables y se exige observación completa para cada registro; no se mezcla una unidad con otra sin escalar.')

with tabs[4]:
    st.subheader('Aprendizaje supervisado · regresiones y SVR')
    if len(indicators)<2:st.warning('Se necesitan al menos dos indicadores simultáneos.')
    else:
        a,b,c=st.columns(3)
        with a:y=st.selectbox('Variable a explicar (Y)',indicators,key='mly')
        with b:algorithm=st.selectbox('Algoritmo',['Ridge','Lasso','ElasticNet','SVR','Bosque aleatorio','Gradient boosting','MCO'])
        with c:alpha=st.number_input('Alpha / C',min_value=0.01,max_value=100.0,value=1.0,step=.25)
        xx=st.multiselect('Explicativas (X)',[x for x in indicators if x!=y],default=[x for x in indicators if x!=y][:2],key='mlx')
        if st.button('Entrenar y validar',type='primary'):
            result=error_box(supervised,panel,y,xx,algorithm,alpha)
            if result:
                metrics,preds,imp,pipe=result
                a,b,c=st.columns(3);a.metric('MAE',f"{metrics['MAE']:,.3f}")
                b.metric('RMSE',f"{metrics['RMSE']:,.3f}")
                c.metric('R² prueba',f"{metrics['R²']:.3f}")
                st.info(f"Entrenamiento: {metrics['Años entrenamiento']} (n={metrics['Entrenamiento']}) · Prueba: {metrics['Años prueba']} (n={metrics['Prueba']}). División temporal, no aleatoria.")
                fig=px.scatter(preds,x='observado',y='predicho',color='anio',hover_name='departamento',
                               title='Observado vs predicho (años fuera de entrenamiento)')
                z=[min(preds.observado.min(),preds.predicho.min()),max(preds.observado.max(),preds.predicho.max())]
                fig.add_trace(go.Scatter(x=z,y=z,mode='lines',name='Predicción perfecta',line=dict(color='#FA6E93',dash='dash')))
                plot(fig,430)
                plot(px.bar(imp,x='aumento_MAE',y='variable',orientation='h',
                            title='Interpretabilidad: aumento del MAE por permutación'),325)
                d_button(preds,'predicciones_validacion_ml.csv')
        st.caption('Las X contemporáneas permiten una estimación condicional; no equivalen a pronóstico futuro sin conocer las X futuras.')

with tabs[5]:
    st.subheader('Modelos econométricos VAR y VECM')
    a,b,c=st.columns(3)
    with a:var_depto=st.selectbox('Unidad territorial',[depto]+[x for x in deptos if x!=depto]+['Promedio departamental (no ponderado)'],key='vard')
    with b:var_type=st.radio('Modelo',['VAR','VECM'],horizontal=True)
    with c:var_lags=st.slider('Máx. rezagos',1,4,2)
    var_variables=st.multiselect('Sistema de variables (2–4)',indicators,default=indicators[:min(2,len(indicators))],key='varvars')
    horizon_var=st.slider('Horizonte multivariante (años)',1,7,4,key='varh')
    if st.button('Estimar sistema e impulsos-respuesta',type='primary'):
        result=error_box(var_vec,panel,var_depto,var_variables,var_type,var_lags,horizon_var)
        if result:
            details,tests,forecast,irf=result
            st.json(details);st.markdown('#### Diagnósticos de estacionariedad')
            st.dataframe(tests,use_container_width=True,hide_index=True)
            fcf=forecast.reset_index().rename(columns={'index':'anio'})
            plot(px.line(fcf,x='anio',y=var_variables,markers=True,title=f'Trayectorias estimadas · {var_type}'),430)
            if irf is not None:
                shock=st.selectbox('Variable del impulso',var_variables,key='shock_irf')
                k=var_variables.index(shock)
                impulse=go.Figure()
                for i,v in enumerate(var_variables):
                    impulse.add_trace(go.Scatter(x=list(range(irf.shape[0])),y=irf[:,i,k],
                                                  mode='lines+markers',name=f'Respuesta {v}'))
                impulse.update_layout(title=f'IRF: choque en {shock} · identificación reducida/no ortogonal')
                plot(impulse,360)
            d_button(fcf,'pronostico_var_vecm.csv')
    st.caption('VAR: ADF y diferenciación uniforme cuando corresponde; VECM: series I(1) y rango Johansen válido. Las IRF reducidas no se interpretan automáticamente como shocks estructurales.')

with tabs[6]:
    st.subheader('Cartografía coroplética y dependencia espacial')
    a,b,c=st.columns(3)
    with a:map_ind=st.selectbox('Indicador a mapear',indicators,key='map_ind')
    with b:map_year=st.selectbox('Año cartográfico',list(reversed(years)),key='map_year')
    with c:w_method=st.selectbox('Matriz espacial',['KNN','Queen','Rook'])
    explain=st.selectbox('Variable explicativa SAR/SEM (opcional)',['Ninguna']+[x for x in indicators if x!=map_ind])
    geo=None
    try:geo=get_geodata()
    except Exception as exc:st.warning(f'No se pudo descargar cartografía oficial: {exc}.')
    datamap=df[(df.indicador==map_ind)&(df.anio==map_year)].copy()
    if geo is not None:
        if datamap.empty:st.warning('No hay observaciones del indicador en el año elegido.')
        else:
            fig=px.choropleth(datamap,geojson=geo,locations='codigo_departamento',
                              featureidkey='properties.CODDPTO',color='valor',hover_name='departamento',
                              color_continuous_scale='Tealgrn',
                              title=f'{map_ind} | {map_year} · polígonos reales')
            fig.update_geos(fitbounds='locations',visible=False,showcoastlines=False)
            plot(fig,570)
            st.caption(f'Cobertura del mapa: {len(datamap)} departamentos con información. Cartografía del servicio oficial geográfico [{GEO_INFO}]({GEO_INFO}).')
            if st.button('Estimar Moran, LISA y modelos espaciales',type='primary'):
                result=error_box(spatial_analysis,geo,df,map_ind,map_year,w_method,
                                 None if explain=='Ninguna' else explain)
                if result:
                    spatial,gstats,models=result
                    c1,c2,c3=st.columns(3)
                    c1.metric('Moran global I',f"{gstats['Moran I']:.3f}")
                    c2.metric('p permutacional',f"{gstats['p_permutacion']:.3f}")
                    c3.metric('Departamentos',gstats['N_departamentos'])
                    st.caption('p de permutación Monte Carlo, no prueba de causalidad. LISA significativo al 5% sin corrección por pruebas múltiples.')
                    lisa=px.choropleth(spatial,geojson=geo,locations='codigo_departamento',
                        featureidkey='properties.CODDPTO',color='LISA_categoria',
                        title='Agrupamientos LISA · clasificaciones locales')
                    lisa.update_geos(fitbounds='locations',visible=False)
                    plot(lisa,490)
                    if not models.empty:
                        st.markdown('#### SAR y SEM (máxima verosimilitud)')
                        st.dataframe(models.round(4),hide_index=True,use_container_width=True)
                        st.info('Para inferencia definitiva se deben revisar significancia, impactos directos/indirectos, diagnóstico residual y especificación espacial; no basta comparar AIC.')
                    if gstats.get('aviso_modelos'):st.warning(gstats['aviso_modelos'])
                    d_button(pd.DataFrame(spatial.drop(columns='geometry')),'resultados_espaciales.csv')
    st.caption('Las matrices Queen/Rook pueden tener islas. KNN proyecta geometrías antes de calcular vecinos. Nunca se usan los centroides aproximados del ejemplo original.')

with tabs[7]:
    st.subheader('Deep learning · Red neuronal multicapas')
    st.write('Modelo **MLP con tres capas ocultas** y rezagos temporales. Incluye comparación con una predicción ingenua de persistencia.')
    a,b,c=st.columns(3)
    with a:dl_depto=st.selectbox('Departamento',[depto]+[x for x in deptos if x!=depto],key='dld')
    with b:dl_indicator=st.selectbox('Variable',indicators,key='dli')
    with c:lookback=st.slider('Rezagos de entrada',2,5,3)
    dl_h=st.slider('Horizonte de red neuronal',1,8,4)
    if st.button('Entrenar red neuronal y proyectar',type='primary'):
        result=error_box(deep_mlp,panel,dl_depto,dl_indicator,lookback,dl_h)
        if result:
            hist,fc,quality=result
            c1,c2=st.columns(2);c1.metric('MAE red',f"{quality['MAE_MLP']:.2f}")
            c2.metric('MAE persistencia',f"{quality['MAE_persistencia']:.2f}")
            plot(px.line(pd.concat([pd.DataFrame({'anio':hist.index,'observado':hist.values,'tipo':'Observado'}),
                pd.DataFrame({'anio':fc.anio,'observado':fc.pronostico,'tipo':'Pronóstico neuronal'})]),
                x='anio',y='observado',color='tipo',markers=True,title='Red neuronal: pasado y escenario proyectado'),430)
            st.caption(quality['arquitectura'])
            if quality['MAE_MLP']>quality['MAE_persistencia']:
                st.warning('La red fue superada por la persistencia en validación; no conviene presentarla como mejor predictor.')
            d_button(fc,'red_neuronal_pronostico.csv')
    st.caption('Con 10–20 observaciones anuales las redes profundas pueden sobreajustarse. Los resultados deben discutirse críticamente.')

with tabs[8]:
    st.subheader('Pronóstico temporal y comparación contra baseline')
    a,b,c=st.columns(3)
    with a:f_depto=st.selectbox('Departamento',[depto]+[x for x in deptos if x!=depto],key='fored')
    with b:f_ind=st.selectbox('Indicador a pronosticar',indicators,key='forei')
    with c:f_h=st.slider('Años a proyectar',1,10,5,key='foreh')
    if st.button('Validar y pronosticar',type='primary'):
        result=error_box(forecast_univariate,panel,f_depto,f_ind,f_h)
        if result:
            hist,fc,quality=result
            st.success(f"Modelo seleccionado: {quality['modelo_seleccionado']}. MAE en validación {quality['MAE_validacion']:,.3f} ({quality['años_prueba']}).")
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=hist.index,y=hist.values,name='Observado',mode='lines+markers'))
            fig.add_trace(go.Scatter(x=fc.anio,y=fc.pronostico,name='Pronóstico',mode='lines+markers',
                                     line=dict(dash='dash',color='#15A997')))
            fig.update_layout(title=f'Pronóstico para {f_depto} · {f_ind}')
            plot(fig,450)
            st.dataframe(fc,hide_index=True,use_container_width=True)
            d_button(fc,'pronostico_serie.csv')
    st.caption('El sistema compara suavizamiento exponencial y persistencia mediante años de prueba anteriores a la estimación final. No genera intervalos de confianza arbitrarios.')

with tabs[9]:
    st.subheader('Simulación de política · escenarios condicionados')
    st.warning('Este módulo NO estima efectos causales de una política. Compara escenarios contrafactuales condicionados al modelo predictivo, manteniendo explícitos sus supuestos.')
    if len(indicators)<2:st.info('Se requiere más de una variable.')
    else:
        a,b=st.columns(2)
        with a:py=st.selectbox('Resultado que interesa (Y)',indicators,key='py')
        with b:pdpto=st.selectbox('Territorio de la intervención',[depto]+[x for x in deptos if x!=depto],key='pd')
        pfeatures=st.multiselect('Variables antecedentes X(t)',[x for x in indicators if x!=py],
                        default=[x for x in indicators if x!=py][:2],key='pfeat')
        c,d,e=st.columns(3)
        with c:shock=st.selectbox('Variable a modificar',pfeatures if pfeatures else ['Seleccione X'])
        with d:pct=st.slider('Cambio persistente (%)',-50,50,10,step=5)
        with e:ph=st.slider('Horizonte simulación',1,8,5)
        pm=st.radio('Modelo dinámico',['Ridge','Gradient boosting'],horizontal=True)
        if st.button('Comparar escenario base vs intervención',type='primary'):
            result=error_box(policy_scenario,panel,pdpto,py,pfeatures,shock,pct,ph,pm)
            if result:
                traj,metadata=result
                st.info(metadata['supuesto'])
                st.write(f"Año de referencia: **{metadata['año_base']}** · MAE predictivo t→t+1: **{metadata['MAE_temporal_1_año']:.3f}** · {metadata['shock']}")
                plot(px.line(traj,x='anio',y=['base','escenario'],markers=True,
                             labels={'value':py,'variable':'Escenario'},
                             title='Diferencia condicional: trayectoria de resultados'),430)
                st.dataframe(traj.round(3),hide_index=True,use_container_width=True)
                d_button(traj,'escenarios_politica.csv')

with tabs[10]:
    st.subheader('Interpretabilidad automatizada · variables relevantes')
    st.write('La importancia por permutación se mide en los **años de prueba no utilizados** en el entrenamiento, para evitar explicar el sobreajuste.')
    if len(indicators)>=2:
        iy=st.selectbox('Resultado',indicators,key='iy')
        ix=st.multiselect('Explicativas',[x for x in indicators if x!=iy],
                          default=[x for x in indicators if x!=iy][:3],key='ix')
        im=st.selectbox('Modelo interpretable',['Bosque aleatorio','Ridge','Lasso','Gradient boosting'])
        if st.button('Explicar resultado predictivo',type='primary'):
            result=error_box(supervised,panel,iy,ix,im,1.)
            if result:
                metrics,preds,imp,pipe=result
                st.metric('MAE prueba',f"{metrics['MAE']:.3f}")
                plot(px.bar(imp.sort_values('aumento_MAE'),y='variable',x='aumento_MAE',
                            error_x='desviacion',orientation='h',title='Permutación: aumento del error al alterar X'),440)
                mdl=pipe.named_steps['model']
                if hasattr(mdl,'coef_'):
                    coef=pd.DataFrame({'variable':ix,'coeficiente_estandarizado_X':mdl.coef_})
                    st.dataframe(coef,hide_index=True,use_container_width=True)
                st.caption('Importancia predictiva no significa efecto causal. Variables colineales pueden distribuirse la importancia.')
                d_button(imp,'importancia_predictiva.csv')

with tabs[11]:
    st.subheader('Auditoría, procedencia y descarga de datos')
    st.write('Datos importados y validados. **No hay datos simulados ni rellenados con valores inventados** en la base principal.')
    show=logs.copy()
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.markdown('#### Catálogo de fuentes')
    catalog=pd.DataFrame([{**v,'clave':k} for k,v in SOURCES.items()])
    st.dataframe(catalog[['nombre','organismo','id','frecuencia','url','advertencia']],hide_index=True,use_container_width=True)
    if uploaded is not None:st.success(f'Datos propios incorporados: {len(uploaded)} observaciones. Son información del usuario, no verificada por el observatorio.')
    st.markdown('#### Cobertura observada por indicador')
    st.dataframe(coverage(df),hide_index=True,use_container_width=True)
    st.markdown('#### Panel maestro (muestra)')
    st.dataframe(df.head(1000),hide_index=True,use_container_width=True)
    x,y=st.columns(2)
    with x:d_button(df,'observatorio_datos_largos.csv','Descargar panel largo completo')
    with y:d_button(panel,'observatorio_panel_ancho.csv','Descargar panel multivariante')
    st.markdown('#### Diccionario de datos')
    st.markdown('''`codigo_departamento`: DIVIPOLA de 2 dígitos · `departamento`: nombre canónico · `anio`: período anual completo · `indicador`: dimensión económica/social · `valor`: valor observado · `unidad`: unidad sin conversión implícita · `fuente`: URL o procedencia declarada.''')
    st.info('Para agregar pobreza, desempleo, educación o informalidad, importa series departamentales oficiales en CSV/XLSX con año, código DIVIPOLA, indicador, valor, unidad y fuente. No se insertan cifras sin verificar.')

st.markdown('---')
st.caption('Observatorio Económico y Social | Herramienta académica reproducible. Resultados de modelos sujetos a supuestos, tamaño de muestra y calidad de la fuente. Ninguna asociación estadística demuestra causalidad por sí sola.')
