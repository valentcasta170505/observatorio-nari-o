"""OBSERVATORIO ECONÓMICO Y SOCIAL | Nariño y los 33 territorios.

Versión autónoma para GitHub y Streamlit Community Cloud.
NO requiere core.py. Datos solicitados a fuentes públicas bajo demanda.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# DATOS, NORMALIZACIÓN Y API (integrados: sin core.py)
# ============================================================
SOURCES = {
    "economia": {
        "nombre": "Actividad económica departamental (constantes de 2015)",
        "organismo": "Gobernación de Caldas / Datos Abiertos Colombia",
        "id": "kgyi-qc7j",
        "url": "https://www.datos.gov.co/Econom-a-y-Finanzas/PIB-Departamental-con-proyecci-n/kgyi-qc7j",
        "frecuencia": "Anual",
        "unidad": "Miles de millones de pesos de 2015",
        "advertencia": "Serie de actividades/proyecciones de un tercero. Su suma NO equivale necesariamente al PIB oficial del DANE.",
    },
    "homicidios": {
        "nombre": "Homicidios intencionales (víctimas)",
        "organismo": "Ministerio de Defensa / Datos Abiertos Colombia",
        "id": "m8fd-ahd9",
        "url": "https://www.datos.gov.co/Seguridad-y-Defensa/HOMICIDIO/m8fd-ahd9",
        "frecuencia": "Registros agregados por año",
        "unidad": "Víctimas registradas",
        "advertencia": "Conteo absoluto; NO es una tasa por 100.000 habitantes. No incluye muertes de tránsito.",
    },
    "transito": {
        "nombre": "Muertes en accidentes de tránsito (víctimas)",
        "organismo": "Ministerio de Defensa / Datos Abiertos Colombia",
        "id": "uav5-b85g",
        "url": "https://www.datos.gov.co/Seguridad-y-Defensa/HOMICIDIO-EN-ACCIDENTE-DE-TR-NSITO/uav5-b85g",
        "frecuencia": "Registros agregados por año",
        "unidad": "Víctimas registradas",
        "advertencia": "Muertes en hechos de tránsito; NO son homicidios intencionales.",
    },
}

DEP_NAMES = [
    "Amazonas", "Antioquia", "Arauca", "Atlántico", "Bogotá D.C.", "Bolívar", "Boyacá",
    "Caldas", "Caquetá", "Casanare", "Cauca", "Cesar", "Chocó", "Córdoba",
    "Cundinamarca", "Guainía", "Guaviare", "Huila", "La Guajira", "Magdalena",
    "Meta", "Nariño", "Norte de Santander", "Putumayo", "Quindío", "Risaralda",
    "San Andrés y Providencia", "Santander", "Sucre", "Tolima", "Valle del Cauca",
    "Vaupés", "Vichada",
]
DEP_CODES = [
    "91", "05", "81", "08", "11", "13", "15", "17", "18", "85", "19", "20",
    "27", "23", "25", "94", "95", "41", "44", "47", "50", "52", "54", "86",
    "63", "66", "88", "68", "70", "73", "76", "97", "99",
]
CODE_TO_NAME = dict(zip(DEP_CODES, DEP_NAMES))
NACIONAL = "Todos los departamentos (33 territorios)"


def _text(value: object) -> str:
    s = str(value if value is not None else "").lower()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.replace(".", "").replace(",", "")).strip()


NAME_TO_CODE = {_text(v): k for k, v in CODE_TO_NAME.items()}
NAME_TO_CODE.update({
    "bogota dc": "11", "bogota d c": "11", "distrito capital": "11",
    "san andres": "88", "san andres providencia y santa catalina": "88",
    "archipielago de san andres providencia y santa catalina": "88", "valle": "76",
})


def normalizar_codigo(value: object) -> str | None:
    """Convierte códigos DIVIPOLA 5, '05', 05001 y '5.0' en código de dos dígitos."""
    if pd.isna(value):
        return None
    text = str(value).strip().split(".")[0]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) == 5:
        digits = digits[:2]
    if len(digits) > 2:
        return None
    code = digits.zfill(2)
    return code if code in CODE_TO_NAME else None


def nombre_a_codigo(value: object) -> str | None:
    return NAME_TO_CODE.get(_text(value))


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    s.headers.update({"User-Agent": "ObservatorioNariño/2.0 (uso academico)", "Accept": "application/json"})
    return s


def socrata_pages(dataset_id: str, params: dict, page_size: int = 20000, max_pages: int = 2) -> pd.DataFrame:
    """Consulta paginada. Las incidencias se descargan previamente agregadas en el servidor."""
    parts: list[pd.DataFrame] = []
    with session() as s:
        for page in range(max_pages):
            query = {**params, "$limit": page_size, "$offset": page * page_size}
            response = s.get(f"https://www.datos.gov.co/resource/{dataset_id}.json", params=query, timeout=(8, 40))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"La fuente {dataset_id} no devolvió una lista JSON válida.")
            if not payload:
                break
            parts.append(pd.DataFrame.from_records(payload))
            if len(payload) < page_size:
                break
        else:
            # Evita truncamiento silencioso al alcanzar el máximo de páginas.
            if parts and len(parts[-1]) == page_size:
                raise RuntimeError(f"Límite de paginación alcanzado para {dataset_id}; aumente max_pages.")
    if not parts:
        raise ValueError("La fuente no tiene registros para los filtros disponibles.")
    return pd.concat(parts, ignore_index=True)


def _normalize_base(raw: pd.DataFrame, code_field: str, date_field: str, value_field: str,
                    indicator: str, activity_field: str | None = None, max_year: int | None = None) -> pd.DataFrame:
    """Normaliza registros y agrupa sin inventar observaciones de departamentos/años faltantes."""
    required = {code_field, date_field, value_field}
    if activity_field:
        required.add(activity_field)
    missing = required - set(raw.columns)
    if missing:
        raise ValueError("Columnas faltantes en respuesta pública: " + ", ".join(sorted(missing)))
    v = raw.copy()
    v["codigo_departamento"] = v[code_field].map(normalizar_codigo)
    if date_field == "fecha_hecho":
        v["anio"] = pd.to_datetime(v[date_field], errors="coerce").dt.year
    else:
        v["anio"] = pd.to_numeric(v[date_field], errors="coerce")
    v["valor"] = pd.to_numeric(v[value_field], errors="coerce")
    v = v.dropna(subset=["codigo_departamento", "anio", "valor"])
    end = max_year if max_year is not None else datetime.now(timezone.utc).year - 1
    v = v[(v["anio"] >= 1990) & (v["anio"] <= end)].copy()
    if v.empty:
        raise ValueError("No se encontraron datos válidos con el intervalo de años especificado.")
    v["anio"] = v["anio"].astype("int16")
    v["valor"] = v["valor"].astype("float64")
    keys = ["codigo_departamento", "anio"]
    if activity_field:
        v["actividad"] = v[activity_field].fillna("Actividad no clasificada").astype(str).str.strip()
        keys.append("actividad")
    v = v.groupby(keys, as_index=False, observed=True)["valor"].sum()
    v["departamento"] = v["codigo_departamento"].map(CODE_TO_NAME)
    v["indicador"] = SOURCES[indicator]["nombre"]
    return v


def fetch_dataset(indicator: str, max_year: int | None = None) -> pd.DataFrame:
    """Solicita solo el indicador elegido; incidentes se agregan por departamento/año en Socrata."""
    if indicator not in SOURCES:
        raise ValueError("Indicador no configurado.")
    end = max_year if max_year is not None else datetime.now(timezone.utc).year - 1
    dataset_id = SOURCES[indicator]["id"]
    if indicator == "economia":
        raw = socrata_pages(dataset_id, {
            "$select": "a_o,c_digo_departamento_divipola,actividad,valor_miles_de_millones_de",
            "$where": "tipo_de_precios = 'PIB a precios constantes de 2015' AND a_o >= 1990 AND a_o <= " + str(int(end)),
            "$order": ":id ASC",
        }, page_size=20000, max_pages=3)
        return _normalize_base(raw, "c_digo_departamento_divipola", "a_o", "valor_miles_de_millones_de",
                               indicator, activity_field="actividad", max_year=end)
    # El servidor de Socrata suma antes de transmitir: ~departamentos x años, no cientos de miles de filas.
    raw = socrata_pages(dataset_id, {
        "$select": "cod_depto as codigo_departamento, date_extract_y(fecha_hecho) as anio, sum(cantidad) as valor",
        "$where": "fecha_hecho >= '1990-01-01T00:00:00' AND fecha_hecho < '" + str(int(end) + 1) + "-01-01T00:00:00'",
        "$group": "cod_depto, date_extract_y(fecha_hecho)",
        "$order": "anio ASC, codigo_departamento ASC",
    }, page_size=10000, max_pages=2)
    return _normalize_base(raw, "codigo_departamento", "anio", "valor", indicator, max_year=end)


def aggregate_indicator(df: pd.DataFrame, activity: str | None = None) -> pd.DataFrame:
    """Agrega actividad económica elegida (o suma exploratoria) o conteos de víctimas."""
    base = df
    if activity and activity != "Todas las actividades (suma exploratoria)" and "actividad" in df.columns:
        base = df[df["actividad"] == activity]
    return base.groupby(["codigo_departamento", "departamento", "anio"], as_index=False, observed=True)["valor"].sum()


def to_series(df: pd.DataFrame, department: str) -> pd.DataFrame:
    base = df if department == NACIONAL else df[df["departamento"] == department]
    return base.groupby("anio", as_index=False)["valor"].sum().sort_values("anio")


def panel_overview(df: pd.DataFrame) -> dict:
    """Cobertura observada, nunca cobertura asumida."""
    return {
        "departamentos": int(df["codigo_departamento"].nunique()),
        "anios": int(df["anio"].nunique()),
        "primer_anio": int(df["anio"].min()),
        "ultimo_anio": int(df["anio"].max()),
        "observaciones": int(len(df)),
    }


# ============================================================
# APLICACIÓN STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Observatorio Económico y Social | Nariño",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .block-container {padding-top:1.5rem; padding-bottom:2.2rem; max-width:1500px;}
  h1,h2,h3 {letter-spacing:-.025em;}
  [data-testid="stMetric"] {background:#ffffff;border:1px solid #dbe4ef;border-radius:13px;padding:16px;}
  [data-testid="stSidebar"] {background-color:#edf4fc;}
  [data-testid="stSidebar"] * {color:#172d46;}
  a {color:#176cd4;}
  .tagline {font-size:.95rem;color:#43627f;margin-top:-8px;margin-bottom:14px;}
</style>
""", unsafe_allow_html=True)

PAGES = [
    "🏠 Resumen", "📈 Series y comparación",
    "🔎 Análisis exploratorio", "🤖 Predicción y ML",
    "🧪 Econometría y PCA", "📚 Datos y metodología",
]
ACTIVITY_ALL = "Todas las actividades (suma exploratoria)"

@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=5)
def load_indicator(key: str, end_year: int) -> pd.DataFrame:
    return fetch_dataset(key, max_year=end_year)


def fmt_number(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def chart_style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=50, b=25),
        font=dict(color="#20354d", size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#e2eaf2"),
    )
    return fig


def show_plot(fig: go.Figure):
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def plot_series(series: pd.DataFrame, unit: str, department: str):
    fig = px.line(series, x="anio", y="valor", markers=True,
                  title=f"Evolución anual · {department}", labels={"anio":"Año", "valor":unit})
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    show_plot(chart_style(fig))


def csv_download(df: pd.DataFrame, filename: str):
    st.download_button("⬇️ Descargar resultados CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv")


def _numpy_linear_model(x_train: np.ndarray, y_train: np.ndarray, choice: str, x_predict: np.ndarray) -> np.ndarray:
    """Ridge/Lasso en una variable (año), con estandarización aprendida solo del train.

    Funciona sin scikit-learn y evita fuga de información en el test cronológico.
    Penalización Ridge L2 y Lasso L1; intercepto sin penalización.
    """
    x_train=np.asarray(x_train,dtype=float).reshape(-1)
    x_predict=np.asarray(x_predict,dtype=float).reshape(-1)
    y_train=np.asarray(y_train,dtype=float).reshape(-1)
    x_mean=float(x_train.mean())
    scale=float(x_train.std()) or 1.0
    z=(x_train-x_mean)/scale
    z_pred=(x_predict-x_mean)/scale
    target_mean=float(y_train.mean())
    centered=y_train-target_mean
    if choice == 'Ridge':
        slope=float(np.dot(z,centered)/(np.dot(z,z)+1.0))
    elif choice == 'Lasso':
        # min (1/2n)*sum((y-a-b*z)^2) + alpha*abs(b), alpha=0.1
        cov=float(np.dot(z,centered)/len(z))
        variance=float(np.dot(z,z)/len(z))
        slope=float(np.sign(cov)*max(abs(cov)-0.1,0.0)/variance) if variance else 0.0
    else:
        raise ValueError('Modelo lineal desconocido.')
    return target_mean+slope*z_pred


def compute_prediction(series: pd.DataFrame, choice: str, horizon: int):
    """Validación cronológica 80/20; Ridge y Lasso sin librerías pesadas.

    Los modelos Random Forest y SVR se ejecutan solo si scikit-learn está instalado.
    """
    frame=series.sort_values('anio').dropna(subset=['anio','valor']).copy()
    if len(frame)<8:
        raise ValueError('Se necesitan al menos 8 años observados para la evaluación temporal.')
    years=frame['anio'].to_numpy(dtype=int)
    if not np.all(np.diff(years)==1):
        raise ValueError('La serie debe tener años consecutivos; no se inventan años faltantes.')
    x=(years-years.min()).reshape(-1,1)
    y=frame['valor'].to_numpy(dtype=float)
    cut=max(5,int(np.floor(len(y)*.8)))
    if len(y)-cut<2: cut=len(y)-2
    future_years=np.arange(years.max()+1,years.max()+horizon+1,dtype=int)
    future_x=(future_years-years.min()).reshape(-1,1)
    if choice in ('Ridge','Lasso'):
        test_pred=_numpy_linear_model(x[:cut],y[:cut],choice,x[cut:])
        future_values=_numpy_linear_model(x,y,choice,future_x)
    elif choice in ('Random Forest','SVR'):
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVR
            from sklearn.compose import TransformedTargetRegressor
        except ImportError as exc:
            raise ValueError('Random Forest y SVR requieren scikit-learn. En la edición ligera usa Ridge o Lasso; consulta el README para instalar la edición completa.') from exc
        def build():
            if choice=='Random Forest':
                return RandomForestRegressor(n_estimators=100,max_depth=4,random_state=42,n_jobs=1)
            return TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(),SVR(kernel='rbf',C=10,epsilon=.1)),
                transformer=StandardScaler())
        model=build()
        model.fit(x[:cut],y[:cut])
        test_pred=model.predict(x[cut:])
        model=build()
        model.fit(x,y)
        future_values=model.predict(future_x)
    else:
        raise ValueError('Modelo desconocido.')
    if not (np.all(np.isfinite(test_pred)) and np.all(np.isfinite(future_values))):
        raise ValueError('El modelo produjo valores no finitos.')
    mae=float(np.mean(np.abs(y[cut:]-test_pred)))
    rmse=float(np.sqrt(np.mean((y[cut:]-test_pred)**2)))
    future_values=np.maximum(0,future_values)
    forecast=pd.DataFrame({'anio':future_years,'proyeccion':future_values})
    evaluation=pd.DataFrame({'anio':years[cut:],'real':y[cut:],'estimado':test_pred})
    return forecast,evaluation,mae,rmse

def extra_panels(end_year: int) -> tuple[pd.DataFrame, dict]:
    """Carga exploratoria explícita y bajo demanda."""
    parts = []
    failures = {}
    for key in SOURCES:
        try:
            raw = load_indicator(key, end_year)
            out = aggregate_indicator(raw)
            out = out[["codigo_departamento", "departamento", "anio", "valor"]].copy()
            out["variable"] = key
            parts.append(out)
        except (requests.RequestException, RuntimeError, ValueError, KeyError, ConnectionError) as exc:
            failures[key] = str(exc)
    if not parts:
        raise ValueError("No fue posible descargar ninguno de los indicadores.")
    return pd.concat(parts, ignore_index=True), failures


def _kmeans_numpy(matrix: np.ndarray, k: int, seed: int=42, restarts: int=10) -> np.ndarray:
    """K-means pequeño para PCA departamental; usa NumPy y reinicios reproducibles."""
    mat=np.asarray(matrix,dtype=float)
    rng=np.random.default_rng(seed)
    best_labels=None
    best_inertia=float('inf')
    for _ in range(restarts):
        centers=mat[rng.choice(len(mat),size=k,replace=False)].copy()
        for __ in range(100):
            d2=((mat[:,None,:]-centers[None,:,:])**2).sum(axis=2)
            labels=np.argmin(d2,axis=1)
            updated=np.vstack([mat[labels==j].mean(axis=0) if np.any(labels==j)
                               else mat[rng.integers(len(mat))] for j in range(k)])
            if np.allclose(updated,centers,atol=1e-7):
                centers=updated
                break
            centers=updated
        d2=((mat[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        labels=np.argmin(d2,axis=1)
        inertia=float(np.sum(np.min(d2,axis=1)))
        if inertia<best_inertia:
            best_labels=labels.copy();best_inertia=inertia
    return best_labels


def pca_panel(panel: pd.DataFrame):
    complete=panel.pivot_table(index=['codigo_departamento','departamento','anio'],
                               columns='variable',values='valor',aggfunc='sum')
    variables=[key for key in SOURCES if key in complete.columns]
    if len(variables)<3:
        st.info('El PCA requiere las tres variables cargadas.')
        return
    complete=complete[variables].dropna().reset_index()
    available=sorted(complete['anio'].unique().tolist())
    if not available:
        st.info('Las fuentes no comparten años y territorios suficientes para PCA.')
        return
    year=st.select_slider('Año para análisis transversal',options=available,value=available[-1])
    sub=complete[complete['anio']==year].copy()
    if len(sub)<8:
        st.info('Hay menos de 8 departamentos con datos conjuntos para ese año.')
        return
    raw_mat=sub[variables].to_numpy(dtype=float)
    means=raw_mat.mean(axis=0)
    stds=raw_mat.std(axis=0)
    if np.any(stds<1e-12):
        st.warning('Una variable es constante para el año elegido; cambia el año de PCA.')
        return
    mat=(raw_mat-means)/stds
    u, singular, vt=np.linalg.svd(mat,full_matrices=False)
    xy=mat@vt[:2,:].T
    sub['PC1'],sub['PC2']=xy[:,0],xy[:,1]
    fractions=singular**2 / np.sum(singular**2)
    k=st.slider('Número de grupos exploratorios',min_value=2,max_value=min(6,len(sub)-1),value=3)
    sub['grupo']=(_kmeans_numpy(mat,k)+1).astype(str)
    fig=px.scatter(sub,x='PC1',y='PC2',color='grupo',hover_name='departamento',
                   title=f'PCA + agrupación descriptiva · {year}',labels={'grupo':'Grupo'})
    fig.update_traces(marker=dict(size=12,opacity=.87))
    show_plot(chart_style(fig))
    st.caption('Varianza explicada por PC1: '+fmt_number(fractions[0]*100)+
               '% · PC2: '+fmt_number(fractions[1]*100)+
               '%. PCA por SVD y k-means implementados con NumPy. Los grupos no implican causalidad.')
    csv_download(sub,'observatorio_pca_departamentos.csv')


def _numpy_var1_forecast(levels: np.ndarray, steps: int=3) -> tuple[np.ndarray,float,float]:
    """VAR(1) en primeras diferencias, con intercepto, ajustado por MCO multivariado.

    AIC por observación aproximado, no compararlo directamente con software distinto.
    Retorna proyección en niveles, criterio de información y radio espectral.
    """
    levels=np.asarray(levels,dtype=float)
    if levels.ndim!=2 or levels.shape[0]<12 or levels.shape[1]<2:
        raise ValueError('VAR exige 12 años y al menos dos variables.')
    delta=np.diff(levels,axis=0)
    response=delta[1:]
    X=np.column_stack([np.ones(len(delta)-1),delta[:-1]])
    if np.linalg.matrix_rank(X)<X.shape[1]:
        raise ValueError('Matriz de regresores no tiene rango completo.')
    beta=np.linalg.lstsq(X,response,rcond=None)[0]
    residual=response-X@beta
    n=len(response)
    p=response.shape[1]
    covariance=residual.T@residual/n
    sign,logdet=np.linalg.slogdet(covariance)
    if sign<=0:
        raise ValueError('Covarianza singular; VAR no identificable con estos datos.')
    aic=float(logdet+2*p*X.shape[1]/n)
    spectral_radius=float(np.max(np.abs(np.linalg.eigvals(beta[1:,:].T))))
    last=delta[-1].copy()
    forecasts=[]
    level_last=levels[-1].copy()
    for _ in range(steps):
        last=np.r_[1.0,last]@beta
        level_last=level_last+last
        forecasts.append(level_last.copy())
    return np.array(forecasts),aic,spectral_radius


def var_panel(panel: pd.DataFrame):
    st.caption('VAR(1) exploratorio sobre primeras diferencias anuales, estimación MCO con NumPy. No permite inferencia causal ni prueba cointegración.')
    departments=sorted(panel['departamento'].unique())
    target=st.selectbox('Departamento para VAR',departments,
                        index=departments.index('Nariño') if 'Nariño' in departments else 0)
    pivot=panel[panel['departamento']==target].pivot_table(index='anio',columns='variable',values='valor')
    vars_available=[k for k in SOURCES if k in pivot.columns]
    chosen=st.multiselect('Variables del VAR',vars_available,default=vars_available[:2],
                          format_func=lambda k:SOURCES[k]['nombre'])
    if len(chosen)<2:
        st.info('Selecciona al menos dos indicadores.')
        return
    time_data=pivot[chosen].dropna().sort_index()
    if len(time_data)<12 or not np.all(np.diff(time_data.index.to_numpy(dtype=int))==1):
        st.warning('Para VAR se exigen al menos 12 años consecutivos y todas las variables, sin completar huecos.')
        return
    delta=time_data.diff().dropna()
    if any(delta[k].std()==0 for k in chosen):
        st.warning('Hay alguna variable constante en diferencias; VAR no identificable.')
        return
    try:
        values,aic,radius=_numpy_var1_forecast(time_data.to_numpy(dtype=float),steps=3)
        frame=pd.DataFrame(values,columns=chosen)
        frame.insert(0,'anio',range(int(time_data.index.max())+1,int(time_data.index.max())+4))
        st.write(f'Muestra: {int(time_data.index.min())}–{int(time_data.index.max())} · Rezagos: 1 · AIC aproximado: {aic:.3f}')
        if radius>=1.0:
            st.warning('La matriz dinámica presenta radio espectral ≥ 1: el VAR estimado no es estable.')
        st.dataframe(frame.rename(columns={k:SOURCES[k]['nombre'] for k in chosen}),
                     use_container_width=True,hide_index=True)
        csv_download(frame,'observatorio_var_exploratorio.csv')
        st.warning('Escenarios estadísticos tentativos: antes de concluir se deben evaluar estacionariedad, residuos y estabilidad. No sustituye un estudio econométrico completo.')
    except (ValueError,np.linalg.LinAlgError,ZeroDivisionError) as exc:
        st.warning(f'El VAR no se pudo estimar con esta muestra: {exc}')

# -------------------------------------------------------------------
# INTERFAZ Y CONEXIÓN
# -------------------------------------------------------------------
st.sidebar.markdown("### Observatorio Nariño")
st.sidebar.caption("Universidad de Nariño · análisis territorial basado en datos públicos")
page = st.sidebar.radio("Módulo", PAGES)
key = st.sidebar.selectbox("Indicador", list(SOURCES), format_func=lambda k:SOURCES[k]["nombre"])
end_year = datetime.now(timezone.utc).year - 1

with st.sidebar:
    if st.button("🔄 Reiniciar la consulta", use_container_width=True):
        load_indicator.clear()
        st.session_state.pop("active_dataset", None)
        st.session_state.pop("lab_panel", None)
        st.session_state.pop("lab_failures", None)
        st.rerun()

# Renderizar la estructura de la página ANTES de realizar cualquier solicitud a Internet.
# Evita la apariencia de una página bloqueada si el servidor público es lento.
st.title("Observatorio Económico y Social")
st.markdown('<p class="tagline">Nariño y Colombia · visualización, seguimiento territorial y análisis reproducible</p>', unsafe_allow_html=True)
st.caption("Universidad de Nariño · Laura Valentina Castañeda Arroyo y Edgar Fernando Domínguez")

if st.session_state.get("active_dataset") != key:
    st.subheader("Panel de consulta de indicadores")
    st.write("La interfaz ya está funcionando. Para cuidar los recursos de Streamlit Cloud, "
             "la descarga de datos reales comienza cuando eliges **Cargar indicador**.")
    st.info(f"**Fuente seleccionada:** {SOURCES[key]['nombre']} · {SOURCES[key]['organismo']}")
    st.markdown(f"[Consultar datos y metodología de la fuente]({SOURCES[key]['url']})")
    if st.button("▶️ Cargar indicador real", type="primary", use_container_width=True):
        st.session_state["active_dataset"] = key
        st.rerun()
    st.markdown("""
    **Módulos disponibles:** panorama territorial, series, EDA,
    predicción, PCA, agrupación y VAR exploratorio. Las cifras se consultan en el
    portal público; no se sustituyen datos faltantes por números simulados.
    """)
    st.stop()

try:
    with st.spinner("Conectando con Datos Abiertos Colombia y validando registros..."):
        raw = load_indicator(key, end_year)
except Exception as exc:
    # En el límite de la API externa es preferible hacer visible el fallo que
    # dejar al visitante con una pantalla sin información.
    st.session_state.pop("active_dataset", None)
    st.error("No fue posible obtener este indicador desde el portal público.")
    st.write("La aplicación está operativa, pero el servidor de datos puede estar temporalmente inaccesible.")
    st.markdown(f"**Fuente:** [{SOURCES[key]['nombre']}]({SOURCES[key]['url']})")
    with st.expander("Detalle del error (para diagnóstico técnico)"):
        st.exception(exc)
    st.info("Prueba otra fuente, o revisa Manage app → Logs en Streamlit Cloud. "
            "Puedes reintentar desde el panel de consulta.")
    st.stop()

activity = None
if key == "economia":
    activities = sorted(raw["actividad"].dropna().unique().tolist())
    activity = st.sidebar.selectbox("Actividad económica", [ACTIVITY_ALL, *activities])
    st.sidebar.caption("La suma de actividades es exploratoria, no el PIB oficial.")

try:
    all_data = aggregate_indicator(raw, activity)
except (ValueError, KeyError, TypeError) as exc:
    st.error("La fuente devolvió una estructura inesperada y no se puede graficar con seguridad.")
    st.exception(exc)
    st.stop()
if all_data.empty:
    st.warning("La selección no tiene observaciones.")
    st.stop()

dep_options = [NACIONAL, "Nariño"] + [n for n in sorted(DEP_NAMES) if n != "Nariño"]
department = st.sidebar.selectbox("Territorio de interés", dep_options, index=1)
years = sorted(int(y) for y in all_data["anio"].unique())
selected_year = st.sidebar.select_slider("Año del mapa y del comparador", options=years, value=years[-1])
info = SOURCES[key]
unit = info["unidad"]
series = to_series(all_data, department)
selected = all_data[all_data["anio"] == selected_year]
if series.empty:
    st.warning(f"No existen registros válidos para {department} en el indicador seleccionado. Prueba otro territorio.")
    st.stop()
latest = series.iloc[-1]
previous = series[series["anio"] < int(latest["anio"])]
change = None
if len(previous) and float(previous.iloc[-1]["valor"]) != 0:
    change = 100*(float(latest["valor"])/float(previous.iloc[-1]["valor"])-1)

if key == "economia":
    st.warning(info["advertencia"])
else:
    st.caption(info["advertencia"])

if page == PAGES[0]:
    st.subheader("Panorama territorial")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Último dato observado", fmt_number(float(latest["valor"])), help=unit)
    c2.metric("Último año", str(int(latest["anio"])))
    c3.metric("Variación respecto al registro previo",
              f"{fmt_number(change)}%" if change is not None else "Sin base comparable")
    c4.metric("Territorios con registro (año elegido)", f"{selected['codigo_departamento'].nunique()} / 33")
    left, right = st.columns([1.2,1])
    with left:
        plot_series(series, unit, department)
    with right:
        ranking = selected.sort_values("valor", ascending=False).head(12)
        fig = px.bar(ranking.sort_values("valor"), x="valor", y="departamento", orientation="h",
                     labels={"valor":unit,"departamento":""}, title=f"12 mayores valores · {selected_year}")
        show_plot(chart_style(fig))
    st.caption(f"Indicador: {info['nombre']} · Unidad: {unit} · Fuente: {info['organismo']}.")

elif page == PAGES[1]:
    st.subheader("Series temporales y comparación")
    plot_series(series, unit, department)
    candidates = [n for n in ["Nariño","Cauca","Putumayo","Valle del Cauca","Bogotá D.C."] if n in set(all_data["departamento"])]
    departments = st.multiselect("Comparar departamentos", sorted(DEP_NAMES), default=candidates[:4])
    if departments:
        compare = all_data[all_data["departamento"].isin(departments)]
        fig = px.line(compare, x="anio", y="valor", color="departamento", markers=True,
                      title="Trayectorias comparadas", labels={"anio":"Año", "valor":unit,"departamento":"Territorio"})
        show_plot(chart_style(fig, 460))
        csv_download(compare, f"observatorio_comparacion_{key}.csv")

elif page == PAGES[2]:
    st.subheader("Análisis exploratorio de datos (EDA)")
    overview = panel_overview(all_data)
    a,b,c,d = st.columns(4)
    a.metric("Territorios observados", str(overview["departamentos"]))
    b.metric("Años distintos", str(overview["anios"]))
    c.metric("Primer año", str(overview["primer_anio"]))
    d.metric("Último año", str(overview["ultimo_anio"]))
    l,r = st.columns(2)
    with l:
        fig = px.histogram(selected, x="valor", nbins=14,
                           title=f"Distribución entre departamentos · {selected_year}",
                           labels={"valor":unit})
        show_plot(chart_style(fig))
    with r:
        fig = px.box(all_data, x="anio", y="valor", points=False,
                     title="Dispersión territorial por año", labels={"anio":"Año","valor":unit})
        show_plot(chart_style(fig))
    st.markdown("**Resumen descriptivo por año elegido**")
    desc = selected["valor"].describe().to_frame("Valor")
    st.dataframe(desc, use_container_width=True)
    st.caption("Los años/territorios ausentes representan datos no observados, no valores iguales a cero. La distribución muestra valores absolutos y no tasas poblacionales.")
    csv_download(all_data, f"observatorio_eda_{key}.csv")

elif page == PAGES[3]:
    st.subheader("Predicción y aprendizaje automático")
    st.write("Proyecciones exploratorias de series anuales para el territorio seleccionado, con evaluación cronológica fuera de muestra.")
    try:
        import sklearn  # Opcional: no se instala en el perfil mínimo de Cloud.
        algorithms = ["Ridge", "Lasso", "Random Forest", "SVR"]
    except ImportError:
        algorithms = ["Ridge", "Lasso"]
        st.caption("Modo ligero: Ridge y Lasso funcionan con NumPy. Random Forest y SVR requieren instalar scikit-learn opcionalmente.")
    model_name = st.selectbox("Algoritmo", algorithms)
    horizon = st.slider("Años a proyectar",1,5,3)
    if st.button("Entrenar y evaluar modelo", type="primary"):
        try:
            pred, ev, mae, rmse = compute_prediction(series,model_name,horizon)
            c1,c2 = st.columns(2)
            c1.metric("MAE fuera de muestra",fmt_number(mae))
            c2.metric("RMSE fuera de muestra",fmt_number(rmse))
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=series["anio"],y=series["valor"],name="Observado",mode="lines+markers"))
            fig.add_trace(go.Scatter(x=ev["anio"],y=ev["estimado"],name="Predicción prueba",mode="lines+markers"))
            fig.add_trace(go.Scatter(x=pred["anio"],y=pred["proyeccion"],name="Proyección",mode="lines+markers",line=dict(dash="dash")))
            fig.update_layout(title=f"{model_name}: entrenamiento, evaluación y proyección",xaxis_title="Año",yaxis_title=unit)
            show_plot(chart_style(fig,480))
            st.dataframe(pred, use_container_width=True, hide_index=True)
            csv_download(pred, f"observatorio_proyeccion_{key}.csv")
            st.warning("No es una predicción causal ni un pronóstico oficial; se basa solo en el tiempo. Random Forest y SVR pueden no extrapolar tendencias de largo plazo. Valores negativos proyectados se acotan a cero.")
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            st.warning(f"No hay condiciones suficientes para modelar esta serie: {exc}")

elif page == PAGES[4]:
    st.subheader("Laboratorio estadístico y econométrico")
    st.write("PCA, agrupamiento y VAR(1) usan NumPy y requieren combinar las fuentes. La descarga adicional se realiza solo cuando se solicita.")
    if st.button("Cargar indicadores adicionales", type="primary"):
        with st.spinner("Agregando fuentes departamentales para el laboratorio..."):
            try:
                panel, failures = extra_panels(end_year)
                st.session_state["lab_panel"] = panel
                st.session_state["lab_failures"] = failures
            except Exception as exc:
                st.error("No se pudieron cargar las fuentes para el laboratorio.")
                with st.expander("Detalle técnico"):
                    st.exception(exc)
    if "lab_panel" in st.session_state:
        panel=st.session_state["lab_panel"]
        errors=st.session_state.get("lab_failures", {})
        if errors:
            st.warning("Algunas fuentes no pudieron cargarse: " + ", ".join(errors))
            with st.expander("Errores de consulta"):
                st.json(errors)
        tab1,tab2=st.tabs(["Componentes principales (PCA)","VAR en diferencias"])
        with tab1:
            try:
                pca_panel(panel)
            except (ValueError, np.linalg.LinAlgError) as exc:
                st.warning(str(exc))
        with tab2:
            try:
                var_panel(panel)
            except (ValueError, np.linalg.LinAlgError) as exc:
                st.warning(str(exc))
    else:
        st.info("Pulsa «Cargar indicadores adicionales» para habilitar los análisis multivariados.")
    st.caption("SAR, SEM y VECM no se estiman automáticamente aquí: requieren especificaciones y diagnósticos adicionales para producir resultados defendibles.")

else:
    st.subheader("Datos, trazabilidad y metodología")
    st.markdown(f"**Indicador activo:** {info['nombre']}")
    st.write(f"**Organismo:** {info['organismo']}  |  **Frecuencia:** {info['frecuencia']}  |  **Unidad:** {unit}")
    st.markdown(f"**Enlace a la fuente original:** {info['url']}")
    st.info(info["advertencia"])
    st.markdown("""
**Tratamiento aplicado:**
- Consulta directa a la API Socrata de Datos Abiertos Colombia, con reintentos limitados.
- Homicidios y muertes de tránsito: agregación de `cantidad` en servidor por código DIVIPOLA y año.
- Actividad económica: exclusivamente observaciones marcadas como precios constantes de 2015;
  sumatoria exploratoria por actividad, departamento y año cuando se seleccionan todas las actividades.
- Se excluye el año calendario vigente porque está incompleto; no se imputan series faltantes.
- Indicadores de conteos absolutos: sus magnitudes no son comparables a tasas de población.
- Predicción: división de entrenamiento/test temporal 80/20; métricas MAE y RMSE.
- VAR: primeras diferencias de series completas, sin convertir automáticamente sus resultados en evidencia causal.
    """)
    csv_download(all_data,f"observatorio_{key}_departamentos.csv")
    with st.expander("Vista previa de datos normalizados"):
        st.dataframe(all_data.head(200),use_container_width=True,hide_index=True)

st.divider()
st.caption("Observatorio Económico y Social · Universidad de Nariño · Laura Valentina Castañeda Arroyo y Edgar Fernando Domínguez · Datos oficiales o públicos identificados en cada fuente. Proyecto académico, no publicación estadística oficial.")