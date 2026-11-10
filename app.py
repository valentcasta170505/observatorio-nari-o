"""Observatorio Económico y Social | Nariño y 33 territorios de Colombia.

Streamlit Cloud: streamlit run app.py
Datos reales. Sin valores de ejemplo ni relleno de observaciones no disponibles.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from core import (
    CODE_TO_NAME, DEP_NAMES, NACIONAL, SOURCES,
    aggregate_indicator, fetch_dataset,
    fetch_geojson, panel_overview, to_series,
)

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
    "🏠 Resumen", "🗺️ Mapa departamental", "📈 Series y comparación",
    "🔎 Análisis exploratorio", "🤖 Predicción y ML",
    "🧪 Econometría y PCA", "📚 Datos y metodología",
]
ACTIVITY_ALL = "Todas las actividades (suma exploratoria)"

@st.cache_data(ttl=24 * 3600, show_spinner=False, max_entries=5)
def load_indicator(key: str, end_year: int) -> pd.DataFrame:
    return fetch_dataset(key, max_year=end_year)

@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False, max_entries=1)
def load_geometry() -> dict:
    return fetch_geojson()


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


def safe_load_geometry():
    try:
        with st.spinner("Cargando cartografía de departamentos..."):
            return load_geometry()
    except (requests.RequestException, ConnectionError, ValueError, RuntimeError) as exc:
        st.warning("La cartografía pública no está disponible en este momento. Los gráficos y la tabla siguen funcionando.")
        with st.expander("Detalle de la conexión cartográfica"):
            st.code(str(exc))
        return None


def choropleth(annual: pd.DataFrame, year: int, unit: str, chosen: str):
    geo = safe_load_geometry()
    if geo is None:
        return
    data = annual[annual["anio"] == year].copy()
    if data.empty:
        st.info("No hay datos para el año seleccionado.")
        return
    fig = px.choropleth_map(
        data, geojson=geo, locations="codigo_departamento", featureidkey="id",
        color="valor", hover_name="departamento",
        hover_data={"codigo_departamento":True, "valor":":,.2f", "anio":False},
        color_continuous_scale="Blues",
        labels={"valor":unit, "codigo_departamento":"DIVIPOLA"},
        map_style="carto-positron", center={"lat":4.25,"lon":-73.8}, zoom=4,
        opacity=.78, title=f"Distribución departamental · {year}",
    )
    if chosen != NACIONAL:
        selected = data[data["departamento"] == chosen]
        if not selected.empty:
            fig.add_annotation(text=f"Territorio de interés: {chosen}", x=.01, y=.01,
                               xref="paper", yref="paper", showarrow=False,
                               bgcolor="#ffffff", bordercolor="#c8d9ee", font=dict(color="#193d68"))
    fig.update_layout(height=590, margin=dict(l=0,r=0,t=42,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", coloraxis_colorbar=dict(title=unit))
    show_plot(fig)
    present = data["codigo_departamento"].nunique()
    st.caption(f"Cobertura observada: {present} de 33 territorios. Los territorios sin registro no se sustituyen por cero. "
               "Límites administrativos descargados de un servicio geográfico público.")


def csv_download(df: pd.DataFrame, filename: str):
    st.download_button("⬇️ Descargar resultados CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv")


def compute_prediction(series: pd.DataFrame, choice: str, horizon: int):
    """Validación cronológica 80/20; sin mezcla de entrenamiento con test."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.compose import TransformedTargetRegressor

    frame = series.sort_values("anio").dropna(subset=["anio","valor"]).copy()
    if len(frame) < 8:
        raise ValueError("Se necesitan al menos 8 años observados para una evaluación temporal básica.")
    years = frame["anio"].to_numpy(dtype=int)
    if not np.all(np.diff(years) == 1):
        raise ValueError("La serie debe tener años consecutivos; no se interpolan años faltantes.")
    x = (years - years.min()).reshape(-1, 1)
    y = frame["valor"].to_numpy(dtype=float)
    cut = max(5, int(np.floor(len(y) * .8)))
    if len(y) - cut < 2:
        cut = len(y) - 2

    def build_model():
        if choice == "Ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        if choice == "Lasso":
            return make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000))
        if choice == "Random Forest":
            return RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, n_jobs=1)
        if choice == "SVR":
            return TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=.1)),
                transformer=StandardScaler(),
            )
        raise ValueError("Modelo desconocido")

    model = build_model()
    model.fit(x[:cut], y[:cut])
    test_pred = model.predict(x[cut:])
    mae = float(mean_absolute_error(y[cut:], test_pred))
    rmse = float(np.sqrt(mean_squared_error(y[cut:], test_pred)))
    # El modelo final se ajusta solo después de evaluar, para proyectar hacia adelante.
    final_model = build_model()
    final_model.fit(x, y)
    future_years = np.arange(years.max()+1, years.max()+horizon+1, dtype=int)
    future_x = (future_years-years.min()).reshape(-1, 1)
    future_values = final_model.predict(future_x)
    # Variables representadas aquí (montos o víctimas) no pueden tener valores negativos.
    future_values = np.maximum(0, future_values)
    forecast = pd.DataFrame({"anio":future_years,"proyeccion":future_values})
    evaluation = pd.DataFrame({"anio":years[cut:], "real":y[cut:], "estimado":test_pred})
    return forecast, evaluation, mae, rmse


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


def pca_panel(panel: pd.DataFrame):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    complete = panel.pivot_table(index=["codigo_departamento", "departamento", "anio"],
                                 columns="variable", values="valor", aggfunc="sum")
    variables = [key for key in SOURCES if key in complete.columns]
    if len(variables) < 3:
        st.info("El PCA requiere las tres variables cargadas.")
        return
    complete = complete[variables].dropna().reset_index()
    available = sorted(complete["anio"].unique().tolist())
    if not available:
        st.info("Las fuentes no comparten años y territorios suficientes para PCA.")
        return
    year = st.select_slider("Año para análisis transversal", options=available, value=available[-1])
    sub = complete[complete["anio"] == year].copy()
    if len(sub) < 8:
        st.info("Hay menos de 8 departamentos con datos conjuntos para ese año.")
        return
    mat = StandardScaler().fit_transform(sub[variables].to_numpy(dtype=float))
    pca = PCA(n_components=2)
    xy = pca.fit_transform(mat)
    sub["PC1"], sub["PC2"] = xy[:,0], xy[:,1]
    k = st.slider("Número de grupos exploratorios", min_value=2, max_value=min(6,len(sub)-1), value=3)
    sub["grupo"] = (KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(mat) + 1).astype(str)
    fig = px.scatter(sub, x="PC1", y="PC2", color="grupo", hover_name="departamento",
                     title=f"PCA + agrupación descriptiva · {year}",
                     labels={"grupo":"Grupo"})
    fig.update_traces(marker=dict(size=12, opacity=.87))
    show_plot(chart_style(fig))
    st.caption("Varianza explicada por PC1: " + fmt_number(pca.explained_variance_ratio_[0]*100) +
               "% · PC2: " + fmt_number(pca.explained_variance_ratio_[1]*100) +
               "%. Las variables se estandarizan antes del PCA; los grupos no implican causalidad.")
    csv_download(sub, "observatorio_pca_departamentos.csv")


def var_panel(panel: pd.DataFrame):
    from statsmodels.tsa.api import VAR
    st.caption("VAR exploratorio en diferencias anuales. No se interpreta como causalidad ni como evidencia de cointegración.")
    target = st.selectbox("Departamento para VAR", sorted(panel["departamento"].unique()),
                          index=sorted(panel["departamento"].unique()).index("Nariño")
                          if "Nariño" in set(panel["departamento"]) else 0)
    pivot = panel[panel["departamento"] == target].pivot_table(index="anio", columns="variable", values="valor")
    vars_available = [k for k in SOURCES if k in pivot.columns]
    chosen = st.multiselect("Variables del VAR", vars_available, default=vars_available[:2],
                            format_func=lambda k:SOURCES[k]["nombre"])
    if len(chosen) < 2:
        st.info("Selecciona al menos dos indicadores.")
        return
    time_data = pivot[chosen].dropna().sort_index()
    if len(time_data) < 12 or not np.all(np.diff(time_data.index.to_numpy(dtype=int)) == 1):
        st.warning("Para VAR se exigen al menos 12 años consecutivos con todas las variables, sin completar huecos.")
        return
    delta = time_data.diff().dropna()
    if any(delta[k].std() == 0 for k in chosen):
        st.warning("Hay alguna variable constante en diferencias; el VAR no es identificable.")
        return
    try:
        model = VAR(delta)
        fit = model.fit(maxlags=1, trend="c")
        fc_diff = fit.forecast(delta.to_numpy(dtype=float)[-fit.k_ar:], steps=3)
        baseline = time_data.iloc[-1].to_numpy(dtype=float)
        levels = baseline + np.cumsum(fc_diff, axis=0)
        f = pd.DataFrame(levels, columns=chosen)
        f.insert(0, "anio", range(int(time_data.index.max())+1, int(time_data.index.max())+4))
        st.write(f"Muestra: {int(time_data.index.min())}–{int(time_data.index.max())} · Rezagos: {fit.k_ar} · AIC: {fit.aic:.3f}")
        st.dataframe(f.rename(columns={k:SOURCES[k]["nombre"] for k in chosen}), use_container_width=True, hide_index=True)
        csv_download(f, "observatorio_var_exploratorio.csv")
        st.warning("Las predicciones VAR son escenarios estadísticos tentativos; se requiere diagnóstico formal de estacionariedad, residuos y estabilidad antes de usarlas para decisiones.")
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        st.warning(f"El VAR no se pudo estimar con esta muestra: {exc}")


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
        load_geometry.clear()
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
    **Módulos disponibles:** panorama territorial, mapas coropléticos, series, EDA,
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
    st.subheader("Mapa coroplético de Colombia")
    choropleth(all_data, selected_year, unit, department)
    st.dataframe(selected.sort_values("valor",ascending=False).rename(columns={
        "codigo_departamento":"DIVIPOLA", "departamento":"Departamento",
        "anio":"Año", "valor":unit}), use_container_width=True, hide_index=True)
    csv_download(selected, f"observatorio_mapa_{key}_{selected_year}.csv")

elif page == PAGES[2]:
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

elif page == PAGES[3]:
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

elif page == PAGES[4]:
    st.subheader("Predicción y aprendizaje automático")
    st.write("Proyecciones exploratorias de series anuales para el territorio seleccionado, con evaluación cronológica fuera de muestra.")
    model_name = st.selectbox("Algoritmo", ["Ridge","Lasso","Random Forest","SVR"])
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

elif page == PAGES[5]:
    st.subheader("Laboratorio estadístico y econométrico")
    st.write("PCA, agrupamiento y VAR requieren combinar las fuentes. La descarga adicional se hace solo cuando se solicita.")
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
- La cartografía se obtiene de SISPRO o DANE; las geometrías deben identificar departamentos.
- Indicadores de conteos absolutos: sus magnitudes no son comparables a tasas de población.
- Predicción: división de entrenamiento/test temporal 80/20; métricas MAE y RMSE.
- VAR: primeras diferencias de series completas, sin convertir automáticamente sus resultados en evidencia causal.
    """)
    csv_download(all_data,f"observatorio_{key}_departamentos.csv")
    with st.expander("Vista previa de datos normalizados"):
        st.dataframe(all_data.head(200),use_container_width=True,hide_index=True)

st.divider()
st.caption("Observatorio Económico y Social · Universidad de Nariño · Laura Valentina Castañeda Arroyo y Edgar Fernando Domínguez · Datos oficiales o públicos identificados en cada fuente. Proyecto académico, no publicación estadística oficial.")
