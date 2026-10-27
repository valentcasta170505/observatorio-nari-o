"""Observatorio Económico y Social · Nariño y Colombia. Streamlit Cloud entrypoint."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from motor import (SOURCES,GEO_INFO,CODE_TO_NAME, load_official,join_data,validate_upload,
                   panel_wide,coverage,eda_stats,compute_outliers,cluster,pca,supervised,
                   var_vec,forecast_univariate,deep_mlp,policy_scenario,geojson_official,spatial_analysis)

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
