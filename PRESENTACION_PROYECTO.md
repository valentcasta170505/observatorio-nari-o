# Observatorio Económico y Social
## Plataforma para el análisis económico, social y espacial de Nariño y los departamentos de Colombia

**Área:** Seminario de Investigación — Economía.  
**Alcance territorial:** Nariño (énfasis); 32 departamentos y Bogotá, D. C. (comparación).  
**Tipo de producto:** aplicación web científica con actualización de datos, análisis cuantitativo, tableros territoriales y modelos predictivos.

### Resumen ejecutivo

El Observatorio Económico y Social es una aplicación de análisis territorial que permite consultar series públicas, comparar los departamentos de Colombia y examinar relaciones temporales y espaciales de indicadores socioeconómicos. A diferencia de un tablero meramente descriptivo, la plataforma integra herramientas de análisis exploratorio, segmentación, reducción dimensional, econometría multivariante, métodos espaciales, aprendizaje automático e interpretabilidad. La propuesta se centra en Nariño como departamento de referencia sin impedir comparaciones con los demás territorios. Su principio operativo es conservar la trazabilidad de cada observación y abstenerse de ejecutar modelos cuando los datos no son suficientes o no cumplen los supuestos de preparación.

### Justificación

El acceso disperso a indicadores económicos, demográficos y sociales dificulta la comparación regional. Además, los mapas y algoritmos de predicción pueden generar conclusiones metodológicamente débiles cuando mezclan escalas, omiten información faltante o ignoran dependencia espacial. Un observatorio reproducible permite sistematizar fuentes, diagnosticar calidad y convertir datos públicos en evidencia susceptible de evaluación.

### Objetivo general

Diseñar e implementar una plataforma interactiva que integre datos socioeconómicos departamentales y aplique herramientas estadísticas, econométricas y computacionales para identificar patrones, diferencias territoriales, dinámicas temporales y escenarios predictivos con enfoque en Nariño.

### Objetivos específicos

1. Unificar fuentes documentadas en un panel departamento–año–indicador, validando codificación, unidad, cobertura y consistencia.
2. Describir distribuciones, tendencias, correlaciones y heterogeneidad territorial mediante EDA y visualización cartográfica.
3. Identificar estructuras comunes mediante clustering y componentes principales.
4. Explorar relaciones dinámicas mediante VAR/VECM y dependencia espacial mediante Moran, LISA, SAR y SEM.
5. Comparar algoritmos predictivos, regularización, SVR y redes neuronales a partir de pruebas temporales fuera de muestra.
6. Construir escenarios condicionados y reportes de interpretabilidad, distinguiendo evidencia asociativa de causalidad.

### Pregunta central

¿Cómo se distribuyen y evolucionan los indicadores económicos y sociales en Nariño respecto de los demás departamentos de Colombia, y qué patrones y escenarios pueden identificarse a partir de métodos estadísticos, econométricos, espaciales y de aprendizaje automático?

### Componente de diseño

**Interfaz:** cabecera de identidad, KPIs, filtros territoriales en barra lateral, visualizaciones interactivas, resultados descargables, panel de fuentes y advertencias visibles. **Interacción:** elegir departamento, período, variables, algoritmo, pesos espaciales y horizonte. **Reproducibilidad:** consultas públicas identificadas, registro de errores, configuración visible y exportación de muestras.

### Metodología científica y precauciones

- **EDA:** medidas de posición, dispersión, outliers, matrices Pearson y gráficos comparativos.
- **Minería y PCA:** estandarización previa y datos completos por observación; evaluación de clusters con silhouette.
- **ML:** validación por años, sin entrenar con registros de años posteriores a los datos de prueba. MAE, RMSE, R² y permutation importance.
- **VAR/VECM:** decisiones condicionadas al diagnóstico ADF y Johansen, muestra suficiente, IRF descriptivas de forma reducida.
- **Espacial:** cartografía poligonal y definición explícita de W por vecindad; autocorrelación Moran/LISA; estimaciones SAR/SEM exploratorias.
- **Deep learning y predicción:** red MLP multicapas con entradas rezagadas y comparación con persistencia; suavizamiento exponencial seleccionado por validación.
- **Escenarios:** modificación porcentual persistente de un predictor, con modelo autoregresivo anual en un entorno asociativo. El resultado no demuestra el impacto causal de una intervención pública.

### Resultados que entrega el software

Gráficos de tendencias, ranking departamental, distribuciones, correlación, registros extremos, agrupaciones, cargas PCA, métricas predictivas, importancia de variables, pronósticos anuales, simulaciones base vs choque, mapas coropléticos, Moran global y local, tablas SAR/SEM, y bases descargables con atributos de procedencia.

### Limitaciones y agenda de ampliación

Los tres conjuntos conectados automáticamente son una **base inicial**, no una representación exhaustiva de bienestar. Para estudiar pobreza, empleo, informalidad, educación y PIB oficial DANE, incorporar archivos fuente concretos con identificación metodológica; no atribuir a las bases disponibles variables que no contienen. El número de períodos y la desagregación condicionan la inferencia. Una fase futura puede incluir un registro de versiones de datos y procesos de evaluación causal con diseño de identificación.

### Conclusión

El producto combina una experiencia visual usable con principios de análisis reproducible. Su utilidad académica proviene tanto de los modelos disponibles como de la claridad con la que presenta **qué información existe, de dónde proviene, qué estimación es válida y qué conclusiones no es posible sostener**.
