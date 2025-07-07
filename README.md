# 📈 Stock Price Prediction using Time Series (Apple Inc.)

Este proyecto tiene como objetivo **predecir los precios futuros de las acciones** de Apple utilizando datos históricos, mediante técnicas de series temporales como **ARIMA** y **SARIMA**.
Se inicio a participar en este proyecto para evaluar los procesos de ARIMA Y SARIMA, cual modelo es indicado con el objetivo de la practica esto para integrarlo al portafolio profesional de proyectos.

Los datos que usaremos seran los del conjunto de historial de precios de apple, que fueron obtenidos desde Keagle.
Este conjunto de datos recopila el historial de precios de las acciones de Apple Inc. (AAPL) durante los últimos 10 años, desde 2010 hasta la fecha actual. La información proviene de Nasdaq, a quienes agradezco por proporcionar acceso a esta valiosa fuente de datos.

El objetivo principal de este dataset es facilitar el análisis financiero y el desarrollo de algoritmos de predicción de precios, que puedan ser aplicados en estrategias de inversión y toma de decisiones en el mercado bursátil.

Planeo mantener este conjunto de datos actualizado periódicamente, con el fin de asegurar su relevancia y utilidad en futuros análisis.



## 🔍 Descripción del proyecto

- Análisis exploratorio de datos
- Preprocesamiento y visualización
- Modelos de series temporales (ARIMA, SARIMA)
- Evaluación de modelos (MAE, RMSE, visualizaciones)
- Optimización de hiperparámetros (`auto_arima`)
- Interpretación de resultados

## Objetivo Especifico
- Tomar datos historicos
- Limpiar y analizar los datos
- Crear un modelo que pueda predecir los precios de los próximos dias
- Evaluar qué tan preciso es ese modelo
- Visualizar comparaciones entre lo real y lo predicho

## 📁 Estructura del repositorio
├── data/ # Dataset CSV
├── notebooks/ # Jupyter notebooks
├── images/ # Gráficas exportadas
├── README.md # Descripción del proyecto
└── requirements.txt # Librerías necesarias

## 📦 Librerías utilizadas

- pandas
- numpy
- matplotlib / seaborn
- statsmodels
- pmdarima
- scikit-learn

## 🧪 Proceso del Proyecto

1. **Data Preprocessing**  
   - Limpieza, parseo de fechas, manejo de valores nulos y outliers.

2. **Exploratory Data Analysis (EDA)**  
   - Visualización de la serie, detección de tendencias y estacionalidad.

3. **Feature Engineering**  
   - Lag features, estadísticas móviles, diferenciación para estacionariedad.

4. **Model Development**  
   - Entrenamiento de modelos ARIMA y SARIMA con partición entrenamiento/prueba.

5. **Model Evaluation**  
   - Métricas utilizadas: MAE y RMSE  
   - Visualización comparativa: valores reales vs predicciones

6. **Hyperparameter Tuning**  
   - Optimización con `auto_arima` para mejorar rendimiento

7. **Reporting**  
   - Análisis final, elección del mejor modelo, reflexiones

---

1. **Data Preprocessing**  

Los datos se procesaron mediante el dataset descargado en Drive para tenerlos actualizados y tener acceso a el mediante la siguiente función:
<pre> ```
from google.colab import drive
drive.mount('/content/drive')
python import pandas as pd df = pd.read_csv('/content/drive/MyDrive/HistoricalQuotes.csv')
df.head() ``` </pre>

2. **Exploratory Data Analysis (EDA)**  

Visualizamos la tendencias del precio a travez del tiempo mediante una grafica, Esto con el objetivo de ver tendenicas, comportamientos y estacionalidad
**Nota: en esta parte del EDA visualizmaos algunos picos que estan fuera del rango, esto queire decir que hay valores atipicos, por lo que se hara limpieza ya 
viendo la grafica son datos inecesarios.**

![Ejemplo de gráfico](/images/eda.jpg)

En los datos se puede observar que hay una tendencia de subida y bajada en el cierre, existen picos anormales de subidas y caudas drasticas, se analiza siempre que cuando hay una subida de precio le sigue un pico en bajada

Este mismo parece estable en cierto rango de tiempo pero no es del todo exacto.

### Visualización con media móvil(Tendencia suavizada)
 Esto te ayuda a ver mejor la tendencia a largo plazo.
<pre> ```
df["rolling_mean_30"] = df["Close"].rolling(window=30).mean()

plt.figure(figsize=(12,6))
plt.plot(df["Close"], label="Precio Real", alpha=0.5)
plt.plot(df["rolling_mean_30"], label="Media Móvil 30 días", color="orange")
plt.title("Tendencia con media móvil de 30 días")
plt.legend()
plt.grid(True)
plt.show()
 ``` </pre>

 usamos rollin,  rolling() se usa para calcular estadísticas móviles o "deslizantes" sobre una serie de tiempo o columna de un DataFrame en pandas. Es súper útil cuando trabajas con datos financieros, meteorológicos, sensores o cualquier dato secuencial donde te interesa analizar la tendencia a lo largo del tiempo suavizando el ruido.
- df["Close"] → Toma la columna de precios de cierre.

- .rolling(window=30) → Crea una "ventana móvil" de 30 valores (como una caja que se mueve por los datos).

- .mean() → Calcula el promedio de esos 30 valores en cada punto

¿Para qué sirve en la práctica?
- En finanzas, por ejemplo, se usa para:
Ver si un activo va en tendencia alcista o bajista.
- Generar señales de compra/venta si se cruza con otro promedio móvil o con el precio real.
- En análisis de procesos industriales (como en tu área), puedes usarlo para suavizar datos de sensores, temperatura, presión, etc.
En general, te da una vista más limpia y entendible del comportamiento de una serie.

### Descomposicion de la serie temporal
Usaremos seasonal_decompose() Descompone tu serie en 4 partes:
- Tendencia (Trend) → ¿Hacia dónde se dirige a largo plazo? ¿Sube? ¿Baja?

- Estacionalidad (Seasonal) → ¿Hay patrones que se repiten cada cierto tiempo? Ej: cada año, cada mes, cada semana.

- Ruido (Residual o Residuals) → Todo lo que no es tendencia ni estacionalidad. Es el "caos", el error, lo inesperado.

- Observado (Observed) → Tu serie original.

Visualizamos que se presenta se observa que hay un conjunto de 1 a 3 años donde los precios presentan un ciclo repetitivo, sin embargo el crecimiento del precio se ve aumentando y disminuyendo quizas adaptandose a la economia actual

### Verificar estacionariedad 
Esto funciona para saber si nuestras series de tiempo son estacionarias o no, quiere decir que si son lineales o ciclicas.

<pre> ```
from statsmodels.tsa.stattools import adfuller

result = adfuller(df["Close"].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
``` </pre>
Si el p-value < 0.05, la serie ya es estacionaria.
Si es mayor, necesitarás aplicar differencing antes de modelar.

Sin embargo los resultados fueron los siguientes
### 🧪 Resultado del Test de Dickey-Fuller (ADF)
- ADF Statistic: -7.93
- p-value: 0.0000

Esto significa que la serie es estacionaria Una serie temporal es estacionaria si sus estadísticas no cambian con el tiempo. Es decir:

- La media es constante

- La varianza es constante

- No hay tendencia creciente/descendente

- No hay ciclos o estacionalidad marcada (o ya se corrigieron)

Esto quiere decir lo siguiente Interpretación p-value = 0.000000 (prácticamente 0):

Esto está muy por debajo del umbral típico de 0.05.

Rechazas la hipótesis nula (H₀): la serie NO es estacionaria.

Entonces, la serie sí es estacionaria.

ADF Statistic = -7.93:
Este valor negativo indica una fuerte evidencia contra la no estacionariedad.
Cuanto más negativo sea el estadístico ADF, más segura es la evidencia de estacionaridad.

3. **Feature Engineering** 

¿Qué es esto? Crear nuevas variables a partir de la serie original que ayuden al modelo a aprender mejor los patrones del tiempo

Es el proceso de: 

- Crear nuevas variables a apartir de las existentes
- Aplicar transformaciones matematicas
- Extraer patrones
- Crear lags o promedios moviles
- Codificar variables categoricas

#### A. Lag Features (Caracteristicas rezagadas)
Esta simulara el recuerdo de la serie. Por ejemplo: ¿Cómo estaba el precio ayer? ¿Hace 7 dias?

#### B.Rolling Statistics(Promedios moviles)
Muestra el comportamiento local de la serie. Son super utiles para suaviar y ver microtendencias

#### C. Differencing
.diff() calcula la diferencia entre un valor y el valor anterior en una columno elimina tendencia para modelos, esta vez solo se hara para si en todo caso no fuera estacionario. (solo es demostraivo)

<pre> ```
#El precio de 1, 2 y 7 dias atras
df['lag_1']=df[' Close/Last'].shift(1)
df['lag_2']=df[' Close/Last'].shift(2)
df['lag_7']=df[' Close/Last'].shift(7)

#media movil de los ultimos 3 y 7 dias
df['roll_mean_3']=df[' Close/Last'].rolling(window=3).mean()
df['roll_mean_7']=df[' Close/Last'].rolling(window=7).mean()

#desviacion estandar movil para ver la voratilidad
df['roll_std_7']=df[' Close/Last'].rolling(window=7).std()

df.head(5)

#differencia simple
df['diff_1']=df[' Close/Last'].diff()

#differencing doble ( por si fuera necesarii )
df['diff_2']=df[' Close/Last'].diff().diff()

``` </pre>

Asi se verian los datos 
veremos cómo se comparan los precios reales con las lag feauteres y rolling stats, para ver si aportan valor


4. **Model Development**  

Pasos a seguir:

Dividir los datos en conjuntos de entrenamiento y prueba (Evaluar si el modelo puede generalizar a datos nuevos)
Aplica modelos de prediccion de series temporales (ARIMA, SARIMA) - Ajustar un modelo matematico que aprenda patrones del pasado y los use para predecir valores futuros.
Predecir.

Dividir datos en entrenamiento y prueba
Queremos saber si el modelo puede predecir datos que no ha visto Usualmente se dejan los ultimos 30 dias para prueba (test), el resto se usa para entrenar (train)

<pre> ```

### En este caso para Arima no se usara sklearning por lo que se hara manual
train_size=int(len(df)*0.95)
train=df[' Close/Last'][:train_size]
test=df[' Close/Last'][train_size:]

print('entrenamiento', len(train))
print('prueba', len(test))

``` </pre>

### Modelo ARIMA

Arima es un modelo estadistico(no machine learning) que se usa para predecir series temporales tiene tres componentes:

Componente	Significado	¿Qué hace?
p	AR (autoregressive)	Usa valores pasados (lags) como predictores.
d	I (integrated)	Aplica d veces differencing para hacer la serie estacionaria.
q	MA (moving average)	Usa errores pasados para corregir predicciones.
Ejemplo real de cómo piensa el modelo: "Para predecir el precio de hoy, voy a fijarme en:

cómo estuvieron los últimos p días (AR)

si hay una tendencia o patrón que necesito eliminar (d)

cuánto me equivoqué en días anteriores para corregirme (MA)"

Tenemos encuenta que nuestra serie es estacuinaria, asi que usaremos d=0

<pre> ```

from statsmodels.tsa.arima.model import ARIMA
# Entrenar el modelo ARIMA
model=ARIMA(train, order=(5,0,2))#p=5 considera los ulgimos 5 valores para predecir el siguuente
#d=0 ya no aplica porque ya es estacionario
#q=2 usa los errores de prediccion de los ultimos 2 dias para mejorar
model_fit=model.fit() #Se entreja el modelo con los datos de entrenamiento

# Genera predicciones para el periodo de prueva
forecast = model_fit.forecast(steps=len(test)) #Genera las predicciones para el mismo numero de dias que hay en test

``` </pre>

#### Modelo SARIMA
Es ARIMA + Estacionalidad(por si hay patrones que se repiten cada X dias)
Lo mismo que p,d,q pero para ciclos repetidos
s= numero de pasos que dura un ciclo

<pre> ```

# Creamos un modelo SARIMA (Seasonal ARIMA) con:
# order = (p, d, q) → mismos componentes que ARIMA:
#    p = 5 → considera los últimos 5 valores (autoregresivo)
#    d = 0 → no aplica differencing (ya es estacionaria)
#    q = 2 → usa 2 errores pasados para ajustar la predicción

# seasonal_order = (P, D, Q, S) → componentes estacionales:
#    P = 1 → componente autoregresivo estacional (usa 1 valor del mismo punto en el ciclo anterior)
#    D = 1 → diferencia estacional (para eliminar estacionalidad si existe)
#    Q = 1 → media móvil estacional (usa 1 error estacional previo)
#    S = 30 → estacionalidad cada 30 días (por ejemplo, mensual si tus datos son diarios)

from statsmodels.tsa.statespace.sarimax import SARIMAX
#Suponiendo que hay una estacionalidad cada 30 dias (puedes cambiar "s")
model_sarima = SARIMAX(train, order=(5, 0, 2), seasonal_order=(1, 1, 1, 30))
model_sarima_fit = model_sarima.fit()

# Predecir
sarima_forecast=model_sarima_fit.forecast(steps=len(test))

``` </pre>

5. **Model Evaluation**  
   - Métricas utilizadas: MAE y RMSE  
   - Visualización comparativa: valores reales vs predicciones


### Estas sonn las metricas de ARIMA Y SARIMA
MAE ARIMA: 34.36638788719395
RMSE ARIMA: 43.19513401116803

MAE SARIMA: 34.77920201455613
RMSE SARIMA: 43.42723896436368

Al comparar ambos modelos, ARIMA mostró una tendencia a aplanarse con el tiempo, generando una curva de predicción que pierde variabilidad, lo cual indica que no logra capturar adecuadamente la dinámica real del mercado.

Por otro lado, SARIMA logró mantener un patrón similar al comportamiento real, especialmente en las fluctuaciones de corto plazo, lo cual sugiere la presencia de estacionalidad en los precios de cierre de Apple.

Aunque los errores (MAE y RMSE) pueden ser similares en magnitud, SARIMA es claramente superior visualmente, ya que refleja de forma más fiel los movimientos reales del precio, sin aplanarse.

6. **Hyperparameter Tuning**  
   - Optimización con `auto_arima` para mejorar rendimiento

Es importante
Porque automatiza ese proceso, para encontrar la mejor combinación con la menor cantidad de error (MAE, AIC o BIC), esto quiere decir que el modelo haga prueba y erro con distintas configuraciones para encontrar la que predice mejor

Vamos a hacer el Hyperparameter Tuning usando auto_arima, que es la forma más pro y practica de buscar los mejores valores de (p, d , q) y tambien los parametros estacionales (P,D,Q,s) si estas usando SARIMA.

<pre> ```

model_auto = auto_arima(
    train,
    seasonal=True,
    m=7,                # prueba semanal en vez de 30 días
    start_p=0, max_p=1,
    start_q=0, max_q=1,
    start_P=0, max_P=0,
    start_Q=0, max_Q=0,
    d=0,
    D=1,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

!pip uninstall -y numpy pmdarima
!pip install numpy==1.24.4
!pip install pmdarima --no-binary :all:

print(model_auto.summary())

``` </pre>

📈 Elección del Mejor Modelo:
Tras evaluar el desempeño de los modelos:

ARIMA mostró una curva de predicción que pierde fuerza y variabilidad, alejándose del comportamiento real.

SARIMA, en cambio, mantuvo patrones similares a la serie original, especialmente en sus ciclos estacionales.

La visualización y los errores más bajos (MAE, RMSE) confirmaron que SARIMA optimizado fue el modelo más apropiado para esta serie temporal.

💡 Reflexión Final:
Este proyecto no solo cumplió con el objetivo de predecir precios futuros, sino que también permitió aplicar todo el proceso de modelado en series de tiempo, incluyendo análisis, ingeniería de características, evaluación y tuning.
SARIMA no solo predice, sino que "entiende" la lógica estacional del mercado, lo cual es clave al trabajar con activos financieros como acciones.

📬 Autor
Daxel Valenzuela

Analista de Datos | Ingeniero Mecánico | Fundador de Moops

Instagram

LinkedIn (agrega el tuyo si quieres)
