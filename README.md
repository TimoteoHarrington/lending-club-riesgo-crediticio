# Modelo de riesgo de default crediticio — Lending Club

**Ciencia de Datos Aplicada · ITBA**
Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim

Queremos estimar la probabilidad de que una persona no devuelva un préstamo (PD), usando solo lo que se sabe cuando lo pide. Con esa probabilidad armamos bandas de riesgo, calculamos la prima de riesgo (cuánto se espera perder) y decidimos si aprobar el préstamo. Usamos los datos históricos de Lending Club, una plataforma de préstamos entre personas de Estados Unidos.

**Probalo:** la interfaz para analistas está en _[link de Streamlit]_ y la API, con su documentación, en _[link de Render]/docs_.

## Estado del proyecto

| Entrega | Contenido | Estado |
|---|---|---|
| E01 — Propuesta | Problema, objetivos y alcance | ✅ Entregado |
| E02 — Recopilación y preparación de datos | Datos, calidad, leakage, separación train/test, exploración, pérdida y transformaciones | ✅ Entregado |
| E03 — Modelado | Regresión logística, Random Forest y LightGBM, validación por año, evaluación, bandas y modelo guardado | ✅ Entregado |
| E04 — Despliegue y presentación | Calibración, severidad por banda, prima, regla de decisión, API e interfaz | ✅ Este repo |

## Qué hay en cada carpeta

```
├── api/             # API REST (FastAPI): recibe una solicitud y devuelve PD, banda, prima y decisión
├── app/             # Interfaz para analistas (Streamlit); llama a la API
├── data/            # Datos (no se suben a GitHub; se descargan con el script)
├── datos_externos/  # Desempleo mensual por estado (BLS), sí está en el repo
├── figures/         # Gráficos que generan los notebooks
├── modelos/         # Modelo guardado (E03) y parámetros de la prima (E04)
├── notebooks/
│   ├── 02_recopilacion_preparacion_datos.ipynb   # Entrega 02
│   ├── 03_modelado.ipynb                         # Entrega 03
│   └── 04_pricing.ipynb                          # Entrega 04: calibración, severidad, prima y decisión
├── scripts/
│   ├── download_data.py         # Descarga el dataset de Kaggle
│   └── download_desempleo.py    # Vuelve a bajar el desempleo del BLS
├── src/
│   ├── datos.py       # Carga los datos y se queda con los préstamos terminados
│   ├── features.py    # Clasifica las columnas, crea variables y calcula la pérdida
│   ├── modelado.py    # Desempleo, modelos, validación por año, guardar/cargar el modelo
│   ├── evaluacion.py  # Métricas (AUC, KS, calibración) y tablas por banda
│   ├── graficos.py    # Colores y estilo de los gráficos
│   ├── solicitud.py   # De una solicitud corta a las 54 variables del modelo (E04)
│   ├── pricing.py     # Calibración, banda, severidad, prima y decisión (E04)
│   └── evaluador.py   # Une modelo y parámetros; lo usan la API y los tests (E04)
├── tests/             # Tests de persistencia, lógica de la prima y API
├── render.yaml        # Despliegue de la API en Render
├── requirements.txt            # Para los notebooks
├── requirements-inference.txt  # Para cargar el modelo
├── requirements-api.txt        # Para la API
└── README.md
```

## Cómo verlo

- **Sin instalar nada:** abrí los notebooks acá en GitHub. Están guardados con todas sus tablas y gráficos.
- **En Google Colab:** abrí el notebook en Colab y ejecutá todo. La primera celda baja este repositorio y te pide un token de Kaggle para descargar los datos (en kaggle.com → Settings → API). El notebook 03 necesita los archivos que genera el 02; si no están, su primera celda ejecuta el 02 sola (~10 minutos más). El 03 tarda ~15 minutos.
- **En tu computadora:**

```bash
git clone https://github.com/lucaskim26/lending-club-riesgo-crediticio
cd lending-club-riesgo-crediticio
pip install -r requirements.txt
python scripts/download_data.py      # necesita el token de Kaggle
jupyter notebook notebooks/02_recopilacion_preparacion_datos.ipynb   # después, 03_modelado.ipynb
```

La primera vez tarda unos minutos y usa ~5 GB de memoria, porque lee el archivo completo. Después guarda una copia en `data/` y las siguientes veces es rápido.

## Usar el modelo: API e interfaz (E04)

```
Analista ──> Interfaz (Streamlit) ──HTTP──> API (FastAPI) ──> evaluador ──> modelo_pd.joblib + modelo_pd.json
```

**Qué hace con una solicitud:**
1. Arma las 54 variables del modelo con los datos principales (monto, ingreso, DTI, FICO, antigüedad, vivienda, destino y estado). Los datos del buró que no vengan se completan con la mediana de train.
2. Calcula la PD con LightGBM y la **calibra** con un ajuste de intercepto (+0,098 en log-odds), estimado en validación.
3. Ubica el préstamo en una de las **7 bandas** de riesgo.
4. Calcula la **prima = PD calibrada × severidad de la banda** (de 45,9% en la banda 1 a 60,2% en la 7). Es la pérdida esperada de toda la vida del préstamo, como % del monto.
5. **Decide:** aprueba las bandas 1 a 4, manda a revisión la 5 y rechaza la 6 y la 7.

**Endpoints de la API** (documentación interactiva en `/docs`):

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/salud` | Estado del servicio |
| GET | `/modelo` | Métricas, bandas con su severidad y decisión, calibración |
| GET | `/opciones` | Valores válidos para armar un formulario |
| POST | `/evaluar` | Evalúa una solicitud |
| POST | `/evaluar/lote` | Evalúa hasta 1000 solicitudes |

Ejemplo:

```bash
curl -X POST https://<api>/evaluar -H "Content-Type: application/json" -d '{
  "monto": 15000, "ingreso_anual": 65000, "dti": 19, "fico": 695,
  "antiguedad_crediticia_anios": 12, "antiguedad_laboral_anios": 4,
  "vivienda": "RENT", "proposito": "debt_consolidation", "estado": "CA"}'
```

Devuelve, entre otros, `"pd": 0.117`, `"banda": "2"`, `"severidad": 0.483`, `"prima": 0.0565`, `"perdida_esperada_usd": 847.74` y `"decision": "aprobar"`. Si un dato es inválido responde 422 con el campo y el motivo en castellano (por ejemplo, `fico: Tiene que ser menor o igual a 850`).

**Correr en local** (dos terminales):

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload                    # API en http://localhost:8000/docs

pip install -r app/requirements.txt
streamlit run app/streamlit_app.py               # interfaz en http://localhost:8501
```

**Desplegar:**
- **API en Render:** en render.com, *New > Blueprint* y elegir este repo. Toma la configuración de `render.yaml`. En el plan gratis el servicio se duerme a los 15 minutos sin uso y tarda ~1 minuto en despertar: abrir `/salud` antes de la demo.
- **Interfaz en Streamlit Community Cloud:** en share.streamlit.io, *Create app*, este repo, archivo `app/streamlit_app.py`. En *Advanced settings > Secrets* poner `API_URL = "https://<url-de-render>"`.

**Tests** (no necesitan el dataset):

```bash
pip install -r requirements-api.txt
python -m unittest discover -s tests -v
```

## Datos

[*Lending Club Loan Data*](https://www.kaggle.com/datasets/wordsforthewise/lending-club) en Kaggle: todos los préstamos que dio Lending Club entre 2007 y 2018 (2,26 millones de préstamos, 151 columnas).

## Cómo lo hicimos (resumen)

1. **Qué predecimos:** si el préstamo se dio por perdido (`Charged Off` = 1) o se pagó (`Fully Paid` = 0).
2. **Qué préstamos usamos:** solo los que ya terminaron, a 36 meses, de 2007 a 2015 (618 mil). En los años recientes muchos préstamos seguían abiertos y el default aparecía inflado.
3. **Qué columnas usamos:** solo las que existen al pedir el préstamo. Sacamos las que aparecen después (leakage) y las que son decisiones de Lending Club, como su nota (`grade`) o si verificó el ingreso.
4. **Train y test:** separados por fecha (train hasta agosto de 2015, test de septiembre a diciembre de 2015), antes de explorar los datos.
5. **Pérdida:** cuando alguien no paga se pierde ~89% de lo que debía, que es el 51,7% del monto prestado.
6. **Transformaciones:** variables nuevas, recorte de valores extremos, relleno de vacíos, misma escala y categorías pasadas a columnas 0/1. Todo lo que aprende de los datos se ajusta solo con train.
7. **Resultado final del sistema:** una pérdida esperada por persona, PD × 51,7%. Es la pérdida de toda la vida del préstamo, no una tasa de interés.
8. **Modelos (E03):** regresión logística, Random Forest y LightGBM, elegidos con validación por año dentro de train (2013, 2014 y 2015). Ganó **LightGBM**: en test tiene ROC-AUC 0,704, contra 0,683 de la nota de Lending Club en los mismos préstamos.
9. **Desempleo por estado (BLS):** su aporte es despreciable: quitarlo reduce el AUC de validación de 0,6837 a 0,6836. Quedó afuera para simplificar, no porque perjudique el modelo.
10. **Bandas:** 7 bandas con cortes definidos en train; en test el default sube de 4% (banda 1) a 57% (banda 7).
11. **Modelo guardado:** `modelos/modelo_pd.joblib` (se carga con `src.modelado.cargar_modelo()`) y `modelos/modelo_pd.json` con los cortes, la severidad y las métricas.
12. **Calibración (E04):** la PD de E03 quedaba un punto abajo del default real en test (13,9% contra 14,8%). Con un ajuste de intercepto estimado en el último año de validación, la PD media pasa a 15,0%. El AUC no cambia.
13. **Prima (E04):** PD calibrada × severidad de la banda. En test cubre el 99% de la pérdida real (la versión de E03, el 92%), y entre 94% y 117% en cada banda.
14. **Decisión (E04):** aprobando las bandas 1 a 4 (94% de los préstamos), la pérdida de los aprobados baja de 7,8% a 6,9% del monto. Con la misma cantidad de aprobados, la nota de Lending Club llega a 7,0%.

## Verificación sin entrenar

El modelo guardado recibe **54 variables ya construidas** y las transforma en **102 columnas** para LightGBM. Se puede probar la carga e inferencia con cinco casos sintéticos sin descargar el dataset ni ejecutar los notebooks:

```bash
python -m pip install -r requirements-inference.txt
python -m unittest discover -s tests -v
```

Usá un entorno aislado. La [guía de verificación de E03](docs/verificacion_e03.md) detalla las versiones originales y verificadas, el contrato de entrada, las pruebas y sus límites.

**Alcance:** la evaluación es retrospectiva, con separación por fecha de originación; no reconstruye qué etiquetas estaban disponibles en cada fecha histórica. La selección de variables de E02 se hizo sobre todo train antes de los folds, aunque los parámetros del pipeline sí se ajustan dentro de cada corte. Test se mantuvo separado. Estas aclaraciones y la validación de pérdidas faltantes no reentrenan el modelo ni cambian los resultados guardados.

## Autores

Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim — ITBA, 2026
