# AquaSense-Agent

Agente de aprendizaje automático para la detección temprana de fugas en redes
de distribución de agua. Prototipo de la **Fase 2** del proyecto.

El agente clasifica cada instante de operación como **fuga** o **operación
normal**, a partir de las presiones y los caudales que medirían los sensores de
la red, y emite una alerta simulada cuando la probabilidad supera un umbral.

---

## 1. Datos

Fuente: benchmark **LeakDB**, subconjunto **Hanoi_CMH**.

| Dato | Valor |
|---|---|
| Escenarios | 10 (`Scenario-1` … `Scenario-10`) |
| Pasos por escenario | 17 520 (año 2017, paso de 30 min) |
| Filas totales | 175 200 |
| Escenarios con fuga | 3, 5, 6, 7, 8, 10 |
| Escenarios de control | 1, 2, 4, 9 (sin fuga en todo el año) |
| Clase 1.0 (fuga) | 27 091 filas = **15,46 %** |

Estructura de cada escenario:

```
Scenario-N/
├── Demands/Node_<n>.csv    32 series   demanda aplicada por nodo
├── Pressures/Node_<n>.csv  32 series   presión en cada nodo
├── Flows/Link_<n>.csv      34 series   caudal en cada conducto
├── Leaks/                             vacío en los escenarios de control
│   ├── Leak_<n>_demand.csv            caudal de fuga realizado
│   └── Leak_<n>_info.csv              nodo, área, intervalo, tipo
├── Labels.csv             (Timestamp, Label)
├── Scenario-N_info.csv    metadatos de la simulación
└── Hanoi_CMH_Scenario-N.inp   red en formato EPANET
```

### Hallazgos verificados sobre el formato

`inspect_data.py` comprueba estas cosas sobre los archivos reales, no de
documentación externa:

1. **Todos los archivos comparten la misma rejilla temporal** (980 archivos
   verificados por hash de la columna `Timestamp`).
2. **`Label = 1` durante `[primer paso con caudal de fuga > 0, Leak End]`.**
   Coincide en los 17 520 pasos de los 6 escenarios con fuga.
3. **`Leaks/Leak_*_demand.csv` no es la etiqueta.** Para fugas de tipo
   `incipient` el caudal se corta a cero en tramos, pero la etiqueta sigue
   marcada. Ejemplo: Scenario-5 tiene 5 930 pasos marcados como fuga y solo
   2 897 con caudal > 0.
4. **`Pressures/Node_1.csv` es 0.0 siempre**: el nodo 1 es el embalse y EPANET
   no define presión ahí. Se excluye de las entradas.
5. **Las fugas no están en el `.inp`** como `EMITTERS`. Viven en `Leaks/`.
6. **Las demandas identifican unívocamente al escenario** (los 10 tienen
   series distintas) y 4 de 10 son controles. Por eso la división de datos
   **debe ser por escenario**.

---

## 2. Instalación

```powershell
python -m pip install -r requirements.txt
```

Versiones verificadas: Python 3.11.9, pandas 3.0.6, numpy 2.4.6,
scikit-learn 1.9.1, joblib 1.6.0.

`inspect_data.py` no necesita ninguna de ellas: usa solo la librería estándar,
a propósito, para poder diagnosticar el entorno antes de instalar nada.

### Datos

El dataset ocupa ~505 MB y **no se versiona** (ver `.gitignore`). Consíguelo de
LeakDB y colócalo en `data/`, o deja que el propio script lo extraiga:

```powershell
python inspect_data.py --zip 'C:\ruta\Hanoi_CMH.zip'
```

---

## 3. Arquitectura

```
main.py            Orquestador: inspect → train → evaluate → agent
├── inspect_data.py   Verifica el formato real del dataset (solo stdlib)
├── data_loader.py    Carga compartida; única fuente de verdad del formato
├── train.py          Entrena + valida + persiste modelo y umbral
├── evaluate.py       Informe oficial de métricas
└── agent.py          Ciclo percepción → decisión → acción → registro
```

| Artefacto | Contenido |
|---|---|
| `models/leak_classifier.joblib` | Modelo, umbral, variables y configuración |
| `logs/train_report.json` | Curvas de compromiso y métricas de **los dos escenarios de despliegue** |
| `logs/evaluation_report.json` | Informe de evaluación |
| `logs/agent_events.csv` | Un registro por observación procesada |
| `logs/data_inspection.json` | Informe de inspección del dataset |

---

## 4. Ejecución

```powershell
# Pipeline completo
python main.py

# Por pasos
python inspect_data.py --zip 'C:\ruta\Hanoi_CMH.zip'   # ~15 s
python train.py                                        # ~3 min
python evaluate.py                                     # ~1 s
python agent.py --escenario Scenario-10                # ~1 min

# Ejecutar un solo paso
python main.py --solo agent --escenario Scenario-7

# Pruebas
python -m pytest tests/ -v
```

Opciones útiles:

```powershell
python train.py --n-estimators 300 --beta 2   # prioriza no perder fugas
python train.py --percentil-calibracion 99    # fuerza el percentil del umbral adaptativo
python train.py --balance none
python agent.py --escenario Scenario-7
python agent.py --escenario Scenario-1 --dias-calibracion 0   # sin calibrar
python evaluate.py --escenario Scenario-10     # contraste en muestra
```

---

## 5. Resultados reales

Validación **Leave-One-Scenario-Out**: se entrena con 9 escenarios y se evalúa
en el décimo, rotando. Ninguna fila de un escenario evaluado aparece en el
entrenamiento de ese mismo fold.

### 5.1 Los dos escenarios de despliegue

No son dos formas de medir lo mismo. Son **dos preguntas distintas**, y por eso
se reportan por separado.

| | A) Red nueva | B) Red conocida |
|---|---|---|
| Situación | El operador recibe una red de la que no tiene datos | El operador ya tiene histórico de esa red |
| ¿Puede calibrar? | **No** | **Sí** |
| Umbral | Global 0,25 | Percentil 99,5 del propio histórico |
| Exactitud | 0,7569 | **0,8906** |
| Precisión | 0,3252 | **0,8385** |
| Recall | **0,5323** | 0,4344 |
| **F1 (fuga)** | 0,4038 | **0,5723** |
| Meta del 80 % | No cumplida | **Cumplida** |
| Frente a la base trivial (84,54 %) | −8,85 puntos | **+4,52 puntos** |

**Conocer la red aporta +41,7 % de F1.**

El caso B es el real: una empresa de agua siempre conoce su red. El caso A
representa desplegar en una red sin ningún dato, donde **no hay forma de
calibrar** porque no existe referencia de qué es normal en esa red. Por eso usa
un umbral global, y por eso rinde peor.

### 5.2 Detalle del escenario A (red nueva)

Matriz de confusión: TP 14 420 · FP 29 918 · FN 12 671 · TN 118 191.

| Escenario | Exactitud | Precisión | Recall | F1 | AUC | Alertas |
|---|---|---|---|---|---|---|
| Scenario-10 | 0,9826 | 1,0000 | 0,9604 | 0,9798 | 0,9999 | 7 399 |
| Scenario-5 | 0,7249 | 1,0000 | 0,1872 | 0,3153 | 0,7136 | 1 110 |
| Scenario-3 | 0,4901 | 0,3633 | 0,4898 | 0,4172 | 0,4808 | 8 800 |
| Scenario-6 | 0,9072 | 1,0000 | 0,0310 | 0,0602 | 0,6117 | 52 |
| Scenario-7 | 0,7408 | 0,0650 | 1,0000 | 0,1221 | 0,9999 | 4 858 |
| Scenario-8 | 0,6780 | 0,4348 | 0,4752 | 0,4541 | 0,6618 | 5 396 |
| Scenario-1 *(control)* | 0,0901 | — | — | — | — | 15 942 |
| Scenario-2 *(control)* | 0,9617 | — | — | — | — | 671 |
| Scenario-4 *(control)* | 0,9937 | — | — | — | — | 110 |
| Scenario-9 *(control)* | 1,0000 | — | — | — | — | 0 |

En los escenarios de control la exactitud mide **cuántas falsas alarms** emite,
no aciertos: Scenario-1 produce 15 942.

### 5.3 Detalle del escenario B (red conocida)

Cada red fija su umbral con el percentil 99,5 de las puntuaciones de sus
primeros 30 días. Matriz de confusión: TP 11 768 · FP 2 266 · FN 15 323 ·
TN 131 443.

| Escenario | Umbral | Exactitud | Precisión | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Scenario-10 | 0,144 | 0,9917 | 0,9839 | 0,9991 | 0,9914 | 0,9999 |
| Scenario-7 | 0,365 | 0,9975 | 0,8876 | 1,0000 | 0,9405 | 0,9999 |
| Scenario-5 | 0,147 | 0,7930 | 0,9738 | 0,4508 | 0,6163 | 0,7136 |
| Scenario-6 | 0,040 | 0,8943 | 0,4823 | 0,1789 | 0,2610 | 0,6117 |
| Scenario-8 | 0,347 | 0,7307 | 0,8816 | 0,1418 | 0,2443 | 0,6618 |
| Scenario-3 | 0,413 | 0,5817 | 0,2253 | 0,0126 | 0,0238 | 0,4808 |
| Scenario-1 *(control)* | 0,472 | 0,9960 | — | — | — | — |
| Scenario-2 *(control)* | 0,299 | 0,9843 | — | — | — | — |
| Scenario-4 *(control)* | 0,233 | 0,9846 | — | — | — | — |
| Scenario-9 *(control)* | 0,080 | 0,9524 | — | — | — | — |

Los umbrales por red van de **0,040 a 0,472**. Un único valor global no cubre
ese rango, y ese es exactamente el motivo del método.

Comparación de A a B en las redes donde más se nota:

| Escenario | F1 en A | F1 en B | Cambio |
|---|---|---|---|
| Scenario-1 *(control, 15 942 falsas alarmas)* | 0,0000 | 0,9960 *(exactitud)* | **arreglado** |
| Scenario-7 | 0,1221 | 0,9405 | **+0,82** |
| Scenario-5 | 0,3153 | 0,6163 | **+0,30** |
| Scenario-6 | 0,0602 | 0,2610 | **+0,20** |
| Scenario-10 | 0,9798 | 0,9914 | +0,01 |
| Scenario-8 | 0,4541 | 0,2443 | −0,21 |
| Scenario-3 | 0,4172 | 0,0238 | **−0,39** |

**Scenario-3 empeora** en el escenario B: su AUC es 0,4808, es decir el modelo
ordena sus instantes *peor que el azar*. Ningún umbral lo arregla.

### 5.4 La curva de compromiso

El coste de detectar fugas es emitir alertas, y toda alerta falsa resta
exactitud. Umbral global (escenario A):

| Umbral | Exactitud | Precisión | Recall | F1 | Alertas |
|---|---|---|---|---|---|
| 0,50 | 0,8470 | 0,8956 | 0,0120 | 0,0237 | 364 |
| 0,30 | 0,7944 | 0,3358 | 0,3373 | 0,3365 | 27 208 |
| **0,25** | 0,7569 | 0,3252 | 0,5323 | 0,4038 | 44 338 |
| 0,15 | 0,6212 | 0,2542 | 0,7498 | 0,3797 | 79 904 |
| 0,10 | 0,4921 | 0,2047 | 0,7917 | 0,3253 | 104 786 |

Percentil del histórico propio (escenario B):

| Percentil | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| 80 | 0,7084 | 0,3054 | 0,5736 | 0,3986 |
| 90 | 0,7911 | 0,4068 | 0,5231 | 0,4577 |
| 95 | 0,8456 | 0,5471 | 0,4853 | 0,5143 |
| 98 | 0,8788 | 0,7241 | 0,4532 | 0,5575 |
| **99,5** | **0,8906** | **0,8385** | **0,4344** | **0,5723** |
| 99,9 | 0,8953 | 0,9346 | 0,4070 | 0,5671 |

⚠️ El percentil 99,5 significa que en operación normal salta **una alerta de
cada 200 pasos** (~87 al año). Que eso sea aceptable lo decide el operador, no
la métrica: un percentil más alto reduce aún más las falsas alarmas a costa del
recall.

### 5.5 La meta del 80 % de exactitud

En el **escenario B se cumple** (89,06 %) y además supera la línea base trivial
(84,54 %), que es lo que hace que la cifra sea informativa: el modelo detecta
fugas *y* acierta en el resto, en lugar de acertar por no avisar.

En el **escenario A no se cumple** (75,69 %), y es ilustrativo: la meta del
80 % está por debajo de la línea base trivial, así que puede "cumplirse" sin
detectar ninguna fuga.

Aun así, la cifra que mide capacidad de detección es el **F1 de la clase fuga**
(0,40 en A y 0,57 en B). Recomendación para la Fase 3: redefinir la meta en
términos de F1 o exactitud balanceada.

### 5.6 El diagnóstico que motivó el umbral adaptativo

El problema **no** era el modelo ni la evaluación. Se comprobó en tres pasos:

**1. La evaluación no era injusta.** Se probó el escenario realista (el operador
entrena con otras redes **más su propio histórico limpio**, y evalúa desde que
empieza la fuga): F1 0,4175 frente a 0,4038 del Leave-One-Scenario-Out.
Prácticamente idéntico.

**2. El umbral óptimo variaba por red.** Si cada escenario tuviera su umbral
óptimo (oracle, no desplegable), el F1 subía de 0,4038 a 0,66: **+0,26 de
ganancia media**. Los umbrales óptimos iban de **0,010 a 0,400**, un factor de
40. El modelo ordenaba bien y fallaba al decidir cuándo avisar.

**3. La causa es que cada red es una realización hidráulica distinta.** La
distancia entre las medias de los escenarios es de 48 a 346 unidades, mientras
que la señal de la fuga dentro de un escenario es de 0,05 a 4,12 desviaciones
estándar. Un umbral absoluto fijo cae en el sitio equivocado en cada red.

**La solución implementada:** cada red fija su umbral con **su propio histórico
limpio**. El agente observa 30 días que el operador declara sin fugas, recoge
las puntuaciones del modelo y toma el percentil indicado. No usa etiquetas en
ningún momento: el periodo de calibración se define por posición en el tiempo
(los primeros 1 440 pasos), y se verificó que están libres de fugas en los 10
escenarios — la fuga más temprana empieza el día 64.

Es exactamente lo que haría un operador real: operar la red un mes,
comprobar que no ocurre nada, y fijar ahí su nivel de alerta. Y es justamente
lo que hace posible el escenario B, donde el 80 % se cumple.

---

## 6. Qué parte es inteligencia artificial

El razonamiento **no** está escrito a mano en ninguna parte. No hay reglas del
tipo "si la presión del nodo X baja de Y, hay fuga".

| Elemento | Rol |
|---|---|
| `RandomForestClassifier` | **El modelo de aprendizaje automático.** Aprende la relación entre las 65 variables y la etiqueta a partir de datos históricos. |
| Umbral de decisión | Parámetro que el agente fija según el despliegue: global si no tiene histórico de la red, o el percentil del propio histórico si lo tiene. |
| `agent.py` | Solo transporta, aplica el umbral y registra. No razona. |

### Cómo funciona cada pieza

- **`perceive()`** recibe la medición y construye un vector con exactamente
  las 65 variables que el modelo espera, en el mismo orden. Cualquier clave
  que no esté en esa lista se descarta.
- **`decide()`** llama a `predict_proba()` del RandomForest. Ahí ocurre el
  aprendizaje automático: el bosque de 150 árboles vota y devuelve una
  probabilidad.
- **`calibrar()`** observa 30 días declarados sin fugas, recoge las
  puntuaciones y fija el umbral en el percentil indicado. **No usa
  etiquetas**: solo mediciones.
- **`act()`** traduce la decisión en una acción: `ALERTA_FUGA` u
  `OPERACION_NORMAL`.
- **`log_event()`** escribe una línea CSV por observación.

### Por qué el umbral tiene que adaptarse a la red

Cada escenario de LeakDB es una **realización hidráulica distinta** de la misma
red: cambian la longitud, el diámetro y la rugosidad. Eso desplaza la
distribución de presiones y caudales, de modo que la escala de las
probabilidades **no es comparable entre redes**. Medido en este dataset, el
umbral óptimo va de 0,010 a 0,400 según la red: un factor de 40.

Un único valor fijo, por tanto, funciona en unas redes y dispara falsas alarmas
en otras. Consecuencia medida: con umbral global, Scenario-1 —una red **sin
ninguna fuga**— produce 15 942 falsas alarmas. Al dejar que cada red fije el
suyo con su propio histórico limpio, baja a 105.

Cuando el operador **no tiene** histórico de la red (escenario A), no hay forma
de calibrar y el agente recurre al umbral global. Por eso ese escenario rinde
peor, y se reporta por separado en vez de esconderlo.

### Qué hace cada biblioteca

- **pandas** — leer los CSV y unir las 175 200 filas en una tabla.
- **NumPy** — operaciones numéricas sobre los arrays.
- **scikit-learn** — `RandomForestClassifier` (el modelo),
  `LeaveOneGroupOut` (la validación) y las métricas.
- **joblib** — serializar el modelo entrenado a disco y recuperarlo.
- **pytest** — pruebas automáticas.

### Cómo se evita la filtración de etiquetas

1. `data_loader.py` carga la etiqueta en un array **separado** de `X`, y la
   mantiene separada en todo momento.
2. Se excluyen de las entradas: `Labels.csv` (es el objetivo), `Leaks/`
   (verdad de terreno) y el identificador `Scenario-N`.
3. La división es **por escenario**, no aleatoria sobre filas: si fuera
   aleatoria, el modelo vería en entrenamiento filas del mismo escenario que
   está evaluando, y con las demandas distintas de cada escenario podría
   reconocerlo y contestar la etiqueta.
4. En `agent.py` la etiqueta real se pasa **después** de actuar, solo para
   escribirla en el registro. `perceive()` y `decide()` nunca la reciben.
5. `tests/test_aquasense.py` incluye una prueba que confirma que cambiar la
   etiqueta no altera la decisión.

---

## 7. Registro de ejecución

Cada observación deja una línea en `logs/agent_events.csv`:

```csv
registro_utc,escenario,observacion_id,timestamp_observacion,n_variables_recibidas,prob_fuga,umbral,decision,accion,etiqueta_real
2026-10-18T10:17:35,Scenario-10,0,2017-01-01T00:00:00,65,0.0,0.006667,NO_FUGA,OPERACION_NORMAL,0
```

Con eso se puede reconstruir qué observó el agente, qué decidió, con qué
probabilidad y qué acción tomó. `etiqueta_real` queda vacío si el evaluador no
la proporcionó.

### Ejecuciones reales

El agente calibró su umbral con los primeros 30 días de cada red y vigiló el
resto del año:

| Escenario | Umbral fijado por el agente | Alertas | Resultado |
|---|---|---|---|
| Scenario-1 *(control)* | 0,0200 | 105 | 105 falsas alarmas (antes: 15 942) |
| Scenario-10 | 0,0067 | 7 752 | TP 7 704 · FP 48 · FN 0 |
| Scenario-7 | 0,0120 | 389 | TP 316 · FP 73 · FN 0 |

En Scenario-1, que no tiene ninguna fuga, el modelo puntúa ese año entero
cerca de cero (máximo 0,04), así que la calibración fija un umbral bajo y el
resultado son 105 avisos en 17 520 pasos en vez de 15 942.

⚠️ Estos resultados son **en muestra**: el modelo se entrenó con los 10
escenarios. No sirven como prueba de generalización. La cifra oficial es la
sección 5.

---

## 8. Reproducibilidad

- Semilla fija (`42`) en `random_state` del modelo y semilla por defecto en
  `data_loader.RANDOM_SEED`.
- Versiones fijadas en `requirements.txt`.
- Toda la configuración (nº de árboles, balanceo, beta, semilla) queda dentro
  de `models/leak_classifier.joblib` y de `logs/train_report.json`, junto con
  las versiones del entorno.
- `logs/train_report.json` guarda la curva de compromiso completa, de modo que
  las métricas se pueden recalcular sin reentrenar.

---

1. **F1 de fuga = 0,57 en el escenario de despliegue real** (red conocida), 0,40
   en red nueva. Útil pero moderado, y muy desigual entre redes.
2. **El modelo no rinde igual en todas las redes.** Scenario-10 y Scenario-7 se
   resuelven casi por completo; Scenario-3 tiene AUC 0,4808 (peor que el azar) y
   ningún umbral lo arregla.
3. **Scenario-3 y Scenario-8 empeoran con el umbral adaptativo** (F1 0,42 → 0,02
   y 0,45 → 0,24). El método es mejor en agregado, pero no gana en todas las
   redes.
4. **El escenario de despliegue cambia la cifra oficial.** Con red conocida se
   cumple la meta del 80 %; con red nueva no se cumple. El README reporta ambos
   porque responder a "cuál es el desempeño" sin decir en qué situación es
   sería engañoso.
5. **Un percentil 99,5 implica ~87 falsas alertas al año** en operación normal.
   El recall cae de 0,53 a 0,43 al subir el umbral: es un compromiso, no una
   mejora gratuita.
6. **No se usan variables temporales.** El modelo mira solo el instante actual.
   Una ventana de las últimas horas debería ayudar sobre todo con las fugas
   `incipient`, cuyo caudal se corta a cero.
7. **Las fugas `incipient` son intrínsecamente difíciles.** Buena parte de los
   positivos está en tramos donde la fuga no descarga caudal.
8. **El ambiente es un conjunto de datos, no una red viva.** No hay conexión
   con EPANET en tiempo real: el agente lee el resultado de simulaciones.
9. **El periodo de calibración lo declara el operador.** El método asume 30
   días sin fugas; si el periodo estuviera contaminado, el umbral saldría
   demasiado alto y se perderían fugas. El agente comprueba y avisa si el
   periodo contiene positivos, pero esa comprobación usa la etiqueta y solo
   sirve para informar, no para decidir.
10. **En el escenario B el modelo se entrena con las otras nueve redes, no con
    datos propios de la red evaluada.** Una empresa de agua también aportaría su
    propio histórico de fugas pasadas, lo que probablemente mejoraría el recall.
    No se ha implementado.
11. **Optimismo residual declarado:** el umbral global y el percentil se eligen
    sobre las mismas predicciones fuera de muestra que se reportan. Son
    escalares únicos derivados de 175 200 filas, así que el optimismo debería ser
    despreciable, pero no es cero.

---

## 10. Tareas pendientes para la Fase 3

- [x] **Umbral adaptativo por red** a partir del histórico limpio (+41,7 % de F1).
- [x] **Separar los dos escenarios de despliegue** (red nueva / red conocida),
      porque dan cifras distintas y antes solo se reportaba una.
- [ ] Añadir el histórico propio de la red al entrenamiento del escenario B
      (hoy solo se usa para calibrar).
- [ ] Añadir variables temporales (medias móviles, deltas, ventana deslizante),
      especialmente para las fugas `incipient`.
- [ ] Probar normalización por perfil horario (mediana propia de cada nodo a
      cada hora del día) como forma alternativa de quitar el efecto de la
      realización hidráulica.
- [ ] Evaluar un modelo que use balance de masa (caudal de entrada menos
      demanda aplicada) en lugar de valores absolutos.
- [ ] Revisar `class_weight`: con prevalencias del 1,8 % al 44 %, `balanced`
      distorsiona las probabilidades.
- [ ] Estudiar Scenario-3: por qué el AUC cae por debajo de 0,5.
- [ ] Probar con otro subconjunto de LeakDB para comprobar que el
      comportamiento se repite fuera de Hanoi.