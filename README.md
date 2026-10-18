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
python train.py --temporal max6-24-48         # ablasion con variables temporales
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

| | A) Red nueva | B) Red conocida | C) Red conocida + entrena con su histórico |
|---|---|---|---|
| Situación | El operador recibe una red de la que no tiene datos | El operador ya tiene histórico de esa red | Igual que B, y además lo usa para entrenar |
| Umbral | Global 0,5 | Percentil 99,9 del propio histórico | Ídem, con 720 pasos de calibración |
| Exactitud | **0,8954** | **0,9056** | 0,8902 |
| Precisión | 0,9832 | 0,9666 | 0,7743 |
| Recall | 0,3293 | **0,4555** | **0,4918** |
| **F1 (fuga)** | 0,4933 | **0,6192** | 0,6016 |
| Meta del 80 % | Cumplida | Cumplida | Cumplida |
| Sobre la base (84,54 %) | +5,00 pts | **+6,02 pts** | +4,48 pts |

**Conocer la red aporta +25,5 % de F1** (de 0,4933 a 0,6192).

El caso B es el recomendado: una empresa de agua conoce su red y la vigila.
El caso A representa desplegar en una red sin ningún dato, donde **no hay forma
de calibrar** porque no existe referencia de qué es normal, así que usa un
umbral global.

### 5.1.1 `class_weight`: quitarlo mejoró el modelo

Decisión tomada midiendo, no por intuición. `class_weight="balanced"` estaba
en el código por la costumbre de "compensar el desbalance". **Medido en este
dataset es contraproducente:**

| | `balanced` | `none` (actual) | Cambio |
|---|---|---|---|
| AUC global (A) | 0,7031 | **0,7638** | **+0,06** |
| F1 (A) | 0,4038 | **0,4933** | **+22 %** |
| F1 (B) | 0,5723 | **0,6192** | **+8,2 %** |
| Falsas alarmas (A) | 29 918 | **152** | **−99,5 %** |

Lo importante es que **el AUC también mejoró** (0,7031 → 0,7638). Eso significa
que `balanced` no solo desplazaba la calibración: hacía que el bosque aprendiera
un criterio **peor**. Con 15,46 % de positivos, ponderar la clase 1 unas 5,5
veces hace que el modelo priorice Prefijo de la clase minoritaria con un criterio
que luego no generaliza.

Ahora `--balance none` es el valor por defecto. Con `balanced` se puede
reproducir con `--balance class_weight`.

### 5.1.2 Escenario C: también un resultado negativo

Entrenar además con el histórico limpio de la propia red **empeora** el
resultado: F1 0,6192 → 0,6016, con la precisión cayendo de 0,97 a 0,77.

La explicación es directa: el histórico limpio solo contiene ejemplos
**negativos**. Añadirlos desplaza el criterio hacia "normal" sin aportar
información sobre cómo es una fuga *en esa red*. Para que ayudara haría falta
histórico propio de fugas anteriores, que este dataset no tiene.

Se separa el tramo en dos mitades (720 para entrenar, 720 para calibrar) para
que el umbral no se contamine con filas que el modelo ya vio. Queda
disponible con `--historial-propio`.

### 5.2 Detalle del escenario A (red nueva)

Matriz de confusión: TP 8 920 · FP 152 · FN 18 171 · TN 147 957.

| Escenario | Exactitud | Precisión | Recall | F1 | AUC | Alertas |
|---|---|---|---|---|---|---|
| Scenario-7 | 0,9997 | 0,9844 | 1,0000 | 0,9922 | 1,0000 | 321 |
| Scenario-10 | 0,9533 | 1,0000 | 0,8937 | 0,9439 | 1,0000 | 6 885 |
| Scenario-8 | 0,7699 | 0,9729 | 0,1888 | 0,3162 | 0,6562 | 958 |
| Scenario-5 | 0,7060 | 1,0000 | 0,1315 | 0,2325 | 0,7233 | 780 |
| Scenario-3 | 0,6267 | 0,2593 | 0,0011 | 0,0021 | 0,4858 | 27 |
| Scenario-6 | 0,9043 | 0,0000 | 0,0000 | 0,0000 | 0,6648 | 0 |
| Scenario-1 *(control)* | 0,9945 | — | — | — | — | 97 |
| Scenario-2 *(control)* | 0,9999 | — | — | — | — | 1 |
| Scenario-4 *(control)* | 0,9998 | — | — | — | — | 3 |
| Scenario-9 *(control)* | 1,0000 | — | — | — | — | 0 |

En los escenarios de control la exactitud mide **cuántas falsas alarmas**
emite. Con `class_weight="balanced"` eran 15 942 en Scenario-1; ahora son 97.

### 5.3 Detalle del escenario B (red conocida)

Cada red fija su umbral con el percentil 99,9 de las puntuaciones de sus
primeros 30 días. Matriz de confusión: TP 12 340 · FP 427 · FN 14 751 ·
TN 133 282.

| Escenario | Umbral | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| Scenario-10 | 0,292 | 0,9982 | 0,9973 | 0,9990 | 0,9981 |
| Scenario-7 | 0,464 | 0,9995 | 0,9753 | 1,0000 | 0,9875 |
| Scenario-5 | 0,180 | 0,8073 | 0,9937 | 0,4804 | 0,6477 |
| Scenario-6 | 0,040 | 0,9137 | 0,7947 | 0,2332 | 0,3605 |
| Scenario-8 | 0,453 | 0,7561 | 0,9409 | 0,2194 | 0,3558 |
| Scenario-3 | 0,504 | 0,5937 | 0,2941 | 0,0008 | 0,0015 |
| Scenario-1 *(control)* | 0,557 | 0,9995 | — | — | — |
| Scenario-2 *(control)* | 0,404 | 0,9989 | — | — | — |
| Scenario-4 *(control)* | 0,450 | 0,9980 | — | — | — |
| Scenario-9 *(control)* | 0,113 | 0,9912 | — | — | — |

Los umbrales por red van de **0,040 a 0,557**. Un único valor global no cubre
ese rango, y ese es exactamente el motivo del método.

### 5.3.1 Por qué Scenario-3 es intratable en este benchmark

Su AUC es 0,4858, es decir **peor que el azar**. Se investigó y **no es un error
del código**:

| Presión en el nodo de la fuga | Normal | Con fuga | Desv. típica | Señal |
|---|---|---|---|---|
| Scenario-10, nodo 28 | 64,07 | 25,70 | 2,93 | **13,1 σ** |
| Scenario-10, nodo 12 | 64,94 | 61,60 | 2,43 | 1,38 σ |
| Scenario-3, nodo 19 | 65,57 | 65,51 | 2,20 | **0,030 σ** |
| Scenario-3, nodo 26 | 62,16 | 62,03 | 3,91 | **0,032 σ** |

La fuga de Scenario-3 mueve la presión del nodo afectado **6 centímetros**, un
3 % de la desviación típica. Es ruido. La de Scenario-10 la baja 38 metros.

Comprobación adicional: entrenando y probando **dentro** de Scenario-3 el AUC
llega a 0,8515, pero desde las otras nueve redes cae a 0,4743. Con 0,03 σ de
señal real, el modelo encuentra en Scenario-3 patrones espurios que no existen
en otras redes, y al transferirlos invierte el orden.

**Conclusión: esa fuga es físicamente indetectable con estos datos.** Ningún
ajuste de modelo, umbral o variables la recupera. Es un límite del benchmark
para fugas de ese tamaño, no del prototipo.

### 5.4 La curva de compromiso

El coste de detectar fugas es emitir alertas, y toda alerta falsa resta
exactitud. Umbral global (escenario A):

| Umbral | Exactitud | Precisión | Recall | F1 | Alertas |
|---|---|---|---|---|---|
| **0,50** | **0,8954** | **0,9832** | 0,3293 | **0,4933** | 9 072 |
| 0,40 | 0,8685 | 0,6183 | 0,3915 | 0,4794 | 17 153 |
| 0,30 | 0,7536 | 0,3456 | 0,6648 | 0,4548 | 52 105 |
| 0,25 | 0,7128 | 0,3143 | 0,7253 | 0,4386 | 62 521 |
| 0,15 | 0,5918 | 0,2422 | 0,7702 | 0,3685 | 86 149 |
| 0,10 | 0,4600 | 0,1999 | 0,8301 | 0,3222 | 112 494 |

Sin `class_weight`, el umbral natural de **0,50** es además el óptimo: las
probabilidades ya están bien calibradas y no hay que forzar un umbral bajo.

Percentil del histórico propio (escenario B):

| Percentil | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|
| 80 | 0,7096 | 0,3147 | 0,6146 | 0,4163 |
| 90 | 0,8048 | 0,4355 | 0,5362 | 0,4806 |
| 95 | 0,8598 | 0,6014 | 0,4973 | 0,5444 |
| 98 | 0,8836 | 0,7395 | 0,4773 | 0,5802 |
| 99 | 0,8934 | 0,8194 | 0,4712 | 0,5983 |
| 99,5 | 0,9022 | 0,9141 | 0,4627 | 0,6144 |
| **99,9** | **0,9056** | **0,9666** | 0,4555 | **0,6192** |

⚠️ El percentil 99,9 significa que en operación normal salta **una alerta de
cada 1 000 pasos** (~17 al año). Que eso sea aceptable lo decide el operador, no
la métrica: bajar el percentil recupera recall a costa de falsas alarmas.

### 5.5 La meta del 80 % de exactitud

**Se cumple en los tres escenarios**: 89,54 % (A), 90,56 % (B), 89,02 % (C).
En todos supera la línea base trivial (84,54 %), que es lo que hace que la cifra
sea informativa: el modelo detecta fugas *y* acierta en el resto, en lugar de
acertar por no avisar.

Aun así, la cifra que mide capacidad de detección es el **F1 de la clase fuga**
(0,40 en A y 0,57 en B). Recomendación para la Fase 3: redefinir la meta en
términos de F1 o exactitud balanceada.

### 5.7 Variables temporales: experimento con resultado NEGATIVO

El modelo mira un solo instante, y las fugas `incipient` son intermitentes, así
que la hipótesis era que el contexto temporal ayudaría. **Se implementó y se
midió. NoAyuda en el escenario de despliegue que importa.**

`data_loader.py` calcula *rolling* causal (ventana trasera, nunca futuro) por
escenario. Se probaron media y máximo en ventanas de 6, 24 y 48 h:

**Cribado por AUC** (solo AUC, porque aísla la calidad de las variables del
efecto del umbral):

| Configuración | nvar | S-3 | S-5 | S-6 | S-7 | S-8 | S-10 | Media |
|---|---|---|---|---|---|---|---|---|
| Solo instantáneas | 65 | 0,4836 | 0,7190 | 0,5839 | 0,9997 | 0,6649 | 0,9999 | 0,7418 |
| + media 6 h y 24 h | 130 | 0,4153 | 0,6616 | 0,5782 | 0,9808 | 0,4785 | 0,9888 | 0,6839 |
| **+ máximo 6/24/48 h** | 260 | 0,4224 | **0,8485** | 0,5615 | 1,0000 | **0,7711** | 0,9981 | **0,7669** |

Hallazgo físico: **la media empeora, el máximo mejora**. Promediar 24 h aplana
la señal intermitente de una fuga `incipient`; el máximo la conserva, porque
recuerda que hubo caudal aunque ahora sea cero.

**Medición completa del pipeline** (`train.py --temporal max6-24-48`):

| Despliegue | F1 sin | F1 con | Cambio |
|---|---|---|---|
| A) Red nueva | 0,4038 | **0,4419** | **+9,4 %** |
| B) Red conocida | 0,5723 | 0,5718 | −0,1 % |

Y por red (escenario B):

| Red | F1 sin | F1 con | Cambio |
|---|---|---|---|
| Scenario-8 | 0,2443 | 0,5441 | **+0,30** |
| Scenario-3 | 0,0238 | 0,2454 | **+0,22** |
| Scenario-6 | 0,2610 | 0,2977 | +0,04 |
| Scenario-10 | 0,9914 | 0,9965 | +0,01 |
| Scenario-5 | 0,6163 | 0,5335 | −0,08 |
| Scenario-7 | 0,9405 | 0,6031 | **−0,34** |

**Decisión: no se activa por defecto.** En el escenario B, que es el
recomendado, el F1 no se mueve (−0,1 %) mientras el modelo crece de 65 a 260
variables y la precisión se desploma de 0,84 a 0,63. La mejora se concentra en
el escenario A, que es el menos relevante, y el resultado es una pérdida neta
por red (Scenario-7 cae 0,34).

### 5.8 El diagnóstico que motivó el umbral adaptativo

Queda disponible como ablación reproducible:

```powershell
python train.py --temporal max6-24-48 \
       --model-out models/leak_temporal.joblib \
       --report-out logs/train_temporal.json
```

Si en la Fase 3 cambia el umbral de decisión (por ejemplo, si perder fugas sale
mucho más caro que avisar de más), conviene volver a medir: con `--beta 2` el
recall pesa más y las temporales podrían volver a compensar.

Por qué el agente sigue soportándolas: mantiene un búfer de las últimas
observaciones y durante el calentamiento (las primeras 96) no alerta, porque
las variables temporales se calculan sobre una ventana parcial y no son
comparables con las de régimen estable. Una prueba verifica que el búfer del
agente reproduce **exactamente** (`diferencia < 1e-9`) lo que calculó
`data_loader`.

El problema **no** era el modelo ni la evaluación. Se comprobó en tres pasos:

**1. La evaluación no era injusta.** Se probó el escenario realista (el operador
entrena con otras redes **más su propio histórico limpio**, y evalúa desde que
empieza la fuga): F1 0,4175 frente a 0,4038 del Leave-One-Scenario-Out.
Prácticamente idéntico.

**2. El umbral óptimo variaba por red.** Si cada escenario tuviera su umbral
óptimo (oracle, no desplegable), el F1 subía **+0,26**. Los umbrales óptimos
iban de **0,010 a 0,400**, un factor de 40. El modelo ordenaba bien y fallaba al
decidir cuándo avisar.

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
| `RandomForestClassifier` | **El modelo de aprendizaje automático.** Aprende la relación entre las 65 variables y la etiqueta a partir de datos históricos. Sin `class_weight`: medido, ponderar las clases empeora el AUC (§5.1.1). |
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

| Escenario | Umbral | Exactitud | Precisión | Recall | F1 |
|---|---|---|---|---|---|
| Scenario-1 *(control)* | 0,0200 | 0,9940 | — | — | — |
| Scenario-10 | 0,0067 | 0,9973 | 0,9938 | 1,0000 | 0,9966 |
| Scenario-7 | 0,0246 | 0,9992 | 0,9576 | 1,0000 | 0,9785 |

En Scenario-1, que no tiene ninguna fuga, el modelo puntúa ese año entero
cerca de cero, así que la calibración fija un umbral bajo y el resultado son
unas 100 alertas en 17 520 pasos, en lugar de las 15 942 que producía antes
de quitar `class_weight="balanced"`.

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

1. **F1 de fuga = 0,62 en el escenario recomendado (red conocida)**, 0,49 en red
   nueva. Util pero moderado, y muy desigual entre redes.
2. **Una de las seis redes con fuga es indetectable.** La de Scenario-3 mueve
   la presion 0,03 sigma (5.3.1). Es un limite del benchmark para fugas de ese
   tamano, no del prototipo. Ningun metodo la recuperara con estos datos.
3. **El recall se sacrifica en favor de la precision.** Con el punto de
   operacion actual se pierde ~55 % de las fugas (FN 14 751) a cambio de 427
   falsas alarmas al ano. Si el coste relativo cambia, hay que mover el
   percentil: la seccion 5.4 da la curva completa.
4. **No se usan variables temporales por defecto.** Se implementaron y midieron
   (5.7): mejoran el escenario de red nueva pero no el de red conocida, que es
   el recomendado. Quedan disponibles con `--temporal`.
5. **Entrenar con el historico propio de la red no aporta** (5.1.2): solo
   contiene ejemplos negativos. Haria falta historico propio de fugas
   anteriores, que este dataset no tiene.
6. **Las fugas `incipient` son intrinsecamente dificiles.** Buena parte de los
   positivos esta en tramos donde la fuga no descarga caudal.
7. **El ambiente es un conjunto de datos, no una red viva.** No hay conexion
   con EPANET en tiempo real: el agente lee el resultado de simulaciones.
8. **El periodo de calibracion lo declara el operador.** El metodo asume 30
   dias sin fugas; si el periodo estuviera contaminado, el umbral saldria
   demasiado alto y se perderian fugas. El agente comprueba y avisa si el
   periodo contiene positivos, pero esa comprobacion usa la etiqueta y solo
   sirve para informar, no para decidir.
9. **Solo un subconjunto.** Todo se ha medido sobre Hanoi_CMH. No se puede
   afirmar que el comportamiento se repita en otras redes de LeakDB, porque no
   hay mas datos disponibles en este entorno.
10. **Optimismo residual declarado:** el umbral global y el percentil se eligen
    sobre las mismas predicciones fuera de muestra que se reportan. Son
    escalares unicos derivados de 175 200 filas, asi que el optimismo deberia ser
    despreciable, pero no es cero.
11. **El coste relativo de una falsa alarma no esta fijado.** Se eligio el
    punto que maximiza F1, es decir, tratando por igual ambos errores. Si
    perder una fuga cuesta mucho mas, hay que fijar beta > 1 y volver a medir.

---

## 10. Tareas pendientes para la Fase 3

Cerradas en esta fase:

- [x] **Umbral adaptativo por red** a partir del histórico limpio.
- [x] **Separar los escenarios de despliegue** (red nueva / red conocida /
      red conocida entrenando también con su histórico).
- [x] **Quitar `class_weight="balanced"`**, que empeoraba el AUC y generaba
      29 918 falsas alarmas (§5.1.1).
- [x] **Variables temporales implementadas y medidas**: no se activan por
      defecto porque en el escenario recomendado el F1 no mejora (§5.7).
- [x] **Diagnóstico de Scenario-3**: fuga indetectable por señal de 0,03 σ (§5.3.1).
- [x] **Histórico propio como dato de entrenamiento**: medido, no aporta (§5.1.2).