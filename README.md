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
| `logs/train_report.json` | Curva de compromiso, métricas, por escenario |
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
python train.py --with-demands                # experimento de ablación
python agent.py --escenario Scenario-7 --limite 2000
python evaluate.py --escenario Scenario-10     # contraste en muestra
```

---

## 5. Resultados reales

Validación **Leave-One-Scenario-Out**: se entrena con 9 escenarios y se evalúa
en el décimo, rotando. Ninguna fila de un escenario evaluado aparece en el
entrenamiento de ese mismo fold.

Punto de operación: **umbral 0,25** (el que maximiza F1 sobre las
predicciones fuera de muestra).

| Métrica | Valor |
|---|---|
| Exactitud | 0,7569 |
| Precisión (fuga) | 0,3252 |
| Recall (fuga) | 0,5323 |
| **F1 (fuga)** | **0,4038** |
| AUC | 0,7031 |
| Matriz de confusión | TP 14 420 · FP 29 918 · FN 12 671 · TN 118 191 |

Media sobre los 6 escenarios **con** fuga: exactitud 0,7539 · precisión 0,6439
· recall 0,5239 · F1 0,3914.

### 5.1 La meta del 80 % de exactitud NO se cumple

Exactitud obtenida **75,69 %**, por debajo del objetivo del 80 %.

Pero el dato relevante es otro: **la línea base de predecir siempre "normal"
es 84,54 %**. La meta del 80 % está *por debajo* de no hacer absolutely nada.
Es decir, el objetivo se supera sin detectar ninguna fuga, y se falla
simplemente por avisar de más.

Detectar fugas obliga a emitir alertas, y toda alerta falsa resta exactitud.
Las dos metas son incompatibles en este dataset. La curva de compromiso
muestra el intercambio:

| Umbral | Exactitud | Precisión | Recall | F1 | Alertas |
|---|---|---|---|---|---|
| 0,50 | 0,8470 | 0,8956 | 0,0120 | 0,0237 | 364 |
| 0,30 | 0,7944 | 0,3358 | 0,3373 | 0,3365 | 27 208 |
| **0,25** | **0,7569** | **0,3252** | **0,5323** | **0,4038** | **44 338** |
| 0,15 | 0,6212 | 0,2542 | 0,7498 | 0,3797 | 79 904 |
| 0,10 | 0,4921 | 0,2047 | 0,7917 | 0,3253 | 104 786 |

**La cifra que demuestra capacidad de detección es el F1 de la clase fuga
(0,40), no la exactitud.** Recomendación para la Fase 3: redefinir la meta en
términos de F1 o exactitud balanceada.

### 5.2 El comportamiento es muy desigual entre escenarios

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

Tres observaciones:

- **Scenario-10 es prácticamente perfecto** (AUC 0,9999).
- **Scenario-3 tiene AUC 0,4808**: el modelo ordena sus instantes *peor que
  al azar* en ese escenario.
- **Scenario-1 dispara 15 942 alertas** sobre un escenario sin ninguna fuga:
  el 91 % de sus pasos se marcan como fuga. El modelo generaliza mal ante
  esa realización hidráulica concreta.

Que el mismo modelo resuelva Scenario-10 y falle en Scenario-3 indica que
cada escenario es una realización hidráulica muy distinta (la incertidumbre de
longitud, diámetro y rugosidad cambia entre ellas) y el modelo no está
capturando la señal relativa a la fuga, sino valores absolutos que no
transferen.

### 5.3 Decisión metodológica: el umbral

Una primera versión usó el umbral por defecto de 0,5. Daba F1 = 0,00 en los
seis escenarios con fuga, pero AUC de 0,93 a 1,00: el modelo ordenaba bien
los instantes con fuga y solo fallaba al decidir cuándo avisar.

Se intentó después ajustar el umbral *anidado* por fold (elegirlo con un
`GroupKFold` interno sobre los escenarios de entrenamiento). **Empeoró el
resultado**: el interior mezclaba probabilidades de modelos distintos, que no
son comparables entre sí, y eligió umbrales erróneos (AUC interior de 0,41
frente a 0,9999 del mismo escenario ya evaluado).

La solución fue una sola pasada Leave-One-Scenario-Out que produce la
probabilidad de cada fila predicha por un modelo que nunca vio su escenario.
Esas mismas predicciones sirven para elegir el umbral y para evaluar, lo que
elimina el problema de comparabilidad.

*Nota declarada:* el umbral se elige sobre las mismas predicciones fuera de
muestra que se reportan. Es un único escalar derivado de 175 200 filas, así que
el optimismo es esperablemente despreciable, pero queda declarado en el
informe en lugar de oculto.

---

## 6. Qué parte es inteligencia artificial

El razonamiento **no** está escrito a mano en ninguna parte. No hay reglas del
tipo "si la presión del nodo X baja de Y, hay fuga".

| Elemento | Rol |
|---|---|
| `RandomForestClassifier` | **El modelo de aprendizaje automático.** Aprende la relación entre las 65 variables y la etiqueta a partir de datos históricos. |
| Umbral 0,25 | Parámetro aprendido sobre las predicciones fuera de muestra, no fijado a ojo. |
| `agent.py` | Solo transporta, aplica el umbral y registra. No razona. |

### Cómo funciona cada pieza

- **`perceive()`** recibe la medición y construye un vector con exactamente
  las 65 variables que el modelo espera, en el mismo orden. Cualquier clave
  que no esté en esa lista se descarta.
- **`decide()`** llama a `predict_proba()` del RandomForest. Ahí ocurre el
  aprendizaje automático: el bosque de 150 árboles vota y devuelve una
  probabilidad.
- **`act()`** traduce la decisión en una acción: `ALERTA_FUGA` u
  `OPERACION_NORMAL`.
- **`log_event()`** escribe una línea CSV por observación.

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
2026-10-18T10:17:35,Scenario-10,0,2017-01-01T00:00:00,65,0.0,0.25,NO_FUGA,OPERACION_NORMAL,0
```

Con eso se puede reconstruir qué observó el agente, qué decidió, con qué
probabilidad y qué acción tomó. `etiqueta_real` queda vacío si el evaluador no
la proporcionó.

Ejemplo de ejecución real (Scenario-10, modelo ya entrenado con ese escenario):

```
Observaciones procesadas: 17,520
Alertas generadas       : 7,704   (43.97 %)
TP=7,704  FP=0  FN=0
```

⚠️ Ese resultado es **en muestra**: el modelo se entrenó con los 10
escenarios. No sirve como prueba de generalización. La cifra oficial es la
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

## 9. Limitaciones

1. **La meta del 80 % de exactitud es inaplicable** en este dataset: la línea
   base trivial es 84,54 %. Hay que redefinirla.
2. **F1 de fuga = 0,40.** Es útil pero moderado, y muy desigual entre
   escenarios.
3. **El modelo no generaliza entre escenarios.** Fallo completo en
   Scenario-1 (91 % de falsas alarmas) y AUC < 0,5 en Scenario-3.
4. **No se usan variables temporales.** El modelo mira solo el instante actual.
   Una ventana de las últimas horas debería ayudar especialmente con las fugas
   `incipient`, cuyo caudal se corta a cero y son prácticamente indetectables
   desde una única medición.
5. **Las fugas `incipient` son intrínsecamente difíciles.** El 25 % de los
   positivos está en tramos donde la fuga no descarga caudal.
6. **El ambiente es un conjunto de datos, no una red viva.** No hay conexión
   con EPANET en tiempo real: el agente lee el resultado de simulaciones.
7. **Sin test automatizado sobre datos sintéticos** ni validación con redes
   distintas de Hanoi.

---

## 10. Tareas pendientes para la Fase 3

- [ ] Añadir variables temporales (medias móviles, deltas, ventana deslizante).
- [ ] Probar normalización por perfil horario (cada escenario tiene su propia
      realización hidráulica; comparar contra la mediana histórica del nodo).
- [ ] Evaluar un modelo que use balance de masa (caudal de entrada menos
      demanda aplicada) en lugar de valores absolutos.
- [ ] Comparar contra una línea base explícita y otros clasificadores.
- [ ] Redefinir la meta de desempeño con métricas Adequadas al desbalance.
- [ ] Probar con otro subconjunto de LeakDB para comprobar que el
      comportamiento se repite fuera de Hanoi.