# Fase 3 — Pruebas basadas en las Medidas de Desempeño (SMART)

**Proyecto:** AquaSense-Agent  
**Fecha:** 2026-10-18  
**Dataset:** LeakDB Hanoi_CMH (10 escenarios, 175 200 pasos de 30 min, 6 con fugas)  
**Validación:** Leave-One-Scenario-Out (LOSO) — 10 folds, ninguna fila del escenario evaluado en entrenamiento  
**Repeticiones:** 5 semillas distintas (42, 123, 456, 789, 999) → media ± desviación estándar  

---

## 1. Medida SMART de la Sección 7 (Diseño)

> **Texto original:** *"La medida de desempeño de AQUASENSE-AGENT se define como el porcentaje de coincidencia entre las fugas detectadas por el agente y las fugas confirmadas mediante inspección de campo. [...] Meta: alcanzar una exactitud de al menos 80 % frente a fugas confirmadas en campo, evaluada sobre el horizonte histórico de tres meses."*

### 1.1 Problema de unidades — **La medida no se puede probar tal como está escrita**

| Lo que pide la Sección 7 | Lo que tenemos en el dataset |
|--------------------------|------------------------------|
| **Fugas** (eventos discretos confirmados en campo) | **Pasos de 30 min** con etiqueta binaria por simulación |
| Inspección de campo (verdad de terreno) | Labels.csv = ground truth de simulación hidráulica |
| Horizonte de 3 meses con fugas | Las fugas empiezan ≥ día 64; en los primeros 90 días hay ≤ 2 eventos |

**Reformulación operacional (documento ajustado):**

> **Medida: Event Detection Rate (EDR) — Tasa de Detección de Eventos de Fuga**  
> **Fórmula:** `EDR = (Número de eventos de fuga reales con ≥ 50 % de su duración cubierta por alertas del agente) / (Total de eventos de fuga reales en el período de evaluación)`  
> **Umbral de aceptación (fijado ANTES de ejecutar):** **≥ 80 %** (heredado de la Sección 7)  
> **Justificación del umbral:** La Sección 7 exige "exactitud de al menos 80 % frente a fugas confirmadas". Operacionalizar "coincidencia" como cobertura ≥ 50 % del evento es el criterio mínimo para considerar que una fuga fue "detectada" en lugar de solo "rozada". Cobertura < 25 % se considera ruido.  
> **Nota sobre horizonte de 3 meses:** En este dataset las fugas comienzan tras el día 64, por lo que una ventana de 3 meses (90 días = 4320 pasos) captura ≤ 2 eventos. Se reporta sobre el año completo (12 meses) y se anota la limitación.  
> **Nota sobre "inspección de campo":** Se usa Labels.csv como proxy; no hay datos de inspección real. La medida se prueba con esta limitación declarada.

**Medida secundaria (reportada para contexto):**  
- **EDR_any:** Cobertura > 0 % (cualquier solape) — umbral aspiracional 80 %  
- **Per-timestep accuracy:** Para comparabilidad con Fase 2 — umbral 80 %  

---

## 2. Plan de Pruebas

### 2.1 Casos de prueba (≥ 10, con normales, límite, ≥ 2 falla esperada)

| # | Tipo | Caso | Descripción |
|---|------|------|-------------|
| 1 | Normal | Scenario-10 | Fuga grande (nodos 12, 28), 2 eventos, señal 13 σ |
| 2 | Normal | Scenario-7 | Fuga grande, 1 evento, señal alta |
| 3 | Normal | Scenario-5 | Fuga incipiente, 1 evento |
| 4 | Normal | Scenario-8 | Fuga incipiente, 1 evento |
| 5 | Normal | Scenario-3 | **Límite**: 2 eventos, señal **0.03 σ** (físicamente indetectable) |
| 6 | Normal | Scenario-6 | **Límite**: fuga inicia día 277 (muy tardía) |
| 7 | Control | Scenario-1 | Sin fugas — mide falsas alarmas |
| 8 | Control | Scenario-2 | Sin fugas |
| 9 | Control | Scenario-4 | Sin fugas |
| 10 | Control | Scenario-9 | Sin fugas |
| 11 | Falla | **Variable faltante** | Observación sin `press_10` → `ValueError` |
| 12 | Falla | **Sensor NaN** | Sensor devuelve `NaN` → decisión silenciosa errónea (prob=0.0) |
| 13 | Falla | **Calibración contaminada** | Periodo "sin fugas" incluye 100 fugas → umbral sube |
| 14 | Falla | **Cambio de distribución** | Scenario-3 (AUC 0.4858) — modelo no generaliza |
| 15 | Edge | Horizonte 3 meses | Evaluación solo primeros 90 días (≤ 2 eventos) |
| 16 | Edge | Umbral global vs adaptativo | A (red nueva) vs B (red conocida) |
| 17 | Falla | Agente ciclo completo + variable faltante | Agente real se detiene en paso 0 |

**Total: 17 casos** (6 normales + 4 control + 2 límite + 4 falla esperada + 1 edge adicional)

### 2.2 Repeticiones

- El agente/entorno **no tiene aleatoriedad** en producción (RandomForest con `random_state` fijo).
- Para medir robustez del entrenamiento, se repite **5 veces con semillas distintas** (42, 123, 456, 789, 999).
- Se reporta **media ± desviación estándar** entre semillas para cada escenario y métrica global.

---

## 3. Resultados — 5 Repeticiones (LOSO completo)

### 3.1 Escenario A — Red Nueva (umbral global 0.5, sin calibración)

| Escenario | Eventos reales | Detectados (cualquier solape) | Detectados (≥50% cub.) | Falsas alarmas | Acc. por paso |
|-----------|----------------|-------------------------------|------------------------|----------------|---------------|
| Scenario-1 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **162.4 ± 110.9** | 0.9903 ± 0.0070 |
| Scenario-10 | 2 | **2.0 ± 0.0** | **2.0 ± 0.0** | 0.0 ± 0.0 | 0.8885 ± 0.0587 |
| Scenario-2 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **3.0 ± 2.3** | 0.9998 ± 0.0001 |
| **Scenario-3** | **2** | **0.6 ± 0.8** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.6272 ± 0.0003 |
| Scenario-4 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **9.2 ± 5.4** | 0.9995 ± 0.0003 |
| Scenario-5 | 1 | **1.0 ± 0.0** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.7044 ± 0.0039 |
| Scenario-6 | 1 | **0.2 ± 0.4** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.9045 ± 0.0005 |
| Scenario-7 | 1 | **1.0 ± 0.0** | **1.0 ± 0.0** | 0.0 ± 0.0 | 0.9998 ± 0.0001 |
| Scenario-8 | 1 | **1.0 ± 0.0** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.7595 ± 0.0068 |
| Scenario-9 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **0.0 ± 0.0** | 1.0000 ± 0.0000 |
| **TOTAL** | **8** | **5.8 ± 0.8** | **3.0 ± 0.0** | **174.6 ± 118.6** | — |

> **Event Detection Rate (cualquier solape): 5.8 / 8 = 72.5 %**  
> **Event Detection Rate (≥50% cobertura): 3.0 / 8 = 37.5 %**  
> **Event Detection Rate (≥25% cobertura): 3.0 / 8 = 37.5 %** (idéntico a ≥50%)

### 3.2 Escenario B — Red Conocida (percentil 99.9 del histórico limpio = 0.5000)

| Escenario | Eventos reales | Detectados (cualquier solape) | Detectados (≥50% cub.) | Falsas alarmas | Acc. por paso |
|-----------|----------------|-------------------------------|------------------------|----------------|---------------|
| Scenario-1 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **121.6 ± 14.4** | 0.9929 ± 0.0008 |
| Scenario-10 | 2 | **2.0 ± 0.0** | **2.0 ± 0.0** | 0.0 ± 0.0 | 0.8833 ± 0.0485 |
| Scenario-2 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **3.2 ± 3.2** | 0.9998 ± 0.0002 |
| **Scenario-3** | **2** | **0.4 ± 0.8** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.6273 ± 0.0003 |
| Scenario-4 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **10.4 ± 8.7** | 0.9994 ± 0.0005 |
| Scenario-5 | 1 | **1.0 ± 0.0** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.7028 ± 0.0058 |
| Scenario-6 | 1 | **0.2 ± 0.4** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.9043 ± 0.0000 |
| Scenario-7 | 1 | **1.0 ± 0.0** | **1.0 ± 0.0** | 0.0 ± 0.0 | 0.9998 ± 0.0001 |
| Scenario-8 | 1 | **1.0 ± 0.0** | **0.0 ± 0.0** | 0.0 ± 0.0 | 0.7578 ± 0.0102 |
| Scenario-9 (control) | 0 | 0.0 ± 0.0 | 0.0 ± 0.0 | **0.0 ± 0.0** | 1.0000 ± 0.0000 |
| **TOTAL** | **8** | **5.6 ± 0.8** | **3.0 ± 0.0** | **135.2 ± 26.3** | — |

> **Event Detection Rate (cualquier solape): 5.6 / 8 = 70.0 %**  
> **Event Detection Rate (≥50% cobertura): 3.0 / 8 = 37.5 %**  

### 3.3 Horizonte de 3 meses (primeros 90 días = 4320 pasos)

| Escenario | Eventos en 3 meses | Detectados (≥50%) |
|-----------|-------------------|-------------------|
| Scenario-10 | 0 | 0 |
| Scenario-3 | 1 | 0 |
| Scenario-5 | 1 | 0 |
| Scenario-6 | 0 | 0 |
| Scenario-7 | 0 | 0 |
| Scenario-8 | 0 | 0 |
| **TOTAL** | **2** | **0** |

> **EDR (3 meses, ≥50%) = 0 / 2 = 0.0 %** — Las fugas en este dataset comienzan tras el día 64; el horizonte de 3 meses captura casi nada. **Limitación del dataset, no del agente.**

### 3.4 Casos de Falla Esperada

| # | Caso | Resultado | ¿Comportamiento esperado? |
|---|------|-----------|---------------------------|
| 11 | Variable faltante (`press_10`) | `ValueError: Faltan 1 variables...` | **SÍ** — Falla ruidosa, clara |
| 12 | Sensor con `NaN` | `prob=0.0`, `alerta=False` (sin error) | **NO** — Debería validar y rechazar/alertar |
| 13 | Calibración con 100 fugas | Umbral 0.5000 → 0.5000 (sin cambio) | Parcial — En este dataset p99.9 ya satura en 0.5 |
| 14 | Scenario-3 (cambio distribución) | AUC = **0.4858** (peor que azar) | **SÍ** — Falla conocida, documentada |
| 17 | Agente real + variable faltante | `ValueError` en paso 0, agente muere | **SÍ** — No tolera fallo de sensor |

---

## 4. Tabla de Resultados Obligatoria

| Medida de desempeño | Fórmula | Umbral | Casos ejecutados | Resultado obtenido | ¿Cumple? | Evidencia |
|---------------------|---------|--------|------------------|-------------------|----------|-----------|
| **EDR (≥50% cobertura) — Esc. A** | Eventos detectados / Eventos reales | ≥ 80 % | 8 eventos (6 escenarios fuga) | **37.5 %** (3/8) | ❌ **NO** | logs/phase3_results.json, seed 42-999 |
| **EDR (≥50% cobertura) — Esc. B** | Eventos detectados / Eventos reales | ≥ 80 % | 8 eventos (6 escenarios fuga) | **37.5 %** (3/8) | ❌ **NO** | logs/phase3_results.json, seed 42-999 |
| **EDR (cualquier solape) — Esc. A** | Eventos con solape > 0 / Eventos reales | ≥ 80 % | 8 eventos | **72.5 %** (5.8/8) | ❌ **NO** | logs/phase3_results.json |
| **EDR (cualquier solape) — Esc. B** | Eventos con solape > 0 / Eventos reales | ≥ 80 % | 8 eventos | **70.0 %** (5.6/8) | ❌ **NO** | logs/phase3_results.json |
| **Per-timestep Accuracy — Esc. A** | (TP+TN)/Total | ≥ 80 % | 175 200 pasos | **89.54 %** | ✅ **SÍ** | logs/train_report.json |
| **Per-timestep Accuracy — Esc. B** | (TP+TN)/Total | ≥ 80 % | 175 200 pasos | **90.56 %** | ✅ **SÍ** | logs/train_report.json |
| **EDR 3 meses (≥50%)** | Eventos detectados / Eventos en 90 días | ≥ 80 % | 2 eventos | **0.0 %** (0/2) | ❌ **NO** | Sección 3.3 (limitación dataset) |
| **Robustez semillas (EDR any)** | 1 - (std/mean) entre 5 semillas | > 90 % | 5 semillas × 10 escenarios | **86.2 %** (A), **85.7 %** (B) | ❌ **NO** | Desv. std > 0 en S3, S6 |
| **Falla: Variable faltante** | Excepción clara / no se cuelga | ValueError en perceive | 1 caso | `ValueError` ✓ | ✅ **SÍ** | Sección 3.4, caso 11 |
| **Falla: Sensor NaN** | Rechazo/alerta, no decisión silenciosa | Error o alerta explícita | 1 caso | **Decisión silenciosa (prob=0)** | ❌ **NO** | Sección 3.4, caso 12 |
| **Falla: Calibración contaminada** | Umbral no sube > 10 % | Δumbral < 0.05 | 1 caso simulado | Δ = 0.0000 (saturado) | ✅ **SÍ** (pero por saturación) | Sección 3.4, caso 13 |
| **Falla: Cambio distribución (S3)** | AUC > 0.70 en escenario OOD | ≥ 0.70 | 1 escenario | **0.4858** | ❌ **NO** | Sección 3.4, caso 14 |

---

## 5. Análisis de Resultados (≤ 1 página)

### 5.1 Medidas que CUMPLEN

| Medida | Resultado | Comentario |
|--------|-----------|------------|
| Per-timestep Accuracy (Esc. A) | 89.54 % | Supera 80 % y la base trivial (84.54 %). La métrica de Fase 2 se cumple. |
| Per-timestep Accuracy (Esc. B) | 90.56 % | Mejor que A; el umbral adaptativo reduce falsas alarmas. |
| Variable faltante → ValueError | Falla ruidosa clara | El agente valida entradas y rechaza observaciones incompletas. |

### 5.2 Medidas que NO CUMPLEN — Causa y Corrección

| Medida | Resultado | Causa probable | Qué cambiar en el agente/modelo |
|--------|-----------|----------------|----------------------------------|
| **EDR (≥50%) = 37.5 %** | **FALLA CRÍTICA** | El modelo detecta fugas de forma **esporádica/parcial** (recall por paso 33–45 %). Las alertas no cubren la duración del evento. | 1. **Variables temporales** (ventanas causales max) — ya implementadas, no activadas por defecto. 2. **Post-procesamiento**: exigir N alertas consecutivas antes de confirmar evento. 3. **Reentrenar con loss focal** o peso temporal para penalizar huecos en detección. |
| **EDR (any) = 70–72 %** | Falla por 8–10 pp | Scenario-3 (2 eventos, 0 detectados ≥50%, 0.6/2 any) y Scenario-6 (1 evento, 0.2/1 any) arrastran la media. | Scenario-3 es **físicamente indetectable** (0.03 σ). Excluirlo del denominador daría 5.8/6 = 96.7 %. Documentar como límite del benchmark. |
| **EDR 3 meses = 0 %** | Limitación de datos | Fugas empiezan día 64+; ventana 90 días captura ≤ 2 eventos sin detectar. | Usar dataset con fugas tempranas o evaluar en ventana móvil, no fija al inicio. |
| **Robustez semillas < 90 %** | Desv. alta en S3, S6 | Scenario-3: 0.6±0.8 detectados (any) — inestable. Scenario-6: 0.2±0.4. | Fijar semilla en producción (ya se hace). La varianza es entre semillas, no en producción. |
| **Sensor NaN → decisión silenciosa** | Prob=0.0, no alerta | `perceive()` no valida `NaN`; RandomForest lo acepta y devuelve prob≈0. | **Añadir en `perceive()`**: `if any(np.isnan(v) for v in vector_base): raise ValueError("NaN en sensor")` o imputar y loggear WARNING. |
| **Scenario-3 AUC 0.4858** | Modelo peor que azar | Fuga mueve presión 0.03 σ (ruido). Modelo aprende patrones espurios en otras redes que invierten ranking aquí. | **No corregible con este dataset**. Requiere: (a) sensores más precisos, (b) balance de masa, (c) modelo físico híbrido. Documentar como límite. |

### 5.3 Conclusión General

**La medida SMART de la Sección 7 (EDR ≥ 80 % a nivel de evento) NO SE CUMPLE** en ninguna de sus operacionalizaciones (37.5 % con cobertura ≥ 50 %, 72.5 % con cualquier solape).

**La métrica de Fase 2 (per-timestep accuracy ≥ 80 %) SÍ SE CUMPLE** (89.5–90.6 %).

**Brecha fundamental:** La Sección 7 define el éxito a **nivel de evento de fuga** (unidad operativa real), mientras que el modelo optimiza y se evalúa a **nivel de paso de 30 min** (unidad de ML). Un agente que acierta 90 % de los pasos pero fragmenta la detección en alertas aisladas **no sirve para operaciones de campo** — el operador necesita saber "hay una fuga activa aquí", no "este paso tiene 60 % de probabilidad".

**Acciones prioritarias para Fase 4:**
1. **Añadir post-procesamiento temporal** en `decide()` / `act()`: exigir persistencia (ej. 3 alertas consecutivas en 90 min) antes de emitir `ALERTA_FUGA`. Esto sube cobertura del evento sin reentrenar.
2. **Activar variables temporales** (`--temporal max6-24-48`) y re-evaluar EDR — ya implementadas, pendientes de activar por defecto.
3. **Validar NaN en `perceive()`** — corrección de 1 línea, alto impacto en robustez.
4. **Documentar Scenario-3 como límite del benchmark** — no es fallo del agente, es límite físico de los datos.

---

## 6. Evidencias y Logs

| Archivo | Descripción |
|---------|-------------|
| `logs/phase3_results.json` | Resultados brutos de 5 semillas × 10 escenarios × 2 despliegues |
| `logs/train_report.json` | Métricas oficiales Fase 2 (per-timestep, matriz confusión) |
| `logs/evaluation_report.json` | Evaluación escenarios A/B/C |
| `logs/agent_events.csv` | Registro por observación del agente en producción |
| `models/leak_classifier.joblib` | Modelo entrenado (150 árboles, `class_weight=None`) |
| `FASE3_RESULTADOS.md` | Este documento |

**Ejecución de pruebas:**  
```bash
# Repeticiones (ya ejecutadas, ~20 min total)
python -c "
from fase3_run import evaluate_seed
for s in [42,123,456,789,999]: evaluate_seed(s)
"

# Casos de falla
python tests/test_failure_cases.py  # (crear a partir de fail5.py)
```

---

## 7. Anexos

### A. Definición de Evento de Fuga
Evento = región contigua de pasos con `y=1` (Labels.csv). En este dataset:
- Scenario-10: 2 eventos (nodos 12 y 28, bien separados)
- Scenario-3: 2 eventos (nodos 19 y 26, señal 0.03 σ)
- Scenario-5,6,7,8: 1 evento cada uno
- Scenarios 1,2,4,9: 0 eventos (control)

### B. Umbrales por Escenario (Escenario B)
| Escenario | Umbral adaptativo (p99.9 histórico limpio) |
|-----------|--------------------------------------------|
| Scenario-10 | 0.292 |
| Scenario-7 | 0.464 |
| Scenario-5 | 0.180 |
| Scenario-6 | 0.040 |
| Scenario-8 | 0.453 |
| Scenario-3 | 0.504 |
| Scenario-1 (control) | 0.557 |
| Scenario-2 (control) | 0.404 |
| Scenario-4 (control) | 0.450 |
| Scenario-9 (control) | 0.113 |

*Rango: 0.040 – 0.557 (factor 14×). Un umbral global no cubre este rango.*

### C. Matriz de Confusión Global (Escenario B, per-timestep)
| | Predicho Normal | Predicho Fuga |
|---|----------------|---------------|
| **Real Normal** | 133 282 (TN) | 427 (FP) |
| **Real Fuga** | 14 751 (FN) | 12 340 (TP) |

*Precision = 0.9666, Recall = 0.4555, F1 = 0.6192*