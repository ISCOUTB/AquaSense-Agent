# AquaSense-Agent - Docker Multi-stage Build
# ==========================================
# Stage 1: Builder - descarga dataset, instala deps, entrena modelo
# Stage 2: Runtime - imagen ligera solo con artefactos necesarios

# ==========================================
# STAGE 1: BUILDER
# ==========================================
FROM python:3.11-slim AS builder

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias de sistema para descargar/extraer LeakDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Descargar LeakDB Hanoi_CMH (capa cacheable - solo cambia si cambia el script)
RUN chmod +x scripts/download_leakdb.sh && ./scripts/download_leakdb.sh

# Entrenar modelo (capa cacheable - solo cambia si cambian datos/código)
RUN python train.py --balance none

# Verificar artefactos generados
RUN ls -la /app/models/ /app/logs/ /app/data/

# ==========================================
# STAGE 2: RUNTIME
# ==========================================
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Solo dependencias runtime mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar SOLO artefactos necesarios del builder
COPY --from=builder /app/models /app/models
COPY --from=builder /app/logs /app/logs
COPY --from=builder /app/data /app/data
COPY --from=builder /app/demo_data /app/demo_data

# Copiar código Python necesario
COPY --from=builder /app/streamlit_app.py /app/
COPY --from=builder /app/agent.py /app/
COPY --from=builder /app/data_loader.py /app/
COPY --from=builder /app/evaluate.py /app/
COPY --from=builder /app/train.py /app/
COPY --from=builder /app/main.py /app/
COPY --from=builder /app/inspect_data.py /app/

# Copiar entrypoint
COPY --from=builder /app/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Puerto Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["streamlit"]