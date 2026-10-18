#!/bin/bash
# Descarga LeakDB Hanoi_CMH desde GitHub KIOS-Research
# Se ejecuta en la etapa BUILD del Dockerfile

set -euo pipefail

DATA_DIR="/app/data"
mkdir -p "$DATA_DIR"

# URL directa al ZIP (raw.githubusercontent.com para descarga directa)
LEAKDB_URL="https://raw.githubusercontent.com/KIOS-Research/LeakDB/master/CCWI-WDSA2018/Benchmarks/Hanoi_CMH.zip"

echo "=========================================="
echo "  Descargando LeakDB Hanoi_CMH"
echo "=========================================="
echo "Fuente: $LEAKDB_URL"
echo "Destino: $DATA_DIR"
echo ""

# Verificar si ya existe (cache de Docker layer)
if [ -f "$DATA_DIR/Nodes.csv" ] && [ -f "$DATA_DIR/Pipes.csv" ] && [ -f "$DATA_DIR/Labels.csv" ]; then
    echo "✅ Dataset ya existe (cache de Docker), saltando descarga"
    ls -la "$DATA_DIR"
    exit 0
fi

echo "⬇️  Descargando (~505 MB)..."
# wget con progreso, timeout y reintentos
wget --progress=bar:force:noscroll \
     --timeout=30 \
     --tries=3 \
     --continue \
     -O /tmp/Hanoi_CMH.zip \
     "$LEAKDB_URL" 2>&1 | tail -20

if [ ! -f /tmp/Hanoi_CMH.zip ]; then
    echo "❌ ERROR: No se pudo descargar el dataset"
    echo "   Verifica la URL: $LEAKDB_URL"
    exit 1
fi

echo ""
echo "📦 Extrayendo..."
unzip -q -o /tmp/Hanoi_CMH.zip -d "$DATA_DIR"

# Verificar estructura esperada
REQUIRED_FILES=("Nodes.csv" "Pipes.csv" "Labels.csv")
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "❌ Falta archivo requerido: $f"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "❌ Estructura inesperada. Contenido de $DATA_DIR:"
    find "$DATA_DIR" -type f | head -20
    exit 1
fi

echo ""
echo "✅ Dataset LeakDB Hanoi_CMH listo:"
ls -la "$DATA_DIR"/*.csv

# Limpiar ZIP temporal
rm -f /tmp/Hanoi_CMH.zip

echo "=========================================="