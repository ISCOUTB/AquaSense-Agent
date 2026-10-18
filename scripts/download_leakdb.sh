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
# Buscar archivos clave en cualquier subdirectorio
if find "$DATA_DIR" -name "Nodes.csv" -o -name "Pipes.csv" -o -name "Labels.csv" 2>/dev/null | grep -q .; then
    echo "✅ Dataset ya existe (cache de Docker), saltando descarga"
    find "$DATA_DIR" -name "*.csv" | head -10
    exit 0
fi

echo "⬇️  Descargando (~76 MB)..."
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
# Extraer a directorio temporal primero
TMP_EXTRACT="/tmp/leakdb_extract"
rm -rf "$TMP_EXTRACT"
mkdir -p "$TMP_EXTRACT"
unzip -q -o /tmp/Hanoi_CMH.zip -d "$TMP_EXTRACT"

echo "🔍 Buscando archivos CSV en la extracción..."
# El ZIP puede tener estructura Hanoi_CMH/Scenario-1/, etc.
# Buscar y copiar todos los CSV al DATA_DIR plano
find "$TMP_EXTRACT" -name "*.csv" -exec cp {} "$DATA_DIR"/ \;

# Verificar archivos clave
REQUIRED_FILES=("Nodes.csv" "Pipes.csv" "Labels.csv")
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "❌ Falta archivo requerido: $f"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo "❌ No se encontraron todos los archivos requeridos."
    echo "   Contenido de $DATA_DIR:"
    ls -la "$DATA_DIR"/*.csv 2>/dev/null | head -20
    echo "   Contenido de extracción temporal:"
    find "$TMP_EXTRACT" -name "*.csv" | head -20
    exit 1
fi

echo ""
echo "✅ Dataset LeakDB Hanoi_CMH listo (archivos CSV en raíz):"
ls -la "$DATA_DIR"/*.csv

# Limpiar
rm -f /tmp/Hanoi_CMH.zip
rm -rf "$TMP_EXTRACT"

echo "=========================================="