#!/bin/bash
# Descarga LeakDB Hanoi_CMH desde GitHub KIOS-Research
# Se ejecuta en la etapa BUILD del Dockerfile
# El data_loader espera la estructura original: Scenario-N/Pressures/, Flows/, Demands/, Labels.csv

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
# Buscar estructura de escenarios
if find "$DATA_DIR" -maxdepth 2 -name "Scenario-*" -type d 2>/dev/null | grep -q .; then
    echo "✅ Dataset ya existe (cache de Docker), saltando descarga"
    find "$DATA_DIR" -maxdepth 2 -name "Scenario-*" -type d | head -5
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
echo "📦 Extrayendo preservando estructura de directorios..."
# Extraer directamente a DATA_DIR (el ZIP contiene Hanoi_CMH/Scenario-*/)
unzip -q -o /tmp/Hanoi_CMH.zip -d "$DATA_DIR"

# El ZIP extrae como Hanoi_CMH/Scenario-*/ - mover contenido a raíz de DATA_DIR
if [ -d "$DATA_DIR/Hanoi_CMH" ]; then
    echo "📂 Moviendo Hanoi_CMH/* a $DATA_DIR/"
    mv "$DATA_DIR/Hanoi_CMH"/* "$DATA_DIR"/
    rmdir "$DATA_DIR/Hanoi_CMH"
fi

# Verificar estructura esperada por data_loader
REQUIRED_DIRS=("Scenario-1/Pressures" "Scenario-1/Flows" "Scenario-1/Demands")
MISSING=0
for d in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$DATA_DIR/$d" ]; then
        echo "❌ Falta directorio requerido: $d"
        MISSING=1
    fi
done

if [ ! -f "$DATA_DIR/Scenario-1/Labels.csv" ]; then
    echo "❌ Falta archivo requerido: Scenario-1/Labels.csv"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo "❌ Estructura incompleta. Contenido de $DATA_DIR:"
    find "$DATA_DIR" -maxdepth 2 -type d | head -20
    exit 1
fi

echo ""
echo "✅ Dataset LeakDB Hanoi_CMH listo con estructura original:"
find "$DATA_DIR" -maxdepth 2 -name "Scenario-*" -type d | head -10
echo ""
echo "Archivos en Scenario-1:"
ls -la "$DATA_DIR/Scenario-1/"

# Limpiar ZIP temporal
rm -f /tmp/Hanoi_CMH.zip

echo "=========================================="