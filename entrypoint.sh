#!/bin/bash
# Entrypoint para AquaSense-Agent Docker
# Modos: streamlit (default) | train | evaluate | agent | bash

set -euo pipefail

echo "=========================================="
echo "  🌊 AquaSense-Agent Docker Runtime"
echo "=========================================="
echo "Modo: ${1:-streamlit}"
echo "Directorio: $(pwd)"
echo ""

# Verificar/Crear directorios necesarios
mkdir -p /app/data /app/models /app/logs /app/demo_data

# Función: verificar si hay dataset completo
check_full_dataset() {
    [ -f "/app/data/Nodes.csv" ] && \
    [ -f "/app/data/Pipes.csv" ] && \
    [ -f "/app/data/Labels.csv" ]
}

# Función: verificar si hay modelo entrenado
check_model() {
    [ -f "/app/models/leak_classifier.joblib" ]
}

# Función: entrenar modelo
train_model() {
    echo "🔧 Entrenando modelo (LOSO 10 folds, balance=none)..."
    python train.py --balance none
    echo "✅ Modelo entrenado y guardado en /app/models/"
}

# Función: evaluar modelo
evaluate_model() {
    echo "📊 Evaluando modelo..."
    python evaluate.py
}

# Verificar datos
if check_full_dataset; then
    echo "📊 Dataset completo detectado: /app/data/"
    export AQUASENSE_DEMO_MODE=0
else
    echo "⚠️  Dataset completo NO encontrado"
    if [ -f "/app/demo_data/X_demo.csv" ]; then
        echo "📦 Usando dataset DEMO (Scenario-7, 2000 pasos)"
        export AQUASENSE_DEMO_MODE=1
    else
        echo "❌ ERROR: Ni dataset completo ni demo_data/"
        echo "   Ejecuta: docker compose build  (para descargar LeakDB)"
        exit 1
    fi
fi

# Verificar modelo
if ! check_model; then
    echo "⚠️  Modelo no encontrado, entrenando..."
    train_model
else
    echo "✅ Modelo encontrado: /app/models/leak_classifier.joblib"
fi

# Parsear modo
MODE="${1:-streamlit}"
shift || true

case "$MODE" in
    streamlit)
        echo "🚀 Iniciando Streamlit en http://0.0.0.0:8501"
        echo "   Abre en navegador: http://localhost:8501"
        exec streamlit run streamlit_app.py \
            --server.address=0.0.0.0 \
            --server.port=8501 \
            --server.headless=true \
            --browser.gatherUsageStats=false
        ;;

    train)
        echo "🔧 Re-entrenando modelo completo..."
        train_model
        ;;

    evaluate)
        evaluate_model
        ;;

    agent)
        echo "🤖 Agente CLI: python agent.py $@"
        exec python agent.py "$@"
        ;;

    bash|sh)
        echo "🐚 Entrando a shell interactivo..."
        exec /bin/bash
        ;;

    *)
        echo "❓ Modo desconocido: $MODE"
        echo "Modos disponibles: streamlit | train | evaluate | agent | bash"
        exit 1
        ;;
esac