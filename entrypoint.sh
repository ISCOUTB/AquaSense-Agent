#!/bin/bash
# Entrypoint para AquaSense-Agent Docker
# Modos: train | evaluate | agent | streamlit | bash

set -euo pipefail

echo "=========================================="
echo "  🌊 AquaSense-Agent Docker Runtime"
echo "=========================================="
echo "Modo: ${1:-train}"
echo "Directorio: $(pwd)"
echo ""

# Verificar/Crear directorios necesarios
mkdir -p /app/data /app/models /app/logs

# Named Docker/Azure volumes mask image contents; seed them on first startup.
if [ ! -f /app/models/leak_classifier.joblib ] && [ -f /app/model-seed/leak_classifier.joblib ]; then
    cp -a /app/model-seed/. /app/models/
fi
if [ ! -f /app/logs/train_report.json ] && [ -d /app/log-seed ]; then
    cp -a /app/log-seed/. /app/logs/
fi

# Función: verificar si hay dataset completo (estructura Scenario-N/)
check_full_dataset() {
    find /app/data -maxdepth 2 -name "Scenario-*" -type d 2>/dev/null | grep -q .
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
else
    echo "❌ ERROR: Dataset completo NO encontrado en /app/data/"
    echo "   El build debería haber descargado LeakDB automáticamente."
    exit 1
fi

# Verificar modelo
if ! check_model; then
    echo "⚠️  Modelo no encontrado, entrenando..."
    train_model
else
    echo "✅ Modelo encontrado: /app/models/leak_classifier.joblib"
fi

# Parsear modo
MODE="${1:-train}"
shift || true

case "$MODE" in
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

    streamlit)
        echo "🌐 Iniciando interfaz Streamlit en el puerto 8501..."
        exec streamlit run /app/streamlit_app.py \
            --server.address=0.0.0.0 \
            --server.port=8501 \
            --server.headless=true
        ;;

    bash|sh)
        echo "🐚 Entrando a shell interactivo..."
        exec /bin/bash
        ;;

    *)
        echo "❓ Modo desconocido: $MODE"
        echo "Modos disponibles: train | evaluate | agent | streamlit | bash"
        exit 1
        ;;
esac