#!/usr/bin/env bash
# Download ONNX models for frame-catalog

set -e

MODELS_DIR="$(dirname "$0")/models"
mkdir -p "$MODELS_DIR"

echo "Downloading MiniLM-L6-v2 ONNX model..."

# Model URLs (HuggingFace)
MODEL_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
TOKENIZER_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
VOCAB_URL="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"

# Download model
if [ ! -f "$MODELS_DIR/all-minilm-l6-v2.onnx" ]; then
    echo "Downloading ONNX model (87MB)..."
    curl -L "$MODEL_URL" -o "$MODELS_DIR/all-minilm-l6-v2.onnx"
else
    echo "Model already exists, skipping download."
fi

# Download tokenizer
if [ ! -f "$MODELS_DIR/all-minilm-l6-v2-tokenizer.json" ]; then
    echo "Downloading tokenizer..."
    curl -L "$TOKENIZER_URL" -o "$MODELS_DIR/all-minilm-l6-v2-tokenizer.json"
else
    echo "Tokenizer already exists, skipping download."
fi

# Download vocab
if [ ! -f "$MODELS_DIR/vocab.txt" ]; then
    echo "Downloading vocab..."
    curl -L "$VOCAB_URL" -o "$MODELS_DIR/vocab.txt"
else
    echo "Vocab already exists, skipping download."
fi

echo "✓ Models downloaded successfully to $MODELS_DIR"
echo "  - all-minilm-l6-v2.onnx (87MB)"
echo "  - all-minilm-l6-v2-tokenizer.json (456KB)"
echo "  - vocab.txt (227KB)"
