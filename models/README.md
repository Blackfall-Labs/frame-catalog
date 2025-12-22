# Embedding Models

This directory contains ONNX models for generating semantic embeddings.

## all-MiniLM-L6-v2

**Model:** sentence-transformers/all-MiniLM-L6-v2
**Size:** ~80MB
**Dimensions:** 384
**Use case:** General-purpose semantic similarity

### Download

```bash
# Download ONNX model (87 MB)
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx -O all-minilm-l6-v2.onnx

# Download vocabulary (227 KB) - REQUIRED
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt -O vocab.txt

# Download tokenizer (456 KB)
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json -O all-minilm-l6-v2-tokenizer.json
```

**Required Files:**
- ✅ all-minilm-l6-v2.onnx (87 MB) - ONNX model
- ✅ vocab.txt (227 KB) - BERT vocabulary (REQUIRED for rust_tokenizers)
- ✅ all-minilm-l6-v2-tokenizer.json (456 KB) - HuggingFace tokenizer config

Or use the download script:

```bash
cd crates/frame-catalog
cargo run --example download_model
```

### License

Apache 2.0 - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### Performance

- **Inference:** ~5-10ms per query on modern CPU
- **Memory:** ~200MB loaded
- **Quality:** 0.85+ on STS benchmark
