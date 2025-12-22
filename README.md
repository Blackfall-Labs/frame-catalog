# SAM Vector - Vector Similarity Search and RAG Infrastructure

**High-performance vector search, embeddings, and retrieval-augmented generation (RAG) for AI systems.**

Extracted from the SAM (Societal Advisory Module) project.

## Features

- **HNSW Vector Search**: Sub-millisecond similarity search using Hierarchical Navigable Small World graphs
- **ONNX Embeddings**: MiniLM-L6-v2 text-to-vector conversion (384-dimensional)
- **Persistent Storage**: SQLite-backed vector store with optional BytePunch compression
- **RAG System**: High-level document indexing and retrieval interface
- **Event Database**: Conversation and event storage with metadata

## Quick Start

```toml
[dependencies]
sam-vector = "0.1.0"
```

```rust
use sam_vector::{VectorStore, VectorStoreConfig, OnnxEmbeddingGenerator, EmbeddingGenerator, DocumentChunk};

// Create embedding generator
let embedder = OnnxEmbeddingGenerator::new()?;

// Create vector store
let config = VectorStoreConfig::default();
let mut store = VectorStore::new(config)?;

// Index documents
let chunk = DocumentChunk {
    id: "doc1".to_string(),
    content: "Rust is a systems programming language".to_string(),
    source: "rust-docs".to_string(),
    metadata: None,
};

let embedding = embedder.generate(&chunk.content)?;
store.add_chunk(chunk, &embedding)?;

// Search
let query_embedding = embedder.generate("programming languages")?;
let results = store.search(&query_embedding, 5)?;

for result in results {
    println!("{:.3}: {}", result.score, result.chunk.content);
}
```

## Modules

- **vector_store** (891 LOC) - HNSW similarity search with RwLock thread safety
- **embeddings** (275 LOC) - ONNX embedding generation + simple hash fallback
- **persistent_store** (324 LOC) - SQLite persistence with BytePunch/DataSpool support
- **retrieval** (189 LOC) - High-level RAG interface with automatic chunking
- **database** (582 LOC) - Event/conversation store for chat history

## Performance

- **Search latency**: 0.5-2ms for 10K documents (HNSW ef=50, M=16)
- **Embedding generation**: ~10-50ms per text (MiniLM-L6-v2)
- **Memory**: ~6KB per document (384-dim float32 vectors + metadata)
- **Throughput**: ~1000 searches/sec single-threaded

## Configuration

```rust
use sam_vector::VectorStoreConfig;

let config = VectorStoreConfig {
    ef_construction: 200,  // Build quality (higher = better recall, slower build)
    max_connections: 32,   // Graph degree (higher = better recall, more memory)
    ef_search: 100,        // Search quality (higher = better recall, slower search)
};
```

## Features

```toml
[dependencies]
sam-vector = { version = "0.1.0", features = ["full"] }
```

- `onnx` (default): ONNX Runtime embedding generation
- `persistence`: SQLite persistence with BytePunch/DataSpool
- `full`: All features enabled

## API Reference

### VectorStore

- `new(config) -> Result<Self>` - Create in-memory HNSW index
- `add_chunk(chunk, embedding) -> Result<usize>` - Add document with vector
- `search(embedding, top_k) -> Result<Vec<SearchResult>>` - Find similar documents
- `clear()` - Remove all documents
- `stats() -> VectorStoreStats` - Get index statistics

### EmbeddingGenerator (trait)

- `generate(&self, text: &str) -> Result<Vec<f32>>` - Generate single embedding
- `generate_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>` - Batch processing
- `dimension(&self) -> usize` - Get embedding dimension (384 for MiniLM)

**Implementations:**
- `OnnxEmbeddingGenerator` - MiniLM-L6-v2 via ONNX Runtime
- `SimpleEmbeddingGenerator` - Deterministic hash-based (testing only)

### RetrievalSystem

- `new(embedder, config) -> Result<Self>` - Create RAG system
- `index_document(id, content, source) -> Result<()>` - Auto-chunk and index
- `retrieve(query, top_k) -> Result<Vec<SearchResult>>` - Search with embedding
- `clear()` - Remove all documents

### Database

- `new(path) -> Result<Self>` - Create/open SQLite database
- `create_conversation(id) -> Result<()>` - Start conversation
- `store_event(event) -> Result<()>` - Store timestamped event
- `search_conversation_history(id, embedding, top_k) -> Result<Vec<(Event, f32)>>` - Semantic search

## Testing

```bash
cargo test                    # Run all tests (40 passing, 3 ignored)
cargo test --features full    # Test with all features
```

Ignored tests require ONNX model file (`models/all-minilm-l6-v2.onnx`).

## Compatibility

- **Rust Edition**: 2021
- **MSRV**: 1.70+
- **Platforms**: All (ONNX runtime supports Windows/Linux/macOS)

## Dependencies

- `hnsw_rs` (0.3) - HNSW implementation
- `ort` (2.0.0-rc.10) - ONNX Runtime
- `rusqlite` (0.31) - SQLite database
- `rust_tokenizers` (8.1) - BERT tokenization
- `ndarray` (0.15) - Array operations
- `bytepunch`, `dataspool` (optional) - Compression and bundling

## License

MIT - See [LICENSE](LICENSE) for details.

## Author

Magnus Trent <magnus@blackfall.dev>

## Links

- **GitHub:** https://github.com/Blackfall-Labs/sam-vector
- **Docs:** https://docs.rs/sam-vector
- **Crates.io:** https://crates.io/crates/sam-vector
- **SAM Project:** https://github.com/Blackfall-Labs/sam
