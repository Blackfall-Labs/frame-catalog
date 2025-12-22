# Changelog

## [0.1.0] - 2025-12-21

### Added
- Initial release extracted from Frame project
- **HNSW Vector Search**: Fast similarity search with sub-millisecond queries
  - Hierarchical Navigable Small World graph index
  - RwLock for thread-safe concurrent reads
  - Configurable ef_construction, max_connections, ef_search parameters
  - 384-dimensional embedding support (MiniLM-L6-v2 compatible)
- **ONNX Embeddings**: Text-to-vector conversion
  - MiniLM-L6-v2 model integration (87MB, 384-dim)
  - Batch processing support
  - L2 normalization
  - Simple hash-based fallback for testing
- **Persistent Storage**: SQLite-backed vector store
  - Document reference storage (file paths or spool offsets)
  - BytePunch compression integration (40-70% savings)
  - DataSpool bundling support
  - Lazy embedding loading
- **RAG System**: High-level retrieval interface
  - Automatic document chunking with overlap
  - One-step indexing and search
  - Configurable chunk sizes
- **Event Database**: Conversation and event storage
  - Session-based organization
  - Timestamped event log
  - JSON metadata support
  - Semantic search over conversation history

### Modules
- vector_store.rs (891 LOC) - HNSW search implementation
- embeddings.rs (275 LOC) - Embedding generation trait + implementations
- persistent_store.rs (324 LOC) - SQLite persistence layer
- retrieval.rs (189 LOC) - High-level RAG interface
- database.rs (582 LOC) - Event/conversation store

### Performance
- Search: 0.5-2ms for 10K documents (HNSW ef=50, M=16)
- Embedding: ~10-50ms per text (MiniLM-L6-v2)
- Memory: ~6KB per document (vectors + metadata)
- Throughput: ~1000 searches/sec single-threaded

### Dependencies
- hnsw_rs 0.3 (HNSW implementation)
- ort 2.0.0-rc.10 (ONNX Runtime)
- rusqlite 0.31 (SQLite)
- rust_tokenizers 8.1 (BERT tokenization)
- ndarray 0.15 (array operations)
- bytepunch, dataspool (optional, for persistence feature)

### Features
- `onnx` (default) - ONNX Runtime embeddings
- `persistence` - SQLite + compression support
- `full` - All features enabled

### Testing
- 40 unit tests passing
- 3 tests ignored (require ONNX model file)
- 1 doctest passing

### Notes
- Extracted from [Frame](https://github.com/Blackfall-Labs/sam)
- Platform-independent with ONNX Runtime
- Production-ready for RAG applications
- Designed for offline-first AI systems
