//! # SAM Vector - Vector Similarity Search and RAG Infrastructure
//!
//! High-performance vector search, embeddings, and retrieval-augmented generation (RAG)
//! for AI systems.
//!
//! ## Features
//!
//! ### 🔍 HNSW Vector Search
//!
//! Fast approximate nearest neighbor search using Hierarchical Navigable Small World graphs:
//!
//! - **Sub-millisecond queries**: ~0.5-2ms for 10K documents
//! - **384-dimensional embeddings**: MiniLM-L6-v2 compatible
//! - **In-memory index**: Optimized for speed
//! - **Thread-safe**: Concurrent read access with RwLock
//!
//! ### 🧠 ONNX Embeddings
//!
//! Text-to-vector conversion using ONNX Runtime:
//!
//! - **MiniLM-L6-v2** model (87MB, 384-dim vectors)
//! - **Batch processing**: Encode multiple texts efficiently
//! - **Normalization**: L2-normalized embeddings
//! - **Fallback**: Simple hash-based embeddings for testing
//!
//! ### 💾 Persistent Storage
//!
//! SQLite-backed vector store with optional compression:
//!
//! - **Document references**: Store file paths or spool offsets
//! - **BytePunch compression**: 40-70% space savings
//! - **DataSpool integration**: Bundle multiple documents
//! - **Lazy loading**: Load embeddings on demand
//!
//! ### 📚 RAG System
//!
//! High-level retrieval interface:
//!
//! - **Automatic chunking**: Split documents with overlap
//! - **Index + search**: One-step document indexing
//! - **Configurable**: Chunk size, overlap, HNSW parameters
//!
//! ### 🗄️ Event Database
//!
//! Conversation and event storage:
//!
//! - **Conversation tracking**: Session-based organization
//! - **Event history**: Timestamped event log
//! - **Metadata storage**: JSON metadata per event
//! - **Search support**: Retrieve events by conversation ID
//!
//! ## Usage
//!
//! ```rust,no_run
//! use sam_vector::{VectorStore, VectorStoreConfig};
//! use sam_vector::{OnnxEmbeddingGenerator, EmbeddingGenerator};
//! use sam_vector::DocumentChunk;
//!
//! // Create embedding generator
//! let embedder = OnnxEmbeddingGenerator::new().unwrap();
//!
//! // Create vector store
//! let config = VectorStoreConfig::default();
//! let mut store = VectorStore::new(config).unwrap();
//!
//! // Index documents
//! let chunk = DocumentChunk {
//!     id: "doc1".to_string(),
//!     content: "Rust is a systems programming language".to_string(),
//!     source: "rust-docs".to_string(),
//!     metadata: None,
//! };
//!
//! let embedding = embedder.generate(&chunk.content).unwrap();
//! store.add_chunk(chunk, &embedding).unwrap();
//!
//! // Search
//! let query_embedding = embedder.generate("programming languages").unwrap();
//! let results = store.search(&query_embedding, 5).unwrap();
//!
//! for result in results {
//!     println!("{:.3}: {}", result.score, result.chunk.content);
//! }
//! ```

pub mod vector_store;
pub mod embeddings;
pub mod database;

#[cfg(feature = "persistence")]
pub mod persistent_store;

pub mod retrieval;

// Re-export main types
pub use vector_store::{
    VectorStore, VectorStoreConfig, VectorStoreError,
    DocumentChunk, SearchResult, EMBEDDING_DIM,
};

pub use embeddings::{
    EmbeddingGenerator, EmbeddingError,
    SimpleEmbeddingGenerator,
};

#[cfg(feature = "onnx")]
pub use embeddings::OnnxEmbeddingGenerator;

pub use database::{Database, DatabaseError, StoredEvent, Conversation};

#[cfg(feature = "persistence")]
pub use persistent_store::{PersistentVectorStore, DocumentRef, FileSearchResult};

pub use retrieval::{RetrievalSystem, RetrievalConfig, RetrievalError};
