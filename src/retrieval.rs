//! Document indexing and retrieval system
//!
//! Provides high-level RAG (Retrieval-Augmented Generation) functionality

use crate::embeddings::{EmbeddingGenerator, SimpleEmbeddingGenerator};
#[cfg(feature = "onnx")]
use crate::embeddings::OnnxEmbeddingGenerator;
use crate::vector_store::{DocumentChunk, SearchResult, VectorStore, VectorStoreConfig};
use std::sync::Arc;

/// Retrieval system error
#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    #[error("Vector store error: {0}")]
    VectorStore(#[from] crate::vector_store::VectorStoreError),

    #[error("Embedding error: {0}")]
    Embedding(#[from] crate::embeddings::EmbeddingError),

    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(usize),
}

pub type Result<T> = std::result::Result<T, RetrievalError>;

/// Configuration for document retrieval
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum chunk size in characters
    pub max_chunk_size: usize,

    /// Overlap between chunks in characters
    pub chunk_overlap: usize,

    /// Vector store configuration
    pub vector_config: VectorStoreConfig,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 512,
            chunk_overlap: 50,
            vector_config: VectorStoreConfig::default(),
        }
    }
}

/// Document retrieval system for RAG
pub struct RetrievalSystem {
    /// Vector store for similarity search
    vector_store: VectorStore,

    /// Embedding generator
    embedder: Arc<dyn EmbeddingGenerator>,

    /// Configuration
    config: RetrievalConfig,
}

impl RetrievalSystem {
    /// Create a new retrieval system
    ///
    /// Attempts to use ONNX embeddings (semantic) with fallback to simple embeddings (hash-based)
    pub fn new(config: RetrievalConfig) -> Result<Self> {
        let vector_store = VectorStore::new(config.vector_config.clone())?;

        // Try to use ONNX embeddings first (real semantic embeddings)
        #[cfg(feature = "onnx")]
        let embedder: Arc<dyn EmbeddingGenerator> = match OnnxEmbeddingGenerator::new() {
            Ok(onnx_gen) => {
                tracing::info!("Using ONNX semantic embeddings (MiniLM-L6-v2, 384d)");
                Arc::new(onnx_gen)
            }
            Err(e) => {
                tracing::warn!(
                    "ONNX embeddings unavailable ({}), falling back to simple hash-based embeddings",
                    e
                );
                Arc::new(SimpleEmbeddingGenerator::new())
            }
        };

        #[cfg(not(feature = "onnx"))]
        let embedder: Arc<dyn EmbeddingGenerator> = {
            tracing::info!("Using simple hash-based embeddings (ONNX feature not enabled)");
            Arc::new(SimpleEmbeddingGenerator::new())
        };

        Ok(Self {
            vector_store,
            embedder,
            config,
        })
    }

    /// Create a new retrieval system with a custom embedding generator
    pub fn with_embedder(
        config: RetrievalConfig,
        embedder: Arc<dyn EmbeddingGenerator>,
    ) -> Result<Self> {
        let vector_store = VectorStore::new(config.vector_config.clone())?;

        Ok(Self {
            vector_store,
            embedder,
            config,
        })
    }

    /// Index a document by splitting it into chunks
    pub fn index_document(
        &self,
        document_id: &str,
        content: &str,
        source: &str,
        metadata: Option<String>,
    ) -> Result<usize> {
        let chunks = self.split_into_chunks(content);
        let mut indexed_count = 0;

        for (i, chunk_text) in chunks.iter().enumerate() {
            let chunk = DocumentChunk {
                id: format!("{}:{}", document_id, i),
                content: chunk_text.to_string(),
                source: source.to_string(),
                metadata: metadata.clone(),
            };

            // Generate embedding
            let embedding = self.embedder.generate(chunk_text)?;

            // Add to vector store
            self.vector_store.add_chunk(chunk, &embedding)?;
            indexed_count += 1;
        }

        Ok(indexed_count)
    }

    /// Retrieve relevant documents for a query
    pub fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let query_embedding = self.embedder.generate(query)?;

        // Search vector store
        let results = self.vector_store.search(&query_embedding, top_k)?;

        Ok(results)
    }

    /// Split text into overlapping chunks
    fn split_into_chunks(&self, text: &str) -> Vec<String> {
        let max_size = self.config.max_chunk_size;
        let overlap = self.config.chunk_overlap;

        if text.len() <= max_size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            let end = (start + max_size).min(text.len());
            let chunk = &text[start..end];
            chunks.push(chunk.to_string());

            if end >= text.len() {
                break;
            }

            // Move forward with overlap
            start += max_size - overlap;
        }

        chunks
    }

    /// Get the number of indexed chunks
    pub fn chunk_count(&self) -> usize {
        self.vector_store.len()
    }

    /// Clear all indexed documents
    pub fn clear(&self) {
        self.vector_store.clear();
    }

    /// Get retrieval statistics
    pub fn stats(&self) -> RetrievalStats {
        let vector_stats = self.vector_store.stats();
        RetrievalStats {
            num_chunks: vector_stats.num_chunks,
            embedding_dim: vector_stats.embedding_dim,
            max_chunk_size: self.config.max_chunk_size,
            chunk_overlap: self.config.chunk_overlap,
        }
    }
}

/// Retrieval system statistics
#[derive(Debug, Clone)]
pub struct RetrievalStats {
    pub num_chunks: usize,
    pub embedding_dim: usize,
    pub max_chunk_size: usize,
    pub chunk_overlap: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrieval_system_creation() {
        let system = RetrievalSystem::new(RetrievalConfig::default()).unwrap();
        assert_eq!(system.chunk_count(), 0);
    }

    #[test]
    fn test_split_into_chunks() {
        let config = RetrievalConfig {
            max_chunk_size: 20,
            chunk_overlap: 5,
            ..Default::default()
        };
        let system = RetrievalSystem::new(config).unwrap();

        let text = "This is a test document that should be split into multiple chunks.";
        let chunks = system.split_into_chunks(text);

        assert!(chunks.len() > 1);
        // Verify each chunk is within size limit
        for chunk in &chunks {
            assert!(chunk.len() <= 20);
        }
    }

    #[test]
    fn test_index_short_document() {
        let system = RetrievalSystem::new(RetrievalConfig::default()).unwrap();

        let count = system
            .index_document("doc1", "This is a short test document.", "test.txt", None)
            .unwrap();

        assert_eq!(count, 1);
        assert_eq!(system.chunk_count(), 1);
    }

    #[test]
    fn test_index_long_document() {
        let config = RetrievalConfig {
            max_chunk_size: 50,
            chunk_overlap: 10,
            ..Default::default()
        };
        let system = RetrievalSystem::new(config).unwrap();

        let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.";

        let count = system
            .index_document("doc1", long_text, "test.txt", None)
            .unwrap();

        assert!(count > 1);
        assert_eq!(system.chunk_count(), count);
    }

    #[test]
    fn test_retrieve() {
        let system = RetrievalSystem::new(RetrievalConfig::default()).unwrap();

        // Index multiple documents
        system
            .index_document(
                "doc1",
                "The quick brown fox jumps over the lazy dog.",
                "animals.txt",
                None,
            )
            .unwrap();

        system
            .index_document(
                "doc2",
                "Rust is a systems programming language.",
                "programming.txt",
                None,
            )
            .unwrap();

        system
            .index_document(
                "doc3",
                "Machine learning and artificial intelligence.",
                "ai.txt",
                None,
            )
            .unwrap();

        // Retrieve relevant documents
        let results = system.retrieve("programming language", 2).unwrap();

        assert!(results.len() > 0);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_retrieve_relevance_order() {
        let system = RetrievalSystem::new(RetrievalConfig::default()).unwrap();

        // Index documents
        system
            .index_document("doc1", "The fox is brown and quick.", "test1.txt", None)
            .unwrap();

        system
            .index_document("doc2", "The fox jumps over the dog.", "test2.txt", None)
            .unwrap();

        system
            .index_document(
                "doc3",
                "Completely unrelated content about programming.",
                "test3.txt",
                None,
            )
            .unwrap();

        // Query should find fox-related documents
        let results = system.retrieve("fox", 3).unwrap();

        // Results should be ordered by relevance (score)
        for i in 0..results.len().saturating_sub(1) {
            assert!(results[i].score >= results[i + 1].score);
        }
    }

    #[test]
    fn test_clear() {
        let system = RetrievalSystem::new(RetrievalConfig::default()).unwrap();

        system
            .index_document("doc1", "Test document", "test.txt", None)
            .unwrap();

        assert_eq!(system.chunk_count(), 1);

        system.clear();

        assert_eq!(system.chunk_count(), 0);
    }

    #[test]
    fn test_stats() {
        let config = RetrievalConfig {
            max_chunk_size: 100,
            chunk_overlap: 20,
            ..Default::default()
        };
        let system = RetrievalSystem::new(config).unwrap();

        system
            .index_document("doc1", "Test document", "test.txt", None)
            .unwrap();

        let stats = system.stats();
        assert_eq!(stats.num_chunks, 1);
        assert_eq!(stats.max_chunk_size, 100);
        assert_eq!(stats.chunk_overlap, 20);
        assert!(stats.embedding_dim > 0);
    }

    #[test]
    fn test_chunk_with_metadata() {
        let system = RetrievalSystem::new(RetrievalConfig::default()).unwrap();

        let metadata = serde_json::json!({
            "author": "Test Author",
            "date": "2024-01-01"
        })
        .to_string();

        let count = system
            .index_document("doc1", "Test content", "test.txt", Some(metadata))
            .unwrap();

        assert_eq!(count, 1);
    }
}
