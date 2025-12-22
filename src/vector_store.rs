//! Vector store for knowledge retrieval using HNSW
//!
//! Provides fast similarity search for RAG (Retrieval-Augmented Generation)

use hnsw_rs::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Vector dimension (typical for small embedding models)
pub const EMBEDDING_DIM: usize = 384;

/// Vector store error
#[derive(Debug, thiserror::Error)]
pub enum VectorStoreError {
    #[error("HNSW error: {0}")]
    Hnsw(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid embedding dimension: expected {EMBEDDING_DIM}, got {0}")]
    InvalidDimension(usize),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, VectorStoreError>;

/// Document chunk stored in the vector store
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    /// Unique identifier
    pub id: String,

    /// Text content
    pub content: String,

    /// Source document
    pub source: String,

    /// Metadata (JSON serialized)
    pub metadata: Option<String>,
}

/// Search result with similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Document chunk
    pub chunk: DocumentChunk,

    /// Similarity score (higher is more similar)
    pub score: f32,
}

/// Vector store using HNSW for fast similarity search
pub struct VectorStore {
    /// HNSW index for fast approximate nearest neighbor search
    hnsw: Arc<RwLock<Hnsw<'static, f32, DistL2>>>,

    /// Document lookup by ID
    documents: Arc<RwLock<HashMap<usize, DocumentChunk>>>,

    /// Next document ID
    next_id: Arc<RwLock<usize>>,

    /// Configuration
    config: VectorStoreConfig,
}

/// Vector store configuration
#[derive(Debug, Clone)]
pub struct VectorStoreConfig {
    /// Maximum number of connections per layer
    pub max_connections: usize,

    /// Size of the dynamic candidate list
    pub ef_construction: usize,

    /// Maximum number of elements
    pub max_elements: usize,

    /// Search ef parameter (higher = more accurate but slower)
    pub ef_search: usize,
}

impl Default for VectorStoreConfig {
    fn default() -> Self {
        Self {
            max_connections: 16,
            ef_construction: 200,
            max_elements: 100_000,
            ef_search: 100,
        }
    }
}

impl VectorStore {
    /// Create a new vector store
    pub fn new(config: VectorStoreConfig) -> Result<Self> {
        let hnsw = Hnsw::<f32, DistL2>::new(
            config.max_connections,
            config.max_elements,
            EMBEDDING_DIM,
            config.ef_construction,
            DistL2 {},
        );

        Ok(Self {
            hnsw: Arc::new(RwLock::new(hnsw)),
            documents: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
            config,
        })
    }

    /// Add a document chunk with its embedding
    pub fn add_chunk(&self, chunk: DocumentChunk, embedding: &[f32]) -> Result<usize> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(VectorStoreError::InvalidDimension(embedding.len()));
        }

        // Get next ID
        let mut next_id = self.next_id.write().unwrap();
        let id = *next_id;
        *next_id += 1;
        drop(next_id);

        // Insert into HNSW
        let hnsw = self.hnsw.write().unwrap();
        hnsw.insert((embedding, id));
        drop(hnsw);

        // Store document
        let mut documents = self.documents.write().unwrap();
        documents.insert(id, chunk);

        Ok(id)
    }

    /// Search for similar chunks
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query_embedding.len() != EMBEDDING_DIM {
            return Err(VectorStoreError::InvalidDimension(query_embedding.len()));
        }

        // Perform search
        let results = {
            let hnsw = self.hnsw.read().unwrap();
            hnsw.search(query_embedding, top_k, self.config.ef_search)
        };

        // Convert to SearchResult
        let documents = self.documents.read().unwrap();
        let search_results: Vec<SearchResult> = results
            .iter()
            .filter_map(|neighbor| {
                let id = neighbor.d_id;
                documents.get(&id).map(|chunk| SearchResult {
                    chunk: chunk.clone(),
                    score: 1.0 / (1.0 + neighbor.distance), // Convert distance to similarity
                })
            })
            .collect();

        Ok(search_results)
    }

    /// Get a chunk by internal ID
    pub fn get_chunk(&self, id: usize) -> Result<DocumentChunk> {
        let documents = self.documents.read().unwrap();
        documents
            .get(&id)
            .cloned()
            .ok_or_else(|| VectorStoreError::NotFound(format!("Document ID {}", id)))
    }

    /// Get total number of chunks
    pub fn len(&self) -> usize {
        self.documents.read().unwrap().len()
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all data
    pub fn clear(&self) {
        let mut documents = self.documents.write().unwrap();
        documents.clear();

        let mut next_id = self.next_id.write().unwrap();
        *next_id = 0;

        // Recreate HNSW index
        let mut hnsw = self.hnsw.write().unwrap();
        *hnsw = Hnsw::<f32, DistL2>::new(
            self.config.max_connections,
            self.config.max_elements,
            EMBEDDING_DIM,
            self.config.ef_construction,
            DistL2 {},
        );
    }

    /// Get statistics
    pub fn stats(&self) -> VectorStoreStats {
        let documents = self.documents.read().unwrap();
        VectorStoreStats {
            num_chunks: documents.len(),
            embedding_dim: EMBEDDING_DIM,
        }
    }
}

/// Vector store statistics
#[derive(Debug, Clone)]
pub struct VectorStoreStats {
    pub num_chunks: usize,
    pub embedding_dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create dummy embeddings
    fn create_embedding(seed: f32) -> Vec<f32> {
        (0..EMBEDDING_DIM)
            .map(|i| (i as f32 * seed).sin())
            .collect()
    }

    #[test]
    fn test_vector_store_creation() {
        let store = VectorStore::new(VectorStoreConfig::default()).unwrap();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_add_chunk() {
        let store = VectorStore::new(VectorStoreConfig::default()).unwrap();

        let chunk = DocumentChunk {
            id: "chunk1".to_string(),
            content: "This is a test document".to_string(),
            source: "test.txt".to_string(),
            metadata: None,
        };

        let embedding = create_embedding(1.0);
        let id = store.add_chunk(chunk, &embedding).unwrap();

        assert_eq!(store.len(), 1);

        let retrieved = store.get_chunk(id).unwrap();
        assert_eq!(retrieved.id, "chunk1");
        assert_eq!(retrieved.content, "This is a test document");
    }

    #[test]
    fn test_invalid_embedding_dimension() {
        let store = VectorStore::new(VectorStoreConfig::default()).unwrap();

        let chunk = DocumentChunk {
            id: "chunk1".to_string(),
            content: "Test".to_string(),
            source: "test.txt".to_string(),
            metadata: None,
        };

        let wrong_embedding = vec![0.0; 128]; // Wrong dimension
        let result = store.add_chunk(chunk, &wrong_embedding);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VectorStoreError::InvalidDimension(_)
        ));
    }

    #[test]
    fn test_search() {
        let store = VectorStore::new(VectorStoreConfig::default()).unwrap();

        // Add some chunks
        for i in 0..5 {
            let chunk = DocumentChunk {
                id: format!("chunk{}", i),
                content: format!("Document number {}", i),
                source: "test.txt".to_string(),
                metadata: None,
            };
            let embedding = create_embedding(i as f32);
            store.add_chunk(chunk, &embedding).unwrap();
        }

        // Search with a query similar to chunk 2
        let query = create_embedding(2.0);
        let results = store.search(&query, 3).unwrap();

        assert!(results.len() > 0);
        assert!(results.len() <= 3);

        // The most similar should be chunk2
        assert_eq!(results[0].chunk.id, "chunk2");
    }

    #[test]
    fn test_clear() {
        let store = VectorStore::new(VectorStoreConfig::default()).unwrap();

        // Add chunks
        for i in 0..3 {
            let chunk = DocumentChunk {
                id: format!("chunk{}", i),
                content: format!("Document {}", i),
                source: "test.txt".to_string(),
                metadata: None,
            };
            let embedding = create_embedding(i as f32);
            store.add_chunk(chunk, &embedding).unwrap();
        }

        assert_eq!(store.len(), 3);

        store.clear();

        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_stats() {
        let store = VectorStore::new(VectorStoreConfig::default()).unwrap();

        // Add chunks
        for i in 0..10 {
            let chunk = DocumentChunk {
                id: format!("chunk{}", i),
                content: format!("Document {}", i),
                source: "test.txt".to_string(),
                metadata: None,
            };
            let embedding = create_embedding(i as f32);
            store.add_chunk(chunk, &embedding).unwrap();
        }

        let stats = store.stats();
        assert_eq!(stats.num_chunks, 10);
        assert_eq!(stats.embedding_dim, EMBEDDING_DIM);
    }

    #[test]
    fn test_search_ordering() {
        let store = VectorStore::new(VectorStoreConfig::default()).unwrap();

        // Add chunks with distinct embeddings
        for i in 0..5 {
            let chunk = DocumentChunk {
                id: format!("chunk{}", i),
                content: format!("Document {}", i),
                source: "test.txt".to_string(),
                metadata: None,
            };
            let embedding = create_embedding(i as f32 * 10.0);
            store.add_chunk(chunk, &embedding).unwrap();
        }

        // Query close to chunk 3
        let query = create_embedding(30.0);
        let results = store.search(&query, 5).unwrap();

        // Results should be ordered by similarity
        for i in 0..results.len() - 1 {
            assert!(results[i].score >= results[i + 1].score);
        }
    }
}
