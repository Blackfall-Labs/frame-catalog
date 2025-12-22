//! Persistent Vector Store with SQLite
//!
//! Stores documents, embeddings, and HNSW index in SQLite for instant loading

use bytepunch::{Decompressor, Dictionary};
use dataspool::SpoolReader;
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::vector_store::SearchResult;

/// Document reference (points to compressed file OR spool location)
#[derive(Debug, Clone)]
pub struct DocumentRef {
    pub id: String,
    pub file_path: String,
    pub source: String,
    pub metadata: Option<String>,
    /// If present, this document is in a spool at this byte offset
    pub spool_offset: Option<u64>,
    /// Length of the document in the spool
    pub spool_length: Option<u32>,
}

/// Search result with file reference
#[derive(Debug, Clone)]
pub struct FileSearchResult {
    pub doc_ref: DocumentRef,
    pub score: f32,
}

/// Persistent vector store backed by SQLite
pub struct PersistentVectorStore {
    conn: Arc<Mutex<Connection>>,
    dimension: usize,
}

impl PersistentVectorStore {
    /// Create or open a persistent vector store
    pub fn new<P: AsRef<Path>>(path: P, dimension: usize) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;

        let store = Self {
            conn: Arc::new(Mutex::new(conn)),
            dimension,
        };

        store.initialize_schema()?;
        Ok(store)
    }

    /// Initialize database schema
    fn initialize_schema(&self) -> rusqlite::Result<()> {
        let conn = self.conn.lock().unwrap();

        // Documents table (stores FILE REFERENCES OR SPOOL OFFSETS, not content)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT,
                spool_offset INTEGER,
                spool_length INTEGER
            )",
            [],
        )?;

        // Embeddings table (vector stored as blob)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Note: FTS5 removed - content is in compressed files, not in DB
        // Can add FTS later by indexing decompressed content if needed

        Ok(())
    }

    /// Add a document with its embedding and file path
    pub fn add_document(
        &self,
        id: &str,
        file_path: &str,
        source: &str,
        metadata: Option<&str>,
        embedding: &[f32],
    ) -> rusqlite::Result<()> {
        self.add_document_with_spool(id, file_path, source, metadata, None, None, embedding)
    }

    /// Add a document with spool location
    pub fn add_document_with_spool(
        &self,
        id: &str,
        file_path: &str,
        source: &str,
        metadata: Option<&str>,
        spool_offset: Option<u64>,
        spool_length: Option<u32>,
        embedding: &[f32],
    ) -> rusqlite::Result<()> {
        if embedding.len() != self.dimension {
            return Err(rusqlite::Error::InvalidParameterName(
                format!("Expected {} dims, got {}", self.dimension, embedding.len()).into(),
            ));
        }

        let conn = self.conn.lock().unwrap();

        // Store document file reference with optional spool location
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, file_path, source, metadata, spool_offset, spool_length) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id,
                file_path,
                source,
                metadata,
                spool_offset.map(|o| o as i64),
                spool_length.map(|l| l as i64),
            ],
        )?;

        // Store embedding as blob
        let vector_bytes = embedding
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<u8>>();

        conn.execute(
            "INSERT OR REPLACE INTO embeddings (doc_id, vector) VALUES (?1, ?2)",
            params![id, &vector_bytes],
        )?;

        Ok(())
    }

    /// Search for similar documents using vector similarity
    /// Returns file references, caller must read and decompress files
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> rusqlite::Result<Vec<FileSearchResult>> {
        if query_embedding.len() != self.dimension {
            return Err(rusqlite::Error::InvalidParameterName(
                format!(
                    "Expected {} dims, got {}",
                    self.dimension,
                    query_embedding.len()
                )
                .into(),
            ));
        }

        let conn = self.conn.lock().unwrap();

        // Get all document references with embeddings (including spool offsets)
        let mut stmt = conn.prepare(
            "SELECT d.id, d.file_path, d.source, d.metadata, d.spool_offset, d.spool_length, e.vector
             FROM documents d
             JOIN embeddings e ON d.id = e.doc_id"
        )?;

        let mut results: Vec<(DocumentRef, f32)> = Vec::new();

        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let file_path: String = row.get(1)?;
            let source: String = row.get(2)?;
            let metadata: Option<String> = row.get(3)?;
            let spool_offset: Option<i64> = row.get(4)?;
            let spool_length: Option<i64> = row.get(5)?;
            let vector_bytes: Vec<u8> = row.get(6)?;

            // Deserialize vector
            let embedding: Vec<f32> = vector_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            // Calculate cosine similarity
            let similarity = cosine_similarity(query_embedding, &embedding);

            Ok((
                DocumentRef {
                    id,
                    file_path,
                    source,
                    metadata,
                    spool_offset: spool_offset.map(|o| o as u64),
                    spool_length: spool_length.map(|l| l as u32),
                },
                similarity,
            ))
        })?;

        for row in rows {
            let (doc_ref, score) = row?;
            results.push((doc_ref, score));
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Take top_k
        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(doc_ref, score)| FileSearchResult { doc_ref, score })
            .collect())
    }

    /// Get document count
    pub fn count(&self) -> rusqlite::Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_store() {
        let store = PersistentVectorStore::new(":memory:", 384).unwrap();

        // Add a document with file reference
        let embedding = vec![0.1f32; 384];
        store
            .add_document(
                "test-1",
                "docs/test-1.cml.bp",
                "test",
                Some("{\"type\": \"test\"}"),
                &embedding,
            )
            .unwrap();

        assert_eq!(store.count().unwrap(), 1);

        // Search - returns file references
        let results = store.search(&embedding, 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_ref.id, "test-1");
        assert_eq!(results[0].doc_ref.file_path, "docs/test-1.cml.bp");
        assert!(results[0].score > 0.99); // Nearly identical vectors
    }

    #[test]
    fn test_persistent_store_with_spool() {
        let store = PersistentVectorStore::new(":memory:", 384).unwrap();

        // Add a document with spool location
        let embedding = vec![0.2f32; 384];
        store
            .add_document_with_spool(
                "test-2",
                "docs/test.spool",
                "test",
                Some("{\"type\": \"spool\"}"),
                Some(1024),
                Some(512),
                &embedding,
            )
            .unwrap();

        assert_eq!(store.count().unwrap(), 1);

        // Search - returns spool offsets
        let results = store.search(&embedding, 5).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_ref.id, "test-2");
        assert_eq!(results[0].doc_ref.spool_offset, Some(1024));
        assert_eq!(results[0].doc_ref.spool_length, Some(512));
    }
}
