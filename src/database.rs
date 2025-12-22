//! SQLite database for persistent storage
//!
//! Stores conversation history, events, and metadata

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Database error
#[derive(Debug, thiserror::Error)]
pub enum DatabaseError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("Not found")]
    NotFound,

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, DatabaseError>;

/// Event stored in the database
#[derive(Debug, Clone)]
pub struct StoredEvent {
    pub id: String,
    pub conversation_id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub content: String,
    pub metadata: Option<String>,
}

/// Conversation metadata
#[derive(Debug, Clone)]
pub struct Conversation {
    pub id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub turn_count: i32,
    pub metadata: Option<String>,
}

/// Database connection pool (simple Arc<Mutex> wrapper for SQLite)
#[derive(Clone)]
pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    /// Create a new database connection
    ///
    /// # Arguments
    /// * `path` - Path to SQLite database file (or ":memory:" for in-memory)
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;

        // Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON", [])?;

        let db = Database {
            conn: Arc::new(Mutex::new(conn)),
        };

        // Initialize schema
        db.initialize_schema()?;

        Ok(db)
    }

    /// Initialize database schema
    pub fn initialize_schema(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        // Conversations table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                end_time TEXT,
                turn_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT
            )",
            [],
        )?;

        // Events table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )",
            [],
        )?;

        // Indices for performance
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_conversation
             ON events(conversation_id, timestamp)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type
             ON events(event_type, timestamp)",
            [],
        )?;

        // Conversation embeddings table for semantic search over conversation history
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversation_embeddings (
                turn_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (turn_id) REFERENCES events(id) ON DELETE CASCADE,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_embeddings
             ON conversation_embeddings(conversation_id)",
            [],
        )?;

        // Users table for identity tracking
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                aliases TEXT,
                hidden_aliases TEXT,
                verification_status TEXT NOT NULL DEFAULT 'unverified',
                pattern_confidence REAL NOT NULL DEFAULT 0.0,
                metadata TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL
            )",
            [],
        )?;

        // User relationships to conversations
        conn.execute(
            "CREATE TABLE IF NOT EXISTS user_relationships (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                role TEXT,
                PRIMARY KEY (user_id, conversation_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_relationships_user
             ON user_relationships(user_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_relationships_conv
             ON user_relationships(conversation_id)",
            [],
        )?;

        // Typing patterns table for behavioral fingerprinting
        conn.execute(
            "CREATE TABLE IF NOT EXISTS typing_patterns (
                user_id TEXT PRIMARY KEY,
                common_words TEXT NOT NULL,
                common_bigrams TEXT NOT NULL,
                common_trigrams TEXT NOT NULL,
                comma_frequency REAL NOT NULL,
                period_frequency REAL NOT NULL,
                exclamation_frequency REAL NOT NULL,
                question_frequency REAL NOT NULL,
                emoji_frequency REAL NOT NULL,
                ellipsis_frequency REAL NOT NULL,
                avg_sentence_length REAL NOT NULL,
                avg_word_length REAL NOT NULL,
                avg_message_length REAL NOT NULL,
                capitalization_frequency REAL NOT NULL,
                all_caps_frequency REAL NOT NULL,
                code_block_frequency REAL NOT NULL,
                technical_terms_frequency REAL NOT NULL,
                formality_score REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                total_characters INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )",
            [],
        )?;

        // Relationships table for social graph
        conn.execute(
            "CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_user_id TEXT NOT NULL,
                to_user_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(from_user_id, to_user_id, relationship_type)
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_from
             ON relationships(from_user_id)",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_relationships_to
             ON relationships(to_user_id)",
            [],
        )?;

        // Chat messages table for admin UI chat persistence
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                instance_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_messages_instance
             ON chat_messages(instance_id, timestamp)",
            [],
        )?;

        Ok(())
    }

    /// Create a new conversation
    pub fn create_conversation(&self, id: Uuid, metadata: Option<String>) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO conversations (id, start_time, turn_count, metadata)
             VALUES (?1, ?2, 0, ?3)",
            params![id.to_string(), now, metadata],
        )?;

        Ok(())
    }

    /// Get conversation by ID
    pub fn get_conversation(&self, id: Uuid) -> Result<Conversation> {
        let conn = self.conn.lock().unwrap();

        let result = conn
            .query_row(
                "SELECT id, start_time, end_time, turn_count, metadata
             FROM conversations WHERE id = ?1",
                params![id.to_string()],
                |row| {
                    Ok(Conversation {
                        id: row.get(0)?,
                        start_time: DateTime::parse_from_rfc3339(&row.get::<_, String>(1)?)
                            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                            .with_timezone(&Utc),
                        end_time: row
                            .get::<_, Option<String>>(2)?
                            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                            .map(|dt| dt.with_timezone(&Utc)),
                        turn_count: row.get(3)?,
                        metadata: row.get(4)?,
                    })
                },
            )
            .optional()?;

        result.ok_or(DatabaseError::NotFound)
    }

    /// End a conversation
    pub fn end_conversation(&self, id: Uuid) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "UPDATE conversations SET end_time = ?1 WHERE id = ?2",
            params![now, id.to_string()],
        )?;

        Ok(())
    }

    /// Increment conversation turn count
    pub fn increment_turn_count(&self, conversation_id: Uuid) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "UPDATE conversations SET turn_count = turn_count + 1 WHERE id = ?1",
            params![conversation_id.to_string()],
        )?;

        Ok(())
    }

    /// Store an event
    pub fn store_event(
        &self,
        id: Uuid,
        conversation_id: Uuid,
        event_type: &str,
        content: &str,
        metadata: Option<String>,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();

        conn.execute(
            "INSERT INTO events (id, conversation_id, timestamp, event_type, content, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                id.to_string(),
                conversation_id.to_string(),
                now,
                event_type,
                content,
                metadata
            ],
        )?;

        Ok(())
    }

    /// Get all events for a conversation
    pub fn get_conversation_events(&self, conversation_id: Uuid) -> Result<Vec<StoredEvent>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, timestamp, event_type, content, metadata
             FROM events
             WHERE conversation_id = ?1
             ORDER BY timestamp ASC",
        )?;

        let events = stmt
            .query_map(params![conversation_id.to_string()], |row| {
                Ok(StoredEvent {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                        .with_timezone(&Utc),
                    event_type: row.get(3)?,
                    content: row.get(4)?,
                    metadata: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(events)
    }

    /// Get recent events by type
    pub fn get_recent_events_by_type(
        &self,
        event_type: &str,
        limit: usize,
    ) -> Result<Vec<StoredEvent>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, timestamp, event_type, content, metadata
             FROM events
             WHERE event_type = ?1
             ORDER BY timestamp DESC
             LIMIT ?2",
        )?;

        let events = stmt
            .query_map(params![event_type, limit as i64], |row| {
                Ok(StoredEvent {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                        .with_timezone(&Utc),
                    event_type: row.get(3)?,
                    content: row.get(4)?,
                    metadata: row.get(5)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(events)
    }

    /// Delete old events (for cleanup)
    pub fn delete_old_events(&self, before: DateTime<Utc>) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let timestamp = before.to_rfc3339();

        let count = conn.execute(
            "DELETE FROM events WHERE timestamp < ?1",
            params![timestamp],
        )?;

        Ok(count)
    }

    /// Store an event with its embedding for conversation memory
    pub fn store_event_with_embedding(
        &self,
        id: Uuid,
        conversation_id: Uuid,
        event_type: &str,
        content: &str,
        metadata: Option<String>,
        embedding: &[f32],
    ) -> Result<()> {
        // Store the event first
        self.store_event(id, conversation_id, event_type, content, metadata)?;

        // Convert embedding to bytes
        let mut embedding_bytes = Vec::with_capacity(embedding.len() * 4);
        for &val in embedding {
            embedding_bytes.extend_from_slice(&val.to_le_bytes());
        }

        // Store the embedding
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO conversation_embeddings (turn_id, conversation_id, embedding)
             VALUES (?1, ?2, ?3)",
            params![id.to_string(), conversation_id.to_string(), embedding_bytes],
        )?;

        Ok(())
    }

    /// Search conversation history by semantic similarity
    ///
    /// # Arguments
    /// * `conversation_id` - Conversation to search within
    /// * `query_embedding` - Embedding vector of the query
    /// * `top_k` - Number of results to return
    ///
    /// # Returns
    /// Vector of (event, similarity_score) tuples, sorted by similarity (highest first)
    pub fn search_conversation_history(
        &self,
        conversation_id: Uuid,
        query_embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<(StoredEvent, f32)>> {
        let conn = self.conn.lock().unwrap();

        // Get all embeddings for this conversation
        let mut stmt = conn.prepare(
            "SELECT ce.turn_id, ce.embedding, e.id, e.conversation_id, e.timestamp,
                    e.event_type, e.content, e.metadata
             FROM conversation_embeddings ce
             JOIN events e ON ce.turn_id = e.id
             WHERE ce.conversation_id = ?1",
        )?;

        let mut results: Vec<(StoredEvent, f32)> = Vec::new();

        let rows = stmt.query_map(params![conversation_id.to_string()], |row| {
            let embedding_bytes: Vec<u8> = row.get(1)?;
            let stored_embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let event = StoredEvent {
                id: row.get(2)?,
                conversation_id: row.get(3)?,
                timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?
                    .with_timezone(&Utc),
                event_type: row.get(5)?,
                content: row.get(6)?,
                metadata: row.get(7)?,
            };

            Ok((event, stored_embedding))
        })?;

        // Calculate cosine similarity for each result
        for row in rows {
            let (event, stored_embedding) = row?;
            let similarity = cosine_similarity(query_embedding, &stored_embedding);
            results.push((event, similarity));
        }

        // Sort by similarity (highest first)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top K results
        results.truncate(top_k);

        Ok(results)
    }

    /// Get database statistics
    pub fn get_stats(&self) -> Result<DatabaseStats> {
        let conn = self.conn.lock().unwrap();

        let conversation_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM conversations", [], |row| row.get(0))?;

        let event_count: i64 =
            conn.query_row("SELECT COUNT(*) FROM events", [], |row| row.get(0))?;

        let active_conversations: i64 = conn.query_row(
            "SELECT COUNT(*) FROM conversations WHERE end_time IS NULL",
            [],
            |row| row.get(0),
        )?;

        Ok(DatabaseStats {
            conversation_count: conversation_count as usize,
            event_count: event_count as usize,
            active_conversations: active_conversations as usize,
        })
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub conversation_count: usize,
    pub event_count: usize,
    pub active_conversations: usize,
}

// Additional Database methods for identity module
impl Database {
    /// Get a clone of the internal connection (for IdentityStore)
    pub fn conn(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    /// Get thought count from thoughtchain table
    pub fn get_thought_count(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM thoughtchain", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Get identity count from identities table
    pub fn get_identity_count(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM identities", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Get relationship count from relationships table
    pub fn get_relationship_count(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 =
            conn.query_row("SELECT COUNT(*) FROM relationships", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Get session count from sessions table
    pub fn get_session_count(&self) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Store a chat message
    pub fn store_chat_message(
        &self,
        id: String,
        instance_id: String,
        role: String,
        content: String,
        timestamp: String,
    ) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO chat_messages (id, instance_id, role, content, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, instance_id, role, content, timestamp],
        )?;
        Ok(())
    }

    /// Get chat messages for a specific instance
    pub fn get_chat_messages(&self, instance_id: &str, limit: usize) -> Result<Vec<ChatMessage>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, instance_id, role, content, timestamp
             FROM chat_messages
             WHERE instance_id = ?1
             ORDER BY timestamp ASC
             LIMIT ?2",
        )?;

        let messages = stmt
            .query_map(params![instance_id, limit as i64], |row| {
                Ok(ChatMessage {
                    id: row.get(0)?,
                    instance_id: row.get(1)?,
                    role: row.get(2)?,
                    content: row.get(3)?,
                    timestamp: row.get(4)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(messages)
    }
}

/// Chat message stored in database
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub id: String,
    pub instance_id: String,
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = Database::new(":memory:").unwrap();
        let stats = db.get_stats().unwrap();
        assert_eq!(stats.conversation_count, 0);
        assert_eq!(stats.event_count, 0);
    }

    #[test]
    fn test_create_conversation() {
        let db = Database::new(":memory:").unwrap();
        let id = Uuid::new_v4();

        db.create_conversation(id, None).unwrap();

        let conv = db.get_conversation(id).unwrap();
        assert_eq!(conv.id, id.to_string());
        assert_eq!(conv.turn_count, 0);
        assert!(conv.end_time.is_none());
    }

    #[test]
    fn test_store_and_retrieve_events() {
        let db = Database::new(":memory:").unwrap();
        let conv_id = Uuid::new_v4();

        db.create_conversation(conv_id, None).unwrap();

        // Store some events
        let event1_id = Uuid::new_v4();
        db.store_event(event1_id, conv_id, "UserMessage", "Hello", None)
            .unwrap();

        let event2_id = Uuid::new_v4();
        db.store_event(event2_id, conv_id, "AssistantMessage", "Hi there", None)
            .unwrap();

        // Retrieve events
        let events = db.get_conversation_events(conv_id).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "UserMessage");
        assert_eq!(events[0].content, "Hello");
        assert_eq!(events[1].event_type, "AssistantMessage");
        assert_eq!(events[1].content, "Hi there");
    }

    #[test]
    fn test_end_conversation() {
        let db = Database::new(":memory:").unwrap();
        let id = Uuid::new_v4();

        db.create_conversation(id, None).unwrap();
        db.end_conversation(id).unwrap();

        let conv = db.get_conversation(id).unwrap();
        assert!(conv.end_time.is_some());
    }

    #[test]
    fn test_increment_turn_count() {
        let db = Database::new(":memory:").unwrap();
        let id = Uuid::new_v4();

        db.create_conversation(id, None).unwrap();
        db.increment_turn_count(id).unwrap();
        db.increment_turn_count(id).unwrap();

        let conv = db.get_conversation(id).unwrap();
        assert_eq!(conv.turn_count, 2);
    }

    #[test]
    fn test_get_recent_events_by_type() {
        let db = Database::new(":memory:").unwrap();
        let conv_id = Uuid::new_v4();

        db.create_conversation(conv_id, None).unwrap();

        // Store events of different types
        for i in 0..5 {
            db.store_event(
                Uuid::new_v4(),
                conv_id,
                "UserMessage",
                &format!("Message {}", i),
                None,
            )
            .unwrap();
        }

        for i in 0..3 {
            db.store_event(
                Uuid::new_v4(),
                conv_id,
                "SystemEvent",
                &format!("Event {}", i),
                None,
            )
            .unwrap();
        }

        let user_msgs = db.get_recent_events_by_type("UserMessage", 10).unwrap();
        assert_eq!(user_msgs.len(), 5);

        let sys_events = db.get_recent_events_by_type("SystemEvent", 2).unwrap();
        assert_eq!(sys_events.len(), 2);
    }

    #[test]
    fn test_database_stats() {
        let db = Database::new(":memory:").unwrap();

        // Create conversations
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        db.create_conversation(id1, None).unwrap();
        db.create_conversation(id2, None).unwrap();

        // Store events
        db.store_event(Uuid::new_v4(), id1, "test", "content", None)
            .unwrap();
        db.store_event(Uuid::new_v4(), id1, "test", "content", None)
            .unwrap();
        db.store_event(Uuid::new_v4(), id2, "test", "content", None)
            .unwrap();

        // End one conversation
        db.end_conversation(id1).unwrap();

        let stats = db.get_stats().unwrap();
        assert_eq!(stats.conversation_count, 2);
        assert_eq!(stats.event_count, 3);
        assert_eq!(stats.active_conversations, 1);
    }

    #[test]
    fn test_store_event_with_embedding() {
        let db = Database::new(":memory:").unwrap();
        let conv_id = Uuid::new_v4();
        let event_id = Uuid::new_v4();

        db.create_conversation(conv_id, None).unwrap();

        // Create a test embedding (384 dimensions, all 0.5)
        let embedding = vec![0.5f32; 384];

        db.store_event_with_embedding(
            event_id,
            conv_id,
            "UserMessage",
            "Test message",
            None,
            &embedding,
        )
        .unwrap();

        // Verify event was stored
        let events = db.get_conversation_events(conv_id).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].content, "Test message");

        // Verify embedding was stored (can't directly query but search will validate)
        let query_embedding = vec![0.5f32; 384];
        let results = db
            .search_conversation_history(conv_id, &query_embedding, 10)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.99); // High similarity since embeddings match
    }

    #[test]
    fn test_search_conversation_history() {
        let db = Database::new(":memory:").unwrap();
        let conv_id = Uuid::new_v4();

        db.create_conversation(conv_id, None).unwrap();

        // Store events with different embeddings
        let embedding1 = vec![1.0f32; 384]; // All 1.0
        let embedding2 = vec![0.0f32; 384]; // All 0.0
        let mut embedding3 = vec![0.0f32; 384];
        embedding3[0] = 1.0; // First element 1.0, rest 0.0

        db.store_event_with_embedding(
            Uuid::new_v4(),
            conv_id,
            "UserMessage",
            "First message",
            None,
            &embedding1,
        )
        .unwrap();

        db.store_event_with_embedding(
            Uuid::new_v4(),
            conv_id,
            "UserMessage",
            "Second message",
            None,
            &embedding2,
        )
        .unwrap();

        db.store_event_with_embedding(
            Uuid::new_v4(),
            conv_id,
            "UserMessage",
            "Third message",
            None,
            &embedding3,
        )
        .unwrap();

        // Search with query matching embedding1
        let query = vec![1.0f32; 384];
        let results = db.search_conversation_history(conv_id, &query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.content, "First message");
        assert!(results[0].1 > 0.99); // Very high similarity
    }

    #[test]
    fn test_search_conversation_history_empty() {
        let db = Database::new(":memory:").unwrap();
        let conv_id = Uuid::new_v4();

        db.create_conversation(conv_id, None).unwrap();

        let query = vec![1.0f32; 384];
        let results = db.search_conversation_history(conv_id, &query, 10).unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_search_conversation_history_top_k() {
        let db = Database::new(":memory:").unwrap();
        let conv_id = Uuid::new_v4();

        db.create_conversation(conv_id, None).unwrap();

        // Store 5 events
        for i in 0..5 {
            let mut embedding = vec![0.0f32; 384];
            embedding[0] = i as f32;
            db.store_event_with_embedding(
                Uuid::new_v4(),
                conv_id,
                "UserMessage",
                &format!("Message {}", i),
                None,
                &embedding,
            )
            .unwrap();
        }

        // Search with top_k = 3
        let query = vec![2.0f32; 384]; // Should match message 2 best
        let results = db.search_conversation_history(conv_id, &query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // Results should be sorted by similarity
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);

        // Opposite vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);

        // Different lengths (should return 0.0)
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);

        // Zero vectors (should return 0.0)
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
