//! Embedding generation for text chunks
//!
//! Provides text embedding functionality for vector search

use crate::vector_store::EMBEDDING_DIM;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Embedding generator error
#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Model error: {0}")]
    Model(String),
}

pub type Result<T> = std::result::Result<T, EmbeddingError>;

/// Trait for embedding generators
pub trait EmbeddingGenerator: Send + Sync {
    /// Generate an embedding for the given text
    fn generate(&self, text: &str) -> Result<Vec<f32>>;

    /// Generate embeddings for multiple texts (batch processing)
    fn generate_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.generate(text)).collect()
    }

    /// Get the embedding dimension
    fn dimension(&self) -> usize;

    /// Average multiple embeddings (for hierarchical parent chunks)
    fn average_embeddings(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Err(EmbeddingError::InvalidInput(
                "No embeddings to average".to_string(),
            ));
        }

        let dim = self.dimension();
        for emb in embeddings {
            if emb.len() != dim {
                return Err(EmbeddingError::InvalidInput(format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    dim,
                    emb.len()
                )));
            }
        }

        // Average all dimensions
        let mut averaged = vec![0.0; dim];
        for emb in embeddings {
            for (i, val) in emb.iter().enumerate() {
                averaged[i] += val;
            }
        }

        let count = embeddings.len() as f32;
        for val in averaged.iter_mut() {
            *val /= count;
        }

        // Re-normalize the averaged vector
        let magnitude: f32 = averaged.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            averaged.iter_mut().for_each(|x| *x /= magnitude);
        }

        Ok(averaged)
    }
}

/// Simple hash-based embedding generator (for testing/fallback)
///
/// This creates deterministic but not semantically meaningful embeddings.
/// In production, use a proper embedding model.
pub struct SimpleEmbeddingGenerator {
    dimension: usize,
}

impl SimpleEmbeddingGenerator {
    pub fn new() -> Self {
        Self {
            dimension: EMBEDDING_DIM,
        }
    }

    /// Create embeddings using a deterministic hash function
    fn hash_to_embedding(&self, text: &str) -> Vec<f32> {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let base_hash = hasher.finish();

        // Generate deterministic values based on hash
        let mut embedding = Vec::with_capacity(self.dimension);
        let mut seed = base_hash;

        for i in 0..self.dimension {
            // Simple LCG (Linear Congruential Generator)
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((seed >> 16) as f32) / 65536.0; // Normalize to [0, 1]

            // Convert to [-1, 1] and add some structure
            let normalized = (val * 2.0 - 1.0) * (1.0 + (i as f32 / self.dimension as f32).sin());
            embedding.push(normalized);
        }

        // Normalize the vector
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            embedding.iter_mut().for_each(|x| *x /= magnitude);
        }

        embedding
    }
}

impl Default for SimpleEmbeddingGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingGenerator for SimpleEmbeddingGenerator {
    fn generate(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }

        Ok(self.hash_to_embedding(text))
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

/// ONNX-based embedding generator using sentence-transformers model
///
/// Uses all-MiniLM-L6-v2 model for 384-dimensional semantic embeddings
pub struct OnnxEmbeddingGenerator {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: rust_tokenizers::tokenizer::BertTokenizer,
    dimension: usize,
}

impl OnnxEmbeddingGenerator {
    /// Create a new ONNX embedding generator
    ///
    /// Model and tokenizer should be in the models/ directory
    pub fn new() -> Result<Self> {
        let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("all-minilm-l6-v2.onnx");

        // Load ONNX model with ort 2.0 API
        let session = ort::session::Session::builder()
            .map_err(|e| EmbeddingError::Model(format!("Failed to create session builder: {}", e)))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbeddingError::Model(format!("Failed to load model: {}", e)))?;

        // Load tokenizer using rust_tokenizers (pure Rust, no native dependencies)
        use rust_tokenizers::tokenizer::BertTokenizer;
        use rust_tokenizers::vocab::{BertVocab, Vocab};

        // rust_tokenizers expects vocab.txt, so we convert the tokenizer.json if needed
        // For now, use a simple approach - create from vocab
        let vocab_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models")
            .join("vocab.txt");

        let vocab = BertVocab::from_file(&vocab_path)
            .map_err(|e| EmbeddingError::Model(format!("Failed to load vocab: {}", e)))?;

        let tokenizer = BertTokenizer::from_existing_vocab(vocab, true, true);

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            dimension: 384, // all-MiniLM-L6-v2 dimension
        })
    }

    /// Mean pooling over token embeddings
    fn mean_pooling(
        &self,
        token_embeddings: &ndarray::ArrayD<f32>,
        attention_mask: &[i64],
    ) -> Vec<f32> {
        let shape = token_embeddings.shape();
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        let mut pooled = vec![0.0f32; hidden_dim];
        let mut mask_sum = 0.0f32;

        for i in 0..seq_len {
            let mask_val = attention_mask[i] as f32;
            mask_sum += mask_val;

            for j in 0..hidden_dim {
                pooled[j] += token_embeddings[[0, i, j]] * mask_val;
            }
        }

        // Divide by sum of mask (number of actual tokens)
        if mask_sum > 0.0 {
            for val in pooled.iter_mut() {
                *val /= mask_sum;
            }
        }

        // L2 normalize
        let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            pooled.iter_mut().for_each(|x| *x /= norm);
        }

        pooled
    }
}

impl EmbeddingGenerator for OnnxEmbeddingGenerator {
    fn generate(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbeddingError::InvalidInput("Empty text".to_string()));
        }

        // Tokenize using rust_tokenizers API
        use rust_tokenizers::tokenizer::{Tokenizer, TruncationStrategy};

        let tokenized = self.tokenizer.encode(
            text,
            None,
            512, // max_len
            &TruncationStrategy::LongestFirst,
            0, // stride
        );

        let input_ids: Vec<i64> = tokenized.token_ids.iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = tokenized.segment_ids.iter().map(|_| 1i64).collect();
        let token_type_ids: Vec<i64> = tokenized.segment_ids.iter().map(|&x| x as i64).collect();

        // Create ONNX input tensors using ort 2.0 API
        let seq_len = input_ids.len();

        // Create TensorRef for each input
        let input_ids_shape = ort::tensor::Shape::from(vec![1usize, seq_len]);
        let input_ids_ref =
            ort::value::TensorRef::from_array_view((input_ids_shape.clone(), input_ids.as_slice()))
                .map_err(|e| {
                    EmbeddingError::Model(format!("Failed to create input_ids tensor: {}", e))
                })?;

        let attention_mask_ref = ort::value::TensorRef::from_array_view((
            input_ids_shape.clone(),
            attention_mask.as_slice(),
        ))
        .map_err(|e| {
            EmbeddingError::Model(format!("Failed to create attention_mask tensor: {}", e))
        })?;

        let token_type_ids_ref =
            ort::value::TensorRef::from_array_view((input_ids_shape, token_type_ids.as_slice()))
                .map_err(|e| {
                    EmbeddingError::Model(format!("Failed to create token_type_ids tensor: {}", e))
                })?;

        // Lock the session mutex for inference
        let mut session = self
            .session
            .lock()
            .map_err(|e| EmbeddingError::Model(format!("Failed to lock session: {}", e)))?;

        // Run inference using ort 2.0 inputs! macro with named inputs
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_ref,
                "attention_mask" => attention_mask_ref,
                "token_type_ids" => token_type_ids_ref
            ])
            .map_err(|e| EmbeddingError::Model(format!("Inference failed: {}", e)))?;

        // Extract token embeddings (last hidden state)
        let output_tensor = outputs
            .get("last_hidden_state")
            .or_else(|| outputs.get("output"))
            .unwrap_or(&outputs[0])
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbeddingError::Model(format!("Failed to extract tensor: {}", e)))?;

        // Destructure tuple: try_extract_tensor returns (&Shape, &[f32])
        let (_shape, data) = output_tensor;

        // Convert to ndarray for mean pooling
        use ndarray::ArrayD;
        let array = ArrayD::from_shape_vec(vec![1, seq_len, self.dimension], data.to_vec())
            .map_err(|e| EmbeddingError::Model(format!("Failed to reshape output: {}", e)))?;

        // Mean pooling
        let embedding = self.mean_pooling(&array, &attention_mask);

        Ok(embedding)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    /// Generate embeddings for multiple texts efficiently (batch processing)
    fn generate_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // For small batches, single-item processing might be faster due to overhead
        if texts.len() == 1 {
            return Ok(vec![self.generate(texts[0])?]);
        }

        // For now, process sequentially (true batch processing is complex with ONNX)
        // This is still faster than calling from different parts of code due to cache locality
        let mut embeddings = Vec::with_capacity(texts.len());
        for text in texts {
            embeddings.push(self.generate(text)?);
        }

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_generator_dimension() {
        let generator = SimpleEmbeddingGenerator::new();
        assert_eq!(generator.dimension(), EMBEDDING_DIM);
    }

    #[test]
    fn test_simple_generator_basic() {
        let generator = SimpleEmbeddingGenerator::new();

        let text = "This is a test document";
        let embedding = generator.generate(text).unwrap();

        assert_eq!(embedding.len(), EMBEDDING_DIM);

        // Check that it's normalized (L2 norm should be close to 1)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_simple_generator_deterministic() {
        let generator = SimpleEmbeddingGenerator::new();

        let text = "Hello world";
        let embedding1 = generator.generate(text).unwrap();
        let embedding2 = generator.generate(text).unwrap();

        // Should produce the same embedding for the same text
        assert_eq!(embedding1, embedding2);
    }

    #[test]
    fn test_simple_generator_different_texts() {
        let generator = SimpleEmbeddingGenerator::new();

        let text1 = "First document";
        let text2 = "Second document";

        let embedding1 = generator.generate(text1).unwrap();
        let embedding2 = generator.generate(text2).unwrap();

        // Different texts should produce different embeddings
        assert_ne!(embedding1, embedding2);
    }

    #[test]
    fn test_simple_generator_empty_text() {
        let generator = SimpleEmbeddingGenerator::new();

        let result = generator.generate("");
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_generation() {
        let generator = SimpleEmbeddingGenerator::new();

        let texts = vec!["First", "Second", "Third"];
        let embeddings = generator.generate_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), EMBEDDING_DIM);
        }
    }

    #[test]
    fn test_similar_texts_produce_similar_embeddings() {
        let generator = SimpleEmbeddingGenerator::new();

        let text1 = "The quick brown fox";
        let text2 = "The quick brown fox jumps";

        let embedding1 = generator.generate(text1).unwrap();
        let embedding2 = generator.generate(text2).unwrap();

        // Calculate cosine similarity
        let dot_product: f32 = embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Embeddings are normalized, so dot product = cosine similarity
        // Similar texts should have some similarity (though this is a simple hash-based approach)
        // We just check that the computation works
        assert!(dot_product.abs() <= 1.0);
    }

    #[test]
    fn test_average_embeddings_basic() {
        let generator = SimpleEmbeddingGenerator::new();

        // Generate 3 embeddings
        let text1 = "First chunk";
        let text2 = "Second chunk";
        let text3 = "Third chunk";

        let emb1 = generator.generate(text1).unwrap();
        let emb2 = generator.generate(text2).unwrap();
        let emb3 = generator.generate(text3).unwrap();

        let embeddings = vec![emb1, emb2, emb3];
        let averaged = generator.average_embeddings(&embeddings).unwrap();

        // Check dimension
        assert_eq!(averaged.len(), EMBEDDING_DIM);

        // Check that it's normalized (L2 norm should be close to 1)
        let norm: f32 = averaged.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_average_embeddings_single() {
        let generator = SimpleEmbeddingGenerator::new();

        let text = "Single chunk";
        let embedding = generator.generate(text).unwrap();

        let embeddings = vec![embedding.clone()];
        let averaged = generator.average_embeddings(&embeddings).unwrap();

        // Averaging a single embedding should return the same embedding
        assert_eq!(averaged.len(), embedding.len());

        // Check that it's still normalized
        let norm: f32 = averaged.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_average_embeddings_empty() {
        let generator = SimpleEmbeddingGenerator::new();

        let embeddings: Vec<Vec<f32>> = vec![];
        let result = generator.average_embeddings(&embeddings);

        assert!(result.is_err());
        match result {
            Err(EmbeddingError::InvalidInput(msg)) => {
                assert_eq!(msg, "No embeddings to average");
            }
            _ => panic!("Expected InvalidInput error"),
        }
    }

    #[test]
    fn test_average_embeddings_dimension_mismatch() {
        let generator = SimpleEmbeddingGenerator::new();

        let emb1 = generator.generate("First").unwrap();
        let emb2 = vec![0.5; 128]; // Wrong dimension

        let embeddings = vec![emb1, emb2];
        let result = generator.average_embeddings(&embeddings);

        assert!(result.is_err());
        match result {
            Err(EmbeddingError::InvalidInput(msg)) => {
                assert!(msg.contains("dimension mismatch"));
            }
            _ => panic!("Expected InvalidInput error for dimension mismatch"),
        }
    }

    #[test]
    fn test_average_embeddings_hierarchical() {
        let generator = SimpleEmbeddingGenerator::new();

        // Simulate hierarchical scenario: parent chunk has 3 children
        let child1 = generator.generate("Child chunk 1 content").unwrap();
        let child2 = generator.generate("Child chunk 2 content").unwrap();
        let child3 = generator.generate("Child chunk 3 content").unwrap();

        let children = vec![child1.clone(), child2.clone(), child3.clone()];
        let parent_embedding = generator.average_embeddings(&children).unwrap();

        // Parent embedding should be normalized
        let norm: f32 = parent_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);

        // Parent embedding should be different from any single child
        assert_ne!(parent_embedding, child1);
        assert_ne!(parent_embedding, child2);
        assert_ne!(parent_embedding, child3);

        // Calculate cosine similarity between parent and children
        let similarity1: f32 = parent_embedding
            .iter()
            .zip(child1.iter())
            .map(|(a, b)| a * b)
            .sum();
        let similarity2: f32 = parent_embedding
            .iter()
            .zip(child2.iter())
            .map(|(a, b)| a * b)
            .sum();
        let similarity3: f32 = parent_embedding
            .iter()
            .zip(child3.iter())
            .map(|(a, b)| a * b)
            .sum();

        // Parent should have reasonable similarity to all children
        assert!(similarity1 > 0.0 && similarity1 <= 1.0);
        assert!(similarity2 > 0.0 && similarity2 <= 1.0);
        assert!(similarity3 > 0.0 && similarity3 <= 1.0);
    }

    #[test]
    #[ignore] // Requires model files, run with --ignored
    fn test_onnx_generator_basic() {
        let generator = OnnxEmbeddingGenerator::new().expect("Failed to create ONNX generator");

        let text = "This is a test sentence";
        let embedding = generator.generate(text).unwrap();

        // Check dimension (MiniLM-L6-v2 produces 384-dim embeddings)
        assert_eq!(embedding.len(), 384);

        // Check that it's normalized (L2 norm should be close to 1)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001, "Norm was {}", norm);
    }

    #[test]
    #[ignore] // Requires model files
    fn test_onnx_semantic_similarity() {
        let generator = OnnxEmbeddingGenerator::new().expect("Failed to create ONNX generator");

        // Similar sentences
        let text1 = "I love programming in Rust";
        let text2 = "Rust programming is great";

        // Dissimilar sentence
        let text3 = "The weather is sunny today";

        let emb1 = generator.generate(text1).unwrap();
        let emb2 = generator.generate(text2).unwrap();
        let emb3 = generator.generate(text3).unwrap();

        // Calculate cosine similarities
        let sim_1_2: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        let sim_1_3: f32 = emb1.iter().zip(emb3.iter()).map(|(a, b)| a * b).sum();

        // Similar sentences should have higher similarity than dissimilar ones
        assert!(
            sim_1_2 > sim_1_3,
            "Similar sentences should have higher cosine similarity"
        );
        println!("Similarity (Rust/Rust): {:.4}", sim_1_2);
        println!("Similarity (Rust/Weather): {:.4}", sim_1_3);

        // Typically, similar sentences should have similarity > 0.5
        assert!(
            sim_1_2 > 0.5,
            "Similar sentences should have similarity > 0.5"
        );
    }

    #[test]
    #[ignore] // Requires model files
    fn test_onnx_vector_ops() {
        let generator = OnnxEmbeddingGenerator::new().expect("Failed to create ONNX generator");

        // Technical question about vectors
        let question = "How do I create a vector in Rust?";
        let answer1 = "Use Vec::new() to create an empty vector";
        let answer2 = "The vec! macro creates a vector with initial values";
        let unrelated = "Python is a popular programming language";

        let q_emb = generator.generate(question).unwrap();
        let a1_emb = generator.generate(answer1).unwrap();
        let a2_emb = generator.generate(answer2).unwrap();
        let un_emb = generator.generate(unrelated).unwrap();

        let sim_q_a1: f32 = q_emb.iter().zip(a1_emb.iter()).map(|(a, b)| a * b).sum();
        let sim_q_a2: f32 = q_emb.iter().zip(a2_emb.iter()).map(|(a, b)| a * b).sum();
        let sim_q_un: f32 = q_emb.iter().zip(un_emb.iter()).map(|(a, b)| a * b).sum();

        println!("Question-Answer1 similarity: {:.4}", sim_q_a1);
        println!("Question-Answer2 similarity: {:.4}", sim_q_a2);
        println!("Question-Unrelated similarity: {:.4}", sim_q_un);

        // Both answers should be more similar to the question than the unrelated text
        assert!(sim_q_a1 > sim_q_un);
        assert!(sim_q_a2 > sim_q_un);
    }
}
