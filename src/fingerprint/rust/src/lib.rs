//! # HDR Hamming - Bitpacked Vector Search with Arrow DataFusion
//!
//! High-performance hyperdimensional computing library using:
//! - **Bitpacked 10Kbit vectors** for compact representation
//! - **Stacked Popcount** for hierarchical Hamming distance
//! - **Vector Field Resonance** via bind/unbind XOR operations
//! - **Arrow DataFusion** for columnar storage (no Parquet needed!)
//! - **DN Tree** - 256-way hierarchical addressing (like LDAP Distinguished Names)
//! - **144 Cognitive Verbs** - Go board topology for semantic relations
//! - **GraphBLAS Mindmap** - Sparse matrix operations with tree structure
//! - **NN-Tree** - O(log n) nearest neighbor with fingerprint clustering
//! - **Epiphany Engine** - SD threshold + centroid radius calibration
//! - **Sentence Crystal** - Transformer embeddings → 5D crystal → fingerprints
//! - **Déjà Vu RL** - Multipass ±3σ overlay for reinforcement patterns
//! - **Truth Markers** - Orthogonal superposition cleaning
//!
//! ## The Core Insight
//!
//! Traditional vector search uses float matrices and cosine similarity.
//! We use pure integer operations:
//!
//! ```text
//! Float Vector Search:          HDR Bitpacked Search:
//! ─────────────────────         ────────────────────────
//! 384 floats × 4 bytes          10,000 bits = 1,250 bytes
//! = 1,536 bytes
//!
//! cosine = Σ(a×b)/|a||b|        hamming = popcount(a⊕b)
//! ~50 cycles per element        ~1 cycle per 64 bits (SIMD)
//! ```
//!
//! ## Vector Field Resonance
//!
//! Instead of matrix multiply, we use XOR binding:
//!
//! ```text
//! Bind:   A ⊗ B = A ⊕ B       (combine concepts)
//! Unbind: A ⊗ B ⊗ B = A       (recover component)
//! Bundle: majority(A, B, C)    (create prototype)
//! ```
//!
//! This enables O(1) retrieval: given edge=A⊗verb⊗B and verb and B,
//! compute A directly without searching!

pub mod bitpack;
pub mod hamming;
pub mod resonance;
pub mod hdr_cascade;
pub mod graphblas;
pub mod representation;
pub mod dntree;
pub mod mindmap;
pub mod nntree;
pub mod epiphany;
pub mod crystal_dejavu;
pub mod slot_encoding;
#[cfg(feature = "datafusion-storage")]
pub mod storage;
#[cfg(feature = "datafusion-storage")]
pub mod query;
#[cfg(feature = "ffi")]
pub mod ffi;

// Re-exports
pub use bitpack::{BitpackedVector, VECTOR_BITS, VECTOR_WORDS};
pub use hamming::{HammingEngine, StackedPopcount};
pub use resonance::{VectorField, Resonator, BoundEdge};
pub use hdr_cascade::{HdrCascade, MexicanHat, SearchResult};
pub use graphblas::{GrBMatrix, GrBVector, HdrSemiring, Semiring};
pub use representation::{GradedVector, StackedBinary, SparseHdr};
pub use dntree::{TreeAddr, DnTree, DnNode, DnEdge, CogVerb, VerbCategory};
pub use mindmap::{GrBMindmap, MindmapBuilder, MindmapNode, NodeType};
pub use nntree::{NnTree, NnTreeConfig, SparseNnTree};
pub use epiphany::{EpiphanyEngine, EpiphanyZone, CentroidStats, ResonanceCalibrator};
pub use crystal_dejavu::{
    SentenceCrystal, Coord5D, CrystalCell,
    DejaVuRL, DejaVuObservation, SigmaBand,
    TruthMarker, SuperpositionCleaner, CrystalDejaVuTruth,
};
pub use slot_encoding::{SlotEncodedNode, SlotKeys, NodeBuilder, StringEncoder};
#[cfg(feature = "datafusion-storage")]
pub use storage::{ArrowStore, VectorBatch};

/// Error types for HDR operations
#[derive(Debug, thiserror::Error)]
pub enum HdrError {
    #[error("Vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid vector data: {0}")]
    InvalidData(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, HdrError>;

/// Global configuration
pub struct HdrConfig {
    /// Number of bits in vectors (default: 10000)
    pub vector_bits: usize,
    /// Enable SIMD acceleration
    pub use_simd: bool,
    /// Batch size for parallel operations
    pub batch_size: usize,
    /// Number of worker threads
    pub num_threads: usize,
}

impl Default for HdrConfig {
    fn default() -> Self {
        Self {
            vector_bits: 10000,
            use_simd: true,
            batch_size: 1024,
            num_threads: num_cpus::get().max(1),
        }
    }
}

// Inline helper for CPU count when num_cpus isn't available
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = HdrConfig::default();
        assert_eq!(config.vector_bits, 10000);
        assert!(config.use_simd);
        assert!(config.num_threads > 0);
    }
}
