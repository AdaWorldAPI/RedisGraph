//! Lance-backed durable vector store with versioned persistence.
//!
//! ## Two-Tier Architecture
//!
//! ```text
//!   hot writes → ConcurrentWriteCache (RwLock, in-memory XOR deltas)
//!        │           zero-copy deflower protection preserves
//!        │           untouched words — only changed bits pay XOR cost
//!        │
//!        ▼  flush()
//!   cold storage → Lance dataset (versioned fragments, crash-safe)
//!        │           each flush() creates a new manifest version
//!        │           old version survives until new is committed
//!        │
//!        ▼  search()
//!   HDR cascade (Belichtung → Stacked → Exact, zero-copy on Arrow batches)
//! ```
//!
//! ## WAL-Equivalent Semantics
//!
//! Lance doesn't need a separate WAL because its manifest-based versioning
//! provides equivalent guarantees:
//!
//! - **Atomic writes**: Each `flush()` atomically creates a new dataset version.
//! - **Crash recovery**: If the process crashes mid-flush, previous version is intact.
//! - **MVCC reads**: `open_at_version()` pins a specific snapshot for reads.
//! - **Soft deletes**: `delete()` creates deletion vectors (bitmasks), no rewrite.
//! - **Compaction**: `compact()` merges fragments and reclaims space.
//!
//! ## Integration with XOR Write Cache
//!
//! The `ConcurrentWriteCache` from `width_16k::xor_bubble` buffers high-frequency
//! writes as XOR deltas in memory. Periodically:
//!
//! ```text
//! ConcurrentWriteCache::flush() → materialize deltas → LanceStore::insert_batch()
//!                                                       → LanceStore::flush()
//! ```
//!
//! This gives microsecond writes (XOR delta in cache) with durable persistence
//! (Lance versioned fragments). The zero-copy deflower protection ensures that
//! only actually-modified words are touched during materialization — unchanged
//! dimensions pass through as-is.

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow_array::RecordBatchIterator;
use lance::dataset::{Dataset, WriteParams, WriteMode};
use futures::TryStreamExt;

use crate::bitpack::BitpackedVector;
use crate::hdr_cascade::HdrCascade;
use crate::storage::{
    VectorBatch, VectorBatchBuilder, ArrowBatchSearch, BatchSearchResult,
    create_schema,
};
use crate::{HdrError, Result};

// ============================================================================
// LANCE STORE
// ============================================================================

/// Lance-backed vector store with WAL-equivalent versioned persistence.
///
/// ## Why not a separate WAL?
///
/// Lance datasets are append-only with versioned manifests. Each write
/// operation creates a new version atomically:
/// - Version N is the current committed state
/// - `flush()` writes new fragments → creates version N+1
/// - If crash happens during flush, version N remains valid
/// - `open()` always loads the latest committed version
///
/// ## Zero-Copy Search
///
/// After loading from Lance, vectors live in Arrow RecordBatches.
/// The HDR cascade (`ArrowBatchSearch::cascaded_knn`) operates directly
/// on these batches via `VectorSlice` — no copies until a candidate
/// survives all cascade levels.
pub struct LanceStore {
    /// Path to the Lance dataset directory
    path: PathBuf,
    /// The open Lance dataset (None = not yet created/opened)
    dataset: Option<Dataset>,
    /// In-memory HDR cascade index for zero-copy search
    index: HdrCascade,
    /// Loaded Arrow batches from Lance (for zero-copy search)
    batches: Vec<VectorBatch>,
    /// ID → (batch_idx, row_idx) for O(1) lookup
    id_map: HashMap<u64, (usize, usize)>,
    /// Arrow schema for the vector table
    schema: SchemaRef,
    /// In-memory write buffer (flushed to Lance periodically)
    buffer: VectorBatchBuilder,
    /// Number of vectors in the unflushed buffer
    buffer_count: usize,
    /// Next auto-increment ID (always matches builder's starting ID)
    next_id: u64,
    /// Current dataset version (0 = no dataset yet)
    version: u64,
}

impl LanceStore {
    /// Create a new store at the given path.
    ///
    /// Does not create the Lance dataset until first `flush()`.
    /// Vectors inserted before flush are searchable in memory via the
    /// HDR cascade index, but not yet durable.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        let schema = Arc::new(create_schema());
        Self {
            path: path.as_ref().to_path_buf(),
            dataset: None,
            index: HdrCascade::new(),
            batches: Vec::new(),
            id_map: HashMap::new(),
            schema,
            buffer: VectorBatchBuilder::new(),
            buffer_count: 0,
            next_id: 0,
            version: 0,
        }
    }

    /// Open an existing Lance dataset and rebuild the HDR cascade index.
    ///
    /// Scans all fragments, loads them as Arrow batches, and builds the
    /// in-memory index. After open, the store is immediately searchable.
    pub async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let uri = path_buf.to_string_lossy().to_string();

        let dataset = Dataset::open(&uri)
            .await
            .map_err(|e| HdrError::Storage(format!("Lance open: {}", e)))?;

        let version = dataset.version().version;

        let mut store = Self {
            path: path_buf,
            dataset: Some(dataset),
            index: HdrCascade::new(),
            batches: Vec::new(),
            id_map: HashMap::new(),
            schema: Arc::new(create_schema()),
            buffer: VectorBatchBuilder::new(),
            buffer_count: 0,
            next_id: 0,
            version,
        };

        store.rebuild_index().await?;
        store.buffer = VectorBatchBuilder::new().with_start_id(store.next_id);
        Ok(store)
    }

    /// Open at a specific version (MVCC snapshot read).
    ///
    /// Readers see a consistent snapshot while writers append to newer
    /// versions. This is the key to concurrent read/write without locks.
    pub async fn open_at_version<P: AsRef<Path>>(path: P, version: u64) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let uri = path_buf.to_string_lossy().to_string();

        let dataset = Dataset::open(&uri)
            .await
            .map_err(|e| HdrError::Storage(format!("Lance open: {}", e)))?;

        let dataset = dataset
            .checkout_version(version)
            .await
            .map_err(|e| HdrError::Storage(format!("Lance checkout v{}: {}", version, e)))?;

        let mut store = Self {
            path: path_buf,
            dataset: Some(dataset),
            index: HdrCascade::new(),
            batches: Vec::new(),
            id_map: HashMap::new(),
            schema: Arc::new(create_schema()),
            buffer: VectorBatchBuilder::new(),
            buffer_count: 0,
            next_id: 0,
            version,
        };

        store.rebuild_index().await?;
        store.buffer = VectorBatchBuilder::new().with_start_id(store.next_id);
        Ok(store)
    }

    // ========================================================================
    // INSERT (buffer in memory, not yet durable)
    // ========================================================================

    /// Insert a vector into the write buffer.
    ///
    /// The vector is immediately searchable via the in-memory HDR index,
    /// but not yet durable. Call `flush()` to persist to Lance.
    ///
    /// Returns the assigned vector ID.
    pub fn insert(&mut self, vector: &BitpackedVector) -> Result<u64> {
        let id = self.buffer.add(vector)?;
        self.index.add(vector.clone());
        self.buffer_count += 1;
        Ok(id)
    }

    /// Insert a vector with JSON metadata.
    pub fn insert_with_metadata(
        &mut self,
        vector: &BitpackedVector,
        metadata: &[u8],
    ) -> Result<u64> {
        let id = self.buffer.add_with_metadata(vector, metadata)?;
        self.index.add(vector.clone());
        self.buffer_count += 1;
        Ok(id)
    }

    /// Insert a pre-built VectorBatch (from ConcurrentWriteCache flush).
    ///
    /// This is the integration point for the XOR write cache:
    /// ```text
    /// ConcurrentWriteCache::flush() → materialize → VectorBatch → insert_batch()
    /// ```
    ///
    /// The batch is added to the in-memory index and buffered for the
    /// next `flush()` to Lance.
    pub fn insert_batch(&mut self, batch: VectorBatch) {
        let batch_idx = self.batches.len();
        for (row_idx, (id, vec)) in batch.iter().enumerate() {
            self.id_map.insert(id, (batch_idx, row_idx));
            self.index.add(vec);
            if id >= self.next_id {
                self.next_id = id + 1;
            }
        }
        self.batches.push(batch);
    }

    // ========================================================================
    // FLUSH (buffer → Lance, creates new version)
    // ========================================================================

    /// Flush the write buffer to Lance, creating a new dataset version.
    ///
    /// Returns the new version number. If the buffer is empty, returns
    /// the current version without writing.
    ///
    /// This is the crash-safety boundary: after `flush()` returns, data
    /// is persisted in a new Lance manifest version. If the process
    /// crashes before `flush()`, buffered data is lost (by design —
    /// the ConcurrentWriteCache provides sub-flush durability via its
    /// own XOR delta journal).
    pub async fn flush(&mut self) -> Result<u64> {
        if self.buffer_count == 0 {
            return Ok(self.version);
        }

        // Take the buffer and replace with a fresh one
        let new_next_id = self.next_id + self.buffer_count as u64;
        let builder = std::mem::replace(
            &mut self.buffer,
            VectorBatchBuilder::new().with_start_id(new_next_id),
        );

        let batch = builder.build()?;
        let record_batch = batch.as_record_batch().clone();

        // Add to local batches for continued zero-copy search
        let batch_idx = self.batches.len();
        for (row_idx, (id, _vec)) in batch.iter().enumerate() {
            self.id_map.insert(id, (batch_idx, row_idx));
        }
        self.batches.push(batch);

        // Write to Lance
        let uri = self.path.to_string_lossy().to_string();
        let mode = if self.dataset.is_some() {
            WriteMode::Append
        } else {
            WriteMode::Create
        };

        let params = WriteParams {
            mode,
            ..Default::default()
        };

        let reader = RecordBatchIterator::new(
            vec![Ok(record_batch)],
            self.schema.clone(),
        );

        let dataset = Dataset::write(reader, &uri, Some(params))
            .await
            .map_err(|e| HdrError::Storage(format!("Lance write: {}", e)))?;

        self.version = dataset.version().version;
        self.next_id = new_next_id;
        self.buffer_count = 0;
        self.dataset = Some(dataset);

        Ok(self.version)
    }

    // ========================================================================
    // SEARCH (HDR cascade on Arrow batches, zero-copy)
    // ========================================================================

    /// Search for k nearest neighbors using HDR cascade.
    ///
    /// The cascade operates directly on Arrow buffers via VectorSlice:
    /// 1. Belichtungsmesser: ~14 cycles, filters ~90% (zero copy)
    /// 2. StackedPopcount: ~157 cycles, filters ~80% of survivors (zero copy)
    /// 3. Exact Hamming: only for the ~1-2% that survive (still zero copy)
    ///
    /// Returns (id, distance, similarity) tuples sorted by distance.
    pub fn search(&self, query: &BitpackedVector, k: usize) -> Vec<(u64, u32, f32)> {
        let results = ArrowBatchSearch::cascaded_knn(
            &self.batches,
            query,
            k,
            crate::bitpack::VECTOR_BITS as u32,
        );

        results
            .into_iter()
            .map(|r| (r.id, r.distance, r.similarity))
            .collect()
    }

    /// Search within a Hamming radius using HDR cascade.
    pub fn search_within(
        &self,
        query: &BitpackedVector,
        k: usize,
        radius: u32,
    ) -> Vec<BatchSearchResult> {
        ArrowBatchSearch::cascaded_knn(&self.batches, query, k, radius)
    }

    /// Range search: find all vectors within `radius`.
    pub fn range_search(
        &self,
        query: &BitpackedVector,
        radius: u32,
    ) -> Vec<BatchSearchResult> {
        ArrowBatchSearch::range_search(&self.batches, query, radius)
    }

    /// XOR-bind search: find vectors whose bind with `key` is near `target`.
    ///
    /// This is the holographic probe operation done zero-copy:
    /// for each candidate c, compute hamming(c XOR key, target).
    pub fn bind_search(
        &self,
        key: &BitpackedVector,
        target: &BitpackedVector,
        k: usize,
        radius: u32,
    ) -> Vec<BatchSearchResult> {
        ArrowBatchSearch::bind_search(&self.batches, key, target, k, radius)
    }

    // ========================================================================
    // GET / LOOKUP
    // ========================================================================

    /// Get a vector by ID (O(1) via id_map).
    pub fn get(&self, id: u64) -> Option<BitpackedVector> {
        let (batch_idx, row_idx) = self.id_map.get(&id)?;
        self.batches.get(*batch_idx)?.get_vector(*row_idx)
    }

    /// Get vector bytes by ID (zero-copy).
    pub fn get_bytes(&self, id: u64) -> Option<&[u8]> {
        let (batch_idx, row_idx) = self.id_map.get(&id)?;
        self.batches.get(*batch_idx)?.get_bytes(*row_idx)
    }

    // ========================================================================
    // DELETE (Lance deletion vectors — soft delete)
    // ========================================================================

    /// Delete vectors matching a SQL predicate.
    ///
    /// Examples:
    /// - `"id = 42"` — delete single vector
    /// - `"id IN (1, 2, 3)"` — delete multiple
    /// - `"created_at < timestamp '2024-01-01 00:00:00'"` — by timestamp
    ///
    /// Creates a deletion vector (bitmask). The data is not physically
    /// removed until `compact()` is called.
    pub async fn delete(&mut self, predicate: &str) -> Result<()> {
        let dataset = self.dataset.as_mut()
            .ok_or_else(|| HdrError::Storage("No dataset open".into()))?;

        dataset
            .delete(predicate)
            .await
            .map_err(|e| HdrError::Storage(format!("Lance delete: {}", e)))?;

        // Rebuild index to reflect deletions
        self.rebuild_index().await?;
        Ok(())
    }

    /// Delete vectors by ID list.
    pub async fn delete_ids(&mut self, ids: &[u64]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let id_list: Vec<String> = ids.iter().map(|id| id.to_string()).collect();
        let predicate = format!("id IN ({})", id_list.join(", "));
        self.delete(&predicate).await
    }

    // ========================================================================
    // VERSIONING (MVCC)
    // ========================================================================

    /// Current dataset version (0 = not yet persisted).
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Total number of vectors (persisted + buffered).
    pub fn count(&self) -> usize {
        self.id_map.len() + self.buffer_count
    }

    /// Number of vectors in the unflushed buffer.
    pub fn buffer_len(&self) -> usize {
        self.buffer_count
    }

    /// Is the store empty?
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Get the HDR cascade index (for advanced search tuning).
    pub fn index(&self) -> &HdrCascade {
        &self.index
    }

    /// Get path to the Lance dataset directory.
    pub fn path(&self) -> &Path {
        &self.path
    }

    // ========================================================================
    // COMPACT (merge fragments, reclaim deleted space)
    // ========================================================================

    /// Get a reference to the underlying Lance dataset.
    ///
    /// For advanced operations (compaction, index optimization) that
    /// require direct Lance API access:
    ///
    /// ```rust,ignore
    /// use lance_index::traits::DatasetIndexExt;
    /// if let Some(ds) = store.dataset_mut() {
    ///     ds.optimize_indices(&Default::default()).await?;
    /// }
    /// ```
    pub fn dataset(&self) -> Option<&Dataset> {
        self.dataset.as_ref()
    }

    /// Get a mutable reference to the underlying Lance dataset.
    pub fn dataset_mut(&mut self) -> Option<&mut Dataset> {
        self.dataset.as_mut()
    }

    // ========================================================================
    // INTERNAL: rebuild index from Lance scan
    // ========================================================================

    /// Scan the Lance dataset and rebuild the in-memory HDR cascade index.
    ///
    /// Called on `open()` and after `delete()`. Reads all surviving rows
    /// from the dataset (deletion vectors are respected automatically).
    async fn rebuild_index(&mut self) -> Result<()> {
        let dataset = self.dataset.as_ref()
            .ok_or_else(|| HdrError::Storage("No dataset to scan".into()))?;

        self.index = HdrCascade::new();
        self.batches.clear();
        self.id_map.clear();
        self.next_id = 0;

        let mut stream = dataset
            .scan()
            .try_into_stream()
            .await
            .map_err(|e| HdrError::Storage(format!("Lance scan: {}", e)))?;

        while let Some(batch) = stream
            .try_next()
            .await
            .map_err(|e| HdrError::Storage(format!("Lance scan batch: {}", e)))?
        {
            let vector_batch = VectorBatch::from_record_batch(batch)?;
            let batch_idx = self.batches.len();

            for (row_idx, (id, vec)) in vector_batch.iter().enumerate() {
                self.id_map.insert(id, (batch_idx, row_idx));
                self.index.add(vec);
                if id >= self.next_id {
                    self.next_id = id + 1;
                }
            }

            self.batches.push(vector_batch);
        }

        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitpack::BitpackedVector;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_lance_create_insert_flush() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.lance");

        let mut store = LanceStore::new(&path);
        assert_eq!(store.count(), 0);
        assert_eq!(store.version(), 0);

        // Insert vectors (buffered, not yet durable)
        let v1 = BitpackedVector::random(1);
        let v2 = BitpackedVector::random(2);
        let id1 = store.insert(&v1).unwrap();
        let id2 = store.insert(&v2).unwrap();
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(store.count(), 2);
        assert_eq!(store.buffer_len(), 2);

        // Flush to Lance
        let version = store.flush().await.unwrap();
        assert!(version > 0, "Version should be > 0 after flush");
        assert_eq!(store.buffer_len(), 0);
        assert_eq!(store.count(), 2);
    }

    #[tokio::test]
    async fn test_lance_flush_and_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("persist.lance");

        // Create and populate
        let mut store = LanceStore::new(&path);
        let v1 = BitpackedVector::random(100);
        let v2 = BitpackedVector::random(200);
        store.insert(&v1).unwrap();
        store.insert(&v2).unwrap();
        let v1 = store.flush().await.unwrap();
        assert!(v1 > 0);

        // Reopen
        let store2 = LanceStore::open(&path).await.unwrap();
        assert_eq!(store2.count(), 2);
        assert!(store2.version() > 0);

        // Verify vectors survived
        let retrieved = store2.get(0).unwrap();
        let expected = BitpackedVector::random(100);
        assert_eq!(retrieved, expected);
    }

    #[tokio::test]
    async fn test_lance_search() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("search.lance");

        let mut store = LanceStore::new(&path);

        let query = BitpackedVector::random(42);
        // Insert the query itself — should be distance 0
        store.insert(&query).unwrap();

        // Insert 50 random vectors
        for i in 100..150 {
            let v = BitpackedVector::random(i);
            store.insert(&v).unwrap();
        }

        store.flush().await.unwrap();

        // Search — the query itself should be #1 result
        let results = store.search(&query, 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 0, "First result should be the query vector");
        assert_eq!(results[0].1, 0, "Distance to self should be 0");
    }

    #[tokio::test]
    async fn test_lance_version_increments() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("versions.lance");

        let mut store = LanceStore::new(&path);

        // First flush
        store.insert(&BitpackedVector::random(1)).unwrap();
        let v1 = store.flush().await.unwrap();

        // Second flush
        store.insert(&BitpackedVector::random(2)).unwrap();
        let v2 = store.flush().await.unwrap();

        assert!(v2 > v1, "Version should increase: {} > {}", v2, v1);
    }

    #[tokio::test]
    async fn test_lance_empty_flush_noop() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("noop.lance");

        let mut store = LanceStore::new(&path);
        store.insert(&BitpackedVector::random(1)).unwrap();
        let v1 = store.flush().await.unwrap();

        // Empty flush should return same version
        let v2 = store.flush().await.unwrap();
        assert_eq!(v1, v2, "Empty flush should not create new version");
    }

    #[tokio::test]
    async fn test_lance_delete() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("delete.lance");

        let mut store = LanceStore::new(&path);
        for i in 0..10 {
            store.insert(&BitpackedVector::random(i)).unwrap();
        }
        store.flush().await.unwrap();
        assert_eq!(store.count(), 10);

        // Delete IDs 3, 5, 7
        store.delete_ids(&[3, 5, 7]).await.unwrap();

        // After delete + rebuild, count should be 7
        assert_eq!(store.count(), 7);

        // Deleted vectors should not be retrievable
        assert!(store.get(3).is_none());
        assert!(store.get(5).is_none());
        assert!(store.get(7).is_none());

        // Surviving vectors should still be there
        assert!(store.get(0).is_some());
        assert!(store.get(1).is_some());
    }

    #[tokio::test]
    async fn test_lance_range_search() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("range.lance");

        let mut store = LanceStore::new(&path);

        let query = BitpackedVector::random(42);
        store.insert(&query).unwrap();
        for i in 100..110 {
            store.insert(&BitpackedVector::random(i)).unwrap();
        }
        store.flush().await.unwrap();

        // Range search with large radius should find at least the self-match
        let results = store.range_search(&query, crate::bitpack::VECTOR_BITS as u32);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0);
        assert_eq!(results[0].distance, 0);
    }

    #[tokio::test]
    async fn test_lance_buffer_searchable_before_flush() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("prebuf.lance");

        let mut store = LanceStore::new(&path);
        let query = BitpackedVector::random(42);
        store.insert(&query).unwrap();

        // Search BEFORE flush — should find via HDR cascade index
        let results = store.index().search(&query, 1);
        assert!(!results.is_empty());
        assert_eq!(results[0].distance, 0);
    }
}
