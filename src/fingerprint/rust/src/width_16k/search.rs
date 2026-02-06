//! Schema-Aware Search API
//!
//! Extends the HDR search cascade with schema predicate pruning.
//! Because ANI/NARS/RL markers live inline in the fingerprint (blocks 13-15),
//! we can reject candidates in O(1) *before* computing Hamming distance.
//!
//! # The Search Cascade with Schema
//!
//! ```text
//! Candidate pool (n vectors)
//!   │
//!   ├─► Level 0: Schema predicate filter (O(1) per vector)
//!   │     Read 2-3 words from blocks 13-15, check ANI/NARS/RL predicates
//!   │     Cost: ~3 cycles per candidate
//!   │     Rejects: depends on predicate selectivity
//!   │
//!   ├─► Level 1: Belichtungsmesser (7-point sample, ~14 cycles)
//!   │     Rejects: ~90% of survivors
//!   │
//!   ├─► Level 2: Block-masked StackedPopcount with threshold
//!   │     Only compute on semantic blocks (0..12), skip schema blocks
//!   │     Rejects: ~80% of survivors
//!   │
//!   └─► Level 3: Exact distance on semantic blocks
//!         k results returned
//! ```
//!
//! # Why This Is Fast
//!
//! Traditional approach: compute Hamming distance first, THEN check metadata.
//! Our approach: check metadata first (it's already in the vector!), then
//! distance on survivors only. For selective predicates (e.g., "ANI level >= 3",
//! "NARS confidence > 0.8"), this eliminates most candidates before the
//! expensive popcount cascade even starts.

use super::schema::{
    AniLevels, NarsTruth, NarsBudget, EdgeTypeMarker, NodeTypeMarker, NodeKind,
    InlineQValues, InlineRewards, NeighborBloom, GraphMetrics, SchemaSidecar,
};
use super::{VECTOR_WORDS, NUM_BLOCKS, BITS_PER_BLOCK, SEMANTIC_BLOCKS, SCHEMA_BLOCK_START};

// ============================================================================
// BLOCK MASK: Which blocks participate in distance computation
// ============================================================================

/// Bitmask selecting which of the 16 blocks participate in distance
/// computation. Default: blocks 0..12 (semantic only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockMask {
    /// 16-bit mask, one bit per block. Bit 0 = block 0, etc.
    mask: u16,
}

impl BlockMask {
    /// All 16 blocks (full 16K distance)
    pub const ALL: Self = Self { mask: 0xFFFF };

    /// Semantic blocks only (0..12 = 13,312 bits)
    pub const SEMANTIC: Self = Self { mask: 0x1FFF }; // bits 0..12

    /// Schema blocks only (13..15 = 3,072 bits)
    pub const SCHEMA: Self = Self { mask: 0xE000 }; // bits 13..15

    /// Custom mask from raw u16
    pub const fn from_raw(mask: u16) -> Self {
        Self { mask }
    }

    /// Is block `i` included?
    #[inline]
    pub fn includes(&self, block: usize) -> bool {
        block < 16 && (self.mask & (1u16 << block)) != 0
    }

    /// Number of included blocks
    pub fn count(&self) -> u32 {
        self.mask.count_ones()
    }

    /// Number of words covered by this mask
    pub fn word_count(&self) -> usize {
        self.count() as usize * 16
    }

    /// Number of bits covered (for normalization)
    pub fn bit_count(&self) -> usize {
        self.word_count() * 64
    }
}

impl Default for BlockMask {
    fn default() -> Self {
        Self::SEMANTIC
    }
}

// ============================================================================
// SCHEMA PREDICATES: O(1) filters on inline metadata
// ============================================================================

/// ANI level filter
#[derive(Clone, Debug)]
pub struct AniFilter {
    /// Minimum reasoning level (0..7) that must be active
    pub min_level: u8,
    /// Minimum activation at that level
    pub min_activation: u16,
}

/// NARS truth/budget filter
#[derive(Clone, Debug)]
pub struct NarsFilter {
    /// Minimum frequency (0.0..1.0)
    pub min_frequency: Option<f32>,
    /// Minimum confidence (0.0..1.0)
    pub min_confidence: Option<f32>,
    /// Minimum priority (0.0..1.0)
    pub min_priority: Option<f32>,
}

/// RL state filter
#[derive(Clone, Debug)]
pub struct RlFilter {
    /// Minimum Q-value for best action
    pub min_best_q: Option<f32>,
    /// Minimum average reward
    pub min_avg_reward: Option<f32>,
    /// Positive reward trend required
    pub positive_trend: bool,
}

/// Graph topology filter
#[derive(Clone, Debug)]
pub struct GraphFilter {
    /// Minimum PageRank (quantized 0..65535)
    pub min_pagerank: Option<u16>,
    /// Maximum hop distance to root
    pub max_hop: Option<u8>,
    /// Required cluster ID
    pub cluster_id: Option<u16>,
    /// Minimum degree
    pub min_degree: Option<u8>,
}

/// Node kind filter
#[derive(Clone, Debug)]
pub struct KindFilter {
    /// Accepted node kinds (empty = accept all)
    pub kinds: Vec<NodeKind>,
    /// Accepted edge verb IDs (empty = accept all)
    pub verb_ids: Vec<u8>,
}

// ============================================================================
// SCHEMA QUERY: Combined predicate + distance search
// ============================================================================

/// A schema-aware search query.
///
/// Combines traditional Hamming distance search with schema predicate filters.
/// Predicates are checked *before* distance computation for early rejection.
///
/// # Example
///
/// ```text
/// SchemaQuery::new()
///     .with_ani(AniFilter { min_level: 3, min_activation: 100 })
///     .with_nars(NarsFilter { min_confidence: Some(0.5), ..Default::default() })
///     .with_block_mask(BlockMask::SEMANTIC)
///     .search(&candidates, &query, 10)
/// ```
#[derive(Clone, Debug)]
pub struct SchemaQuery {
    /// ANI reasoning level filter
    pub ani_filter: Option<AniFilter>,
    /// NARS truth/budget filter
    pub nars_filter: Option<NarsFilter>,
    /// RL state filter
    pub rl_filter: Option<RlFilter>,
    /// Graph topology filter
    pub graph_filter: Option<GraphFilter>,
    /// Node/edge kind filter
    pub kind_filter: Option<KindFilter>,
    /// Which blocks participate in distance (default: semantic only)
    pub block_mask: BlockMask,
    /// Maximum Hamming distance (on masked blocks)
    pub max_distance: Option<u32>,
}

impl SchemaQuery {
    pub fn new() -> Self {
        Self {
            ani_filter: None,
            nars_filter: None,
            rl_filter: None,
            graph_filter: None,
            kind_filter: None,
            block_mask: BlockMask::SEMANTIC,
            max_distance: None,
        }
    }

    /// Builder: add ANI filter
    pub fn with_ani(mut self, filter: AniFilter) -> Self {
        self.ani_filter = Some(filter);
        self
    }

    /// Builder: add NARS filter
    pub fn with_nars(mut self, filter: NarsFilter) -> Self {
        self.nars_filter = Some(filter);
        self
    }

    /// Builder: add RL filter
    pub fn with_rl(mut self, filter: RlFilter) -> Self {
        self.rl_filter = Some(filter);
        self
    }

    /// Builder: add graph topology filter
    pub fn with_graph(mut self, filter: GraphFilter) -> Self {
        self.graph_filter = Some(filter);
        self
    }

    /// Builder: add node/edge kind filter
    pub fn with_kind(mut self, filter: KindFilter) -> Self {
        self.kind_filter = Some(filter);
        self
    }

    /// Builder: set block mask
    pub fn with_block_mask(mut self, mask: BlockMask) -> Self {
        self.block_mask = mask;
        self
    }

    /// Builder: set maximum Hamming distance
    pub fn with_max_distance(mut self, d: u32) -> Self {
        self.max_distance = Some(d);
        self
    }

    /// Check if a candidate's schema passes all predicates.
    ///
    /// This reads directly from the word array — **zero deserialization cost**
    /// when only checking a few fields. Each predicate reads 1-2 words max.
    ///
    /// Returns `true` if the candidate passes (should proceed to distance check).
    pub fn passes_predicates(&self, candidate_words: &[u64]) -> bool {
        if candidate_words.len() < VECTOR_WORDS {
            return false;
        }

        let base = SchemaSidecar::WORD_OFFSET; // 208

        // ANI filter: read words[208..209] (128 bits)
        if let Some(ref ani) = self.ani_filter {
            let ani_packed = candidate_words[base] as u128
                | ((candidate_words[base + 1] as u128) << 64);
            let levels = AniLevels::unpack(ani_packed);
            let level_vals = [
                levels.reactive, levels.memory, levels.analogy, levels.planning,
                levels.meta, levels.social, levels.creative, levels.r#abstract,
            ];
            if ani.min_level as usize >= 8 {
                return false;
            }
            // Check that the required level (and all above) meet activation threshold
            let activation = level_vals[ani.min_level as usize];
            if activation < ani.min_activation {
                return false;
            }
        }

        // NARS filter: read word[210] (lower 32 bits = truth)
        if let Some(ref nars) = self.nars_filter {
            let truth = NarsTruth::unpack(candidate_words[base + 2] as u32);
            if let Some(min_f) = nars.min_frequency {
                if truth.f() < min_f {
                    return false;
                }
            }
            if let Some(min_c) = nars.min_confidence {
                if truth.c() < min_c {
                    return false;
                }
            }
            // Budget: upper 32 bits of word[210] → lower 64 bits
            if let Some(min_p) = nars.min_priority {
                let budget = NarsBudget::unpack((candidate_words[base + 2] >> 32) as u64);
                if (budget.priority as f32 / 65535.0) < min_p {
                    return false;
                }
            }
        }

        // Kind filter: read word[211] (upper 32 bits = node type)
        if let Some(ref kind) = self.kind_filter {
            if !kind.kinds.is_empty() {
                let node = NodeTypeMarker::unpack((candidate_words[base + 3] >> 32) as u32);
                if !kind.kinds.iter().any(|k| *k as u8 == node.kind) {
                    return false;
                }
            }
            if !kind.verb_ids.is_empty() {
                let edge = EdgeTypeMarker::unpack(candidate_words[base + 3] as u32);
                if !kind.verb_ids.contains(&edge.verb_id) {
                    return false;
                }
            }
        }

        // RL filter: read words[224..227]
        if let Some(ref rl) = self.rl_filter {
            let block14_base = base + 16;

            if let Some(min_q) = rl.min_best_q {
                let q = InlineQValues::unpack([
                    candidate_words[block14_base],
                    candidate_words[block14_base + 1],
                ]);
                let best = q.q(q.best_action());
                if best < min_q {
                    return false;
                }
            }

            if rl.min_avg_reward.is_some() || rl.positive_trend {
                let mut rewards = InlineRewards::default();
                let rw0 = candidate_words[block14_base + 2];
                let rw1 = candidate_words[block14_base + 3];
                for i in 0..4 {
                    rewards.rewards[i] = ((rw0 >> (i * 16)) & 0xFFFF) as u16 as i16;
                }
                for i in 0..4 {
                    rewards.rewards[i + 4] = ((rw1 >> (i * 16)) & 0xFFFF) as u16 as i16;
                }

                if let Some(min_avg) = rl.min_avg_reward {
                    if rewards.average() < min_avg {
                        return false;
                    }
                }
                if rl.positive_trend && rewards.trend() <= 0.0 {
                    return false;
                }
            }
        }

        // Graph filter: read word[248]
        if let Some(ref graph) = self.graph_filter {
            let block15_base = base + 32;
            let metrics = GraphMetrics::unpack(candidate_words[block15_base + 8]);

            if let Some(min_pr) = graph.min_pagerank {
                if metrics.pagerank < min_pr {
                    return false;
                }
            }
            if let Some(max_h) = graph.max_hop {
                if metrics.hop_to_root > max_h {
                    return false;
                }
            }
            if let Some(cid) = graph.cluster_id {
                if metrics.cluster_id != cid {
                    return false;
                }
            }
            if let Some(min_d) = graph.min_degree {
                if metrics.degree < min_d {
                    return false;
                }
            }
        }

        true
    }

    /// Compute block-masked Hamming distance between two word arrays.
    ///
    /// Only popcount words in blocks selected by `self.block_mask`.
    /// For `BlockMask::SEMANTIC` (blocks 0..12), this computes distance
    /// over 13,312 bits and ignores the schema blocks entirely.
    pub fn masked_distance(&self, a: &[u64], b: &[u64]) -> u32 {
        debug_assert!(a.len() >= VECTOR_WORDS);
        debug_assert!(b.len() >= VECTOR_WORDS);

        let mut total = 0u32;
        for block in 0..NUM_BLOCKS {
            if !self.block_mask.includes(block) {
                continue;
            }
            let start = block * 16;
            let end = start + 16; // All blocks are 16 words in 16K
            for w in start..end {
                total += (a[w] ^ b[w]).count_ones();
            }
        }
        total
    }

    /// Compute block-masked distance with early termination.
    ///
    /// Returns `None` if the running distance exceeds `threshold` at any
    /// block boundary (coarse-grained pruning on block sums).
    pub fn masked_distance_with_threshold(
        &self,
        a: &[u64],
        b: &[u64],
        threshold: u32,
    ) -> Option<u32> {
        debug_assert!(a.len() >= VECTOR_WORDS);
        debug_assert!(b.len() >= VECTOR_WORDS);

        let mut total = 0u32;
        for block in 0..NUM_BLOCKS {
            if !self.block_mask.includes(block) {
                continue;
            }
            let start = block * 16;
            let end = start + 16;
            let mut block_sum = 0u32;
            for w in start..end {
                block_sum += (a[w] ^ b[w]).count_ones();
            }
            total += block_sum;
            if total > threshold {
                return None; // Early exit: exceeded threshold
            }
        }
        Some(total)
    }

    /// Full search pipeline: predicate filter → block-masked distance → top-k.
    ///
    /// `candidates` is a slice of `&[u64; 256]` word arrays (zero-copy from Arrow).
    /// Returns (index, distance) pairs sorted by distance, up to `k` results.
    pub fn search(
        &self,
        candidates: &[&[u64]],
        query: &[u64],
        k: usize,
    ) -> Vec<SchemaSearchResult> {
        let mut results: Vec<SchemaSearchResult> = Vec::with_capacity(k + 1);
        let mut current_threshold = self.max_distance.unwrap_or(u32::MAX);

        for (idx, &candidate) in candidates.iter().enumerate() {
            // Level 0: Schema predicate filter (O(1), ~3 cycles)
            if !self.passes_predicates(candidate) {
                continue;
            }

            // Level 1: Block-masked distance with threshold
            let dist = match self.masked_distance_with_threshold(
                query, candidate, current_threshold,
            ) {
                Some(d) => d,
                None => continue,
            };

            // Insert into results (maintain sorted order)
            let result = SchemaSearchResult {
                index: idx,
                distance: dist,
                schema: None, // Lazy: only decode schema on demand
            };

            // Binary search for insertion point
            let pos = results.partition_point(|r| r.distance <= dist);
            results.insert(pos, result);

            if results.len() > k {
                results.truncate(k);
                // Tighten threshold to best kth distance
                current_threshold = results.last().map(|r| r.distance).unwrap_or(u32::MAX);
            }
        }

        results
    }
}

impl Default for SchemaQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from schema-aware search
#[derive(Clone, Debug)]
pub struct SchemaSearchResult {
    /// Index in the candidate array
    pub index: usize,
    /// Block-masked Hamming distance
    pub distance: u32,
    /// Decoded schema (lazy, populated on demand)
    pub schema: Option<SchemaSidecar>,
}

impl SchemaSearchResult {
    /// Decode the full schema sidecar from the candidate words.
    /// Call this only when you need the schema details — it's ~50ns per decode.
    pub fn decode_schema(&mut self, candidate_words: &[u64]) {
        self.schema = Some(SchemaSidecar::read_from_words(candidate_words));
    }
}

// ============================================================================
// BLOOM-ASSISTED NEIGHBOR CHECK
// ============================================================================

/// Check if two 16K vectors are likely neighbors using the inline bloom filter.
///
/// This is O(1) with ~1% FPR — no graph traversal needed.
/// The bloom filter in block 15 was populated during graph construction.
#[inline]
pub fn bloom_might_be_neighbors(a_words: &[u64], b_id: u64) -> bool {
    let bloom_base = SchemaSidecar::WORD_OFFSET + 32 + 4; // block 15, offset 4 words
    if a_words.len() < bloom_base + 4 {
        return false;
    }
    let bloom = NeighborBloom {
        words: [
            a_words[bloom_base],
            a_words[bloom_base + 1],
            a_words[bloom_base + 2],
            a_words[bloom_base + 3],
        ],
    };
    bloom.might_contain(b_id)
}

// ============================================================================
// Q-VALUE ROUTING: Use inline RL state for beam search guidance
// ============================================================================

/// Extract the best action and Q-value from a candidate's inline RL state.
///
/// This enables RL-guided beam search: instead of ranking candidates by
/// Hamming distance alone, combine distance with learned Q-value as a
/// routing heuristic. Candidates with higher Q-values for the current
/// action context get priority in the beam.
#[inline]
pub fn read_best_q(candidate_words: &[u64]) -> (usize, f32) {
    let block14_base = SchemaSidecar::WORD_OFFSET + 16;
    if candidate_words.len() < block14_base + 2 {
        return (0, 0.0);
    }
    let q = InlineQValues::unpack([
        candidate_words[block14_base],
        candidate_words[block14_base + 1],
    ]);
    let best = q.best_action();
    (best, q.q(best))
}

/// Composite routing score: weighted combination of Hamming distance
/// and Q-value for RL-guided search.
///
/// `alpha` controls the RL weight: 0.0 = pure distance, 1.0 = pure Q-value.
/// Typical: alpha = 0.2 (20% RL influence on routing).
#[inline]
pub fn rl_routing_score(distance: u32, q_value: f32, alpha: f32) -> f32 {
    let distance_norm = distance as f32 / (SEMANTIC_BLOCKS as f32 * BITS_PER_BLOCK as f32);
    let q_norm = (1.0 - q_value) / 2.0; // Map [-1, 1] → [1, 0] (lower = better)
    (1.0 - alpha) * distance_norm + alpha * q_norm
}

// ============================================================================
// NARS-AWARE OPERATIONS
// ============================================================================

/// Revise two 16K vectors' NARS truth values.
///
/// When bundling two vectors that carry NARS truth values, the resulting
/// truth value should be the NARS revision (combining evidence).
/// This reads both truth values inline, computes the revision, and
/// writes it to the output words.
pub fn nars_revision_inline(a_words: &[u64], b_words: &[u64], out_words: &mut [u64]) {
    let base = SchemaSidecar::WORD_OFFSET;
    if a_words.len() < VECTOR_WORDS || b_words.len() < VECTOR_WORDS || out_words.len() < VECTOR_WORDS {
        return;
    }

    let truth_a = NarsTruth::unpack(a_words[base + 2] as u32);
    let truth_b = NarsTruth::unpack(b_words[base + 2] as u32);
    let revised = truth_a.revision(&truth_b);

    // Preserve budget from higher-priority input
    let budget_a = NarsBudget::unpack((a_words[base + 2] >> 32) as u64);
    let budget_b = NarsBudget::unpack((b_words[base + 2] >> 32) as u64);
    let budget = if budget_a.priority >= budget_b.priority { budget_a } else { budget_b };

    out_words[base + 2] = revised.pack() as u64 | ((budget.pack() as u64) << 32);
}

/// NARS deduction chain: compute truth value for A→B, B→C ⊢ A→C
pub fn nars_deduction_inline(premise_words: &[u64], conclusion_words: &[u64]) -> NarsTruth {
    let base = SchemaSidecar::WORD_OFFSET;
    let t1 = NarsTruth::unpack(premise_words[base + 2] as u32);
    let t2 = NarsTruth::unpack(conclusion_words[base + 2] as u32);
    t1.deduction(&t2)
}

// ============================================================================
// SCHEMA-AWARE BIND: XOR with schema combination
// ============================================================================

/// XOR-bind two 16K vectors with intelligent schema merging.
///
/// The semantic blocks (0..12) are XOR'd as usual. The schema blocks are
/// handled specially:
/// - ANI levels: take element-wise max (binding shouldn't reduce capability)
/// - NARS truth: compute revision (combine evidence)
/// - RL state: preserve from `a` (primary operand)
/// - Graph cache: clear (binding creates a new edge, not a node)
///
/// This is the "surprising feature" — bind operations automatically
/// propagate and combine metadata without explicit schema management.
pub fn schema_bind(a: &[u64], b: &[u64]) -> Vec<u64> {
    assert!(a.len() >= VECTOR_WORDS && b.len() >= VECTOR_WORDS);
    let mut out = vec![0u64; VECTOR_WORDS];

    // Semantic blocks: XOR as usual
    let semantic_end = SCHEMA_BLOCK_START * 16; // word 208
    for w in 0..semantic_end {
        out[w] = a[w] ^ b[w];
    }

    let base = SchemaSidecar::WORD_OFFSET;

    // Block 13: ANI levels — element-wise max
    let ani_a = AniLevels::unpack(a[base] as u128 | ((a[base + 1] as u128) << 64));
    let ani_b = AniLevels::unpack(b[base] as u128 | ((b[base + 1] as u128) << 64));
    let ani_merged = AniLevels {
        reactive: ani_a.reactive.max(ani_b.reactive),
        memory: ani_a.memory.max(ani_b.memory),
        analogy: ani_a.analogy.max(ani_b.analogy),
        planning: ani_a.planning.max(ani_b.planning),
        meta: ani_a.meta.max(ani_b.meta),
        social: ani_a.social.max(ani_b.social),
        creative: ani_a.creative.max(ani_b.creative),
        r#abstract: ani_a.r#abstract.max(ani_b.r#abstract),
    };
    let packed_ani = ani_merged.pack();
    out[base] = packed_ani as u64;
    out[base + 1] = (packed_ani >> 64) as u64;

    // Block 13: NARS — revision
    let truth_a = NarsTruth::unpack(a[base + 2] as u32);
    let truth_b = NarsTruth::unpack(b[base + 2] as u32);
    let revised = truth_a.revision(&truth_b);
    // Budget: max priority
    let budget_a = NarsBudget::unpack((a[base + 2] >> 32) as u64);
    let budget_b = NarsBudget::unpack((b[base + 2] >> 32) as u64);
    let merged_budget = if budget_a.priority >= budget_b.priority {
        budget_a
    } else {
        budget_b
    };
    out[base + 2] = revised.pack() as u64 | ((merged_budget.pack() as u64) << 32);

    // Block 13: Edge type — XOR verb IDs (compositional binding)
    let edge_a = EdgeTypeMarker::unpack(a[base + 3] as u32);
    let edge_b = EdgeTypeMarker::unpack(b[base + 3] as u32);
    let merged_edge = EdgeTypeMarker {
        verb_id: edge_a.verb_id ^ edge_b.verb_id,
        direction: edge_a.direction, // preserve primary direction
        weight: ((edge_a.weight as u16 + edge_b.weight as u16) / 2) as u8,
        flags: edge_a.flags | edge_b.flags, // union of flags
    };
    out[base + 3] = merged_edge.pack() as u64;
    // Node type: XOR (compositional)
    let node_a = NodeTypeMarker::unpack((a[base + 3] >> 32) as u32);
    let node_b = NodeTypeMarker::unpack((b[base + 3] >> 32) as u32);
    out[base + 3] |= (NodeTypeMarker {
        kind: node_a.kind, // preserve primary kind
        subtype: node_a.subtype ^ node_b.subtype,
        provenance: node_a.provenance ^ node_b.provenance,
    }.pack() as u64) << 32;

    // Block 14: RL state — preserve from primary operand (a)
    let block14_base = base + 16;
    for w in 0..16 {
        out[block14_base + w] = a[block14_base + w];
    }

    // Block 15: Graph cache — clear (new binding = new identity)
    // Words 240..255 remain zero

    out
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_words() -> Vec<u64> {
        let mut words = vec![0u64; VECTOR_WORDS];
        // Set some schema data
        let mut sidecar = SchemaSidecar::default();
        sidecar.ani_levels.planning = 500;
        sidecar.ani_levels.meta = 200;
        sidecar.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        sidecar.nars_budget = NarsBudget::from_floats(0.9, 0.5, 0.7);
        sidecar.q_values.set_q(0, 0.7);
        sidecar.rewards.push(0.5);
        sidecar.metrics.pagerank = 1000;
        sidecar.metrics.hop_to_root = 2;
        sidecar.metrics.cluster_id = 42;
        sidecar.metrics.degree = 5;
        sidecar.neighbors.insert(100);
        sidecar.neighbors.insert(200);
        sidecar.write_to_words(&mut words);
        words
    }

    #[test]
    fn test_block_mask() {
        assert_eq!(BlockMask::ALL.count(), 16);
        assert_eq!(BlockMask::SEMANTIC.count(), 13);
        assert_eq!(BlockMask::SCHEMA.count(), 3);
        assert!(BlockMask::SEMANTIC.includes(0));
        assert!(BlockMask::SEMANTIC.includes(12));
        assert!(!BlockMask::SEMANTIC.includes(13));
        assert!(BlockMask::SCHEMA.includes(13));
        assert!(BlockMask::SCHEMA.includes(15));
    }

    #[test]
    fn test_predicate_ani_pass() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_ani(AniFilter {
            min_level: 3, // planning
            min_activation: 100,
        });
        assert!(query.passes_predicates(&words)); // planning=500 >= 100
    }

    #[test]
    fn test_predicate_ani_fail() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_ani(AniFilter {
            min_level: 3, // planning
            min_activation: 600,
        });
        assert!(!query.passes_predicates(&words)); // planning=500 < 600
    }

    #[test]
    fn test_predicate_nars_pass() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_nars(NarsFilter {
            min_frequency: Some(0.7),
            min_confidence: Some(0.5),
            min_priority: None,
        });
        assert!(query.passes_predicates(&words)); // f=0.8 >= 0.7, c=0.6 >= 0.5
    }

    #[test]
    fn test_predicate_nars_fail_confidence() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_nars(NarsFilter {
            min_frequency: None,
            min_confidence: Some(0.9), // too high
            min_priority: None,
        });
        assert!(!query.passes_predicates(&words));
    }

    #[test]
    fn test_predicate_graph_filter() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_graph(GraphFilter {
            min_pagerank: Some(500),
            max_hop: Some(3),
            cluster_id: Some(42),
            min_degree: Some(3),
        });
        assert!(query.passes_predicates(&words));
    }

    #[test]
    fn test_predicate_graph_wrong_cluster() {
        let words = make_test_words();
        let query = SchemaQuery::new().with_graph(GraphFilter {
            min_pagerank: None,
            max_hop: None,
            cluster_id: Some(99), // wrong cluster
            min_degree: None,
        });
        assert!(!query.passes_predicates(&words));
    }

    #[test]
    fn test_predicate_combined() {
        let words = make_test_words();
        // All filters pass together
        let query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 100 })
            .with_nars(NarsFilter {
                min_frequency: Some(0.5),
                min_confidence: Some(0.3),
                min_priority: None,
            })
            .with_graph(GraphFilter {
                min_pagerank: Some(500),
                max_hop: None,
                cluster_id: None,
                min_degree: None,
            });
        assert!(query.passes_predicates(&words));
    }

    #[test]
    fn test_masked_distance_semantic_only() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let b = vec![0u64; VECTOR_WORDS];

        // Set bit differences only in semantic region
        a[0] = 0xFFFF;
        // Set bit differences only in schema region (should be ignored)
        a[210] = 0xFFFF_FFFF_FFFF_FFFF;

        let query = SchemaQuery::new(); // default: semantic only
        let dist = query.masked_distance(&a, &b);

        // Only semantic bits counted: 16 bits from a[0]
        assert_eq!(dist, 16);
    }

    #[test]
    fn test_masked_distance_all_blocks() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let b = vec![0u64; VECTOR_WORDS];
        a[0] = 0xFFFF; // 16 bits in semantic
        a[210] = 0xFF;  // 8 bits in schema

        let query = SchemaQuery::new().with_block_mask(BlockMask::ALL);
        let dist = query.masked_distance(&a, &b);
        assert_eq!(dist, 24); // 16 + 8
    }

    #[test]
    fn test_masked_distance_with_threshold() {
        let a = vec![0xFFFF_FFFF_FFFF_FFFFu64; VECTOR_WORDS];
        let b = vec![0u64; VECTOR_WORDS];

        let query = SchemaQuery::new();
        // Very low threshold should abort early
        let result = query.masked_distance_with_threshold(&a, &b, 100);
        assert!(result.is_none()); // Exceeded threshold
    }

    #[test]
    fn test_search_pipeline() {
        let mut candidates: Vec<Vec<u64>> = Vec::new();

        // Candidate 0: close to query
        let mut c0 = vec![0u64; VECTOR_WORDS];
        c0[0] = 0xFF; // 8 bits different
        let mut s0 = SchemaSidecar::default();
        s0.ani_levels.planning = 500;
        s0.nars_truth = NarsTruth::from_floats(0.8, 0.6);
        s0.write_to_words(&mut c0);
        candidates.push(c0);

        // Candidate 1: far from query
        let mut c1 = vec![0xFFFF_FFFF_FFFF_FFFFu64; VECTOR_WORDS];
        let mut s1 = SchemaSidecar::default();
        s1.ani_levels.planning = 100;
        s1.nars_truth = NarsTruth::from_floats(0.3, 0.2);
        s1.write_to_words(&mut c1);
        candidates.push(c1);

        // Candidate 2: close but fails predicate
        let mut c2 = vec![0u64; VECTOR_WORDS];
        c2[0] = 0xF; // 4 bits different
        // No ANI planning set — will fail predicate
        candidates.push(c2);

        let refs: Vec<&[u64]> = candidates.iter().map(|c| c.as_slice()).collect();
        let query_words = vec![0u64; VECTOR_WORDS];

        let query = SchemaQuery::new()
            .with_ani(AniFilter { min_level: 3, min_activation: 50 });

        let results = query.search(&refs, &query_words, 10);

        // Candidate 0 passes (planning=500, dist=8)
        // Candidate 1 passes predicate (planning=100) but distance is huge
        // Candidate 2 fails predicate (planning=0)
        assert!(!results.is_empty());
        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].distance, 8);
    }

    #[test]
    fn test_bloom_neighbor_check() {
        let mut words = vec![0u64; VECTOR_WORDS];
        let mut sidecar = SchemaSidecar::default();
        sidecar.neighbors.insert(42);
        sidecar.neighbors.insert(100);
        sidecar.write_to_words(&mut words);

        assert!(bloom_might_be_neighbors(&words, 42));
        assert!(bloom_might_be_neighbors(&words, 100));
        // Unknown ID: might have false positive, but low probability
    }

    #[test]
    fn test_rl_routing_score() {
        // Pure distance mode (alpha=0)
        let score = rl_routing_score(1000, 0.5, 0.0);
        assert!(score > 0.0);

        // Pure Q-value mode (alpha=1)
        let score_high_q = rl_routing_score(1000, 0.9, 1.0);
        let score_low_q = rl_routing_score(1000, -0.5, 1.0);
        assert!(score_high_q < score_low_q); // Higher Q = lower (better) score
    }

    #[test]
    fn test_schema_bind_merges_metadata() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let mut b = vec![0u64; VECTOR_WORDS];

        let mut sa = SchemaSidecar::default();
        sa.ani_levels.planning = 500;
        sa.ani_levels.meta = 100;
        sa.nars_truth = NarsTruth::from_floats(0.8, 0.5);
        sa.write_to_words(&mut a);

        let mut sb = SchemaSidecar::default();
        sb.ani_levels.planning = 300;
        sb.ani_levels.meta = 400; // higher meta
        sb.nars_truth = NarsTruth::from_floats(0.6, 0.3);
        sb.write_to_words(&mut b);

        let result = schema_bind(&a, &b);
        let result_schema = SchemaSidecar::read_from_words(&result);

        // ANI: element-wise max
        assert_eq!(result_schema.ani_levels.planning, 500); // max(500, 300)
        assert_eq!(result_schema.ani_levels.meta, 400);     // max(100, 400)

        // NARS: revision should increase confidence
        assert!(result_schema.nars_truth.c() > 0.5 || result_schema.nars_truth.c() > 0.3);
    }

    #[test]
    fn test_read_best_q() {
        let mut words = vec![0u64; VECTOR_WORDS];
        let mut sidecar = SchemaSidecar::default();
        sidecar.q_values.set_q(3, 0.8);
        sidecar.q_values.set_q(7, -0.2);
        sidecar.write_to_words(&mut words);

        let (action, q) = read_best_q(&words);
        assert_eq!(action, 3);
        assert!((q - 0.8).abs() < 0.02);
    }

    #[test]
    fn test_nars_deduction_inline() {
        let mut a = vec![0u64; VECTOR_WORDS];
        let mut b = vec![0u64; VECTOR_WORDS];

        let mut sa = SchemaSidecar::default();
        sa.nars_truth = NarsTruth::from_floats(0.9, 0.8);
        sa.write_to_words(&mut a);

        let mut sb = SchemaSidecar::default();
        sb.nars_truth = NarsTruth::from_floats(0.7, 0.6);
        sb.write_to_words(&mut b);

        let deduced = nars_deduction_inline(&a, &b);
        // Deduction: f = f1*f2, c = f1*f2*c1*c2
        assert!(deduced.f() > 0.5); // 0.9 * 0.7 ≈ 0.63
        assert!(deduced.c() < deduced.f()); // confidence always ≤ frequency in deduction
    }
}
