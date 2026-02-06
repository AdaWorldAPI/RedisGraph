# Implementation Plan: Hardening Fixes + Exploration Features

**Branch**: `claude/review-rust-graph-port-XZVrr`
**Date**: 2026-02-06
**Status**: All items below are to be implemented in this session

---

## PART 1: Hardening Fixes (from HARDENING_REVIEW.md)

### 1.1 Schema Versioning Byte
**File**: `width_16k/schema.rs`
**Change**: Reserve top 8 bits of `words[208]` (first schema word) as version tag.
- `SchemaSidecar::SCHEMA_VERSION = 1`
- `write_to_words()`: Write version into `words[208] |= (version << 56)`
- `read_from_words()`: Read version, currently accept only 0 or 1
- ANI pack/unpack must mask off the top 8 bits of word[208]
- **Backward compatible**: Version 0 (all existing data) reads as before

### 1.2 Bounded Top-K Search with BinaryHeap
**File**: `width_16k/search.rs`
**Change**: The current `search()` uses `Vec::insert()` with binary search which is O(n) per insert due to shifting. Replace with `std::collections::BinaryHeap<Reverse<(u32, usize)>>` capped at k.
- Actually the current approach is already reasonable (insert + truncate at k), but `Vec::insert` shifts elements. For k<=100 this is fine. The real fix is just documenting this is already bounded.
- **Actual issue**: No cap on the initial candidate pool scanning. Add a `max_candidates` builder option that limits total scans.

### 1.3 RwLock Wrapper for XorWriteCache
**File**: `width_16k/xor_bubble.rs`
**Change**: Add `ConcurrentWriteCache` wrapper:
```rust
pub struct ConcurrentWriteCache {
    inner: std::sync::RwLock<XorWriteCache>,
}
```
Methods: `read_through()` takes read lock, `record_delta()` takes write lock, `flush()` takes write lock.

### 1.4 Fix RNG Seed=0 in probabilistic_mask
**File**: `width_16k/xor_bubble.rs`
**Change**: In `XorBubble::apply_to_parent()`, add seed fixup:
```rust
let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15)...;
if rng == 0 { rng = 1; }  // Prevent degenerate xorshift
```
Also in `probabilistic_mask()`, add the same guard.

### 1.5 DeltaChain Depth Limit
**File**: `width_16k/xor_bubble.rs`
**Change**: Add `MAX_CHAIN_DEPTH = 256` constant. `DeltaChain::from_path()` truncates at this depth. `reconstruct()` already clamps via `.min()`.

### 1.6 GraphBLAS Fan-In Cap
**File**: `navigator.rs`
**Change**: In `graphblas_spmv()`, cap `output[row].len()` at a configurable max (default 10000). Drop lowest-priority messages when cap is hit.
- Actually this is complex and changes semantics. Better approach: add `with_max_fan_in(n)` to Navigator builder, document the limit.

---

## PART 2: Quick Wins

### 2.1 Bloom-Accelerated Traversal
**File**: `width_16k/search.rs`
**New function**: `bloom_accelerated_search()`
```rust
pub fn bloom_accelerated_search(
    candidates: &[&[u64]],
    query: &[u64],
    source_id: u64,  // Node we're searching from
    k: usize,
) -> Vec<SchemaSearchResult>
```
After schema predicate check, also check `bloom_might_be_neighbors(candidate, source_id)` — candidates that are bloom-confirmed neighbors get a distance bonus (lower distance score) because they have a known 1-hop path.

### 2.2 NARS Confidence Pruning in SchemaQuery
**File**: `width_16k/search.rs`
**Change**: Add `min_evidence: Option<f32>` to `NarsFilter`. This filters nodes that have too little evidence to be trusted, regardless of frequency.

Already covered by `min_confidence`. No change needed — just document that `min_confidence` serves this purpose.

### 2.3 RL-Routed Tree Descent
**File**: `width_16k/search.rs`
**New function**: `rl_guided_search()`
Uses `rl_routing_score()` to combine Hamming distance with Q-value during candidate ranking. The SchemaQuery gains `rl_alpha: Option<f32>` to enable this.

### 2.4 Federated Schema Merge
**File**: `width_16k/search.rs`
**New function**: `schema_merge(a, b) -> Vec<u64>`
Like `schema_bind` but with federation semantics:
- Semantic blocks: majority vote (bundle of 2 → undefined for binary; use `a` as default)
- ANI: element-wise max
- NARS: revision (combine evidence from both instances)
- RL: average Q-values
- Bloom: OR (union of known neighbors)
- Metrics: max pagerank, min hop_to_root

---

## PART 3: Files to Modify

| File | Changes |
|------|---------|
| `width_16k/schema.rs` | Schema version byte in pack/unpack |
| `width_16k/search.rs` | bloom_accelerated_search, rl_guided_search, schema_merge, max_candidates |
| `width_16k/xor_bubble.rs` | ConcurrentWriteCache, RNG seed guard, depth limit |
| `navigator.rs` | Wire new search functions into Cypher, fan-in docs |
| `width_16k/demo.rs` | Add tests for new features |
| `HARDENING_REVIEW.md` | Update with fixes applied |

---

## PART 4: Build & Test

```bash
cargo test --no-default-features --features simd 2>&1
```

Expected: 239+ pass (new tests added), 10 pre-existing failures unchanged.

Then:
```bash
git add -A src/fingerprint/rust/src/width_16k/ src/fingerprint/rust/src/navigator.rs src/fingerprint/rust/src/HARDENING_REVIEW.md
git commit -m "Harden: schema versioning, concurrent cache, bounded search, bloom traversal, RL routing, federated merge"
git push -u origin claude/review-rust-graph-port-XZVrr
```
