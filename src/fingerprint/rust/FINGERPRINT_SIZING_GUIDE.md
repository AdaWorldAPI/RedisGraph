# Fingerprint Sizing Guide for ladybug-rs

## The Core Question

How much fingerprint bandwidth do you need? The answer depends on what you're doing:

```
Use Case                  Minimum    Recommended    Overkill
────────────────────────────────────────────────────────────
Unique ID / Hash          64 bit     128 bit       256 bit
Graph address             128 bit    256 bit       512 bit
Semantic similarity       1024 bit   4096 bit      10000 bit
Full-text search          4096 bit   10000 bit     65536 bit
Multimodal embedding      10000 bit  16384 bit     32768 bit
```

---

## 1. Mathematical Foundation

### 1.1 Orthogonal Capacity

How many **independent concepts** can a fingerprint hold?

```
Capacity ≈ √N  for N-bit random binary vectors

   N bits    Capacity    Use Case
   ───────────────────────────────
   64        ~8          IDs only
   128       ~11         Simple graph
   256       ~16         Rich graph
   512       ~23         Light semantic
   1024      ~32         Semantic search
   4096      ~64         Good semantic
   10000     ~100        Excellent semantic
   65536     ~256        Research-grade
```

### 1.2 Bundling Capacity

How many items can you **bundle together** before saturation?

```
Bundle capacity ≈ N / (2 × log₂(k))

where N = bits, k = items to bundle

   N bits    Bundle 10    Bundle 50    Bundle 100
   ─────────────────────────────────────────────────
   128       19           11           6       ← Very limited!
   256       38           23           13
   1024      154          91           51
   4096      615          364          204
   10000     1505         893          501     ← Good headroom
```

**Key insight:** If you want to bundle 50+ concepts, you need at least 1024 bits.

### 1.3 Collision Probability

What's the chance two random fingerprints collide?

```
P(collision) ≈ 1/2^(N/2)  (birthday paradox)

   N bits    P(collision)    Safe population
   ──────────────────────────────────────────
   64        1 in 2^32       ~65K items
   128       1 in 2^64       ~4 billion
   256       1 in 2^128      ~10^19 items
   10000     1 in 2^5000     Infinite for practical purposes
```

### 1.4 Hamming Distance Statistics

For random N-bit vectors:

```
Expected distance = N/2
Standard deviation = √(N/4) = √N / 2

   N bits    μ (mean)    σ (std dev)    1σ range
   ────────────────────────────────────────────────
   128       64          5.7            58-70
   256       128         8.0            120-136
   1024      512         16.0           496-528
   10000     5000        50.0           4950-5050
```

**Key insight:** Larger N gives better discrimination (larger σ relative to search space).

---

## 2. The Tiered Architecture

### Recommended: 64 + 64 + Variable

```
┌─────────────────────────────────────────────────────────────────┐
│                     FINGERPRINT STRUCTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┬──────────┬────────────────────────────────────┐   │
│  │ TIER 0   │ TIER 1   │ TIER 2 (Variable)                  │   │
│  │ 64 bits  │ 64 bits  │ 128 / 1024 / 10000 bits           │   │
│  ├──────────┼──────────┼────────────────────────────────────┤   │
│  │ Identity │ Metadata │ Semantic Content                   │   │
│  │ Hash     │ Slots    │                                    │   │
│  └──────────┴──────────┴────────────────────────────────────┘   │
│                                                                  │
│  Total: 128 bits minimum, up to 10128 bits for rich semantic    │
└─────────────────────────────────────────────────────────────────┘
```

### Tier Breakdown

```rust
/// Tiered fingerprint with configurable semantic depth
pub struct TieredFingerprint {
    /// Tier 0: Identity hash (always present)
    /// - Unique ID derived from tree address
    /// - Used for exact matching, deduplication
    /// - Collision-safe for billions of items
    identity: u64,

    /// Tier 1: Metadata slots (always present)
    /// - 8 × 8-bit slots OR 4 × 16-bit slots
    /// - Encodes: type, rung, flags, version, etc.
    /// - Fast filtering before semantic search
    metadata: u64,

    /// Tier 2: Semantic content (variable size)
    /// - None: ID-only mode (128 bits total)
    /// - Small: 128 bits (256 total) - light semantic
    /// - Medium: 1024 bits (1152 total) - standard semantic
    /// - Large: 10000 bits (10128 total) - full HDR
    semantic: SemanticTier,
}

pub enum SemanticTier {
    None,                           // 0 bits
    Micro([u64; 2]),               // 128 bits
    Small([u64; 4]),               // 256 bits
    Medium([u64; 16]),             // 1024 bits
    Large([u64; 64]),              // 4096 bits
    Full([u64; 157]),              // 10048 bits (≈10K)
}
```

---

## 3. Use Case Recommendations

### 3.1 Graph Database (RedisGraph replacement)

**Recommended: 64 + 64 + 256 = 384 bits (48 bytes)**

```
Identity (64):  Tree address hash
Metadata (64):  Node type, rung, flags, timestamp
Semantic (256): Relationship patterns, nearby concepts

Total: 48 bytes per node/edge
Memory for 1M nodes: 48 MB (excellent)
```

Why 256 for semantic?
- Graph ops mostly care about structure, not deep meaning
- Edge = From ⊕ Verb ⊕ To needs only ~50 bits with 144 verbs
- 256 gives headroom for small bundles (~16 concepts)

### 3.2 Semantic Search (ladybug-rs core)

**Recommended: 64 + 64 + 4096 = 4224 bits (528 bytes)**

```
Identity (64):   Content hash
Metadata (64):   Source, date, language, domain
Semantic (4096): Full document/sentence meaning

Total: 528 bytes per document
Memory for 1M docs: 528 MB (reasonable)
```

Why 4096?
- Captures ~64 orthogonal concepts
- Bundles up to ~300 items well
- Good balance of precision and storage
- Matches common embedding dimensions (512-1024 floats → 4096 bits)

### 3.3 Cognitive Architecture (NARS-style)

**Recommended: 64 + 64 + 10000 = 10128 bits (1266 bytes)**

```
Identity (64):   Term/concept ID
Metadata (64):   Truth value (f, c), attention, decay
Semantic (10K):  Full hyperdimensional representation

Total: 1266 bytes per belief
Memory for 1M beliefs: 1.27 GB (acceptable for cognitive system)
```

Why 10K?
- Maximum bundling capacity (~1000 items)
- 100 orthogonal concepts
- Matches neuroscience-inspired VSA research
- Headroom for compositional structures

### 3.4 Lightweight/Embedded

**Recommended: 64 + 64 = 128 bits (16 bytes)**

```
Identity (64):  Unique ID
Metadata (64):  All essential flags/types

Total: 16 bytes per item
Memory for 1M items: 16 MB (tiny!)
```

When to use:
- Mobile devices
- IoT sensors
- Simple key-value stores
- When you have external embedding storage

---

## 4. The Sparse vs Dense Trade-off

### Dense (Current 10K bit approach)

```
Storage:    1250 bytes per fingerprint
Operations: popcount(XOR) = O(N/64) = O(157)
Bundling:   Majority voting
Similarity: Hamming distance

Pros:
  ✓ Fast SIMD operations
  ✓ Simple implementation
  ✓ Proven in VSA research

Cons:
  ✗ Fixed memory regardless of content
  ✗ Wasted bits for sparse concepts
```

### Sparse (Alternative)

```
Storage:    O(k) where k = non-zero bits
Operations: Set intersection/union
Bundling:   Union + threshold
Similarity: Jaccard index

Example: 10K dimensional, 1% density
  Dense:  1250 bytes
  Sparse: ~100 indices × 2 bytes = 200 bytes (6x smaller)
```

```rust
/// Sparse fingerprint representation
pub struct SparseFingerprint {
    /// Non-zero bit indices (sorted)
    indices: Vec<u16>,  // For 10K bits, u16 suffices
    /// Total dimensionality
    dims: u16,
    /// Target density (for normalization)
    density: f32,
}

impl SparseFingerprint {
    /// Jaccard similarity
    pub fn similarity(&self, other: &Self) -> f32 {
        let intersection = self.intersect_count(other);
        let union = self.indices.len() + other.indices.len() - intersection;
        intersection as f32 / union as f32
    }

    /// Convert to dense when needed
    pub fn to_dense(&self) -> BitpackedVector {
        let mut dense = BitpackedVector::zero();
        for &idx in &self.indices {
            dense.set_bit(idx as usize, true);
        }
        dense
    }
}
```

### When to Use Sparse

| Scenario | Dense | Sparse |
|----------|-------|--------|
| Bundling many items | ✓ | ✗ |
| Memory constrained | ✗ | ✓ |
| Very high dimensions (100K+) | ✗ | ✓ |
| SIMD acceleration | ✓ | ✗ |
| Variable-density data | ✗ | ✓ |

---

## 5. Concrete Recommendations for ladybug-rs

### Option A: Simple & Fast (Recommended for v1)

```rust
/// 256-bit fingerprint for graph + light semantic
pub struct LadybugFingerprint {
    words: [u64; 4],  // 256 bits = 32 bytes
}

// Breakdown:
//   words[0]: Identity (tree addr hash)
//   words[1]: Metadata (type, rung, flags, timestamp)
//   words[2]: Semantic low (bound attributes)
//   words[3]: Semantic high (relationship patterns)
```

**Storage:** 32 bytes per node
**Memory for 10M nodes:** 320 MB
**Bundling capacity:** ~23 concepts
**Orthogonal capacity:** ~16 concepts

### Option B: Balanced (Recommended for v2)

```rust
/// 1024-bit fingerprint for good semantic search
pub struct LadybugFingerprint {
    identity: u64,        // 64 bits
    metadata: u64,        // 64 bits
    semantic: [u64; 14],  // 896 bits (rounds to 1024 total)
}
```

**Storage:** 128 bytes per node
**Memory for 10M nodes:** 1.28 GB
**Bundling capacity:** ~150 concepts
**Orthogonal capacity:** ~32 concepts

### Option C: Full Power (For research/cognitive)

```rust
/// Full 10K-bit fingerprint
pub struct LadybugFingerprint {
    identity: u64,         // 64 bits
    metadata: u64,         // 64 bits
    semantic: [u64; 155],  // 9920 bits (≈10K total)
}
```

**Storage:** 1256 bytes per node
**Memory for 10M nodes:** 12.56 GB
**Bundling capacity:** ~1500 concepts
**Orthogonal capacity:** ~100 concepts

---

## 6. Metadata Slot Encoding

### 64-bit Metadata Layout

```
┌────────────────────────────────────────────────────────────────┐
│                    METADATA (64 bits)                          │
├────────┬────────┬────────┬────────┬────────┬────────┬─────────┤
│ Byte 0 │ Byte 1 │ Byte 2 │ Byte 3 │ Byte 4 │ Byte 5 │ Byte 6-7│
├────────┼────────┼────────┼────────┼────────┼────────┼─────────┤
│ Type   │ Rung   │ Flags  │ Ver    │ Src    │ Trust  │Timestamp│
│ 0-255  │ 0-255  │ 8 bits │ 0-255  │ 0-255  │ 0-255  │ 16 bits │
└────────┴────────┴────────┴────────┴────────┴────────┴─────────┘

Type:     Node type (concept, entity, event, etc.)
Rung:     Abstraction level (0=concrete, 255=most abstract)
Flags:    8 boolean flags (active, verified, locked, etc.)
Version:  Schema version for forward compatibility
Source:   Data source ID (0-255 sources)
Trust:    Trust/confidence level (0-255 → 0.0-1.0)
Timestamp: Compressed timestamp (days since epoch mod 65536)
```

### Alternative: 4 × 16-bit Slots

```
┌────────────────────────────────────────────────────────────────┐
│                    METADATA (64 bits)                          │
├─────────────────┬─────────────────┬─────────────────┬──────────┤
│ Slot 0 (16 bit) │ Slot 1 (16 bit) │ Slot 2 (16 bit) │ Slot 3   │
├─────────────────┼─────────────────┼─────────────────┼──────────┤
│ Type + Rung     │ Trust + Flags   │ Source + Ver    │ Timestamp│
│ 8+8 bits        │ 8+8 bits        │ 8+8 bits        │ 16 bits  │
└─────────────────┴─────────────────┴─────────────────┴──────────┘
```

---

## 7. Transformer Embedding → Fingerprint

### FP32 1024D → Fingerprint Conversion

```rust
/// Convert 1024D float embedding to fingerprint
pub fn embedding_to_fingerprint(embedding: &[f32; 1024]) -> [u64; 16] {
    let mut fp = [0u64; 16];

    // Method 1: Sign thresholding (fastest)
    // Each float → 1 bit based on sign
    for (i, &val) in embedding.iter().enumerate() {
        if val > 0.0 {
            fp[i / 64] |= 1 << (i % 64);
        }
    }

    // Method 2: Multi-threshold (richer, needs more bits)
    // Each float → 4 bits based on quantile
    // Requires 4096 bits for 1024D

    // Method 3: Random projection (for larger output)
    // Project 1024D → 10000 bits via random matrix

    fp
}
```

### Conversion Matrix

| Embedding | Method | Fingerprint Size |
|-----------|--------|------------------|
| 384D (MiniLM) | Sign | 384 bits |
| 768D (BERT) | Sign | 768 bits |
| 1024D (Jina) | Sign | 1024 bits |
| 1024D (Jina) | 4-bit quantile | 4096 bits |
| 1024D (Jina) | Random projection | 10000 bits |

---

## 8. Final Recommendation

### For ladybug-rs specifically:

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   DEFAULT: 64 + 64 + 1024 = 1152 bits (144 bytes)          │
│                                                              │
│   Identity:  64 bits  - Tree address / content hash         │
│   Metadata:  64 bits  - Type, rung, trust, timestamp        │
│   Semantic: 1024 bits - From transformer or VSA ops         │
│                                                              │
│   This gives you:                                            │
│     ✓ ~32 orthogonal concepts                               │
│     ✓ ~150 items bundling capacity                          │
│     ✓ Collision-safe for billions                           │
│     ✓ 144 bytes = reasonable storage                        │
│     ✓ Direct 1:1 from 1024D embeddings                      │
│                                                              │
│   UPGRADE PATH:                                              │
│     v1 → v2: Just extend semantic to 4096 bits              │
│     v2 → v3: Extend to 10000 bits if needed                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Code Template

```rust
/// ladybug-rs recommended fingerprint
#[repr(C, align(64))]
pub struct Fingerprint {
    /// Tier 0: Identity (64 bits)
    pub identity: u64,

    /// Tier 1: Metadata (64 bits)
    pub metadata: u64,

    /// Tier 2: Semantic (1024 bits = 16 × u64)
    pub semantic: [u64; 16],
}

impl Fingerprint {
    pub const BITS: usize = 64 + 64 + 1024;  // 1152
    pub const BYTES: usize = 144;

    /// From transformer embedding (1024D → 1024 bits)
    pub fn from_embedding(embedding: &[f32; 1024]) -> Self {
        let mut semantic = [0u64; 16];
        for (i, &val) in embedding.iter().enumerate() {
            if val > 0.0 {
                semantic[i / 64] |= 1 << (i % 64);
            }
        }
        Self {
            identity: 0,  // Set separately
            metadata: 0,  // Set separately
            semantic,
        }
    }

    /// Hamming distance (semantic only)
    pub fn semantic_distance(&self, other: &Self) -> u32 {
        self.semantic.iter()
            .zip(other.semantic.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Full distance (identity + metadata + semantic weighted)
    pub fn weighted_distance(&self, other: &Self, semantic_weight: f32) -> f32 {
        let id_match = (self.identity == other.identity) as u32 as f32;
        let meta_dist = (self.metadata ^ other.metadata).count_ones() as f32 / 64.0;
        let sem_dist = self.semantic_distance(other) as f32 / 1024.0;

        (1.0 - id_match) * 0.1 + meta_dist * 0.2 + sem_dist * semantic_weight
    }
}
```

---

## Summary: How Much Bandwidth?

| Question | Answer |
|----------|--------|
| **Just need IDs?** | 64-128 bits |
| **Graph structure?** | 256-512 bits |
| **Semantic search?** | 1024-4096 bits |
| **Cognitive/bundling?** | 10000+ bits |
| **Memory tight?** | Use sparse representation |
| **Transformer embedding?** | Match embedding dim (1024→1024) |

**The golden rule:** Start with 64+64+1024 (144 bytes) and only increase if you hit capacity limits.
