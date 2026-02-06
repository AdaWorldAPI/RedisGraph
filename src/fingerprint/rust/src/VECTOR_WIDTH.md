# Vector Width: 10Kbit vs 16Kbit (2^14) Architecture Comparison

## Status Quo

The current codebase uses **10,000-bit** vectors packed into 157 × u64 words.
This document compares the two architectures and explains the rationale for
maintaining both variants in `10k/` and `16k/` subfolders.

---

## Side-by-Side Comparison

| Property | 10K (current) | 16K (2^14 = 16,384) |
|---|---|---|
| **Bits** | 10,000 | 16,384 |
| **u64 words** | 157 | 256 |
| **Bytes** | 1,256 | 2,048 |
| **Padded bytes** | 1,280 (20 cache lines) | 2,048 (32 cache lines) |
| **Padding waste** | 24 bytes (48 bits) | **0** (exact power-of-2) |
| **SIMD alignment** | 157 words → remainder handling | 256 = 4×64 → perfect AVX-512 |
| **AVX-512 loads** | 19 full + 1 partial (5 words) | **32 full, zero remainder** |
| **AVX2 loads** | 39 full + 1 partial | **64 full, zero remainder** |
| **σ (std dev)** | √(10000/4) = 50 | √(16384/4) = **64** (one word!) |
| **Expected random d** | 5,000 | 8,192 |
| **1σ** | 50 | 64 |
| **2σ** | 100 | 128 |
| **3σ** | 150 | 192 |
| **Neural blocks** | 10 (9×16 + 1×13) | **16** (16×16, uniform) |
| **Bits per block** | 1024 (last: 832) | **1024** (all equal) |
| **Crystal dims** | 5D → 2 blocks each | 5D → 3.2 blocks each (or 8D×2) |
| **Arrow column** | FixedSizeBinary(1280) | FixedSizeBinary(2048) |
| **Memory per 1M vecs** | 1.19 GiB | 1.91 GiB (+60%) |

---

## Why 16K Is Architecturally Superior

### 1. Perfect Power-of-2 Alignment
- 256 words = 2^8 → all SIMD widths divide evenly
- Zero remainder loops in all distance computations
- No `LAST_WORD_MASK` needed — all 64 bits of word[255] are used
- `from_words([u64; 256])` — compiler can optimize perfectly

### 2. σ = 64 = Exactly One Word
- This is the "magic" number: one standard deviation of Hamming distance
  equals exactly one u64 word of popcount
- Block-level sigma calculations become exact integer arithmetic
- Stacked popcount crossing 1σ boundary = crossing exactly one word boundary
- Epiphany zone thresholds: 64, 128, 192 — all powers of 2 × 64

### 3. 16 Uniform Blocks
- 256 words ÷ 16 words/block = **16 blocks of 1024 bits each**
- No short last block (10K has block[9] = 13 words / 832 bits)
- Block sums are directly comparable (all same denominator)
- Maps perfectly to 16-element SIMD registers
- 4 blocks per AVX-512 register for block-level filtering

### 4. Zero-Waste SIMD
```
AVX-512: 256 words ÷ 8 words/reg = 32 iterations, 0 remainder
AVX-2:   256 words ÷ 4 words/reg = 64 iterations, 0 remainder
NEON:    256 words ÷ 2 words/reg = 128 iterations, 0 remainder
```
Compared to 10K where every SIMD loop needs a remainder epilogue.

### 5. Crystal ↔ Block Mapping
With 16 blocks, the crystal lattice can be mapped more granularly:
- **5D × 3 blocks** = 15 blocks (leave 1 for global metadata)
- **8D × 2 blocks** = 16 blocks (enable 8-dimensional crystal: 2^8 = 256 cells if binary, or 5^8 = 390,625 if pentary)
- **4D × 4 blocks** = 16 blocks (coarser but deeper crystal)

---

## ANI / NARS / RL Schema Markers

The extra 6,384 bits (16,384 − 10,000) over 10K can carry structured
schema metadata when an application needs it. Rather than "wasting" the
extra space, each 16K fingerprint can optionally embed markers for:

### Proposed Schema Sidecar (optional, metadata-in-vector)

With 16 uniform blocks of 1024 bits, blocks 0..12 carry the semantic
fingerprint (13,312 bits ≈ 10K equivalent information) and blocks 13..15
carry structured markers:

```text
┌────────────────────────────────────────────────────────────────┐
│ Blocks 0..12: Semantic fingerprint (13,312 bits)               │
├────────────────────────────────────────────────────────────────┤
│ Block 13: Node / Edge Type Markers (1024 bits)                 │
│   ├─ bits 0..127:   ANI reasoning level (8 levels × 16 bits)  │
│   ├─ bits 128..255: NARS truth value {f, c} quantized          │
│   ├─ bits 256..383: NARS budget {p, d, q} quantized            │
│   ├─ bits 384..511: Edge type (144 cognitive verbs → 8-bit ID  │
│   │                 + 120 bits of XOR-bound context)            │
│   ├─ bits 512..639: Node kind (entity/concept/event/rule/goal) │
│   │                 + provenance hash                           │
│   └─ bits 640..1023: Reserved for user-defined schema          │
├────────────────────────────────────────────────────────────────┤
│ Block 14: RL / Temporal State (1024 bits)                      │
│   ├─ bits 0..127:   Q-value quantized (16 actions × 8 bits)   │
│   ├─ bits 128..255: Reward history (8 steps × 16-bit reward)  │
│   ├─ bits 256..383: Visit count (log-scaled, 16-bit)           │
│   ├─ bits 384..511: Causal binding hash (state⊕action)         │
│   ├─ bits 512..639: STDP timing markers (8 × 16-bit stamps)   │
│   ├─ bits 640..767: Hebbian weight vector (8 neighbors × 16b) │
│   └─ bits 768..1023: Temporal attention / priority flags       │
├────────────────────────────────────────────────────────────────┤
│ Block 15: Traversal / Graph Cache (1024 bits)                  │
│   ├─ bits 0..255:   DN address (compressed TreeAddr, 32 bytes)│
│   ├─ bits 256..511: Parent centroid fingerprint (XOR-folded    │
│   │                 from 16K to 256 bits via block XOR)         │
│   ├─ bits 512..767: Neighbor bloom filter (256 bits → ~7       │
│   │                 neighbors with 1% FPR)                     │
│   └─ bits 768..1023: PageRank / centrality score (16-bit) +   │
│                       hop distance to root (8-bit) +            │
│                       cluster ID (16-bit) + reserved            │
└────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

1. **Blocks 0..12 remain pure HDR**: XOR binding, majority bundling,
   and Hamming distance operate on the full 13K semantic region. The
   schema blocks participate in distance only when explicitly included
   (configurable via block mask).

2. **Schema blocks are XOR-bindable too**: When you XOR-bind two
   edges, the schema markers also combine — NARS truth values get
   XORed (which is meaningful: XOR of quantized {f,c} represents the
   symmetric difference of belief), and RL state inherits from both
   participants.

3. **Backward compatible**: A 10K fingerprint can be zero-extended to
   16K (blocks 10..15 = 0) with no semantic change. A 16K fingerprint
   can be truncated to 13K by dropping the schema blocks.

4. **All-information mode**: If schema markers are not needed, all 16
   blocks carry semantic information — giving 63.8% more capacity
   than 10K for higher-fidelity fingerprints.

### ANI Reasoning Levels (Block 13, bits 0..127)

Encoded as 8 × 16-bit slots for the AGI capability spectrum:

| Slot | Level | Description |
|------|-------|-------------|
| 0 | Reactive | Stimulus→response, no internal state |
| 1 | Memory | Pattern recognition from stored examples |
| 2 | Analogy | Transfer learning across domains |
| 3 | Planning | Multi-step goal decomposition |
| 4 | Meta | Reasoning about own reasoning |
| 5 | Social | Theory of mind, intent modeling |
| 6 | Creative | Novel combination of existing concepts |
| 7 | Abstract | Mathematical / logical abstraction |

Each 16-bit value represents activation level (0..65535) at that
reasoning tier. XOR-binding two vectors at the same tier creates the
compositional binding of their reasoning capabilities.

### NARS Truth Values (Block 13, bits 128..383)

Non-Axiomatic Reasoning System (NARS) truth values are embedded for
evidential reasoning:

- **Frequency (f)**: 16-bit quantized [0,1] — proportion of positive evidence
- **Confidence (c)**: 16-bit quantized [0,1) — ratio of evidence to total
- **Priority (p)**: 16-bit — urgency of processing
- **Durability (d)**: 16-bit — resistance to forgetting
- **Quality (q)**: 16-bit — usefulness of the item

These enable the graph to perform NARS-style inference directly on
fingerprints: revision (bundle with confidence-weighted voting),
deduction (XOR-bind with truth value combination), and abduction
(unbind + truth value relaxation).

### RL Temporal State (Block 14)

Each node/edge carries its own RL state, eliminating the need for
external Q-tables and reward trackers:

- **Q-values inline**: 16 actions × 8-bit Q ≈ same as PolicyGradient
  but embedded in the fingerprint itself
- **Reward history**: Last 8 rewards for temporal-difference updates
  without external storage
- **Causal hash**: Compressed state⊕action binding for O(1) lookup
- **STDP timestamps**: For plasticity engine spike-timing without
  external HashMap

This is significant because it makes the fingerprint **self-contained
for learning** — the RL state travels with the data, enabling
distributed learning without shared mutable state.

---

## Trade-offs

| Concern | 10K | 16K |
|---|---|---|
| Memory per vector | 1.25 KB | 2.0 KB (+60%) |
| 1M vectors in RAM | 1.19 GiB | 1.91 GiB |
| Cache efficiency | Good (20 lines) | Acceptable (32 lines) |
| L1 cache (32KB) | 25 vectors | 16 vectors |
| L2 cache (256KB) | 200 vectors | 128 vectors |
| Hamming compute | ~157 popcnt | ~256 popcnt (+63%) |
| SIMD efficiency | ~95% (remainder) | **100%** (no remainder) |
| Information capacity | 10,000 bits | 16,384 bits (+63.8%) |
| Schema capacity | None (all semantic) | 3,072 bits for markers |

---

## Recommendation

**Use 16K (2^14) for new deployments**, especially on AVX-512 hardware
(Claude backend, Railway). The perfect alignment, zero-waste SIMD, and
embedded schema markers more than compensate for the 60% memory increase.

**Keep 10K as a lightweight alternative** for constrained environments
(embedded, WASM, mobile) where memory pressure is the primary concern.

Both variants share the same algorithmic structure (XOR binding, majority
bundling, stacked popcount, crystal lattice, DN tree). Only the constants
and array sizes differ.

---

## File Organization

```
src/
├── width_10k/
│   └── mod.rs          # VECTOR_BITS=10000, VECTOR_WORDS=157, σ=50
├── width_16k/
│   ├── mod.rs          # VECTOR_BITS=16384, VECTOR_WORDS=256, σ=64
│   │                    # + schema block offsets
│   └── schema.rs       # ANI levels, NARS truth/budget, RL inline state,
│                        # neighbor bloom filter, graph metrics, SchemaSidecar
├── bitpack.rs          # Uses width_10k constants (current default)
├── hamming.rs          # Generic distance computation
├── epiphany.rs         # Zone thresholds from σ constants
└── ...                 # All other modules unchanged
```
