# The Holy Grail: DN-Sparse Graph Architecture

## The Question That Drove This

> Where do you store `domain:tree:branch:twig:leaf` so that vertical
> traversal (`domain:tree:branch:twig` minus `leaf`) is a hash-table
> operation like Active Directory's Distinguished Name lookup,
> and you NEVER scan nodes?

The answer is: **the DN address IS the key into every data structure**.

Not a secondary index. Not a label. Not an integer that maps to a DN.
The DN itself, packed into a u64, is the primary identity of every node
in the graph. Every lookup, every traversal, every edge check goes through
that u64 directly.

---

## The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: HDR SEMANTICS                                          │
│                                                                 │
│ Fingerprints (Arc<BitpackedVector>) cached per node             │
│ Edge fp = src XOR verb XOR dst (computed on demand, ~5ns)       │
│ Resonance unbinding, similarity search, superposition           │
│ Hamming cascade with early exit for nearest-neighbor            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ uses
┌─────────────────────────────────▼───────────────────────────────┐
│ Layer 2: SPARSE TOPOLOGY (DeltaDnMatrix)                        │
│                                                                 │
│ main: DnCsr (sorted PackedDn rows, binary search)               │
│ delta_plus: HashMap<PackedDn, Vec<(PackedDn, EdgeDescriptor)>>  │
│ delta_minus: HashSet<(PackedDn, PackedDn)>                      │
│                                                                 │
│ Edge = 8 bytes (verb:u16 + weight:u16 + offset:u32)             │
│ NOT 1,256 bytes. 157x smaller than the old Rust port.           │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ indexed by
┌─────────────────────────────────▼───────────────────────────────┐
│ Layer 1: DN NODE STORE (HashMap<PackedDn, NodeSlot>)            │
│                                                                 │
│ PackedDn: u64, hierarchically sorted, 7 levels x 255 values    │
│ children: HashMap<PackedDn, Vec<PackedDn>>  ← O(1) children    │
│ walk_to_root: chain of parent() bit ops + hash lookups          │
│                                                                 │
│ This is Active Directory's DN index in 8 bytes.                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## PackedDn: The u64 That Changes Everything

### Encoding

```
Byte:  7      6      5      4      3      2      1      0
     [lv0+1][lv1+1][lv2+1][lv3+1][lv4+1][lv5+1][lv6+1][ 00 ]

/animals/mammals/cat  →  components [0x01, 0x0F, 0x03]
                      →  packed: 0x02_10_04_00_00_00_00_00

Why +1?  So that 0x00 means "no component at this level".
         This lets us distinguish /a (depth 1) from /a/0 (depth 2).
```

### Why This Sort Order Is Magic

Because components are MSB-first and unused levels are 0x00:

```
/0         = 0x01_00_00_00_00_00_00_00
/0/0       = 0x01_01_00_00_00_00_00_00
/0/0/0     = 0x01_01_01_00_00_00_00_00
/0/1       = 0x01_02_00_00_00_00_00_00
/1         = 0x02_00_00_00_00_00_00_00

Sort:  /0 < /0/0 < /0/0/0 < /0/1 < /1
       ▲ parent before children, siblings together
```

This means in any sorted array of PackedDns:
- **All nodes in a subtree are contiguous**
- **Binary search finds any subtree in O(log n)**
- **No scanning, ever**

### The Operations That Fall Out For Free

| Operation | How | Cost |
|-----------|-----|------|
| Parent of `/a/b/c` | Zero out byte at position `depth-1` | O(1) bit mask |
| Child `/a/b` → `/a/b/5` | OR in `0x06` at position `depth` | O(1) bit OR |
| Sibling | parent() then child(new) | O(1) two bit ops |
| All ancestors | Chain of parent() calls | O(depth) ≤ 7 |
| Subtree range | `[self|0x01<<shift, self|0xFF...FF]` | O(1) bit ops |
| Is ancestor? | Compare masked prefix | O(1) bit AND + compare |
| Tree distance | `depth(a) + depth(b) - 2*common_depth` | O(1) |

---

## Where Everything Is Stored

### The Node Store Problem

**Neo4j**: Nodes are in a linked file. Finding "children of X" means following
relationship chains. O(degree).

**RedisGraph**: Nodes are integer IDs in a DataBlock. Finding children means
iterating the adjacency matrix row. O(nnz in row).

**This architecture**: Nodes are in a HashMap keyed by PackedDn. Children are
in a separate HashMap `parent → Vec<children>`, maintained on insert/remove.

```rust
// Finding children of /animals/mammals:
let parent = PackedDn::new(&[ANIMALS, MAMMALS]);
let children: &[PackedDn] = store.children_of(parent);  // O(1)
// Returns: [/animals/mammals/cat, /animals/mammals/dog, ...]
```

No scanning. No matrix multiplication. One hash lookup.

### The Edge Store Problem

**Old Rust port**: Each edge is a `HdrScalar::Vector(BitpackedVector)` = 1,256 bytes
inside the sparse matrix. 1M edges = 1.2 GB.

**Original RedisGraph**: Each edge is a boolean or u64 in a GraphBLAS matrix.
1M edges = 8 MB.

**This architecture**: Each edge is an `EdgeDescriptor` = 8 bytes.
1M edges = 8 MB. Same as RedisGraph. But we also get:

```rust
EdgeDescriptor (u64):
  bits 63-48: verb_id (u16)     — which cognitive verb
  bits 47-32: weight (u16)      — 0.0-1.0 as fixed-point
  bits 31-0:  offset (u32)      — into Arrow property batch
```

If you need the semantic fingerprint of an edge, compute it on demand:
```rust
let fp = src_fp.xor(&verb.to_fingerprint()).xor(&dst_fp);
// 3 XORs over 157 u64 words = ~5ns. Cheaper than a cache miss.
```

### The Adjacency Store Problem

**RedisGraph's delta matrix** (the good part we were missing):

```
main (CSR)         +  delta_plus (HashMap)  -  delta_minus (HashSet)
─────────────         ─────────────────        ──────────────────
sorted, immutable     fast insert O(1)         fast delete O(1)
binary search O(lg n) unsorted, small          (src,dst) pairs

Read  = check delta_minus → check delta_plus → check main
Write = insert into delta_plus or delta_minus (never touch main)
Flush = rebuild CSR from main ± deltas (background, batch)
```

This gives:
- **Snapshot isolation**: readers see consistent main + deltas
- **Non-blocking writes**: writers only touch hash maps
- **Efficient flush**: rebuild CSR once, not per-mutation

### The DN-Ordered CSR

The key innovation over RedisGraph's integer-indexed CSR:

```
RedisGraph CSR:
  row_ptrs indexed by integer node ID (0, 1, 2, 3, ...)
  subtree query = mxm (matrix multiply) = O(nnz)

DN-Ordered CSR:
  row_ptrs indexed by sorted PackedDn
  subtree query = binary search for [lo, hi] range = O(log n + edges)

  row_dns: [ /0,  /0/1,  /0/2,  /1,  /1/0,  /1/0/3 ]
  row_ptrs: [ 0,    2,     4,    5,    7,     8,    9 ]
  col_dns:  [/0/1, /0/2, /1, /0, /0/1, /1/0, /1/0/3, /0, /1]
  edges:    [ e0,   e1,  e2,  e3,  e4,   e5,    e6,  e7,  e8]

  "All edges from subtree /0":
    subtree_range(/0) = [/0/0/0..., /0/FF/FF...]
    binary_search(row_dns, lo) → position 1 (/0/1)
    binary_search(row_dns, hi) → position 2 (/0/2)
    Also include /0 itself → positions 0, 1, 2
    Result: edges e0, e1, e2, e3 (4 edges from 3 rows)
    Cost: O(log 6) + O(4) = O(log n + edges_in_subtree)
```

---

## The Vertical Traversal (The Actual Answer)

The user's core question: how to traverse `domain:tree:branch:twig` (minus leaf)
without scanning.

```rust
let leaf = PackedDn::new(&[domain, tree, branch, twig, leaf]);

// "domain:tree:branch:twig minus leaf" = walk ancestors
for (ancestor_dn, ancestor_data) in graph.walk_to_root(leaf) {
    // Each step: parent() = bit mask on u64, then HashMap::get()
    // Total: 4 hash lookups for depth-5 node
    // Zero scanning. Zero matrix operations.
    println!("{}: {}", ancestor_dn, ancestor_data.label);
}

// Output:
//   /domain/tree/branch/twig: "Twig Node"
//   /domain/tree/branch: "Branch Node"
//   /domain/tree: "Tree Node"
//   /domain: "Domain Node"
```

**How parent() works (bit-level)**:
```
leaf    = 0x_02_0A_05_03_01_00_00_00   depth=5
                              ▲
                              zero this byte
twig    = 0x_02_0A_05_03_00_00_00_00   depth=4
                        ▲
                        zero this byte
branch  = 0x_02_0A_05_00_00_00_00_00   depth=3
...and so on.
```

Each step is one AND instruction. The result is a HashMap key. O(1) per level.

---

## Graduated Hamming Similarity

The old approach (XOR-bind all levels) gives ~50% Hamming distance for ANY
difference, whether sibling or unrelated. Not useful for tree-based similarity.

The new approach: **bit-range partitioning**.

```
10,000 bits divided into 7 zones (one per tree level):

Bits:  [0──1428] [1429──2856] [2857──4284] ... [8572──9999]
Level:     0          1            2                6

Each zone's bits are set by: random(seed = component_at_this_level)

Siblings share 6 of 7 zones → differ in ~1428/10000 = 14.3% of bits
  → Hamming distance ≈ 714

Cousins share 5 of 7 zones → differ in ~2856/10000 = 28.6%
  → Hamming distance ≈ 1428

Depth-3 relatives share 4/7 → ≈ 2142
Unrelated share 0/7 → ≈ 5000 (random chance)
```

This gives a clean gradient:
```
distance 0:    exact match (same node)
distance ~714: siblings (same parent)
distance ~1428: cousins (same grandparent)
distance ~2142: 2nd cousins
distance ~5000: unrelated (random)
```

The stacked popcount with early exit from `hamming.rs` still applies.
In the first 7-word sample, we can estimate which zone differs and
reject 90% of candidates before touching the rest.

---

## Superposition (Multiple Classifications)

A whale is both a mammal and a marine animal:

```rust
let whale_mammal = PackedDn::new(&[ANIMALS, MAMMALS, WHALE]);
let whale_marine = PackedDn::new(&[ANIMALS, MARINE, WHALE]);

// Primary DN is the "canonical" address
let mut whale = NodeSlot::new(whale_mammal, "Whale");

// Add alias: the fingerprint becomes BUNDLE(mammal_fp, marine_fp)
whale.add_alias(whale_mammal, whale_marine);

// Now: whale.fingerprint resonates with BOTH classifications
// "Is this a marine animal?" → Hamming(whale_fp, marine_subtree_fp) < threshold → YES
// "Is this a mammal?" → Hamming(whale_fp, mammal_subtree_fp) < threshold → YES
```

The bundle (majority vote) preserves bits shared by both paths and
randomizes bits where they differ. With 2 paths, ~75% of bits are
preserved from each path (compared to ~50% for unrelated vectors).

---

## Comparison Table

| Feature | Neo4j | RedisGraph | Kuzu | **DN-Sparse** |
|---------|-------|------------|------|---------------|
| Node lookup | O(1) by ID | O(1) by ID | O(1) by ID | **O(1) by DN** |
| Children | O(degree) | O(nnz) | O(degree) | **O(1) hash** |
| Subtree query | BFS O(V+E) | mxm O(nnz) | BFS O(V+E) | **O(log n + k)** |
| Vertical walk | O(depth) ptrs | O(depth) lookups | O(depth) ptrs | **O(depth) bit ops** |
| Edge memory | 34 bytes | 8 bytes | 16 bytes | **8 bytes** |
| Semantic sim | N/A | N/A | N/A | **O(1) XOR+popcount** |
| Transactions | MVCC (heavy) | delta matrix | MVCC | **delta matrix** |
| Delete edge | O(degree) | O(1) delta | O(1) | **O(1) delta** |
| Storage format | native | native | native | **Arrow (zero-copy)** |
| Fingerprint cost | N/A | 1,256 bytes/edge | N/A | **0 bytes/edge** (computed) |
| Subtree bundle | N/A | N/A | N/A | **majority vote** |

---

## The Spirit of GraphBLAS, Preserved

GraphBLAS taught us: graph algorithms are linear algebra on sparse matrices
over user-defined semirings.

We keep that:
- **CSR format**: same as GraphBLAS internally uses for `GrB_Matrix`
- **Sorted row keys**: same as GraphBLAS's column indices within a row
- **Semiring compatibility**: `EdgeDescriptor` can be the "scalar" in a semiring
  where multiply = compose verbs, add = choose best edge
- **Matrix-vector multiply**: BFS is still "frontier = adj * frontier"
  but the adj is indexed by PackedDn, not integer

What we add:
- **DN keys instead of integers**: hierarchical sort gives free subtree operations
- **Delta isolation**: reads don't block writes, writes don't block reads
- **HDR layer**: semantic fingerprints computed on demand, not stored in the matrix
- **Graduated similarity**: tree proximity maps to Hamming proximity

---

## File: `src/fingerprint/rust/src/dn_sparse.rs`

The implementation contains:

| Component | Lines | What It Does |
|-----------|-------|-------------|
| `PackedDn` | ~200 | u64 DN encoding with bit-level parent/child/sibling |
| `hierarchical_fingerprint()` | ~40 | Bit-zone partitioned fingerprints for graduated similarity |
| `xor_bind_fingerprint()` | ~12 | Classic XOR-bind for resonance operations |
| `EdgeDescriptor` | ~50 | 8-byte packed edge (verb + weight + offset) |
| `NodeSlot` | ~40 | Node data with Arc'd fingerprints and superposition aliases |
| `DnNodeStore` | ~120 | O(1) node lookup + O(1) children + O(depth) vertical walk |
| `DnCsr` | ~130 | DN-ordered CSR with binary-search subtree queries |
| `DeltaDnMatrix` | ~100 | main + delta_plus + delta_minus transactional isolation |
| `DnGraph` | ~200 | Unified graph: nodes + forward + reverse + typed adjacency |
| Tests | ~200 | 12 tests covering every claim |

Every O(1) claim is backed by a HashMap or bit operation.
Every "no scanning" claim uses either hash lookup or binary search.
Every test is concrete, not aspirational.
