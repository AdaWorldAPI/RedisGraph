# The CAM Prefix Fits at 256 Words — No Separate Address Space Needed

> The 4096 CAM (Content-Addressable Methods) operation dictionary is
> ladybug-rs's most innovative idea and its biggest architectural headache.
> At 256 words, the headache disappears: CAM operations map directly to
> schema metadata blocks.

---

## The Problem

The CAM dictionary defines 4096 operations across 16 categories:

```
0x000-0x0FF: LanceDB Core
0x100-0x1FF: SQL
0x200-0x2FF: Cypher
0x300-0x3FF: Hamming/VSA
0x400-0x4FF: NARS
...
0xF00-0xFFF: User-Defined
```

Each operation has a 12-bit ID (0x000-0xFFF), metadata, and a fingerprint.
The original design tried to fit this 4096-entry dictionary into the
"surplus" bits between 10,000 and 16,384. This didn't work because:

1. 4096 entries × any reasonable per-entry size > 6,384 surplus bits
2. The operation fingerprint itself is 10K bits — you can't store an
   operation definition inside the surplus of another fingerprint
3. The CAM prefix is an *addressing* concept, not a *data* concept

## The Reframe

**The CAM dictionary is not data to store in a fingerprint.** It's a routing
table that maps operation IDs to execution handlers. It belongs in the
Surface zone of BindSpace (prefixes 0x00-0x0F, 4,096 addresses).

But the CAM operation's *parameters* — what to filter by, what schema
predicates to check — these DO map to the schema metadata blocks in a 256-word
fingerprint.

---

## The Solution: CAM → Schema Query Translation

Each CAM operation, when executed, translates to a `SchemaQuery` that
leverages the metadata packed in blocks 13-15 of 16K fingerprints.

### Category 0x300: Hamming/VSA Operations

```
CAM 0x300: HAMMING.DISTANCE → full 256-word distance
CAM 0x301: HAMMING.SEMANTIC → blocks 0-12 only (semantic_distance)
CAM 0x302: HAMMING.BIND     → XOR bind two 256-word fingerprints
CAM 0x303: HAMMING.UNBIND   → XOR unbind (same as bind, self-inverse)
CAM 0x304: HAMMING.BUNDLE   → majority vote across N fingerprints
CAM 0x305: HAMMING.MEXICANHAT → excite/inhibit with sigma=64 thresholds
CAM 0x306: HAMMING.POPCOUNT → popcount of words 0-207 (semantic)
CAM 0x307: HAMMING.BLOOM    → check neighbor bloom at words 244-247
```

Implementation: each of these is a direct function call on the 256-word
array. No lookup table. No hash. Just: read the CAM opcode, call the
corresponding function on the word array.

### Category 0x400: NARS Operations

```
CAM 0x400: NARS.DEDUCTION   → read truth from word 210, compute f=f1*f2, c=f*c1*c2
CAM 0x401: NARS.INDUCTION   → read truth, compute abduced truth
CAM 0x402: NARS.ABDUCTION   → read truth, compute inducted truth
CAM 0x403: NARS.REVISION    → combine evidence: weighted average by confidence
```

Implementation: read NarsTruth from word 210 of both inputs, apply
NARS formula, write result to word 210 of output. All integer arithmetic
on the u64 word — no float, no separate column.

### Category 0x500: Search with Schema Predicates

```
CAM 0x500: SEARCH.ANN        → semantic_distance top-k, no predicates
CAM 0x501: SEARCH.SCHEMA     → semantic_distance + passes_predicates()
CAM 0x502: SEARCH.BLOOM      → bloom_accelerated_search() (neighbor bonus)
CAM 0x503: SEARCH.RL         → rl_guided_search() (Q-value composite)
CAM 0x504: SEARCH.NARS_TRUST → filter by NARS confidence > threshold
CAM 0x505: SEARCH.ANI_LEVEL  → filter by ANI reasoning level ≥ min
CAM 0x506: SEARCH.CLUSTER    → filter by cluster_id in graph metrics
```

Implementation: each search variant constructs a `SchemaQuery` and calls
the same underlying search function with different predicates. The schema
metadata in blocks 13-15 is the predicate target.

### Category 0x900: RL Operations

```
CAM 0x900: RL.BEST_ACTION   → read Q-values from words 224-225, return argmax
CAM 0x901: RL.UPDATE_Q      → write new Q-value for action at index
CAM 0x902: RL.PUSH_REWARD   → push reward to history at words 226-227
CAM 0x903: RL.TREND         → compute reward trend from words 226-227
CAM 0x904: RL.ROUTING_SCORE → composite: alpha*distance + (1-alpha)*q_cost
```

Implementation: direct word read/write at the RL block (words 224-231).
No external Q-table needed — the policy travels with the fingerprint.

---

## The CAM Surface Zone as Router

The 4096 Surface addresses (prefixes 0x00-0x0F) become the CAM dispatch
table. Each address contains a fingerprint that encodes the operation's
*signature* — what arguments it takes and what it returns:

```rust
// Surface address 0x00:0x03 = CAM 0x003 (HAMMING.UNBIND)
// The fingerprint at this address encodes:
// - Block 13: ANI level = which cognitive tier this op serves
// - Block 14: RL Q-value = usage frequency / success rate
// - Block 15: Graph metrics = how many nodes this op has touched

/// Execute a CAM operation
pub fn cam_execute(
    op_id: u16,
    bind_space: &BindSpace16K,
    args: &[Addr],
) -> CamResult {
    let category = (op_id >> 8) as u8;
    let operation = (op_id & 0xFF) as u8;

    match category {
        0x03 => hamming_dispatch(operation, bind_space, args),
        0x04 => nars_dispatch(operation, bind_space, args),
        0x05 => search_dispatch(operation, bind_space, args),
        0x09 => rl_dispatch(operation, bind_space, args),
        _ => CamResult::Error(format!("Unknown category: 0x{:02X}", category)),
    }
}

fn search_dispatch(
    op: u8,
    bs: &BindSpace16K,
    args: &[Addr],
) -> CamResult {
    let query_addr = args[0];
    let query_fp = bs.read(query_addr);
    let k = 10; // or read from args

    match op {
        0x00 => { // SEARCH.ANN
            let query = SchemaQuery::new();
            let results = query.search(bs.node_slice(), &query_fp, k);
            CamResult::Many(results.into_iter().map(|r| r.addr).collect())
        }
        0x01 => { // SEARCH.SCHEMA
            let schema_json = /* read from args[1] or operation metadata */;
            let query = SchemaQuery::from_json(&schema_json);
            let results = query.search(bs.node_slice(), &query_fp, k);
            CamResult::Many(results.into_iter().map(|r| r.addr).collect())
        }
        0x02 => { // SEARCH.BLOOM
            let source_id = query_addr.0 as u64;
            let results = bloom_accelerated_search(
                bs.node_slice(), &query_fp, source_id, k, 0.3,
                &SchemaQuery::new(),
            );
            CamResult::Many(results.into_iter().map(|r| Addr(r.index as u16)).collect())
        }
        _ => CamResult::Error(format!("Unknown search op: 0x{:02X}", op)),
    }
}
```

---

## What This Means for cam_ops.rs

The 4,661-line cam_ops.rs doesn't need to grow. It needs to *shrink*:

1. **Remove stubs** — operations that return `Error("not yet implemented")`
   get replaced with one-line dispatches to search/schema/rl functions

2. **Remove OpResult enum complexity** — most operations return either a
   fingerprint (the 256-word array) or a scalar read from a schema block.
   The enum can be simplified.

3. **Remove per-operation fingerprint generation** — the "fingerprint of
   the operation" concept is useful for content-addressing the operation
   dictionary, but it doesn't need to be computed at runtime. Pre-compute
   once, store at the Surface address.

### Before (cam_ops.rs today, ~4600 lines):

```rust
fn execute(&self, op: u16, args: Vec<Fingerprint>) -> OpResult {
    match op {
        0x300 => OpResult::Scalar(hamming_distance(&args[0], &args[1]) as f64),
        0x301 => OpResult::Error("HAMMING.BIND not yet implemented".into()),
        // ... 4000 lines of stubs and partial implementations
    }
}
```

### After (~500 lines):

```rust
fn execute(&self, op: u16, args: Vec<Addr>) -> CamResult {
    let category = (op >> 8) as u8;
    match category {
        0x03 => self.hamming_ops(op & 0xFF, args),  // ~50 lines
        0x04 => self.nars_ops(op & 0xFF, args),     // ~50 lines
        0x05 => self.search_ops(op & 0xFF, args),   // ~80 lines
        0x09 => self.rl_ops(op & 0xFF, args),       // ~50 lines
        0x0E => self.learning_ops(op & 0xFF, args), // ~50 lines
        _ => CamResult::Error(format!("Unknown: 0x{:03X}", op)),
    }
}
```

Each dispatch function delegates to the proven schema/search/rl operations
that operate on the 256-word fingerprint. The CAM dictionary becomes a thin
routing layer, not a monolithic 4600-line match statement.

---

## The Key Insight

The CAM prefix was never a data-fitting problem. It was a routing problem.
The operations don't need to *live inside* the fingerprint surplus — they
need to *operate on* the fingerprint's schema blocks.

At 256 words, the schema blocks exist. The operations have targets. The
routing is trivial. The fitting problem dissolves.
