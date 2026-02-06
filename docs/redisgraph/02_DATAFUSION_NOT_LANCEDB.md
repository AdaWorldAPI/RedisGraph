# Stop Rewriting LanceDB — Extend DataFusion Instead

> The ladybug-rs codebase has two storage paths: a broken Lance integration
> (lance.rs, API mismatch) and a working Arrow zero-copy layer
> (lance_zero_copy/). The correct move is neither — it's DataFusion
> extensions that make BindSpace queryable as a SQL/Cypher data source.

---

## The Problem

### What Happened

1. `lance.rs` was written against Lance 1.0 API
2. Vendor ships Lance 2.1.0-beta.0
3. API mismatch: Dataset::query() changed, Schema types moved
4. `database.rs` is blocked waiting for lance.rs
5. Meanwhile, `lance_zero_copy/` works fine with pure Arrow buffers
6. Attempted DataFusion 51→52 upgrade is blocked by liblzma conflict

### The Trap

The temptation is to fix lance.rs to match 2.1 API, wire database.rs through
it, then build query capabilities on top of Lance. This is a rewrite of LanceDB's
query layer in application code. **Don't do this.**

LanceDB is a storage format + index. DataFusion is a query engine. The right
architecture is:

```
SQL/Cypher query
      │
      ▼
  DataFusion (query planning + optimization)
      │
      ▼
  Custom TableProvider (reads from BindSpace)
      │
      ▼
  BindSpace arrays (65,536 × 256 u64, direct addressing)
      │
      ▼
  ArrowZeroCopy (reads/writes Arrow columnar batches)
      │
      ▼
  Parquet files (persistence, optional Lance format)
```

**BindSpace is already an efficient in-memory store.** It doesn't need LanceDB
for query capabilities — it needs DataFusion for SQL/Cypher planning and
optimization. Lance/Parquet can be the persistence layer beneath BindSpace,
not the query layer above it.

---

## What to Build: 4 DataFusion Extensions

### Extension 1: BindSpaceTableProvider

Make BindSpace look like a SQL table to DataFusion.

```rust
use datafusion::catalog::TableProvider;
use datafusion::arrow::datatypes::{Schema, Field, DataType};

pub struct BindSpaceTable {
    bind_space: Arc<BindSpace16K>,
    zone: Zone, // Surface, Fluid, or Nodes
}

impl TableProvider for BindSpaceTable {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("addr", DataType::UInt16, false),
            Field::new("prefix", DataType::UInt8, false),
            Field::new("slot", DataType::UInt8, false),
            // The fingerprint is FixedSizeBinary(2048) — one column, zero overhead
            Field::new("fingerprint", DataType::FixedSizeBinary(2048), false),
            // Pre-computed from schema blocks — read from fingerprint, no storage cost
            Field::new("popcount", DataType::UInt16, false),
            Field::new("nars_f", DataType::Float32, true),
            Field::new("nars_c", DataType::Float32, true),
            Field::new("ani_dominant", DataType::UInt8, true),
            Field::new("schema_version", DataType::UInt8, false),
        ]))
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Return a custom ExecutionPlan that reads from BindSpace
        // The filters get pushed down for predicate evaluation
        Ok(Arc::new(BindSpaceScan::new(
            self.bind_space.clone(),
            self.zone,
            projection.cloned(),
            filters.to_vec(),
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        // Push down schema predicate filters
        filters.iter().map(|f| {
            match f {
                // These can be evaluated inline during scan
                Expr::BinaryExpr { .. } => Ok(TableProviderFilterPushDown::Inexact),
                _ => Ok(TableProviderFilterPushDown::Unsupported),
            }
        }).collect()
    }
}
```

### Extension 2: HDR UDFs (User-Defined Functions)

Register fingerprint operations as SQL functions.

```rust
pub fn register_hdr_udfs(ctx: &SessionContext) {
    // hamming_distance(a, b) → UInt32
    ctx.register_udf(create_udf(
        "hamming_distance",
        vec![DataType::FixedSizeBinary(2048), DataType::FixedSizeBinary(2048)],
        DataType::UInt32,
        Volatility::Immutable,
        Arc::new(|args| {
            let a = args[0].as_fixed_size_binary();
            let b = args[1].as_fixed_size_binary();
            // Vectorized: process entire column pair
            let mut builder = UInt32Builder::with_capacity(a.len());
            for i in 0..a.len() {
                let dist = hamming_distance_bytes(a.value(i), b.value(i));
                builder.append_value(dist);
            }
            Ok(Arc::new(builder.finish()))
        }),
    ));

    // xor_bind(a, b) → FixedSizeBinary(2048)
    ctx.register_udf(create_udf(
        "xor_bind",
        vec![DataType::FixedSizeBinary(2048), DataType::FixedSizeBinary(2048)],
        DataType::FixedSizeBinary(2048),
        Volatility::Immutable,
        Arc::new(xor_bind_udf),
    ));

    // schema_predicate(fp, predicate_json) → Boolean
    // Reads schema blocks inline, evaluates predicate
    ctx.register_udf(create_udf(
        "schema_passes",
        vec![DataType::FixedSizeBinary(2048), DataType::Utf8],
        DataType::Boolean,
        Volatility::Immutable,
        Arc::new(schema_predicate_udf),
    ));

    // semantic_distance(a, b) → UInt32
    // Only blocks 0-12, ignores schema blocks
    ctx.register_udf(create_udf(
        "semantic_distance",
        vec![DataType::FixedSizeBinary(2048), DataType::FixedSizeBinary(2048)],
        DataType::UInt32,
        Volatility::Immutable,
        Arc::new(semantic_distance_udf),
    ));

    // read_ani_level(fp, level_index) → UInt16
    ctx.register_udf(create_udf(
        "ani_level",
        vec![DataType::FixedSizeBinary(2048), DataType::UInt8],
        DataType::UInt16,
        Volatility::Immutable,
        Arc::new(ani_level_udf),
    ));

    // read_nars_truth(fp) → Struct{f: Float32, c: Float32}
    ctx.register_udf(create_udf(
        "nars_truth",
        vec![DataType::FixedSizeBinary(2048)],
        DataType::Struct(Fields::from(vec![
            Field::new("f", DataType::Float32, false),
            Field::new("c", DataType::Float32, false),
        ])),
        Volatility::Immutable,
        Arc::new(nars_truth_udf),
    ));
}
```

### Extension 3: PhysicalOptimizer Rule (HDR Cascade Pushdown)

The key optimization: push schema predicate filtering *below* the sort
in a top-k query. Without this, DataFusion computes all distances, sorts,
then filters. With this, candidates are rejected during the distance scan.

```rust
pub struct HdrCascadePushdown;

impl PhysicalOptimizerRule for HdrCascadePushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Pattern match: SortExec(ProjectionExec(FilterExec(BindSpaceScan)))
        // Transform to: TopKExec(HdrCascadeScan)
        //
        // HdrCascadeScan evaluates predicates inline with distance computation,
        // maintains a bounded BinaryHeap of size k, and exits early when
        // remaining blocks can't beat the current k-th best.
        plan.transform_up(|node| {
            if let Some(sort) = node.as_any().downcast_ref::<SortExec>() {
                if is_hamming_sort(sort) {
                    // Replace with HDR cascade
                    return Ok(Transformed::yes(create_hdr_cascade_plan(sort)));
                }
            }
            Ok(Transformed::no(node))
        })
    }

    fn name(&self) -> &str { "hdr_cascade_pushdown" }
    fn schema_check(&self) -> bool { true }
}
```

### Extension 4: Cypher-to-SQL Transpiler Enhancement

The existing `cypher.rs` transpiler should map Cypher patterns to SQL queries
that use the HDR UDFs:

```sql
-- Cypher:
-- MATCH (a:Person)-[:KNOWS]->(b:Person)
-- WHERE a.trust > 0.5 AND b.social > 300
-- RETURN b ORDER BY similarity DESC LIMIT 10

-- Transpiled SQL:
SELECT b.addr, semantic_distance(a.fingerprint, b.fingerprint) as dist,
       ani_level(b.fingerprint, 5) as social,
       nars_truth(b.fingerprint).f as trust
FROM nodes a, nodes b
WHERE schema_passes(a.fingerprint, '{"nars": {"min_frequency": 0.5}}')
  AND schema_passes(b.fingerprint, '{"ani": {"min_level": 5, "min_activation": 300}}')
ORDER BY dist ASC
LIMIT 10
```

---

## What NOT to Do

1. **Don't fix lance.rs API mismatch** — it's not on the critical path.
   ArrowZeroCopy works. When you need persistence, write Parquet directly.

2. **Don't build a custom query planner** — DataFusion already has one.
   Register your data as a TableProvider and your operations as UDFs.
   The planner handles joins, projections, filter pushdown, and top-k.

3. **Don't store metadata in separate columns** — put it in the fingerprint.
   DataFusion UDFs can extract any field from the FixedSizeBinary column.
   This eliminates joins and simplifies the schema.

4. **Don't upgrade DataFusion 51→52 yet** — the liblzma conflict is a
   packaging issue, not a feature gap. DF 51 has everything you need.
   Upgrade when the conflict is resolved upstream.

---

## Implementation Priority

1. **BindSpaceTableProvider** — makes BindSpace queryable via SQL (2-3 files)
2. **HDR UDFs** — hamming_distance, xor_bind, schema_passes (1 file)
3. **HdrCascadePushdown** — optimizer rule for efficient top-k (1 file)
4. **Cypher transpiler update** — maps to new UDFs (modify existing)

This is approximately 500-800 lines of Rust. It replaces the need for:
- lance.rs (200 lines, broken)
- database.rs (blocked)
- Any LanceDB query reimplementation
- Separate metadata column management

---

## Proven in RedisGraph

The RedisGraph HDR engine implements this exact pattern:
- `SchemaQuery` with `passes_predicates()` checks inline during search
- `BlockMask` controls which blocks participate in distance (semantic-only vs full)
- `search()` evaluates predicates and distance in a single pass
- Navigator's `cypher_call()` maps Cypher procedures to search operations

The RedisGraph implementation runs 259 tests, all passing. The approach works.
