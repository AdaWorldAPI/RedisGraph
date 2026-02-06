# Prompt for Ladybug-RS Claude Code Session

> Copy-paste this into a fresh Claude Code session working on the ladybug-rs
> repository. It transfers the learning curve from the RedisGraph HDR
> fingerprint engine review so the session starts at full understanding
> instead of cold.

---

## Context Prompt

```
I need you to help refactor ladybug-rs using architectural insights from a
parallel Rust codebase (RedisGraph HDR fingerprint engine) that solved the
same core problems ladybug-rs is struggling with. The insights are documented
in docs/redisgraph/ — read ALL files there before making any changes.

Key problems to solve, in priority order:

### 1. The 156/157 Word Bug — Use 256 Words (16K Bits)

The codebase has FINGERPRINT_WORDS=156 in bind_space.rs and FINGERPRINT_U64=157
in lib.rs. Neither is correct. The RedisGraph engine proved that 256 words
(16,384 bits = 2^14) is the right choice because:
- sigma = sqrt(16384/4) = 64 = exactly one u64 word
- 256 / 8 = 32 AVX-512 iterations with ZERO remainder
- 16 uniform blocks of 1024 bits each (no short last block)
- Blocks 0-12 carry semantic content (13,312 bits > 10K requirement)
- Blocks 13-15 carry structured metadata (ANI, NARS, RL, graph metrics)

Read docs/redisgraph/01_THE_256_WORD_SOLUTION.md for the complete analysis.

### 2. Stop Rewriting LanceDB — Extend DataFusion Instead

The lance.rs / database.rs path is broken and unnecessary. ArrowZeroCopy already
works. The right move is DataFusion extensions:
- Custom TableProvider that wraps BindSpace as an Arrow data source
- UDFs for hamming_distance, xor_bind, schema_predicate that operate on
  FixedSizeBinary(2048) columns
- PhysicalOptimizer rule that pushes HDR cascade filtering below sort
- ExecutionPlan that does block-masked distance with early exit

Read docs/redisgraph/02_DATAFUSION_NOT_LANCEDB.md for the implementation guide.

### 3. The 4096 CAM Prefix Fits Naturally at 256 Words

The CAM operation dictionary (0x000-0xFFF) maps directly to schema metadata
blocks in 16K vectors. The operation fingerprint IS the schema query:
- CAM category → Block 13 ANI level filter
- CAM operation → Block 14 Q-value / RL action
- CAM arguments → Schema predicates passed to search

Read docs/redisgraph/03_CAM_PREFIX_SOLUTION.md for how this works.

### 4. Race Conditions Have Known Fixes

All 9 documented race conditions follow the same pattern: lock released between
check and commit. The ConcurrentWriteCache pattern from RedisGraph (RwLock with
owned return values) solves most of them.

Read docs/redisgraph/04_RACE_CONDITION_PATTERNS.md for the fix templates.

### 5. Don't Overwrite Anything — Additive Changes Only

Create new files alongside existing ones. The migration from 156-word to 256-word
should be a separate module (width_16k/) that coexists with the current code.
Wire it in gradually, test both paths, then deprecate the old one.

Read docs/redisgraph/05_MIGRATION_STRATEGY.md for the step-by-step plan.

IMPORTANT: Read ALL docs/redisgraph/*.md files before starting. They contain
proven, tested solutions from a working implementation — not speculation.
```

---

## What This Prompt Does

1. **Transfers the learning curve** — The receiving session understands
   *why* 256 words, not just *that* 256 words
2. **Prevents the LanceDB trap** — Explicitly redirects to DataFusion
   extensions, which is where the leverage actually is
3. **Solves the CAM fitting problem** — Shows that 4096 CAM ops map to
   schema metadata blocks, eliminating the "where do we put them" question
4. **Provides fix templates** — Not just "fix the race conditions" but
   exact code patterns proven in another codebase
5. **Protects existing work** — Additive migration, no overwrites

## Prerequisite

The docs/redisgraph/ directory must exist in the ladybug-rs repo. Copy it
from the RedisGraph repo or ensure both repos are accessible.
