/*
 * HDR Hamming - Bitpacked Vector Search Library
 *
 * C header for integrating with RedisGraph and GraphBLAS
 *
 * Copyright (c) RedisGraph Contributors
 * Licensed under AGPL-3.0
 */

#ifndef HDR_HAMMING_H
#define HDR_HAMMING_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

#define HDR_VECTOR_BITS  10000
#define HDR_VECTOR_WORDS 157
#define HDR_VECTOR_BYTES 1256

/* ============================================================================
 * OPAQUE TYPES
 * ============================================================================ */

/** Opaque 10Kbit vector handle */
typedef struct HdrVector HdrVector;

/** Opaque HDR cascade index handle */
typedef struct HdrCascadeIndex HdrCascadeIndex;

/** Opaque vector field handle */
typedef struct HdrField HdrField;

/** Opaque resonator (cleanup memory) handle */
typedef struct HdrResonator HdrResonator;

/* ============================================================================
 * RESULT TYPES
 * ============================================================================ */

/** Search result */
typedef struct {
    uint64_t index;      /* Index in corpus */
    uint32_t distance;   /* Hamming distance */
    float similarity;    /* Similarity score (0.0-1.0) */
    float response;      /* Mexican hat response */
} HdrSearchResult;

/** Stacked popcount result */
typedef struct {
    uint8_t per_word[HDR_VECTOR_WORDS];  /* Per-word bit counts */
    uint32_t total;                       /* Total Hamming distance */
} HdrStackedPopcount;

/** Belichtungsmesser (exposure meter) result */
typedef struct {
    uint8_t mean;    /* Mean of 7 sample points (0-7) */
    uint8_t sd_100;  /* Standard deviation × 100 */
} HdrBelichtung;

/** Sparse matrix entry for GraphBLAS interop */
typedef struct {
    uint64_t row;
    uint64_t col;
    float value;
} HdrSparseEntry;

/* ============================================================================
 * VECTOR OPERATIONS
 * ============================================================================ */

/**
 * Create a zero vector (all bits 0)
 * @return New vector handle, NULL on error
 */
HdrVector* hdr_vector_zero(void);

/**
 * Create a random vector from seed
 * @param seed Random seed
 * @return New vector handle, NULL on error
 */
HdrVector* hdr_vector_random(uint64_t seed);

/**
 * Create vector from raw bytes
 * @param data Byte array (must be HDR_VECTOR_BYTES long)
 * @param len Length of data (must equal HDR_VECTOR_BYTES)
 * @return New vector handle, NULL on error
 */
HdrVector* hdr_vector_from_bytes(const uint8_t* data, size_t len);

/**
 * Create vector from u64 words
 * @param words Array of u64 (must be HDR_VECTOR_WORDS long)
 * @param len Number of words (must equal HDR_VECTOR_WORDS)
 * @return New vector handle, NULL on error
 */
HdrVector* hdr_vector_from_words(const uint64_t* words, size_t len);

/**
 * Create vector from hash of arbitrary data
 * @param data Data to hash
 * @param len Length of data
 * @return New vector handle
 */
HdrVector* hdr_vector_from_hash(const uint8_t* data, size_t len);

/**
 * Clone a vector
 * @param vec Vector to clone
 * @return New vector handle, NULL on error
 */
HdrVector* hdr_vector_clone(const HdrVector* vec);

/**
 * Free a vector
 * @param vec Vector to free (safe to call with NULL)
 */
void hdr_vector_free(HdrVector* vec);

/**
 * Export vector to bytes
 * @param vec Source vector
 * @param out Output buffer (must be at least HDR_VECTOR_BYTES)
 * @param out_len Buffer length
 * @return Number of bytes written, -1 on error
 */
int32_t hdr_vector_to_bytes(const HdrVector* vec, uint8_t* out, size_t out_len);

/**
 * Export vector to words
 * @param vec Source vector
 * @param out Output buffer (must be at least HDR_VECTOR_WORDS)
 * @param out_len Buffer length in words
 * @return Number of words written, -1 on error
 */
int32_t hdr_vector_to_words(const HdrVector* vec, uint64_t* out, size_t out_len);

/**
 * Get population count (number of set bits)
 * @param vec Vector
 * @return Number of set bits (0 to HDR_VECTOR_BITS)
 */
uint32_t hdr_vector_popcount(const HdrVector* vec);

/**
 * Get vector density (fraction of bits set)
 * @param vec Vector
 * @return Density (0.0 to 1.0)
 */
float hdr_vector_density(const HdrVector* vec);

/* ============================================================================
 * BINDING OPERATIONS (Vector Field)
 * ============================================================================ */

/**
 * Bind two vectors: A ⊗ B (XOR operation)
 * @param a First vector
 * @param b Second vector
 * @return Bound vector, NULL on error
 */
HdrVector* hdr_vector_bind(const HdrVector* a, const HdrVector* b);

/**
 * Unbind: bound ⊗ key = result
 * Since XOR is self-inverse: A ⊗ B ⊗ B = A
 * @param bound Bound vector
 * @param key Key to unbind with
 * @return Unbound result, NULL on error
 */
HdrVector* hdr_vector_unbind(const HdrVector* bound, const HdrVector* key);

/**
 * Bind three vectors: A ⊗ B ⊗ C (for typed edges)
 * Used for: src ⊗ verb ⊗ dst
 * @param a First vector (e.g., source node)
 * @param b Second vector (e.g., verb/relationship)
 * @param c Third vector (e.g., destination node)
 * @return Bound vector, NULL on error
 */
HdrVector* hdr_vector_bind3(const HdrVector* a, const HdrVector* b, const HdrVector* c);

/**
 * Bundle multiple vectors using majority voting
 * Creates a prototype from multiple examples
 * @param vecs Array of vector pointers
 * @param count Number of vectors
 * @return Bundled vector, NULL on error
 */
HdrVector* hdr_vector_bundle(const HdrVector* const* vecs, size_t count);

/**
 * Permute (rotate) vector bits
 * Used for positional encoding in sequences
 * @param vec Vector to permute
 * @param positions Rotation amount (positive=left, negative=right)
 * @return Permuted vector, NULL on error
 */
HdrVector* hdr_vector_permute(const HdrVector* vec, int32_t positions);

/* ============================================================================
 * HAMMING DISTANCE OPERATIONS
 * ============================================================================ */

/**
 * Compute exact Hamming distance between two vectors
 * @param a First vector
 * @param b Second vector
 * @return Hamming distance (0 to HDR_VECTOR_BITS), UINT32_MAX on error
 */
uint32_t hdr_hamming_distance(const HdrVector* a, const HdrVector* b);

/**
 * Compute similarity score (0.0 to 1.0)
 * similarity = 1.0 - (distance / HDR_VECTOR_BITS)
 * @param a First vector
 * @param b Second vector
 * @return Similarity score
 */
float hdr_similarity(const HdrVector* a, const HdrVector* b);

/**
 * Compute stacked popcount (per-word distances)
 * Useful for hierarchical filtering
 * @param a First vector
 * @param b Second vector
 * @param out Output structure
 * @return 0 on success, -1 on error
 */
int32_t hdr_stacked_popcount(const HdrVector* a, const HdrVector* b, HdrStackedPopcount* out);

/**
 * Compute stacked popcount with early termination
 * Returns early if threshold is exceeded
 * @param a First vector
 * @param b Second vector
 * @param threshold Maximum distance threshold
 * @param out Output structure
 * @return 0 on success within threshold, 1 if exceeded, -1 on error
 */
int32_t hdr_stacked_popcount_threshold(
    const HdrVector* a,
    const HdrVector* b,
    uint32_t threshold,
    HdrStackedPopcount* out
);

/**
 * Quick exposure meter (Belichtungsmesser)
 * 7-point sample for fast distance estimation
 * @param a First vector
 * @param b Second vector
 * @param out Output structure
 * @return 0 on success, -1 on error
 */
int32_t hdr_belichtung_meter(const HdrVector* a, const HdrVector* b, HdrBelichtung* out);

/* ============================================================================
 * CASCADE INDEX OPERATIONS
 * ============================================================================ */

/**
 * Create cascade index
 * @param capacity Initial capacity
 * @return Index handle, NULL on error
 */
HdrCascadeIndex* hdr_cascade_create(size_t capacity);

/**
 * Free cascade index
 * @param cascade Index to free
 */
void hdr_cascade_free(HdrCascadeIndex* cascade);

/**
 * Add vector to cascade index
 * @param cascade Target index
 * @param vec Vector to add
 * @return 0 on success, -1 on error
 */
int32_t hdr_cascade_add(HdrCascadeIndex* cascade, const HdrVector* vec);

/**
 * Get number of vectors in index
 * @param cascade Index
 * @return Number of vectors
 */
size_t hdr_cascade_len(const HdrCascadeIndex* cascade);

/**
 * Search cascade index for k nearest neighbors
 * @param cascade Index to search
 * @param query Query vector
 * @param k Maximum results
 * @param out Output array
 * @param out_len Output array capacity
 * @return Number of results, -1 on error
 */
int32_t hdr_cascade_search(
    const HdrCascadeIndex* cascade,
    const HdrVector* query,
    size_t k,
    HdrSearchResult* out,
    size_t out_len
);

/**
 * Set Mexican hat parameters for discrimination
 * @param cascade Target index
 * @param excite Excitation threshold (distance below = positive response)
 * @param inhibit Inhibition threshold (distance above = zero response)
 * @return 0 on success, -1 on error
 */
int32_t hdr_cascade_set_mexican_hat(
    HdrCascadeIndex* cascade,
    uint32_t excite,
    uint32_t inhibit
);

/* ============================================================================
 * RESONATOR (CLEANUP MEMORY) OPERATIONS
 * ============================================================================ */

/**
 * Create resonator for cleanup matching
 * @param capacity Initial capacity
 * @return Resonator handle, NULL on error
 */
HdrResonator* hdr_resonator_create(size_t capacity);

/**
 * Free resonator
 * @param resonator Resonator to free
 */
void hdr_resonator_free(HdrResonator* resonator);

/**
 * Add concept vector to resonator
 * @param resonator Target resonator
 * @param vec Concept vector
 * @return Index of added concept, -1 on error
 */
int32_t hdr_resonator_add(HdrResonator* resonator, const HdrVector* vec);

/**
 * Add named concept to resonator
 * @param resonator Target resonator
 * @param name Concept name (null-terminated string)
 * @param vec Concept vector
 * @return Index of added concept, -1 on error
 */
int32_t hdr_resonator_add_named(
    HdrResonator* resonator,
    const char* name,
    const HdrVector* vec
);

/**
 * Set resonator match threshold
 * @param resonator Target resonator
 * @param threshold Maximum distance for match
 * @return 0 on success, -1 on error
 */
int32_t hdr_resonator_set_threshold(HdrResonator* resonator, uint32_t threshold);

/**
 * Find best matching concept (resonate)
 * @param resonator Resonator to search
 * @param query Query vector
 * @param out_index Output: matched concept index (optional, can be NULL)
 * @param out_distance Output: distance to match (optional)
 * @param out_similarity Output: similarity score (optional)
 * @return 0 if match found, 1 if no match, -1 on error
 */
int32_t hdr_resonator_resonate(
    const HdrResonator* resonator,
    const HdrVector* query,
    size_t* out_index,
    uint32_t* out_distance,
    float* out_similarity
);

/* ============================================================================
 * GRAPHBLAS INTEGRATION
 * ============================================================================ */

/**
 * Convert vector similarities to sparse matrix entries
 * For building GraphBLAS adjacency matrices from search results
 * @param cascade Index to search
 * @param queries Array of query vectors
 * @param n_queries Number of queries
 * @param k Results per query
 * @param out Output sparse entries
 * @param out_capacity Maximum entries
 * @return Number of entries written, -1 on error
 */
int32_t hdr_to_sparse_matrix(
    const HdrCascadeIndex* cascade,
    const HdrVector* const* queries,
    size_t n_queries,
    size_t k,
    HdrSparseEntry* out,
    size_t out_capacity
);

/**
 * Batch bind edges for graph construction
 * Computes: edge[i] = source[i] ⊗ verb[i] ⊗ target[i]
 * @param sources Source node vectors
 * @param verbs Relationship vectors
 * @param targets Target node vectors
 * @param count Number of edges
 * @param out Output edge vectors (caller must free each)
 * @return Number of edges created, -1 on error
 */
int32_t hdr_batch_bind_edges(
    const HdrVector* const* sources,
    const HdrVector* const* verbs,
    const HdrVector* const* targets,
    size_t count,
    HdrVector** out
);

/* ============================================================================
 * VERSION INFO
 * ============================================================================ */

/**
 * Get library version string
 * @return Static version string
 */
const char* hdr_version(void);

/**
 * Get vector size in bits
 * @return HDR_VECTOR_BITS
 */
size_t hdr_vector_bits(void);

/**
 * Get vector size in bytes
 * @return HDR_VECTOR_BYTES
 */
size_t hdr_vector_bytes(void);

/**
 * Get vector size in 64-bit words
 * @return HDR_VECTOR_WORDS
 */
size_t hdr_vector_words(void);

#ifdef __cplusplus
}
#endif

#endif /* HDR_HAMMING_H */
