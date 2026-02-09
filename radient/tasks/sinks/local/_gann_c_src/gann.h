#ifndef GANN_H
#define GANN_H

#include <stdint.h>

/* Pointer-based tree node (40 bytes).
   Internal: weight_bias points to (dim+1) floats, left/right point to children.
   Leaf:     candidates points to candidate IDs, n_candidates gives count.
   Discriminant: is_leaf flag. */
typedef struct GANNNode {
    struct GANNNode *left;            /* child pointers for internal nodes */
    struct GANNNode *right;
    float *weight_bias;               /* internal: pointer to weights+bias; NULL for leaf */
    int64_t *candidates;              /* leaf: pointer to candidate IDs; NULL for internal */
    int32_t n_candidates;             /* leaf: number of candidates */
    int32_t is_leaf;                  /* 1 = leaf, 0 = internal */
} GANNNode;

/* Top-level index. */
typedef struct {
    GANNNode **trees;                 /* array of tree root pointers */
    int n_trees;
    float *dataset;                   /* flat [n_vectors * dim], 64-byte aligned */
    float *norms;                     /* precomputed ||v||^2, 64-byte aligned */
    int n_vectors;
    int dim;
} GANNIndex;

/* Search result entry. */
typedef struct {
    int64_t id;
    float distance;
} GANNResult;

/* Pre-allocated search context for zero-malloc queries. */
typedef struct {
    int64_t *hs_keys;                 /* hash set key storage */
    int *hs_present;                  /* hash set occupancy flags */
    int hs_capacity;                  /* hash set capacity (power of 2) */
} GANNSearchCtx;

/* Load a GANN index from a directory. */
GANNIndex *gann_load(const char *path);

/* Create a reusable search context for the given index. */
GANNSearchCtx *gann_search_ctx_create(const GANNIndex *index);
void gann_search_ctx_free(GANNSearchCtx *ctx);

/* Search with pre-allocated context. */
int gann_search(const GANNIndex *index, GANNSearchCtx *ctx,
                const float *query, int top_k,
                GANNResult *results);

/* Free all memory. */
void gann_free(GANNIndex *index);

#endif /* GANN_H */
