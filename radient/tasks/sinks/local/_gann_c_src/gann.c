#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>

#include <immintrin.h>

#include "gann.h"

/* ==========================================================================
   Utility
   ========================================================================== */

static char *path_join(const char *dir, const char *name)
{
    size_t dlen = strlen(dir);
    size_t nlen = strlen(name);
    char *out = malloc(dlen + 1 + nlen + 1);
    if (!out) {
        return NULL;
    }
    memcpy(out, dir, dlen);
    out[dlen] = '/';
    memcpy(out + dlen + 1, name, nlen + 1);
    return out;
}

static char *read_file(const char *path, long *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(len + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if ((long)fread(buf, 1, len, f) != len) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[len] = '\0';
    if (out_len) {
        *out_len = len;
    }
    fclose(f);
    return buf;
}

/* ==========================================================================
   Minimal JSON parser
   ========================================================================== */

static const char *json_skip_ws(const char *p)
{
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') {
        p++;
    }
    return p;
}

static const char *json_parse_string(const char *p, char **out)
{
    if (*p != '"') {
        return NULL;
    }
    p++;
    const char *start = p;
    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
        }
        p++;
    }
    size_t len = p - start;
    *out = malloc(len + 1);
    memcpy(*out, start, len);
    (*out)[len] = '\0';
    if (*p == '"') {
        p++;
    }
    return p;
}

static const char *json_parse_number(const char *p, double *out)
{
    char *end;
    *out = strtod(p, &end);
    return end;
}

typedef struct JSONValue JSONValue;
typedef struct { char *key; JSONValue *value; } JSONKeyValue;
typedef enum { JSON_NULL, JSON_NUMBER, JSON_STRING, JSON_BOOL, JSON_ARRAY, JSON_OBJECT } JSONType;

struct JSONValue {
    JSONType type;
    union {
        double number;
        char *string;
        int boolean;
        struct { JSONValue **items; int count; } array;
        struct { JSONKeyValue *pairs; int count; } object;
    };
};

static void json_free(JSONValue *v);
static const char *json_parse_value(const char *p, JSONValue **out);

static const char *json_parse_array(const char *p, JSONValue **out)
{
    if (*p != '[') {
        return NULL;
    }
    p = json_skip_ws(p + 1);
    JSONValue *arr = calloc(1, sizeof(JSONValue));
    arr->type = JSON_ARRAY;
    if (*p == ']') {
        p++;
        *out = arr;
        return p;
    }
    int cap = 16;
    arr->array.items = malloc(cap * sizeof(JSONValue *));
    while (1) {
        p = json_skip_ws(p);
        JSONValue *item = NULL;
        p = json_parse_value(p, &item);
        if (!p) {
            json_free(arr);
            return NULL;
        }
        if (arr->array.count >= cap) {
            cap *= 2;
            arr->array.items = realloc(arr->array.items, cap * sizeof(JSONValue *));
        }
        arr->array.items[arr->array.count++] = item;
        p = json_skip_ws(p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == ']') {
            p++;
            break;
        }
        json_free(arr);
        return NULL;
    }
    *out = arr;
    return p;
}

static const char *json_parse_object(const char *p, JSONValue **out)
{
    if (*p != '{') {
        return NULL;
    }
    p = json_skip_ws(p + 1);
    JSONValue *obj = calloc(1, sizeof(JSONValue));
    obj->type = JSON_OBJECT;
    if (*p == '}') {
        p++;
        *out = obj;
        return p;
    }
    int cap = 16;
    obj->object.pairs = malloc(cap * sizeof(JSONKeyValue));
    while (1) {
        p = json_skip_ws(p);
        char *key = NULL;
        p = json_parse_string(p, &key);
        if (!p) {
            json_free(obj);
            return NULL;
        }
        p = json_skip_ws(p);
        if (*p != ':') {
            free(key);
            json_free(obj);
            return NULL;
        }
        p = json_skip_ws(p + 1);
        JSONValue *val = NULL;
        p = json_parse_value(p, &val);
        if (!p) {
            free(key);
            json_free(obj);
            return NULL;
        }
        if (obj->object.count >= cap) {
            cap *= 2;
            obj->object.pairs = realloc(obj->object.pairs, cap * sizeof(JSONKeyValue));
        }
        obj->object.pairs[obj->object.count].key = key;
        obj->object.pairs[obj->object.count].value = val;
        obj->object.count++;
        p = json_skip_ws(p);
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == '}') {
            p++;
            break;
        }
        json_free(obj);
        return NULL;
    }
    *out = obj;
    return p;
}

static const char *json_parse_value(const char *p, JSONValue **out)
{
    p = json_skip_ws(p);
    if (*p == '"') {
        JSONValue *v = calloc(1, sizeof(JSONValue));
        v->type = JSON_STRING;
        p = json_parse_string(p, &v->string);
        *out = v;
        return p;
    }
    if (*p == '{') {
        return json_parse_object(p, out);
    }
    if (*p == '[') {
        return json_parse_array(p, out);
    }
    if (*p == 't' && strncmp(p, "true", 4) == 0) {
        JSONValue *v = calloc(1, sizeof(JSONValue));
        v->type = JSON_BOOL;
        v->boolean = 1;
        *out = v;
        return p + 4;
    }
    if (*p == 'f' && strncmp(p, "false", 5) == 0) {
        JSONValue *v = calloc(1, sizeof(JSONValue));
        v->type = JSON_BOOL;
        v->boolean = 0;
        *out = v;
        return p + 5;
    }
    if (*p == 'n' && strncmp(p, "null", 4) == 0) {
        JSONValue *v = calloc(1, sizeof(JSONValue));
        v->type = JSON_NULL;
        *out = v;
        return p + 4;
    }
    if (*p == '-' || (*p >= '0' && *p <= '9')) {
        JSONValue *v = calloc(1, sizeof(JSONValue));
        v->type = JSON_NUMBER;
        p = json_parse_number(p, &v->number);
        *out = v;
        return p;
    }
    return NULL;
}

static void json_free(JSONValue *v)
{
    if (!v) {
        return;
    }
    switch (v->type) {
    case JSON_STRING:
        free(v->string);
        break;
    case JSON_ARRAY:
        for (int i = 0; i < v->array.count; i++) {
            json_free(v->array.items[i]);
        }
        free(v->array.items);
        break;
    case JSON_OBJECT:
        for (int i = 0; i < v->object.count; i++) {
            free(v->object.pairs[i].key);
            json_free(v->object.pairs[i].value);
        }
        free(v->object.pairs);
        break;
    default:
        break;
    }
    free(v);
}

static JSONValue *json_object_get(const JSONValue *obj, const char *key)
{
    if (!obj || obj->type != JSON_OBJECT) {
        return NULL;
    }
    for (int i = 0; i < obj->object.count; i++) {
        if (strcmp(obj->object.pairs[i].key, key) == 0) {
            return obj->object.pairs[i].value;
        }
    }
    return NULL;
}

/* ==========================================================================
   NPY file parsing
   ========================================================================== */

typedef struct {
    void *data;
    int ndim;
    int64_t shape[4];
    int dtype_size;
    char dtype;
} NPYArray;

static int npy_load(const char *path, NPYArray *arr)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6) {
        fclose(f);
        return -1;
    }
    if (magic[0] != 0x93 || memcmp(magic + 1, "NUMPY", 5) != 0) {
        fclose(f);
        return -1;
    }

    unsigned char ver[2];
    if (fread(ver, 1, 2, f) != 2) {
        fclose(f);
        return -1;
    }

    uint32_t header_len = 0;
    if (ver[0] == 1) {
        uint16_t hl;
        if (fread(&hl, 2, 1, f) != 1) {
            fclose(f);
            return -1;
        }
        header_len = hl;
    } else {
        if (fread(&header_len, 4, 1, f) != 1) {
            fclose(f);
            return -1;
        }
    }

    char *header = malloc(header_len + 1);
    if (fread(header, 1, header_len, f) != header_len) {
        free(header);
        fclose(f);
        return -1;
    }
    header[header_len] = '\0';

    memset(arr, 0, sizeof(*arr));

    char *dp = strstr(header, "'descr'");
    if (!dp) {
        dp = strstr(header, "\"descr\"");
    }
    if (dp) {
        char *q = strchr(dp + 7, '\'');
        if (!q) {
            q = strchr(dp + 7, '"');
        }
        if (q) {
            q++;
            if (*q == '<' || *q == '>' || *q == '=' || *q == '|') {
                q++;
            }
            arr->dtype = *q;
            q++;
            arr->dtype_size = atoi(q);
        }
    }

    char *sp = strstr(header, "'shape'");
    if (!sp) {
        sp = strstr(header, "\"shape\"");
    }
    if (sp) {
        char *paren = strchr(sp, '(');
        if (paren) {
            paren++;
            arr->ndim = 0;
            while (*paren && *paren != ')') {
                while (*paren == ' ' || *paren == ',') {
                    paren++;
                }
                if (*paren == ')') {
                    break;
                }
                arr->shape[arr->ndim++] = strtoll(paren, &paren, 10);
            }
        }
    }
    free(header);

    int64_t total = 1;
    for (int i = 0; i < arr->ndim; i++) {
        total *= arr->shape[i];
    }
    size_t data_size = total * arr->dtype_size;
    arr->data = malloc(data_size);
    if (fread(arr->data, 1, data_size, f) != data_size) {
        free(arr->data);
        arr->data = NULL;
        fclose(f);
        return -1;
    }

    fclose(f);
    return 0;
}

/* ==========================================================================
   SIMD: Inline AVX2+FMA operations
   ========================================================================== */

static inline float hsum_avx2(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

static inline float dot_avx2(const float *a, const float *b, int dim)
{
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    int i = 0;
    int dim32 = dim & ~31;
    for (; i < dim32; i += 32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
    }
    sum0 = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));

    for (; i + 8 <= dim; i += 8) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum0);
    }

    float result = hsum_avx2(sum0);
    for (; i < dim; i++) {
        result += a[i] * b[i];
    }

    return result;
}

static inline float sq_norm_avx2(const float *v, int dim)
{
    return dot_avx2(v, v, dim);
}

/* ==========================================================================
   Max-heap for top-k selection
   ========================================================================== */

typedef struct {
    GANNResult *data;
    int capacity;
    int size;
} MaxHeap;

static inline void maxheap_init(MaxHeap *h, GANNResult *buf, int cap)
{
    h->data = buf;
    h->capacity = cap;
    h->size = 0;
}

static inline void maxheap_sift_down(MaxHeap *h, int i)
{
    GANNResult *d = h->data;
    int n = h->size;
    while (1) {
        int largest = i, l = 2*i+1, r = 2*i+2;
        if (l < n && d[l].distance > d[largest].distance) {
            largest = l;
        }
        if (r < n && d[r].distance > d[largest].distance) {
            largest = r;
        }
        if (largest == i) {
            break;
        }
        GANNResult tmp = d[i];
        d[i] = d[largest];
        d[largest] = tmp;
        i = largest;
    }
}

static inline void maxheap_sift_up(MaxHeap *h, int i)
{
    GANNResult *d = h->data;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (d[i].distance <= d[parent].distance) {
            break;
        }
        GANNResult tmp = d[i];
        d[i] = d[parent];
        d[parent] = tmp;
        i = parent;
    }
}

static inline float maxheap_push(MaxHeap *h, int64_t id, float dist)
{
    if (h->size < h->capacity) {
        h->data[h->size].id = id;
        h->data[h->size].distance = dist;
        h->size++;
        maxheap_sift_up(h, h->size - 1);
    } else if (dist < h->data[0].distance) {
        h->data[0].id = id;
        h->data[0].distance = dist;
        maxheap_sift_down(h, 0);
    }
    return (h->size == h->capacity) ? h->data[0].distance : INFINITY;
}

static void maxheap_sort(MaxHeap *h)
{
    int orig = h->size;
    while (h->size > 1) {
        h->size--;
        GANNResult tmp = h->data[0];
        h->data[0] = h->data[h->size];
        h->data[h->size] = tmp;
        maxheap_sift_down(h, 0);
    }
    h->size = orig;
}

/* ==========================================================================
   Tree construction: pointer-based tree
   ========================================================================== */

typedef struct {
    int64_t **candidate_ids;
    int *n_candidates;
    int base_id;
    int n_leaves;
} LeafLookup;

static LeafLookup *leaf_lookup_build(const JSONValue *leaves_json, int num_internal)
{
    if (!leaves_json || leaves_json->type != JSON_OBJECT) {
        return NULL;
    }
    int n_entries = leaves_json->object.count;
    LeafLookup *lk = calloc(1, sizeof(LeafLookup));
    lk->base_id = num_internal;

    int max_offset = 0;
    for (int i = 0; i < n_entries; i++) {
        int key = atoi(leaves_json->object.pairs[i].key);
        int offset = key - num_internal;
        if (offset > max_offset) {
            max_offset = offset;
        }
    }
    lk->n_leaves = max_offset + 1;
    lk->candidate_ids = calloc(lk->n_leaves, sizeof(int64_t *));
    lk->n_candidates = calloc(lk->n_leaves, sizeof(int));

    for (int i = 0; i < n_entries; i++) {
        int key = atoi(leaves_json->object.pairs[i].key);
        int offset = key - num_internal;
        JSONValue *arr = leaves_json->object.pairs[i].value;
        if (arr && arr->type == JSON_ARRAY) {
            lk->n_candidates[offset] = arr->array.count;
            lk->candidate_ids[offset] = malloc(arr->array.count * sizeof(int64_t));
            for (int j = 0; j < arr->array.count; j++) {
                lk->candidate_ids[offset][j] = (int64_t)arr->array.items[j]->number;
            }
        }
    }
    return lk;
}

static void leaf_lookup_free(LeafLookup *lk)
{
    if (!lk) {
        return;
    }
    for (int i = 0; i < lk->n_leaves; i++) {
        free(lk->candidate_ids[i]);
    }
    free(lk->candidate_ids);
    free(lk->n_candidates);
    free(lk);
}

/* Build tree with per-node allocation, linked via left/right pointers. */
static GANNNode *build_tree_node(
    int64_t nid, int num_internal, int dim,
    const float *weights, const float *biases,
    const int64_t *children, const LeafLookup *lk)
{
    GANNNode *node = calloc(1, sizeof(GANNNode));

    if (nid < num_internal) {
        /* Internal node: allocate weight+bias per node (64-byte aligned for SIMD) */
        size_t wb_size = (size_t)(dim + 1) * sizeof(float);
        wb_size = (wb_size + 63) & ~(size_t)63;
        node->weight_bias = (float *)aligned_alloc(64, wb_size);
        memcpy(node->weight_bias, weights + nid * dim, dim * sizeof(float));
        node->weight_bias[dim] = biases[nid];
        node->is_leaf = 0;

        node->left = build_tree_node(
            children[nid * 2], num_internal, dim,
            weights, biases, children, lk);
        node->right = build_tree_node(
            children[nid * 2 + 1], num_internal, dim,
            weights, biases, children, lk);
    } else {
        /* Leaf node: allocate candidates per node */
        node->is_leaf = 1;

        int offset = (int)(nid - lk->base_id);
        if (offset >= 0 && offset < lk->n_leaves && lk->candidate_ids[offset]) {
            int nc = lk->n_candidates[offset];
            node->candidates = malloc(nc * sizeof(int64_t));
            memcpy(node->candidates, lk->candidate_ids[offset], nc * sizeof(int64_t));
            node->n_candidates = nc;
        }
    }

    return node;
}

static void free_tree_nodes(GANNNode *node)
{
    if (!node) {
        return;
    }
    free_tree_nodes(node->left);
    free_tree_nodes(node->right);
    if (node->weight_bias) {
        free(node->weight_bias);
    }
    free(node->candidates);
    free(node);
}

/* ==========================================================================
   Parallel tree loading
   ========================================================================== */

typedef struct {
    const char *index_path;
    int tree_idx;
    int dim;
    GANNNode **tree;
    int status;
} TreeLoadTask;

static void *load_tree_worker(void *arg)
{
    TreeLoadTask *task = (TreeLoadTask *)arg;
    int t = task->tree_idx;
    int dim = task->dim;
    const char *path = task->index_path;

    char tree_dir[64];
    snprintf(tree_dir, sizeof(tree_dir), "tree_%d", t);
    char *tpath = path_join(path, tree_dir);
    char *w_path = path_join(tpath, "weights.npy");
    char *b_path = path_join(tpath, "biases.npy");
    char *c_path = path_join(tpath, "children.npy");
    char *l_path = path_join(tpath, "leaves.json");

    NPYArray weights_arr, biases_arr, children_arr;
    if (npy_load(w_path, &weights_arr) != 0 ||
        npy_load(b_path, &biases_arr) != 0 ||
        npy_load(c_path, &children_arr) != 0) {
        fprintf(stderr, "Failed to load .npy files for tree %d\n", t);
        task->status = -1;
        free(w_path); free(b_path); free(c_path); free(l_path); free(tpath);
        return NULL;
    }

    int num_internal = (int)weights_arr.shape[0];

    int64_t *children_i64 = NULL;
    int children_i64_allocd = 0;
    if (children_arr.dtype == 'i' && children_arr.dtype_size == 4) {
        int32_t *src = (int32_t *)children_arr.data;
        int64_t total_elems = children_arr.shape[0] * children_arr.shape[1];
        children_i64 = malloc(total_elems * sizeof(int64_t));
        for (int64_t i = 0; i < total_elems; i++) {
            children_i64[i] = (int64_t)src[i];
        }
        children_i64_allocd = 1;
    } else {
        children_i64 = (int64_t *)children_arr.data;
        children_arr.data = NULL;
    }

    long leaves_len;
    char *leaves_str = read_file(l_path, &leaves_len);
    if (!leaves_str) {
        fprintf(stderr, "Failed to read leaves.json for tree %d\n", t);
        task->status = -1;
        free(weights_arr.data);
        free(biases_arr.data);
        free(children_arr.data);
        if (children_i64_allocd) {
            free(children_i64);
        }
        free(w_path); free(b_path); free(c_path); free(l_path); free(tpath);
        return NULL;
    }

    JSONValue *leaves_json = NULL;
    json_parse_value(leaves_str, &leaves_json);
    free(leaves_str);
    if (!leaves_json) {
        fprintf(stderr, "Failed to parse leaves.json for tree %d\n", t);
        task->status = -1;
        free(weights_arr.data);
        free(biases_arr.data);
        free(children_arr.data);
        if (children_i64_allocd) {
            free(children_i64);
        }
        free(w_path); free(b_path); free(c_path); free(l_path); free(tpath);
        return NULL;
    }

    LeafLookup *lk = leaf_lookup_build(leaves_json, num_internal);
    json_free(leaves_json);

    *task->tree = build_tree_node(
        0, num_internal, dim,
        (float *)weights_arr.data, (float *)biases_arr.data,
        children_i64, lk);

    fprintf(stderr, "Loaded tree %d: %d internal\n", t, num_internal);

    leaf_lookup_free(lk);
    task->status = 0;

    free(weights_arr.data);
    free(biases_arr.data);
    free(children_arr.data);
    if (children_i64_allocd) {
        free(children_i64);
    }
    free(w_path); free(b_path); free(c_path); free(l_path); free(tpath);
    return NULL;
}

/* ==========================================================================
   Public API: gann_load
   ========================================================================== */

GANNIndex *gann_load(const char *path)
{
    char *meta_path = path_join(path, "meta.json");
    long meta_len;
    char *meta_str = read_file(meta_path, &meta_len);
    free(meta_path);
    if (!meta_str) {
        fprintf(stderr, "Failed to read meta.json\n");
        return NULL;
    }

    JSONValue *meta = NULL;
    json_parse_value(meta_str, &meta);
    free(meta_str);
    if (!meta) {
        fprintf(stderr, "Failed to parse meta.json\n");
        return NULL;
    }

    JSONValue *v_dim = json_object_get(meta, "dim");
    JSONValue *v_ntrees = json_object_get(meta, "n_trees");
    if (!v_dim || !v_ntrees) {
        json_free(meta);
        return NULL;
    }

    int dim = (int)v_dim->number;
    int n_trees = (int)v_ntrees->number;
    json_free(meta);

    char *ds_path = path_join(path, "dataset.npy");
    NPYArray ds_arr;
    if (npy_load(ds_path, &ds_arr) != 0) {
        fprintf(stderr, "Failed to load dataset.npy\n");
        free(ds_path);
        return NULL;
    }
    free(ds_path);

    int n_vectors = (int)ds_arr.shape[0];
    size_t ds_size = (size_t)n_vectors * dim * sizeof(float);
    ds_size = (ds_size + 63) & ~(size_t)63;
    float *dataset = (float *)aligned_alloc(64, ds_size);
    if (!dataset) {
        free(ds_arr.data);
        return NULL;
    }
    memcpy(dataset, ds_arr.data, (size_t)n_vectors * dim * sizeof(float));
    free(ds_arr.data);
    fprintf(stderr, "Loaded dataset: %d vectors, dim=%d\n", n_vectors, dim);

    size_t norms_size = (size_t)n_vectors * sizeof(float);
    norms_size = (norms_size + 63) & ~(size_t)63;
    float *norms = (float *)aligned_alloc(64, norms_size);
    if (!norms) {
        free(dataset);
        return NULL;
    }
    for (int i = 0; i < n_vectors; i++) {
        norms[i] = sq_norm_avx2(dataset + (size_t)i * dim, dim);
    }

    GANNIndex *index = calloc(1, sizeof(GANNIndex));
    index->dataset = dataset;
    index->norms = norms;
    index->n_vectors = n_vectors;
    index->dim = dim;
    index->n_trees = n_trees;
    index->trees = calloc(n_trees, sizeof(GANNNode *));

    TreeLoadTask *tasks = calloc(n_trees, sizeof(TreeLoadTask));
    pthread_t *threads = calloc(n_trees, sizeof(pthread_t));

    for (int t = 0; t < n_trees; t++) {
        tasks[t].index_path = path;
        tasks[t].tree_idx = t;
        tasks[t].dim = dim;
        tasks[t].tree = &index->trees[t];
        tasks[t].status = -1;
        pthread_create(&threads[t], NULL, load_tree_worker, &tasks[t]);
    }

    int any_failed = 0;
    for (int t = 0; t < n_trees; t++) {
        pthread_join(threads[t], NULL);
        if (tasks[t].status != 0) {
            any_failed = 1;
        }
    }

    free(tasks);
    free(threads);

    if (any_failed) {
        fprintf(stderr, "Some trees failed to load\n");
        gann_free(index);
        return NULL;
    }

    return index;
}

/* ==========================================================================
   Public API: gann_search_ctx
   ========================================================================== */

GANNSearchCtx *gann_search_ctx_create(const GANNIndex *index)
{
    GANNSearchCtx *ctx = calloc(1, sizeof(GANNSearchCtx));

    int est = 300 * index->n_trees * 2;
    int cap = 16;
    while (cap < est) {
        cap <<= 1;
    }
    ctx->hs_capacity = cap;
    ctx->hs_keys = malloc(cap * sizeof(int64_t));
    ctx->hs_present = calloc(cap, sizeof(int));

    return ctx;
}

void gann_search_ctx_free(GANNSearchCtx *ctx)
{
    if (!ctx) {
        return;
    }
    free(ctx->hs_keys);
    free(ctx->hs_present);
    free(ctx);
}

/* ==========================================================================
   Tree traversal: pointer chasing (follow left/right pointers)
   ========================================================================== */

static inline const GANNNode *traverse_tree(
    const GANNNode *node,
    const float *query, int dim)
{
    while (!node->is_leaf) {
        /* Prefetch both children before the dot product */
        if (node->left && !node->left->is_leaf) {
            _mm_prefetch((const char *)node->left->weight_bias, _MM_HINT_T0);
        }
        if (node->right && !node->right->is_leaf) {
            _mm_prefetch((const char *)node->right->weight_bias, _MM_HINT_T0);
        }

        const float *wb = node->weight_bias;
        float val = dot_avx2(wb, query, dim) + wb[dim];

        if (val < 0.0f) {
            node = node->left;
        } else {
            node = node->right;
        }
    }
    return node;
}

/* ==========================================================================
   Hash set for deduplication
   ========================================================================== */

static inline int hs_hash(int64_t key)
{
    uint64_t h = (uint64_t)key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (int)h;
}

static void hs_ensure_capacity(GANNSearchCtx *ctx, int min_cap)
{
    int needed = min_cap * 2;
    if (needed <= ctx->hs_capacity) {
        return;
    }

    int cap = ctx->hs_capacity;
    while (cap < needed) {
        cap <<= 1;
    }
    ctx->hs_capacity = cap;
    ctx->hs_keys = realloc(ctx->hs_keys, cap * sizeof(int64_t));
    ctx->hs_present = realloc(ctx->hs_present, cap * sizeof(int));
}

static inline void hs_clear(GANNSearchCtx *ctx)
{
    memset(ctx->hs_present, 0, ctx->hs_capacity * sizeof(int));
}

static inline int hs_insert(GANNSearchCtx *ctx, int64_t key)
{
    int mask = ctx->hs_capacity - 1;
    int idx = hs_hash(key) & mask;
    while (1) {
        if (!ctx->hs_present[idx]) {
            ctx->hs_keys[idx] = key;
            ctx->hs_present[idx] = 1;
            return 1;
        }
        if (ctx->hs_keys[idx] == key) {
            return 0;
        }
        idx = (idx + 1) & mask;
    }
}

/* ==========================================================================
   Public API: gann_search
   ========================================================================== */

int gann_search(const GANNIndex *index, GANNSearchCtx *ctx,
                const float *query, int top_k,
                GANNResult *results)
{
    int dim = index->dim;
    int n_trees = index->n_trees;

    int est_candidates = 300 * n_trees;
    hs_ensure_capacity(ctx, est_candidates);
    hs_clear(ctx);

    MaxHeap heap;
    maxheap_init(&heap, results, top_k);
    float query_norm = sq_norm_avx2(query, dim);

    int mask = ctx->hs_capacity - 1;

    for (int t = 0; t < n_trees; t++) {
        const GANNNode *leaf = traverse_tree(index->trees[t], query, dim);
        int nc = leaf->n_candidates;
        const int64_t *cands = leaf->candidates;

        /* Seed: prefetch first 8 candidates (full vector + HS bucket) */
        for (int p = 0; p < 8 && p < nc; p++) {
            const char *base = (const char *)(index->dataset + cands[p] * dim);
            _mm_prefetch(base, _MM_HINT_T0);
            _mm_prefetch(base + 64, _MM_HINT_T0);
            _mm_prefetch(base + 128, _MM_HINT_T0);
            _mm_prefetch(base + 192, _MM_HINT_T0);
            _mm_prefetch(base + 256, _MM_HINT_T0);
            _mm_prefetch(base + 320, _MM_HINT_T0);
            _mm_prefetch(base + 384, _MM_HINT_T0);
            _mm_prefetch(base + 448, _MM_HINT_T0);

            int hs_idx = hs_hash(cands[p]) & mask;
            _mm_prefetch((const char *)(ctx->hs_present + hs_idx), _MM_HINT_T0);
            _mm_prefetch((const char *)(ctx->hs_keys + hs_idx), _MM_HINT_T0);
        }

        for (int i = 0; i < nc; i++) {
            int64_t cid = cands[i];

            /* Dataset: 8 ahead, all 8 cache lines (full vector) */
            if (i + 8 < nc) {
                const char *base = (const char *)(index->dataset + cands[i+8] * dim);
                _mm_prefetch(base, _MM_HINT_T0);
                _mm_prefetch(base + 64, _MM_HINT_T0);
                _mm_prefetch(base + 128, _MM_HINT_T0);
                _mm_prefetch(base + 192, _MM_HINT_T0);
                _mm_prefetch(base + 256, _MM_HINT_T0);
                _mm_prefetch(base + 320, _MM_HINT_T0);
                _mm_prefetch(base + 384, _MM_HINT_T0);
                _mm_prefetch(base + 448, _MM_HINT_T0);

                /* Hash set bucket for i+8 */
                int hs_idx = hs_hash(cands[i+8]) & mask;
                _mm_prefetch((const char *)(ctx->hs_present + hs_idx), _MM_HINT_T0);
                _mm_prefetch((const char *)(ctx->hs_keys + hs_idx), _MM_HINT_T0);
            }

            /* Candidate array: 16 ahead */
            if (i + 16 < nc) {
                _mm_prefetch((const char *)(cands + i + 16), _MM_HINT_T0);
            }

            if (!hs_insert(ctx, cid)) {
                continue;
            }

            const float *vec = index->dataset + cid * dim;
            float d = query_norm + index->norms[cid] - 2.0f * dot_avx2(query, vec, dim);
            if (d < 0.0f) {
                d = 0.0f;
            }

            maxheap_push(&heap, cid, d);
        }
    }

    maxheap_sort(&heap);
    return heap.size;
}

/* ==========================================================================
   Public API: gann_free
   ========================================================================== */

void gann_free(GANNIndex *index)
{
    if (!index) {
        return;
    }

    free(index->dataset);
    free(index->norms);

    for (int t = 0; t < index->n_trees; t++) {
        free_tree_nodes(index->trees[t]);
    }
    free(index->trees);
    free(index);
}
