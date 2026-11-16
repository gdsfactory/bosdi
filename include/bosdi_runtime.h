#ifndef BOSDI_RUNTIME_H
#define BOSDI_RUNTIME_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Model registration and handle lifetime. Version is 4 or 5 (OSDI 0.4/0.5).
 * Parameter values are in descriptor order; NaN selects provider defaults.
 * Metadata num_nodes includes all raw unknowns; num_pins is terminal count.
 */
typedef struct BosdiModelMetadata {
    uint32_t model_id;
    size_t num_pins, num_nodes, num_params, num_states;
    uint32_t osdi_version;
    bool success;
} BosdiModelMetadata;
BosdiModelMetadata load_osdi_library_with_analysis(const char *path, uint32_t version,
    double temperature, uint32_t analysis);
BosdiModelMetadata load_osdi_library_with_simparams(const char *path, uint32_t version,
    double temperature, uint32_t analysis, const char *const *names,
    const double *values, size_t count);
/* Thread-local loader diagnostic: returns total byte length, copies at most
 * buf_len bytes without a NUL terminator. NULL/0 queries length. */
size_t get_last_osdi_error(uint8_t *buf, size_t buf_len);
void osdi_free_handle_ffi(uint64_t handle);
size_t osdi_handle_jac_count_ffi(uint64_t handle, bool reactive);
/* len is exactly count*num_entries*2 integers. Ground targets are (-1,-1).
 * Duplicates must be summed, and collapsed equality derivatives are separate. */
int32_t osdi_handle_jac_pairs_ffi(uint64_t handle, bool reactive, int32_t *out, size_t len);

bool bosdi_set_num_threads(size_t threads); /* 0 restores once-read defaults */
size_t bosdi_num_threads(void);
bool bosdi_set_parallel_policy(size_t min_lanes, size_t min_chunk);
/* Same provider-qualification contract as bosdi_qualify_binding below.
 * Applies only to non-tiled evaluations; tiled calls always restore snapshots. */
bool bosdi_qualify_handle(uint64_t handle, bool pure_recomputing_eval);
bool bosdi_set_handle_parallel_policy(uint64_t handle, size_t min_lanes, size_t min_chunk);

/* Disjoint contiguous double arrays, lengths in elements. F/Q have stride
 * num_nodes+1: the final slot is a private ground sink. G/C contain raw entries
 * (#36); equality derivatives come from the existing handle sparse pattern.
 * All buffers remain alive and exclusively accessible through unbind/rebind.
 * Registered buffers cannot be shared by simultaneously executing bindings.
 * Input voltages contain all raw unknowns, stride num_nodes. No vmap tiling.
 */
typedef struct BosdiBufferBinding {
    const double *voltages; size_t voltages_len;
    double *currents; size_t currents_len;
    double *conductances; size_t conductances_len;
    double *charges; size_t charges_len;
    double *capacitances; size_t capacitances_len;
} BosdiBufferBinding;
uint64_t bosdi_bind_buffers(uint64_t handle, const BosdiBufferBinding *buffers);
uint64_t bosdi_rebind_buffers(uint64_t binding, const BosdiBufferBinding *buffers);
void bosdi_unbind_buffers(uint64_t binding); /* waits for in-flight evaluation */
uint64_t bosdi_binding_generation(uint64_t binding);
/* Caller certifies eval fully recomputes all intermediate values, preserves
 * the node map, reads prev_solve without modifying it, and reads no history. ABI-state models (including limiting slots) cannot be qualified.
 * Defaults to false: snapshot restoration remains the generic exact path.
 * Changing qualification advances the binding generation, invalidating caches.
 */
bool bosdi_qualify_binding(uint64_t binding, bool pure_recomputing_eval);

typedef struct BosdiBypass {
    const double *previous_voltages; size_t previous_voltages_len;
    const uint8_t *valid; size_t valid_len;
    uint8_t *bypassed; size_t bypassed_len;
    uint64_t generation;
    double reltol, vntol;
    bool dry_run;
} BosdiBypass;
typedef struct BosdiEvalStats { size_t evaluated, bypassed, eligible; } BosdiEvalStats;
/* Bypass is optional and zero-order. Validity: 1=F/Q, 2=G/C, 3=both.
 * Previous inputs/validity are read-only; caller updates them only for outputs
 * it accepts. Cache generation must match the binding. Rebind always
 * invalidates validity, even if addresses do not change. Nonfinite inputs
 * never bypass. Dry-run measures eligibility with exact evaluation, including
 * on unqualified providers. Exact evaluations never consume cached values.
 * All cache regions are disjoint from registered buffers and each other.
 * Status: 0 success; -1 stale/unbound; -2 invalid cache/alias/tolerances;
 * -3 unqualified provider; -4 nonfinite outputs. Optional stats describe a
 * completed pass. Residual-only calls preserve both Jacobian arrays.
 */
int32_t bosdi_eval_bound(uint64_t binding, uint64_t generation,
    bool residual_only, bool check_finite);
int32_t bosdi_eval_bound_bypass(uint64_t binding, uint64_t generation,
    bool residual_only, bool check_finite, const BosdiBypass *bypass,
    BosdiEvalStats *stats);

/* Additive embedder ABI v1. Existing entry points retain their defaults. */
uint32_t bosdi_runtime_abi_version(void);
#define BOSDI_REACTIVE_AT_DC 1u
#define BOSDI_RESIST_ALWAYS 2u
typedef struct BosdiEvalContext {
    uint32_t analysis; /* 0 DC/static, 1 AC/static, 2 transient */
    uint32_t options;  /* only the two bits above are accepted */
    double abstime;   /* finite, non-negative seconds */
} BosdiEvalContext;
/* Status: 0 success, -1 unknown/stale, -2 invalid argument, -4 nonfinite,
 * -5 setup failure, -6 eval failure. lane=SIZE_MAX for success/argument errors.
 * Lowest failing lane wins. message is NUL-terminated, at most 255 bytes.
 * Any failed call's outputs must be discarded; setup publishes no handle.
 */
typedef struct BosdiCallError {
    int32_t status;
    size_t lane;
    char message[256];
} BosdiCallError;
/* Optional synchronous callback. Can run concurrently on BOSDI workers.
 * Message is borrowed for the callback only. Must not reenter BOSDI, mutate
 * registered arrays, unwind, or acquire locks needed by the calling thread.
 */
typedef void (*BosdiLogCallback)(void *user, size_t lane, const char *message, uint32_t level);
int32_t bosdi_setup_batch_checked(uint32_t model, size_t count, size_t num_params,
    const double *params, uint64_t *handle, BosdiCallError *error,
    BosdiLogCallback callback, void *user);
/* Uses the same disjoint raw-entry buffer sizes as a binding. Restores setup
 * snapshots each call, regardless of qualification. Residual-only preserves G/C.
 * Context is required and affects only this call; no bypass caches are consumed.
 * error and callback are optional. Caller owns all pointers for the entire call.
 */
int32_t bosdi_eval_handle_checked(uint64_t handle, const BosdiBufferBinding *buffers,
    const BosdiEvalContext *context, bool residual_only, bool check_finite,
    BosdiCallError *error, BosdiLogCallback callback, void *user);
int32_t bosdi_eval_bound_checked(uint64_t binding, uint64_t generation,
    const BosdiEvalContext *context, bool residual_only, bool check_finite,
    BosdiCallError *error, BosdiLogCallback callback, void *user);
/* Actual post-setup maps, device-major, count*num_nodes entries. -1 is ground.
 * count and num_nodes must exactly match the handle. Returns 0/-1/-2.
 */
int32_t bosdi_handle_node_maps(uint64_t handle, size_t count, size_t num_nodes, int32_t *out);
/* False for unknown model, live handles, or concurrent registry activity; retry
 * after quiescing the host. Free handles and unbind buffers before unloading.
 */
bool bosdi_free_model(uint32_t model);
#ifdef __cplusplus
}
#endif
#endif
