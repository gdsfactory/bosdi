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
/* Capability registry v1 (ABI v1), introduced checked-groups-2026-11-24.
 * Value Symbol                              Entry points governed                 Introduced
 * 1     BOSDI_CAP_DELAY_OBSERVATION           model_delays, handle_prepared_delays, CG1
 *                                           setup_batch_delays_checked,
 *                                           eval_handle_delays_checked
 * 2     BOSDI_CAP_OBSERVED_WORKERS            handle_observed_workers               CG1
 * 4     BOSDI_CAP_TRANSACTIONAL_SETUP         update_lanes_checked                  CG1
 * 8     BOSDI_CAP_STAGED_SETUP                stage_lanes_checked,                  CG2
 *                                           stage_prepared_delays,
 *                                           commit_stage, discard_stage
 * 16    BOSDI_CAP_NAN_DEFAULT_SETUP           stage_lanes_checked NaN defaults,     CG2
 *                                           update_lanes_defaults_checked
 * 32    BOSDI_RESERVED_CHECKED_BOUND_RETAINED       NONE: reserved, NOT advertised         CG2
 * All function names above have the bosdi_ prefix. CG1 is the first local
 * checked-groups candidate (mask 7); CG2 is its Circulox review follow-up (31).
 * No assigned/reserved registry bit may be reused for another feature.
 * Unknown bits confer no guarantees. Gate features on BOTH bit and symbol
 * presence. Legacy local candidate 98626261 used bit 1 for the unregistered
 * BOSDI_CHECKED_BOUND_RETAINED contract: that contract/name is retired. It must
 * NEVER be inferred from this registry's bit 1. If retained storage is ever
 * shipped, its feature is permanently reassigned bit 32 with new symbols;
 * this runtime does not implement or advertise it. Checked calls restore snapshots.
 */
#define BOSDI_CAP_DELAY_OBSERVATION 1ull
#define BOSDI_CAP_OBSERVED_WORKERS 2ull
#define BOSDI_CAP_TRANSACTIONAL_SETUP 4ull
#define BOSDI_CAP_STAGED_SETUP 8ull
#define BOSDI_CAP_NAN_DEFAULT_SETUP 16ull
#define BOSDI_RESERVED_CHECKED_BOUND_RETAINED 32ull
uint64_t bosdi_runtime_capabilities(void);
typedef struct BosdiDelayMetadata {
    uint32_t y_node, z_node; /* raw hidden nodes; never terminals or collapsed */
    uint32_t has_maxdelay;  /* 0 or 1; duration observation supports either */
} BosdiDelayMetadata;
/* NULL/0 queries count, otherwise len must equal count. Returns 0/-1/-2;
 * -7 reports unsupported delay metadata (bad nodes/offsets). No truncation.
 * All outputs are caller-owned. 0.4 models have zero delays. */
int32_t bosdi_model_delays(uint32_t model, BosdiDelayMetadata *out,
    size_t len, size_t *count);
/* Opt-in setup runs full checked eval on restored scratch at the supplied
 * X/context, and copies only evaluated td slots into prepared instance storage.
 * Needed for OpenVAF builds which initialize td only in eval. Other model and
 * instance bytes retain setup values. No handle is returned on failure. Raw td
 * may be nonfinite: the host must validate it. Existing v1 setup is unchanged.
 */
int32_t bosdi_setup_batch_delays_checked(uint32_t model, size_t count,
    size_t num_params, const double *params, const double *voltages,
    size_t voltages_len, const BosdiEvalContext *context, uint64_t *handle,
    BosdiCallError *error, BosdiLogCallback callback, void *user);
/* Device-major raw durations, len = lane count * delay count. No finite/sign
 * validation: host owns delay acceptance/history. Zero length permits NULL. */
int32_t bosdi_handle_prepared_delays(uint64_t handle, double *out, size_t len);
/* Same checked contract as v1; optional NULL/0 disables delay observation.
 * Otherwise len must be lane count * delay count. delays must be aligned and
 * disjoint from all buffers. Read from evaluated scratch, including nonfinite
 * values. On any failure discard all outputs. Context applies to every lane. */
int32_t bosdi_eval_handle_delays_checked(uint64_t handle,
    const BosdiBufferBinding *buffers, const BosdiEvalContext *context,
    bool residual_only, bool check_finite, double *delays, size_t len,
    BosdiCallError *error, BosdiLogCallback callback, void *user);
/* Distinct threads executing lanes in the most recent checked call. Reset at
 * evaluation pass start; zero on failed pass/update/unknown handle. Serialize checked calls
 * on a handle if querying this value; independent handles may run concurrently. */
size_t bosdi_handle_observed_workers(uint64_t handle);
/* Atomically replace selected lanes using fresh model+instance setup. Rows of
 * params and validation voltages follow the supplied lane list, not lane order.
 * Lanes must be unique. All params/voltages must be finite (no NaN defaults).
 * Context is required; full outputs at supplied voltages and prepared/evaluated
 * durations must be finite. Node maps AND collapse flags must remain identical.
 * Both setup callbacks run. No library reload. No accepted state changes on
 * failure. -7 structural/unsupported layout; -8 busy (retry after quiescence).
 * Lowest failing raw lane wins. All pointer ownership ends when call returns.
 * Takes a nonblocking global handle write lock: no publication during eval.
 * Success invalidates every existing binding: old generations fail until rebind,
 * which refreshes lane workspaces and returns a new generation. Unbound handle
 * calls immediately use the new snapshots. Prepared durations can be queried.
 */
int32_t bosdi_update_lanes_checked(uint64_t handle, size_t count,
    const size_t *lanes, size_t num_params, const double *params,
    const double *voltages, size_t voltages_len, const BosdiEvalContext *context,
    BosdiCallError *error, BosdiLogCallback callback, void *user);
/* Staged candidates (cap 8, NaN parameter-default semantics additionally cap 16).
 * Stage runs the same fresh setup, full finite F/Q/G/C and td validation, node-map
 * and collapse checks as immediate update. Rows follow the unique lane list.
 * NaN means provider default exactly as v1 setup; all other parameters and all X
 * values must be finite. Explicit -0.0 is a value, never a default.
 * Inputs/context/callback storage are borrowed only until stage returns. On
 * failure *stage=0; error uses the lowest raw lane. Success returns one owned
 * token, which the caller must commit or discard, including after staleness.
 * Stage takes a nonblocking global handle READ lock during preparation; -8 if
 * unavailable. It may run concurrently with evaluation of the SAME handle.
 * It changes no accepted bytes, bindings, generations, workspaces or worker stats.
 * Several stages per handle may coexist. No lock is held between calls.
 * Durations have supplied-lane-list order, with len=count*num_delays. -2 for
 * wrong length/alignment, -1 unknown/consumed, -9 stale. NULL allowed for len=0.
 * Query takes handle read + stage-table locks for its duration.
 * Commit takes a nonblocking global handle WRITE lock then the stage-table lock;
 * no publication overlaps evaluation or stage creation on any handle. -8 busy
 * leaves token intact and retryable; -9 stale leaves an uncommittable tombstone;
 * -1 unknown/consumed. No setup, eval, validation or allocation occurs at commit.
 * Success atomically publishes this handle, consumes token, clears qualified
 * workspace/stats and invalidates bindings exactly as immediate update does.
 * Any successful update/commit, or handle free, makes other stages stale and
 * immediately releases their payloads; their tombstones live until discard.
 * Discard is thread-safe and idempotent (including 0/unknown/freed/stale tokens).
 * It takes only the stage-table lock and releases all token storage. Calls on
 * different tokens may be concurrent; discard racing commit is ordered by that
 * lock. A stale token can NEVER publish. Library loading is never repeated.
 * Cross-handle publication is NOT a BOSDI atomic operation: the host must quiesce
 * evaluations, retain exclusive ownership of handles, and serialize all writes
 * across its stage/commit sequence. Busy commits are retryable; cancellation
 * should be honored before the first commit in that sequence.
 */
int32_t bosdi_stage_lanes_checked(uint64_t handle, size_t count,
    const size_t *lanes, size_t num_params, const double *params,
    const double *voltages, size_t voltages_len, const BosdiEvalContext *context,
    uint64_t *stage, BosdiCallError *error, BosdiLogCallback callback, void *user);
int32_t bosdi_stage_prepared_delays(uint64_t stage, double *out, size_t len);
int32_t bosdi_commit_stage(uint64_t stage);
void bosdi_discard_stage(uint64_t stage);
/* Immediate defaults-aware counterpart (cap 16). Same validation/publication
 * as stage+commit with one exclusive write-lock interval. Existing
 * bosdi_update_lanes_checked retains its finite-parameters-only contract. */
int32_t bosdi_update_lanes_defaults_checked(uint64_t handle, size_t count,
    const size_t *lanes, size_t num_params, const double *params,
    const double *voltages, size_t voltages_len, const BosdiEvalContext *context,
    BosdiCallError *error, BosdiLogCallback callback, void *user);
#ifdef __cplusplus
}
#endif
#endif
