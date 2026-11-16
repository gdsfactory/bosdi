#ifndef BOSDI_RUNTIME_H
#define BOSDI_RUNTIME_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

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
 * Changing qualification advances the binding generation and invalidates prior
 * evaluation assumptions.
 */
bool bosdi_qualify_binding(uint64_t binding, bool pure_recomputing_eval);

/* Exact evaluation is the default. Residual-only calls preserve both Jacobian arrays.
 * Status: 0 success; -1 stale/unbound; -4 nonfinite outputs.
 */
int32_t bosdi_eval_bound(uint64_t binding, uint64_t generation,
    bool residual_only, bool check_finite);
#ifdef __cplusplus
}
#endif
#endif
