/* Build and link an embedder using only the published runtime header. */
#include "bosdi_runtime.h"
#include <assert.h>
#include <math.h>

int main(int argc, char **argv) {
    assert(argc == 2);
    assert(bosdi_runtime_abi_version() == 1);
    BosdiModelMetadata m = load_osdi_library_with_analysis(argv[1], 5, 300, 0);
    assert(m.success && m.num_nodes == 3 && m.num_params == 2);
    double params[] = {1000, 1e-9};
    uint64_t handle = 0;
    BosdiCallError error;
    assert(bosdi_setup_batch_checked(m.model_id, 1, 2, params, &handle, &error, NULL, NULL) == 0);
    assert(!bosdi_free_model(m.model_id));
    int32_t maps[3];
    assert(bosdi_handle_node_maps(handle, 1, 3, maps) == 0);
    assert(maps[0] == 0 && maps[1] == 1 && maps[2] == 2);
    double v[] = {1, 0, 0}, f[4], g[4], q[4], c[4];
    BosdiBufferBinding buffers = {v, 3, f, 4, g, 4, q, 4, c, 4};
    BosdiEvalContext context = {2, BOSDI_REACTIVE_AT_DC | BOSDI_RESIST_ALWAYS, 2};
    assert(bosdi_eval_handle_checked(handle, &buffers, &context, false, true, &error, NULL, NULL) == 0);
    assert(error.status == 0 && error.lane == SIZE_MAX);
    assert(fabs(f[0] - 2.001) < 1e-15 && q[0] == 1e-9 && c[0] == 1e-9);
    uint64_t binding = bosdi_bind_buffers(handle, &buffers);
    assert(binding);
    assert(bosdi_eval_bound_checked(binding, bosdi_binding_generation(binding), &context,
        false, true, &error, NULL, NULL) == 0);
    bosdi_unbind_buffers(binding);
    osdi_free_handle_ffi(handle);
    assert(bosdi_free_model(m.model_id));
    assert(!bosdi_free_model(m.model_id));
    return 0;
}
