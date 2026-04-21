#include <cstdint>
#include <cstddef>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;

// ---------------------------------------------------------
// 1. DATA STRUCTURES (Must match Rust #[repr(C)])
// ---------------------------------------------------------

struct ModelMetadata {
    uint32_t model_id;
    size_t   num_pins;
    size_t   num_nodes;
    size_t   num_params;
    size_t   num_states;
    uint32_t osdi_version;
    bool     success;
};

extern "C" {
    // Phase 1: The loader — version selects the ABI layout in Rust
    ModelMetadata load_osdi_library(const char* path_ptr, uint32_t version);

    // Diagnostic
    void dump_model_info(uint32_t model_id);

    // Resistive mask (Vec<bool> not repr(C), so separate two-call pattern)
    size_t get_resistive_mask_len(uint32_t model_id);
    void   get_resistive_mask_ffi(uint32_t model_id, uint8_t* out);

    // Structural introspection — paired len + copy-out functions.
    // Pair accessors write 2*len u32s (node_1, node_2, node_1, node_2, ...).
    size_t get_resist_jac_pairs_len(uint32_t model_id);
    void   get_resist_jac_pairs_ffi(uint32_t model_id, uint32_t* out);
    size_t get_react_jac_pairs_len(uint32_t model_id);
    void   get_react_jac_pairs_ffi(uint32_t model_id, uint32_t* out);
    size_t get_collapsible_pairs_len(uint32_t model_id);
    void   get_collapsible_pairs_ffi(uint32_t model_id, uint32_t* out);
    size_t get_param_flags_len(uint32_t model_id);
    void   get_param_flags_ffi(uint32_t model_id, uint32_t* out);
    size_t get_param_name_len(uint32_t model_id, size_t idx);
    void   get_param_name_ffi(uint32_t model_id, size_t idx, uint8_t* out);

    // Phase 2: The batched execution (Rayon zipper in Rust)
    void batched_osdi_eval_ffi(
        uint32_t model_id,
        size_t num_devices,
        size_t num_pins,
        size_t num_params,
        size_t num_states,
        const double* voltages,
        const double* params,
        const double* old_state,
        double* currents,
        double* conductances,
        double* charges,
        double* capacitances,
        double* new_state
    );

    // Phase 2 (residual-only): skips Jacobian work; returns only
    // (currents, charges, new_state). For Newton inner iterations that reuse
    // a frozen Jacobian from the first iter of the timestep.
    void batched_osdi_residual_eval_ffi(
        uint32_t model_id,
        size_t num_devices,
        size_t num_pins,
        size_t num_params,
        size_t num_states,
        const double* voltages,
        const double* params,
        const double* old_state,
        double* currents,
        double* charges,
        double* new_state
    );

    // Batch handle API: runs setup_model + setup_instance once for N devices
    // and stores the resulting snapshots keyed by a handle id. Eval calls
    // against the handle skip setup entirely.
    uint64_t osdi_setup_batch_ffi(
        uint32_t model_id,
        size_t num_devices,
        size_t num_params,
        const double* params
    );
    void osdi_free_handle_ffi(uint64_t handle_id);
    size_t osdi_handle_num_devices(uint64_t handle_id);

    // Handle-based full eval: skips setup entirely. Rust tiles the handle's
    // snapshots across num_devices (must be a multiple of handle.num_devices).
    void batched_osdi_eval_handle_ffi(
        uint64_t handle_id,
        size_t num_devices,
        size_t num_pins,
        size_t num_states,
        const double* voltages,
        const double* old_state,
        double* currents,
        double* conductances,
        double* charges,
        double* capacitances,
        double* new_state
    );

    // Handle-based residual-only eval: skips setup + Jacobian pass.
    void batched_osdi_residual_eval_handle_ffi(
        uint64_t handle_id,
        size_t num_devices,
        size_t num_pins,
        size_t num_states,
        const double* voltages,
        const double* old_state,
        double* currents,
        double* charges,
        double* new_state
    );
}

// ---------------------------------------------------------
// 2. THE XLA FFI HANDLER (The bridge between JAX and Rust)
// ---------------------------------------------------------

ffi::Error batched_osdi_eval_impl(
    ffi::Buffer<ffi::DataType::U32> model_id_buf,
    ffi::Buffer<ffi::DataType::F64> voltages,
    ffi::Buffer<ffi::DataType::F64> params,
    ffi::Buffer<ffi::DataType::F64> old_state,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> currents,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> conductances,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> charges,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> capacitances,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> new_state
) {
    uint32_t model_id = model_id_buf.typed_data()[0];

    auto v_dims = voltages.dimensions();
    auto p_dims = params.dimensions();
    auto s_dims = old_state.dimensions();

    size_t num_devices = v_dims[0];
    size_t num_pins    = v_dims[1];
    size_t num_params  = p_dims[1];
    size_t num_states  = s_dims[1];

    batched_osdi_eval_ffi(
        model_id,
        num_devices,
        num_pins,
        num_params,
        num_states,
        voltages.typed_data(),
        params.typed_data(),
        old_state.typed_data(),
        currents->typed_data(),
        conductances->typed_data(),
        charges->typed_data(),
        capacitances->typed_data(),
        new_state->typed_data()
    );

    return ffi::Error::Success();
}

// ---------------------------------------------------------
// 3. REGISTER THE FFI SYMBOL
// ---------------------------------------------------------

XLA_FFI_DEFINE_HANDLER_SYMBOL(OsdiEvalCpu, batched_osdi_eval_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

// ---------------------------------------------------------
// 3b. RESIDUAL-ONLY XLA FFI HANDLER
// ---------------------------------------------------------

ffi::Error batched_osdi_residual_eval_impl(
    ffi::Buffer<ffi::DataType::U32> model_id_buf,
    ffi::Buffer<ffi::DataType::F64> voltages,
    ffi::Buffer<ffi::DataType::F64> params,
    ffi::Buffer<ffi::DataType::F64> old_state,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> currents,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> charges,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> new_state
) {
    uint32_t model_id = model_id_buf.typed_data()[0];

    auto v_dims = voltages.dimensions();
    auto p_dims = params.dimensions();
    auto s_dims = old_state.dimensions();

    size_t num_devices = v_dims[0];
    size_t num_pins    = v_dims[1];
    size_t num_params  = p_dims[1];
    size_t num_states  = s_dims[1];

    batched_osdi_residual_eval_ffi(
        model_id,
        num_devices,
        num_pins,
        num_params,
        num_states,
        voltages.typed_data(),
        params.typed_data(),
        old_state.typed_data(),
        currents->typed_data(),
        charges->typed_data(),
        new_state->typed_data()
    );

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OsdiResidualEvalCpu, batched_osdi_residual_eval_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

// ---------------------------------------------------------
// 3c. HANDLE-BASED XLA FFI HANDLERS
// ---------------------------------------------------------
//
// The handle_id is a scalar u64 passed as a 1-element U64 buffer — XLA FFI
// doesn't have scalar attribute support that works cleanly from ffi_call, so
// we wrap it in a buffer like we do for model_id.

ffi::Error batched_osdi_eval_handle_impl(
    ffi::Buffer<ffi::DataType::U64> handle_id_buf,
    ffi::Buffer<ffi::DataType::F64> voltages,
    ffi::Buffer<ffi::DataType::F64> old_state,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> currents,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> conductances,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> charges,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> capacitances,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> new_state
) {
    uint64_t handle_id = handle_id_buf.typed_data()[0];

    auto v_dims = voltages.dimensions();
    auto s_dims = old_state.dimensions();

    size_t num_devices = v_dims[0];
    size_t num_pins    = v_dims[1];
    size_t num_states  = s_dims[1];

    batched_osdi_eval_handle_ffi(
        handle_id,
        num_devices,
        num_pins,
        num_states,
        voltages.typed_data(),
        old_state.typed_data(),
        currents->typed_data(),
        conductances->typed_data(),
        charges->typed_data(),
        capacitances->typed_data(),
        new_state->typed_data()
    );
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OsdiEvalHandleCpu, batched_osdi_eval_handle_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

ffi::Error batched_osdi_residual_eval_handle_impl(
    ffi::Buffer<ffi::DataType::U64> handle_id_buf,
    ffi::Buffer<ffi::DataType::F64> voltages,
    ffi::Buffer<ffi::DataType::F64> old_state,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> currents,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> charges,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> new_state
) {
    uint64_t handle_id = handle_id_buf.typed_data()[0];

    auto v_dims = voltages.dimensions();
    auto s_dims = old_state.dimensions();

    size_t num_devices = v_dims[0];
    size_t num_pins    = v_dims[1];
    size_t num_states  = s_dims[1];

    batched_osdi_residual_eval_handle_ffi(
        handle_id,
        num_devices,
        num_pins,
        num_states,
        voltages.typed_data(),
        old_state.typed_data(),
        currents->typed_data(),
        charges->typed_data(),
        new_state->typed_data()
    );
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(OsdiResidualEvalHandleCpu, batched_osdi_residual_eval_handle_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
        .Ret<ffi::Buffer<ffi::DataType::F64>>()
);

// ---------------------------------------------------------
// 4. NANOBIND EXPORT
// ---------------------------------------------------------

NB_MODULE(osdi_shim_nb, m) {
    m.doc() = "OSDI Batched Evaluator for JAX";

    nb::class_<ModelMetadata>(m, "ModelMetadata")
        .def_ro("model_id",     &ModelMetadata::model_id)
        .def_ro("num_pins",     &ModelMetadata::num_pins)
        .def_ro("num_nodes",    &ModelMetadata::num_nodes)
        .def_ro("num_params",   &ModelMetadata::num_params)
        .def_ro("num_states",   &ModelMetadata::num_states)
        .def_ro("osdi_version", &ModelMetadata::osdi_version)
        .def_ro("success",      &ModelMetadata::success);

    m.def("load_osdi_library", [](const std::string& path, uint32_t version) {
        return load_osdi_library(path.c_str(), version);
    }, nb::arg("path"), nb::arg("version") = 4u);

    m.def("batched_osdi_eval", []() {
        return nb::capsule((void*)&OsdiEvalCpu, "xla._CUSTOM_CALL_TARGET");
    });

    m.def("batched_osdi_residual_eval", []() {
        return nb::capsule((void*)&OsdiResidualEvalCpu, "xla._CUSTOM_CALL_TARGET");
    });

    m.def("batched_osdi_eval_handle", []() {
        return nb::capsule((void*)&OsdiEvalHandleCpu, "xla._CUSTOM_CALL_TARGET");
    });

    m.def("batched_osdi_residual_eval_handle", []() {
        return nb::capsule((void*)&OsdiResidualEvalHandleCpu, "xla._CUSTOM_CALL_TARGET");
    });

    m.def("osdi_setup_batch", [](uint32_t model_id, size_t num_devices,
                                  size_t num_params, uintptr_t params_addr) {
        return osdi_setup_batch_ffi(
            model_id, num_devices, num_params,
            reinterpret_cast<const double*>(params_addr)
        );
    }, nb::arg("model_id"), nb::arg("num_devices"),
       nb::arg("num_params"), nb::arg("params_addr"));

    m.def("osdi_free_handle", [](uint64_t handle_id) {
        osdi_free_handle_ffi(handle_id);
    }, nb::arg("handle_id"));

    m.def("osdi_handle_num_devices", [](uint64_t handle_id) {
        return osdi_handle_num_devices(handle_id);
    }, nb::arg("handle_id"));

    m.def("dump_model_info", [](uint32_t model_id) {
        dump_model_info(model_id);
    }, nb::arg("model_id"));

    m.def("get_resistive_mask", [](uint32_t model_id) {
        size_t n = get_resistive_mask_len(model_id);
        std::vector<uint8_t> buf(n, 0);
        if (n > 0) get_resistive_mask_ffi(model_id, buf.data());
        nb::list result;
        for (size_t i = 0; i < n; i++) result.append(buf[i] != 0);
        return result;
    }, nb::arg("model_id"));

    auto fetch_pairs = [](size_t n, auto&& filler) {
        std::vector<uint32_t> buf(n * 2, 0);
        if (n > 0) filler(buf.data());
        nb::list result;
        for (size_t i = 0; i < n; i++) {
            result.append(nb::make_tuple((int)buf[i * 2], (int)buf[i * 2 + 1]));
        }
        return result;
    };

    m.def("get_resist_jac_pairs", [fetch_pairs](uint32_t model_id) {
        size_t n = get_resist_jac_pairs_len(model_id);
        return fetch_pairs(n, [=](uint32_t* out) { get_resist_jac_pairs_ffi(model_id, out); });
    }, nb::arg("model_id"));

    m.def("get_react_jac_pairs", [fetch_pairs](uint32_t model_id) {
        size_t n = get_react_jac_pairs_len(model_id);
        return fetch_pairs(n, [=](uint32_t* out) { get_react_jac_pairs_ffi(model_id, out); });
    }, nb::arg("model_id"));

    m.def("get_collapsible_pairs", [fetch_pairs](uint32_t model_id) {
        size_t n = get_collapsible_pairs_len(model_id);
        return fetch_pairs(n, [=](uint32_t* out) { get_collapsible_pairs_ffi(model_id, out); });
    }, nb::arg("model_id"));

    m.def("get_param_flags", [](uint32_t model_id) {
        size_t n = get_param_flags_len(model_id);
        std::vector<uint32_t> buf(n, 0);
        if (n > 0) get_param_flags_ffi(model_id, buf.data());
        nb::list result;
        for (size_t i = 0; i < n; i++) result.append(buf[i]);
        return result;
    }, nb::arg("model_id"));

    m.def("get_param_name", [](uint32_t model_id, size_t idx) {
        size_t n = get_param_name_len(model_id, idx);
        if (n == 0) return std::string();
        std::vector<uint8_t> buf(n, 0);
        get_param_name_ffi(model_id, idx, buf.data());
        return std::string(reinterpret_cast<const char*>(buf.data()), n);
    }, nb::arg("model_id"), nb::arg("idx"));

    m.def("get_param_names", [](uint32_t model_id) {
        size_t total = get_param_flags_len(model_id);
        nb::list result;
        for (size_t i = 0; i < total; i++) {
            size_t n = get_param_name_len(model_id, i);
            if (n == 0) { result.append(std::string()); continue; }
            std::vector<uint8_t> buf(n, 0);
            get_param_name_ffi(model_id, i, buf.data());
            result.append(std::string(reinterpret_cast<const char*>(buf.data()), n));
        }
        return result;
    }, nb::arg("model_id"));
}
