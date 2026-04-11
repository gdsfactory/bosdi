#include <cstdint>
#include <cstddef>
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
// 4. NANOBIND EXPORT
// ---------------------------------------------------------

NB_MODULE(osdi_shim_nb, m) {
    m.doc() = "OSDI Batched Evaluator for JAX";

    nb::class_<ModelMetadata>(m, "ModelMetadata")
        .def_ro("model_id",     &ModelMetadata::model_id)
        .def_ro("num_pins",     &ModelMetadata::num_pins)
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

    m.def("dump_model_info", [](uint32_t model_id) {
        dump_model_info(model_id);
    }, nb::arg("model_id"));
}
