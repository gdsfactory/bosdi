use libloading::{Library, Symbol};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::sync::RwLock;

// ─────────────────────────────────────────────────────────────────────────────
// 0. RAYON DISPATCH POLICY
// ─────────────────────────────────────────────────────────────────────────────
//
// Rayon's task-stealing scheduler has a fixed per-task overhead (∼5–20 µs on
// typical x86_64 desktops) that dominates when the per-item work is small.
// For PSP103 N=9 ring (18 devices) on 8 cores, ``par_chunks_exact_mut`` is
// 25 % SLOWER than sequential — the scheduling cost exceeds the work being
// distributed.  Empirical measurements (2026-04-27, PSP103 ring, OSDI handle
// path):
//
//     N=9   (18 devices)   1-thr 219, 2-thr 193, 4-thr 211, 8-thr 241 µs/step
//     N=27  (54 devices)   1-thr 514, 2-thr 366, 4-thr 329, 8-thr 357 µs/step
//
// Below ~32 devices, sequential beats any parallelism.  Above that, parallelism
// wins but defaults still over-schedule for moderate sizes.  Switch
// ``par_chunks_exact_mut`` for ``chunks_exact_mut`` when ``num_devices`` is
// below ``RAYON_DEVICE_THRESHOLD``.  Override at runtime via the
// ``BOSDI_RAYON_THRESHOLD`` env var (handy for benchmarking).
const RAYON_DEVICE_THRESHOLD_DEFAULT: usize = 32;

fn rayon_threshold() -> usize {
    std::env::var("BOSDI_RAYON_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(RAYON_DEVICE_THRESHOLD_DEFAULT)
}

fn use_parallel(num_devices: usize) -> bool {
    num_devices >= rayon_threshold()
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. OSDI VERSION ABSTRACTION
// ─────────────────────────────────────────────────────────────────────────────

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OsdiVersion {
    V04 = 4,
    V05 = 5,  // placeholder — layout not yet defined
}

impl OsdiVersion {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            4 => Some(Self::V04),
            5 => Some(Self::V05),
            _ => None,
        }
    }
}

/// All byte offsets and flag constants that vary between OSDI standard versions.
/// Field names match OSDI spec terminology.
#[derive(Clone, Copy)]
struct AbiLayout {
    // OsdiDescriptor byte offsets
    desc_num_nodes:            usize,
    desc_num_terminals:        usize,
    desc_num_params:           usize,
    desc_node_mapping_off:     usize,
    desc_num_states:           usize,
    desc_instance_size:        usize,
    desc_model_size:           usize,
    desc_fn_load_resid:        usize,
    desc_fn_load_resid_react:  usize,
    desc_num_resist_jac:       usize,
    desc_num_react_jac:        usize,
    desc_fn_write_jac_resist:  usize,
    desc_fn_write_jac_react:   usize,
    // Collapsible node pairs (internal node → terminal short-circuit when element = 0)
    desc_num_collapsible:      usize,  // +40: count of collapsible pairs
    desc_collapsible:          usize,  // +48: ptr to OsdiCollapsibleNode[]
    // Jacobian entry array (for mapping write_jacobian output to terminal pairs)
    desc_num_jac_entries:      usize,  // +24: total count (resist + react)
    desc_jac_entries:          usize,  // +32: ptr to OsdiJacobianEntry[]
    // param_opvar and access/given_flag function pointers
    desc_param_opvar:          usize,
    desc_fn_access:            usize,
    desc_fn_given_flag_model:  usize,
    desc_fn_given_flag_inst:   usize,
    // OsdiSimInfo shape (for layout-driven construction in future versions)
    sim_info_prev_solve_off:   usize,
    sim_info_flags_off:        usize,
    sim_info_total_size:       usize,
    // Simulation flags
    flag_calc_resist_residual:  u32,
    flag_calc_react_residual:   u32,
    flag_calc_resist_jacobian:  u32,
    flag_calc_react_jacobian:   u32,
    // Exported descriptor symbol
    descriptor_symbol:         &'static [u8],
}

impl AbiLayout {
    fn for_version(ver: OsdiVersion) -> Option<Self> {
        match ver {
            OsdiVersion::V04 => Some(Self::v04()),
            OsdiVersion::V05 => None,  // not yet implemented
        }
    }

    fn v04() -> Self {
        Self {
            desc_num_nodes:            8,
            desc_num_terminals:        12,
            desc_num_params:           76,
            desc_node_mapping_off:     96,
            desc_num_states:           104,
            desc_instance_size:        116,
            desc_model_size:           120,
            desc_fn_load_resid:        168,
            desc_fn_load_resid_react:  176,
            desc_num_resist_jac:       256,
            desc_num_react_jac:        260,
            desc_fn_write_jac_resist:  264,
            desc_fn_write_jac_react:   272,
            desc_num_collapsible:      40,
            desc_collapsible:          48,
            desc_num_jac_entries:      24,
            desc_jac_entries:          32,
            desc_param_opvar:          88,
            desc_fn_access:            128,
            desc_fn_given_flag_model:  240,
            desc_fn_given_flag_inst:   248,
            sim_info_prev_solve_off:   40,
            sim_info_flags_off:        64,
            sim_info_total_size:       72,
            flag_calc_resist_residual:  1,
            flag_calc_react_residual:   2,
            flag_calc_resist_jacobian:  4,
            flag_calc_react_jacobian:   8,
            descriptor_symbol:         b"OSDI_DESCRIPTORS\0",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. OSDI 0.4 ABI TYPES  (must match osdi_0_4.h layout exactly)
// ─────────────────────────────────────────────────────────────────────────────

/// OsdiSimParas — 4 pointers, 32 bytes total.
#[repr(C)]
struct OsdiSimParas {
    names:     *mut *mut i8,
    vals:      *mut f64,
    names_str: *mut *mut i8,
    vals_str:  *mut *mut i8,
}
unsafe impl Send for OsdiSimParas {}
impl OsdiSimParas {
    fn null() -> Self {
        Self {
            names:     std::ptr::null_mut(),
            vals:      std::ptr::null_mut(),
            names_str: std::ptr::null_mut(),
            vals_str:  std::ptr::null_mut(),
        }
    }

    /// Returns an empty-but-valid OsdiSimParas whose names pointer addresses the
    /// provided null sentinel slot.  Models that read `sim_paras->names[0]` to
    /// iterate the parameter list (e.g. the compiled OpenVAF diode) see a null
    /// first entry and skip the lookup — instead of crashing on a null `names`.
    ///
    /// # Safety
    /// `names_sentinel` must remain valid for the lifetime of the returned struct.
    unsafe fn with_null_sentinel(names_sentinel: *mut *mut i8) -> Self {
        Self {
            names:     names_sentinel,
            vals:      std::ptr::null_mut(),
            names_str: std::ptr::null_mut(),
            vals_str:  std::ptr::null_mut(),
        }
    }
}

/// OsdiSimInfo layout (confirmed by disassembly of eval_0):
///   offset  0: paras     (32 bytes)
///   offset 32: abstime   (f64)
///   offset 40: prev_solve (*mut f64)   ← eval reads voltages from here
///   offset 48: prev_state
///   offset 56: next_state
///   offset 64: flags     (u32)        ← eval checks CALC_RESIST_RESIDUAL here
///
/// OSDI 0.4 only. For V05: use layout-driven raw buffer (see AbiLayout::sim_info_*).
#[repr(C)]
struct OsdiSimInfo {
    paras:      OsdiSimParas,
    abstime:    f64,
    prev_solve: *mut f64,
    prev_state: *mut f64,
    next_state: *mut f64,
    flags:      u32,
    _pad:       u32,
}

#[repr(C)]
struct OsdiInitInfo {
    flags:      u32,
    num_errors: u32,
    errors:     *mut (),
}
impl Default for OsdiInitInfo {
    fn default() -> Self {
        Self { flags: 0, num_errors: 0, errors: std::ptr::null_mut() }
    }
}

/// OsdiParamOpvar — 40 bytes on 64-bit (from osdi_0_4.h).
/// Only the `flags` field is needed at runtime; the rest are metadata strings.
#[repr(C)]
struct OsdiParamOpvar {
    name:        *const *const c_char,  // +0   ptr
    num_alias:   u32,                   // +8
    _pad:        u32,                   // +12  alignment padding
    description: *const c_char,         // +16  ptr
    units:       *const c_char,         // +24  ptr
    pub flags:   u32,                   // +32
    pub len:     u32,                   // +36
}
// Safety assertion checked in load_osdi_library via debug_assert.

// ─────────────────────────────────────────────────────────────────────────────
// OSDI simulator callbacks
// ─────────────────────────────────────────────────────────────────────────────

/// No-op osdi_log callback installed into models that export the `osdi_log`
/// symbol.  Some compiled OpenVAF models (e.g. the diode) call this slot when
/// a simulator parameter like `gmin` is not found in `OsdiSimParas.names`.
/// They fall back to 0.0 for the missing value — which is acceptable — but they
/// crash if the function pointer is null (its BSS-initialized default).
///
/// Signature (from OSDI 0.4 calling convention in disassembly):
///   void osdi_log(void* handle, const char* message, uint32_t level)
/// The message string is heap-allocated by the model; we deliberately do not
/// free it here to keep this callback simple and stateless.  The leak is
/// at most one allocation per eval() call and only when gmin is requested.
unsafe extern "C" fn noop_osdi_log(_handle: *mut c_void, _msg: *const c_char, _level: u32) {}

// OSDI 0.4 parameter flags (from osdi_0_4.h)
const PARA_TY_MASK:    u32 = 3;
const PARA_TY_INT:     u32 = 1;
const PARA_TY_STR:     u32 = 2;
const PARA_KIND_MASK:  u32 = 3 << 30;
const PARA_KIND_MODEL: u32 = 0 << 30;
const PARA_KIND_INST:  u32 = 1 << 30;
const PARA_KIND_OPVAR: u32 = 2 << 30;
const ACCESS_FLAG_SET:      u32 = 1;
const ACCESS_FLAG_INSTANCE: u32 = 4;

// ─────────────────────────────────────────────────────────────────────────────
// 3. OSDI FUNCTION POINTER TYPES
// ─────────────────────────────────────────────────────────────────────────────

/// Called once per model to fill default model params.
type SetupModelFn = unsafe extern "C" fn(
    handle:    *mut c_void,
    model:     *mut c_void,
    sim_paras: *mut OsdiSimParas,
    init_info: *mut OsdiInitInfo,
);

/// Called once per instance to precompute conductances into the inst block.
type SetupInstanceFn = unsafe extern "C" fn(
    handle:        *mut c_void,
    inst:          *mut c_void,
    model:         *mut c_void,
    temperature:   f64,
    num_terminals: u32,
    sim_paras:     *mut OsdiSimParas,
    init_info:     *mut OsdiInitInfo,
);

/// Called each Newton step to compute residuals.
/// Returns EVAL_RET_FLAG_* bits (0 = ok).
/// Last arg is *mut c_void (not *mut OsdiSimInfo) so the 0.4 typed path and a
/// future layout-driven raw-buffer path can both use the same fn pointer type.
type EvalFn = unsafe extern "C" fn(
    handle:   *mut c_void,
    inst:     *mut c_void,
    model:    *mut c_void,
    sim_info: *mut c_void,
) -> u32;

/// Adds this element's currents into dst[node_index] (accumulates).
type LoadResidualFn = unsafe extern "C" fn(inst: *mut c_void, model: *mut c_void, dst: *mut f64);

/// Writes the flat Jacobian array (num_resistive_jac doubles) to dst.
type WriteJacobianFn = unsafe extern "C" fn(inst: *mut c_void, model: *mut c_void, dst: *mut f64);

/// Returns a pointer to parameter slot `id` in the model or instance struct.
/// flags: ACCESS_FLAG_SET(1) to write; OR ACCESS_FLAG_INSTANCE(4) for inst params.
type AccessFn = unsafe extern "C" fn(
    inst:  *mut c_void,
    model: *mut c_void,
    id:    u32,
    flags: u32,
) -> *mut c_void;

/// Sets the "parameter given" bit for model parameter `id`. Returns the flag value.
type GivenFlagModelFn = unsafe extern "C" fn(model: *mut c_void, id: u32) -> u32;

/// Sets the "parameter given" bit for instance parameter `id`. Returns the flag value.
type GivenFlagInstFn = unsafe extern "C" fn(inst: *mut c_void, id: u32) -> u32;

// ─────────────────────────────────────────────────────────────────────────────
// 4. OsdiDescriptor field layout  (derived from osdi_0_4.h, 64-bit ABI)
// ─────────────────────────────────────────────────────────────────────────────
//
// struct OsdiDescriptor layout (abbreviated):
//   +  0  name             (ptr 8)
//   +  8  num_nodes        (u32)
//   + 12  num_terminals    (u32)
//   + 16  nodes            (ptr 8)
//   + 24  num_jacobian_entries (u32) + [4 pad]
//   + 32  jacobian_entries (ptr 8)
//   + 40  num_collapsible  (u32) + [4 pad]
//   + 48  collapsible      (ptr 8)
//   + 56  collapsed_offset (u32) + [4 pad]
//   + 64  noise_sources    (ptr 8)
//   + 72  num_noise_src    (u32)
//   + 76  num_params       (u32)   ← total user params (model + instance)
//   + 80  num_instance_params (u32)
//   + 84  num_opvars       (u32)
//   + 88  param_opvar      (ptr 8)
//   + 96  node_mapping_offset       (u32)  ← byte offset of node indices in inst
//   +100  jacobian_ptr_resist_offset (u32)
//   +104  num_states       (u32)
//   +108  state_idx_off    (u32)
//   +112  bound_step_offset (u32)
//   +116  instance_size    (u32)
//   +120  model_size       (u32)
//   +124  [4 pad]
//   +128  access           (fn ptr — NULL in compiled binary, look up by name)
//   +136  setup_model      (fn ptr — NULL in compiled binary, look up by name)
//   +144  setup_instance   (fn ptr — NULL in compiled binary, look up by name)
//   +152  eval             (fn ptr — NULL in compiled binary, look up by name)
//   +160  load_noise       (fn ptr)
//   +168  load_residual_resist (fn ptr)  ← present in descriptor
//   ...
//   +256  num_resistive_jacobian_entries (u32)
//   +260  num_reactive_jacobian_entries  (u32)
//   +264  write_jacobian_array_resist    (fn ptr)  ← present in descriptor
//
// Byte offsets for OSDI 0.4 are stored in AbiLayout::v04().

unsafe fn read_u32(base: *const u8, offset: usize) -> u32 {
    (base.add(offset) as *const u32).read_unaligned()
}

/// Read a function pointer stored in the descriptor (relocated by the dynamic linker).
unsafe fn read_fn<T: Copy>(base: *const u8, offset: usize) -> Option<T> {
    let addr = (base.add(offset) as *const usize).read_unaligned();
    if addr == 0 { None } else { Some(std::mem::transmute_copy(&addr)) }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. LOADED MODEL & REGISTRY
// ─────────────────────────────────────────────────────────────────────────────

struct LoadedOsdi {
    _lib:                    Library,
    layout:                  AbiLayout,
    pub num_terminals:       u32,
    pub num_nodes:           u32,
    pub num_resist_jac:      u32,
    pub num_react_jac:       u32,
    pub instance_size:       usize,
    pub model_size:          usize,
    /// Byte offset within inst where the u32 node-index array begins.
    pub node_map_off:        usize,
    // Functions with NULL descriptor slots — looked up by name:
    pub setup_model:         SetupModelFn,
    pub setup_instance:      SetupInstanceFn,
    pub eval:                EvalFn,
    // Resistive functions read from descriptor (relocated by dynamic linker):
    pub load_residual:       LoadResidualFn,
    pub write_jacobian:      WriteJacobianFn,
    // Reactive functions (may be no-ops for resistive-only models):
    pub load_residual_react:  Option<LoadResidualFn>,
    pub write_jacobian_react: Option<WriteJacobianFn>,
    // Parameter mapping — derived from param_opvar at load time.
    /// access(inst, model, id, flags) → raw pointer to the parameter slot.
    pub access:               AccessFn,
    /// given_flag_model(model, id) — marks model parameter id as user-provided.
    pub given_flag_model:     GivenFlagModelFn,
    /// given_flag_inst(inst, id) — marks instance parameter id as user-provided.
    pub given_flag_inst:      GivenFlagInstFn,
    /// param_opvar[i].flags for i in 0..num_params (kind + type bits).
    pub param_flags:          Vec<u32>,
    // Jacobian entry pairs — used to route write_jacobian output to terminal
    // pairs in the dense num_terminals^2 output matrix.
    /// (node_1, node_2) for each resistive Jacobian entry, in write order.
    pub resist_jac_pairs:     Vec<(u32, u32)>,
    /// (node_1, node_2) for each reactive Jacobian entry, in write order.
    pub react_jac_pairs:      Vec<(u32, u32)>,
    /// Collapsible node pairs from the descriptor. When a coupling element
    /// (e.g. drain series resistance) is zero, these two nodes become
    /// electrically identical. bosdi always collapses them so internal node
    /// voltages match their terminal counterparts.
    pub collapsible_pairs:    Vec<(u32, u32)>,
    /// resistive_mask[out_idx] = true iff output node out_idx appears as the row
    /// in at least one resist_jac_pair (i.e. F[out_idx] can be non-zero at DC).
    /// Nodes where this is false have G[i,:] = 0 always; exposing them as Newton
    /// unknowns makes DC singular — the caller should regularise those rows.
    pub resistive_mask:       Vec<bool>,
    /// Canonical (alias 0) name of each parameter in param_opvar order.
    /// Empty strings for params with a null name pointer (shouldn't happen in
    /// well-formed .osdi binaries).
    pub param_names:          Vec<String>,
    // ── precomputed collapse topology (depends on collapsible_pairs only) ────
    /// node_map[raw_node_idx] → slot_idx after running the collapse algorithm.
    /// Written verbatim into inst_data[node_map_off ..] during setup.
    pub node_map:             Vec<i32>,
    /// slot_to_out[slot_idx] → output index, -1 for phantom slots.
    pub slot_to_out:          Vec<i32>,
    /// Number of distinct slots after collapse (max of node_map + 1).
    pub num_slots:             usize,
    /// num_terminals + num_non_collapsed_internal.
    pub num_all_nodes:         usize,
    /// Number of NQS charge-partition state variables (e.g. 5 for BSIM3v3/4).
    /// Zero for purely resistive models like the diode.
    pub num_states:            usize,
}
unsafe impl Send for LoadedOsdi {}
unsafe impl Sync for LoadedOsdi {}

/// Metadata returned to C++ and then to Python.
#[repr(C)]
pub struct ModelMetadata {
    pub model_id:      u32,
    pub num_pins:      usize,   // = num_terminals (external pins only)
    pub num_nodes:     usize,   // = num_terminals + num_non_collapsed_internal
    pub num_params:    usize,
    pub num_states:    usize,
    pub osdi_version:  u32,
    pub success:       bool,
}

lazy_static::lazy_static! {
    static ref OSDI_REGISTRY: RwLock<HashMap<u32, LoadedOsdi>> =
        RwLock::new(HashMap::new());
    static ref NEXT_MODEL_ID: RwLock<u32> = RwLock::new(1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. PHASE 1: LOADING
// ─────────────────────────────────────────────────────────────────────────────

fn fail() -> ModelMetadata {
    ModelMetadata {
        model_id: 0, num_pins: 0, num_nodes: 0, num_params: 0,
        num_states: 0, osdi_version: 0, success: false,
    }
}

#[no_mangle]
pub extern "C" fn load_osdi_library(path_ptr: *const c_char, version: u32) -> ModelMetadata {
    let ver = match OsdiVersion::from_u32(version) {
        Some(v) => v,
        None    => { eprintln!("OSDI: unknown version {version}"); return fail(); }
    };
    let layout = match AbiLayout::for_version(ver) {
        Some(l) => l,
        None    => { eprintln!("OSDI: version {:?} not yet implemented", ver); return fail(); }
    };

    let path = unsafe {
        assert!(!path_ptr.is_null());
        match CStr::from_ptr(path_ptr).to_str() {
            Ok(s) => s,
            Err(_) => return fail(),
        }
    };

    let lib = match unsafe { Library::new(path) } {
        Ok(l)  => l,
        Err(e) => { eprintln!("OSDI load error: {e}"); return fail(); }
    };

    macro_rules! sym {
        ($lib:expr, $name:expr, $ty:ty) => {{
            let s: Symbol<$ty> = match unsafe { $lib.get($name) } {
                Ok(s)  => s,
                Err(e) => {
                    eprintln!("OSDI missing '{}': {e}",
                              std::str::from_utf8($name).unwrap_or("?"));
                    return fail();
                }
            };
            *s
        }};
    }

    // ── read descriptor metadata ──────────────────────────────────────────────
    let desc: *const u8 = {
        let desc_sym: Symbol<*const u8> =
            match unsafe { lib.get(layout.descriptor_symbol) } {
                Ok(s)  => s,
                Err(e) => { eprintln!("OSDI missing descriptor symbol: {e}"); return fail(); }
            };
        unsafe { *desc_sym }
    };

    let num_nodes      = unsafe { read_u32(desc, layout.desc_num_nodes) };
    let num_terminals  = unsafe { read_u32(desc, layout.desc_num_terminals) };
    let num_params     = unsafe { read_u32(desc, layout.desc_num_params) };
    let node_map_off   = unsafe { read_u32(desc, layout.desc_node_mapping_off) } as usize;
    let num_states     = unsafe { read_u32(desc, layout.desc_num_states) };
    let instance_size  = unsafe { read_u32(desc, layout.desc_instance_size) } as usize;
    let model_size     = unsafe { read_u32(desc, layout.desc_model_size) } as usize;
    let num_resist_jac = unsafe { read_u32(desc, layout.desc_num_resist_jac) };
    let num_react_jac  = unsafe { read_u32(desc, layout.desc_num_react_jac) };

    // ── function pointers present in descriptor (filled by dynamic linker) ───
    let load_residual: LoadResidualFn =
        match unsafe { read_fn(desc, layout.desc_fn_load_resid) } {
            Some(f) => f,
            None    => { eprintln!("OSDI: load_residual_resist fn is null"); return fail(); }
        };
    let write_jacobian: WriteJacobianFn =
        match unsafe { read_fn(desc, layout.desc_fn_write_jac_resist) } {
            Some(f) => f,
            None    => { eprintln!("OSDI: write_jacobian_array_resist fn is null"); return fail(); }
        };
    let load_residual_react: Option<LoadResidualFn> =
        unsafe { read_fn(desc, layout.desc_fn_load_resid_react) };
    let write_jacobian_react: Option<WriteJacobianFn> =
        unsafe { read_fn(desc, layout.desc_fn_write_jac_react) };

    // ── param_opvar: cache flags for each parameter ───────────────────────────
    // Each OsdiParamOpvar entry is 40 bytes; flags is at byte +32 within each entry.
    debug_assert_eq!(std::mem::size_of::<OsdiParamOpvar>(), 40,
                     "OsdiParamOpvar size mismatch — check repr(C) layout");
    let param_opvar_ptr = unsafe {
        let ptr_val = (desc.add(layout.desc_param_opvar) as *const usize).read_unaligned();
        ptr_val as *const u8
    };
    let param_flags: Vec<u32> = (0..num_params as usize).map(|i| {
        unsafe { read_u32(param_opvar_ptr.add(i * 40), 32) }
    }).collect();
    // Also cache the canonical (alias 0) parameter name for each OSDI param.
    // OsdiParamOpvar.name is a char** at offset +0; name[0] is the canonical name.
    let param_names: Vec<String> = (0..num_params as usize).map(|i| unsafe {
        let name_pp = (param_opvar_ptr.add(i * 40) as *const *const *const c_char).read_unaligned();
        if name_pp.is_null() { return String::new(); }
        let name_p = *name_pp;
        if name_p.is_null() { return String::new(); }
        CStr::from_ptr(name_p).to_string_lossy().into_owned()
    }).collect();

    // ── jacobian entry pairs: route write_jacobian output to terminal pairs ───
    // OsdiJacobianEntry: {node_1: u32, node_2: u32, react: u32} = 12 bytes.
    // react=0 → resistive entry; react=1 → reactive entry.
    let num_jac_entries = unsafe { read_u32(desc, layout.desc_num_jac_entries) };
    let jac_entries_ptr = unsafe {
        let ptr_val = (desc.add(layout.desc_jac_entries) as *const usize).read_unaligned();
        ptr_val as *const u8
    };
    let mut resist_jac_pairs: Vec<(u32, u32)> = Vec::new();
    let mut react_jac_pairs:  Vec<(u32, u32)> = Vec::new();
    // OsdiJacobianEntry (16 bytes), per the OSDI 0.4 header:
    //     {OsdiNodePair nodes; uint32_t react_ptr_off; uint32_t flags}
    // Flag bits: _CONST entries carry a pre-computed contribution loaded by
    // load_jacobian_{resist,react}; the non-_CONST RESIST / REACT bits mark
    // entries written per-iteration by write_jacobian_array_{resist,react}.
    // Our scatter reads from the write_jacobian_array outputs, so we index
    // only by the variable (non-_CONST) bits — those counts match the
    // descriptor's num_resistive_jacobian_entries / num_reactive_jacobian_entries.
    // Entries can be dual-flagged (both RESIST and REACT set).
    const RESIST_MASK: u32 = 4;  // JACOBIAN_ENTRY_RESIST
    const REACT_MASK:  u32 = 8;  // JACOBIAN_ENTRY_REACT
    for i in 0..num_jac_entries as usize {
        let node_1 = unsafe { read_u32(jac_entries_ptr.add(i * 16), 0) };
        let node_2 = unsafe { read_u32(jac_entries_ptr.add(i * 16), 4) };
        let flags  = unsafe { read_u32(jac_entries_ptr.add(i * 16), 12) };
        if flags & RESIST_MASK != 0 {
            resist_jac_pairs.push((node_1, node_2));
        }
        if flags & REACT_MASK != 0 {
            react_jac_pairs.push((node_1, node_2));
        }
    }
    // Sanity-check against the descriptor's counts. A mismatch means either the
    // flag layout changed or num_resist_jac/num_react_jac were miscounted.
    debug_assert_eq!(resist_jac_pairs.len(), num_resist_jac as usize,
        "resist_jac_pairs length disagrees with descriptor num_resist_jac");
    debug_assert_eq!(react_jac_pairs.len(),  num_react_jac as usize,
        "react_jac_pairs length disagrees with descriptor num_react_jac");

    // ── collapsible node pairs ────────────────────────────────────────────────
    // OsdiCollapsibleNode: {node_1: u32, node_2: u32} = 8 bytes.
    // When the coupling element between node_1 and node_2 is zero, the simulator
    // collapses them to the same MNA row. bosdi always collapses them so internal
    // node voltages match their terminal counterparts and device current reaches
    // the terminal outputs correctly.
    let num_collapsible = unsafe { read_u32(desc, layout.desc_num_collapsible) };
    let collapsible_pairs: Vec<(u32, u32)> = if num_collapsible > 0 {
        let coll_ptr = unsafe {
            let ptr_val = (desc.add(layout.desc_collapsible) as *const usize).read_unaligned();
            ptr_val as *const u8
        };
        (0..num_collapsible as usize).map(|i| {
            let n1 = unsafe { read_u32(coll_ptr.add(i * 8), 0) };
            let n2 = unsafe { read_u32(coll_ptr.add(i * 8), 4) };
            (n1, n2)
        }).collect()
    } else {
        Vec::new()
    };

    // ── collapse topology + resistive_mask: one pass at load time ───────────
    // Run the deterministic collapse algorithm once per model; all subsequent
    // evaluations reuse these buffers. Phantom slots (numbered but not mapped to
    // by any node, e.g. PSP103's gap between slots 3 and 12) stay at -1 in
    // slot_to_out and are skipped in every scatter operation.
    let (node_map, slot_to_out, num_slots, num_all_nodes) =
        compute_collapse_topology(num_nodes as usize, num_terminals as usize, &collapsible_pairs);

    let resistive_mask: Vec<bool> = {
        let mut mask = vec![false; num_all_nodes];
        for &(n1, _) in &resist_jac_pairs {
            let n1 = n1 as usize;
            if n1 < node_map.len() {
                let slot = node_map[n1] as usize;
                if slot < slot_to_out.len() {
                    let out = slot_to_out[slot];
                    if out >= 0 { mask[out as usize] = true; }
                }
            }
        }
        mask
    };

    // ── access and given_flag function pointers ───────────────────────────────
    let access: AccessFn = match unsafe { read_fn(desc, layout.desc_fn_access) } {
        Some(f) => f,
        None    => { eprintln!("OSDI: access fn is null"); return fail(); }
    };
    unsafe extern "C" fn noop_given_model(_: *mut c_void, _: u32) -> u32 { 0 }
    unsafe extern "C" fn noop_given_inst(_: *mut c_void, _: u32) -> u32 { 0 }
    let given_flag_model: GivenFlagModelFn =
        unsafe { read_fn(desc, layout.desc_fn_given_flag_model) }
            .unwrap_or(noop_given_model);
    let given_flag_inst: GivenFlagInstFn =
        unsafe { read_fn(desc, layout.desc_fn_given_flag_inst) }
            .unwrap_or(noop_given_inst);

    // ── function pointers with NULL descriptor slots — look up by name ────────
    // OpenVAF exports these as `fname_0` (index 0 = first model in the binary).
    let setup_model:    SetupModelFn    = sym!(lib, b"setup_model_0\0",    SetupModelFn);
    let setup_instance: SetupInstanceFn = sym!(lib, b"setup_instance_0\0", SetupInstanceFn);
    let eval:           EvalFn          = sym!(lib, b"eval_0\0",           EvalFn);

    // ── install simulator callbacks if the model exports them ─────────────────
    // `osdi_log` is a BSS function-pointer slot (null by default) that some
    // models call to report missing simulator parameters (e.g. $simparam("gmin")).
    // Without a callback they crash; a no-op is sufficient since we fall back to
    // 0.0 for missing params anyway.
    unsafe {
        if let Ok(sym) = lib.get::<*mut c_void>(b"osdi_log\0") {
            // *sym is the BSS address cast as *mut c_void; reinterpret as *mut fn ptr.
            let slot = *sym as *mut unsafe extern "C" fn(*mut c_void, *const c_char, u32);
            if !slot.is_null() {
                *slot = noop_osdi_log;
            }
        }
    }

    let model_id = {
        let mut id = NEXT_MODEL_ID.write().unwrap();
        let cur = *id;
        *id += 1;
        cur
    };

    OSDI_REGISTRY.write().unwrap().insert(model_id, LoadedOsdi {
        _lib: lib,
        layout,
        num_terminals,
        num_nodes,
        num_resist_jac,
        num_react_jac,
        instance_size,
        model_size,
        node_map_off,
        setup_model,
        setup_instance,
        eval,
        load_residual,
        write_jacobian,
        load_residual_react,
        write_jacobian_react,
        access,
        given_flag_model,
        given_flag_inst,
        param_flags,
        resist_jac_pairs,
        react_jac_pairs,
        collapsible_pairs,
        resistive_mask,
        param_names,
        node_map,
        slot_to_out,
        num_slots,
        num_all_nodes,
        num_states: num_states as usize,
    });

    ModelMetadata {
        model_id,
        num_pins:     num_terminals as usize,
        num_nodes:    num_all_nodes,
        num_params:   num_params as usize,
        num_states:   num_states as usize,
        osdi_version: version,
        success:      true,
    }
}

/// Deterministic collapse: fold `collapsible_pairs` into an index map, build
/// the slot→output index array, and report the total distinct-slots count.
/// Shared by the loader (for resistive_mask) and the evaluator (via cache).
fn compute_collapse_topology(
    num_nodes: usize,
    num_terminals: usize,
    collapsible_pairs: &[(u32, u32)],
) -> (Vec<i32>, Vec<i32>, usize, usize) {
    let mut nm: Vec<i32> = (0..num_nodes as i32).collect();
    for &(n1, n2) in collapsible_pairs {
        let (n1, n2) = (n1 as usize, n2 as usize);
        if n1 >= num_nodes || n2 >= num_nodes { continue; }
        let s1 = nm[n1]; let s2 = nm[n2];
        let merged = s1.min(s2); let higher = s1.max(s2);
        for s in nm.iter_mut() { if *s == higher { *s = merged; } }
    }
    let num_slots = nm.iter().map(|&s| s as usize + 1).max().unwrap_or(num_terminals);

    let mut slot_occupied = vec![false; num_slots];
    for &s in &nm { slot_occupied[s as usize] = true; }

    let mut slot_to_out = vec![-1i32; num_slots];
    for t in 0..num_terminals {
        slot_to_out[nm[t] as usize] = t as i32;
    }
    let mut next_out = num_terminals;
    for slot in 0..num_slots {
        if slot_occupied[slot] && slot_to_out[slot] < 0 {
            slot_to_out[slot] = next_out as i32;
            next_out += 1;
        }
    }
    let num_all_nodes = next_out;

    (nm, slot_to_out, num_slots, num_all_nodes)
}

// ─────────────────────────────────────────────────────────────────────────────
// 6b. RESISTIVE MASK FFI  (separate from ModelMetadata — Vec<bool> is not repr(C))
// ─────────────────────────────────────────────────────────────────────────────

/// Returns the length of the resistive_mask for model_id (= num_nodes).
/// Returns 0 for unknown model IDs.
#[no_mangle]
pub extern "C" fn get_resistive_mask_len(model_id: u32) -> usize {
    OSDI_REGISTRY.read().unwrap()
        .get(&model_id)
        .map(|m| m.resistive_mask.len())
        .unwrap_or(0)
}

/// Copies resistive_mask[0..n] into out[0..n] as u8 (1 = true, 0 = false).
/// Caller must ensure out points to at least get_resistive_mask_len() bytes.
#[no_mangle]
pub extern "C" fn get_resistive_mask_ffi(model_id: u32, out: *mut u8) {
    if let Some(m) = OSDI_REGISTRY.read().unwrap().get(&model_id) {
        for (i, &b) in m.resistive_mask.iter().enumerate() {
            unsafe { *out.add(i) = b as u8; }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6c. STRUCTURAL INTROSPECTION FFI
// Jacobian and collapsible pairs use (node_1, node_2) in raw OSDI indices
// (0..num_nodes, before collapse). Paired 2-call pattern: len first, then fill.
// ─────────────────────────────────────────────────────────────────────────────

fn copy_pairs(pairs: &[(u32, u32)], out: *mut u32) {
    for (i, &(a, b)) in pairs.iter().enumerate() {
        unsafe {
            *out.add(i * 2) = a;
            *out.add(i * 2 + 1) = b;
        }
    }
}

#[no_mangle]
pub extern "C" fn get_resist_jac_pairs_len(model_id: u32) -> usize {
    OSDI_REGISTRY.read().unwrap().get(&model_id).map(|m| m.resist_jac_pairs.len()).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn get_resist_jac_pairs_ffi(model_id: u32, out: *mut u32) {
    if let Some(m) = OSDI_REGISTRY.read().unwrap().get(&model_id) {
        copy_pairs(&m.resist_jac_pairs, out);
    }
}

#[no_mangle]
pub extern "C" fn get_react_jac_pairs_len(model_id: u32) -> usize {
    OSDI_REGISTRY.read().unwrap().get(&model_id).map(|m| m.react_jac_pairs.len()).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn get_react_jac_pairs_ffi(model_id: u32, out: *mut u32) {
    if let Some(m) = OSDI_REGISTRY.read().unwrap().get(&model_id) {
        copy_pairs(&m.react_jac_pairs, out);
    }
}

#[no_mangle]
pub extern "C" fn get_collapsible_pairs_len(model_id: u32) -> usize {
    OSDI_REGISTRY.read().unwrap().get(&model_id).map(|m| m.collapsible_pairs.len()).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn get_collapsible_pairs_ffi(model_id: u32, out: *mut u32) {
    if let Some(m) = OSDI_REGISTRY.read().unwrap().get(&model_id) {
        copy_pairs(&m.collapsible_pairs, out);
    }
}

/// Get parameter name length (in bytes, excluding NUL) for bounds-checking before get_param_name.
/// Returns 0 if model_id or idx is out of range, or the param has no name.
#[no_mangle]
pub extern "C" fn get_param_name_len(model_id: u32, idx: usize) -> usize {
    OSDI_REGISTRY.read().unwrap().get(&model_id)
        .and_then(|m| m.param_names.get(idx))
        .map(|s| s.len())
        .unwrap_or(0)
}

/// Copy the UTF-8 bytes of the param name into `out` (no NUL terminator).
/// `out` must be at least get_param_name_len bytes.
#[no_mangle]
pub extern "C" fn get_param_name_ffi(model_id: u32, idx: usize, out: *mut u8) {
    if let Some(m) = OSDI_REGISTRY.read().unwrap().get(&model_id) {
        if let Some(s) = m.param_names.get(idx) {
            let bytes = s.as_bytes();
            for (i, &b) in bytes.iter().enumerate() {
                unsafe { *out.add(i) = b; }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn get_param_flags_len(model_id: u32) -> usize {
    OSDI_REGISTRY.read().unwrap().get(&model_id).map(|m| m.param_flags.len()).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn get_param_flags_ffi(model_id: u32, out: *mut u32) {
    if let Some(m) = OSDI_REGISTRY.read().unwrap().get(&model_id) {
        for (i, &f) in m.param_flags.iter().enumerate() {
            unsafe { *out.add(i) = f; }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. PHASE 2: SINGLE-DEVICE EVALUATION
// ─────────────────────────────────────────────────────────────────────────────
//
// OSDI 0.4 evaluation protocol. Per device, the sequence is:
//   1. Write params via access() + given_flag_*()       ─┐
//   2. setup_model()                                      │ setup_device()
//   3. setup_instance()                                   ┘  (once per param change)
//   4. eval()                                            ─┐
//   5. load_residual / write_jacobian (and reactive pair)─┘ eval_device_from_setup()
//                                                            (once per Newton iter)
//
// Splitting lets callers (via the OsdiBatchHandle API) keep a snapshot of
// (model_data, inst_data) after step 3 and re-run steps 4–5 any number of times
// with different voltages — the expensive setup_instance doesn't repeat.

/// Grow-to-fit scratch buffers reused across evaluations by one Rayon worker.
/// `clear() + resize(n, v)` preserves capacity — only the first iter per thread
/// allocates.
#[derive(Default)]
struct EvalScratch {
    /// Opaque model struct (written by setup_model, read by eval).
    model_data: Vec<u8>,
    /// Opaque instance struct (written by setup_instance + eval).
    inst_data: Vec<u8>,
    small: SmallScratch,
}

#[derive(Default)]
struct SmallScratch {
    /// num_slots f64s — voltages indexed by post-collapse slot.
    vol_buf: Vec<f64>,
    /// num_slots f64s — written by load_residual / load_residual_react.
    node_buf: Vec<f64>,
    /// num_resist_jac or num_react_jac f64s — written by write_jacobian_*.
    jac_buf: Vec<f64>,
    /// 2 * num_states f64s — prev_state (first half) and next_state (second half)
    /// for stateful models (BSIM3v3/4, etc.).  Both halves are zeroed before each
    /// eval so DC analysis starts from zero charges.
    state_buf: Vec<f64>,
}

thread_local! {
    static SCRATCH: RefCell<EvalScratch> = RefCell::new(EvalScratch::default());
}

/// Phase 1: write params, run setup_model + setup_instance, write node_map
/// into inst_data. After this, (model_data, inst_data) is a valid device
/// instance and can be eval'd any number of times.
///
/// Caller is responsible for sizing: model_data = m.model_size bytes (zeroed),
/// inst_data = m.instance_size bytes (zeroed).
fn setup_device(m: &LoadedOsdi, param: &[f64], model_data: &mut [u8], inst_data: &mut [u8]) {
    debug_assert_eq!(model_data.len(), m.model_size);
    debug_assert_eq!(inst_data.len(), m.instance_size);

    // Write the cached post-collapse node_map into inst_data.
    for (i, &slot) in m.node_map.iter().enumerate() {
        unsafe {
            *(inst_data.as_mut_ptr().add(m.node_map_off + i * 4) as *mut i32) = slot;
        }
    }

    let model_ptr = model_data.as_mut_ptr() as *mut c_void;
    let inst_ptr  = inst_data.as_mut_ptr() as *mut c_void;

    // Helper: write one param via access() + given_flag_*().
    let write_param = |i: usize, val: f64, kind: u32, ty: u32| {
        if val.is_nan()            { return; }  // NaN → Verilog-A default
        if kind == PARA_KIND_OPVAR { return; }  // output-only
        if ty   == PARA_TY_STR     { return; }  // can't map str from f64 array

        let access_flags = if kind == PARA_KIND_MODEL {
            ACCESS_FLAG_SET
        } else {
            ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE
        };

        let ptr = unsafe { (m.access)(inst_ptr, model_ptr, i as u32, access_flags) };
        if ptr.is_null() { return; }

        unsafe {
            if ty == PARA_TY_INT {
                *(ptr as *mut i32) = val as i32;
            } else {
                *(ptr as *mut f64) = val;
            }
        }

        if kind == PARA_KIND_MODEL {
            unsafe { (m.given_flag_model)(model_ptr, i as u32); }
        } else {
            unsafe { (m.given_flag_inst)(inst_ptr, i as u32); }
        }
    };

    // Pass 1: model params
    for (i, &val) in param.iter().enumerate() {
        let flags = match m.param_flags.get(i) { Some(&f) => f, None => break };
        let kind = flags & PARA_KIND_MASK;
        if kind != PARA_KIND_MODEL { continue; }
        write_param(i, val, kind, flags & PARA_TY_MASK);
    }

    let mut init1 = OsdiInitInfo::default();
    unsafe {
        (m.setup_model)(std::ptr::null_mut(), model_ptr, std::ptr::null_mut(), &mut init1);
    }

    // Pass 2: instance params (after setup_model, before setup_instance)
    for (i, &val) in param.iter().enumerate() {
        let flags = match m.param_flags.get(i) { Some(&f) => f, None => break };
        let kind = flags & PARA_KIND_MASK;
        if kind != PARA_KIND_INST { continue; }
        write_param(i, val, kind, flags & PARA_TY_MASK);
    }

    let mut init2 = OsdiInitInfo::default();
    unsafe {
        (m.setup_instance)(
            std::ptr::null_mut(), inst_ptr, model_ptr,
            300.0, m.num_terminals, std::ptr::null_mut(), &mut init2,
        );
    }
}

/// Phase 2: given an already-set-up (model_data, inst_data), set voltages,
/// call eval(), and extract outputs. May mutate inst_data (OSDI models write
/// op-vars during eval) — callers that reuse setup state across iterations
/// should snapshot inst_data before the first eval and restore before each
/// subsequent eval.
fn eval_device_from_setup(
    m: &LoadedOsdi,
    model_data: &mut [u8],
    inst_data: &mut [u8],
    vol: &[f64],
    cur: &mut [f64],
    cond: &mut [f64],
    chg: &mut [f64],
    cap: &mut [f64],
    residual_only: bool,
    scratch: &mut SmallScratch,
) {
    let num_slots = m.num_slots;
    let num_all_nodes = m.num_all_nodes;

    let mut flags = 0u32;
    if m.num_resist_jac > 0 {
        flags |= m.layout.flag_calc_resist_residual;
        if !residual_only { flags |= m.layout.flag_calc_resist_jacobian; }
    }
    if m.num_react_jac > 0 {
        flags |= m.layout.flag_calc_react_residual;
        if !residual_only { flags |= m.layout.flag_calc_react_jacobian; }
    }

    // Voltages indexed by slot (many-to-one collapse preserves terminal values).
    scratch.vol_buf.clear();
    scratch.vol_buf.resize(num_slots, 0.0);
    for slot in 0..num_slots {
        let out_idx = m.slot_to_out[slot];
        if out_idx >= 0 && (out_idx as usize) < vol.len() {
            scratch.vol_buf[slot] = vol[out_idx as usize];
        }
    }

    let model_ptr = model_data.as_mut_ptr() as *mut c_void;
    let inst_ptr  = inst_data.as_mut_ptr() as *mut c_void;

    let mut names_sentinel: *mut i8 = std::ptr::null_mut();
    let sim_paras = unsafe { OsdiSimParas::with_null_sentinel(&mut names_sentinel) };

    // Stateful models (BSIM3v3/4, etc.) write NQS charge-partition states to
    // next_state and may read prev_state.  Pass zero-initialised scratch buffers
    // so the model has valid pointers.  For DC analysis the states start at zero
    // and the written values are discarded; for transient the caller propagates
    // them explicitly (future work).
    let (prev_state_ptr, next_state_ptr) = if m.num_states > 0 {
        scratch.state_buf.clear();
        scratch.state_buf.resize(2 * m.num_states as usize, 0.0);
        let mid = m.num_states as usize;
        (scratch.state_buf[..mid].as_mut_ptr(), scratch.state_buf[mid..].as_mut_ptr())
    } else {
        (std::ptr::null_mut(), std::ptr::null_mut())
    };

    let mut sim_info = OsdiSimInfo {
        paras:      sim_paras,
        abstime:    0.0,
        prev_solve: scratch.vol_buf.as_mut_ptr(),
        prev_state: prev_state_ptr,
        next_state: next_state_ptr,
        flags,
        _pad:       0,
    };
    unsafe {
        (m.eval)(
            std::ptr::null_mut(), inst_ptr, model_ptr,
            &mut sim_info as *mut _ as *mut c_void,
        );
    }

    // Resistive extraction
    if m.num_resist_jac > 0 {
        scratch.node_buf.clear();
        scratch.node_buf.resize(num_slots, 0.0);
        unsafe { (m.load_residual)(inst_ptr, model_ptr, scratch.node_buf.as_mut_ptr()); }
        for slot in 0..num_slots {
            let out = m.slot_to_out[slot];
            if out >= 0 {
                cur[out as usize] = scratch.node_buf[slot];
            }
        }

        if !residual_only {
            scratch.jac_buf.clear();
            scratch.jac_buf.resize(m.num_resist_jac as usize, 0.0);
            unsafe { (m.write_jacobian)(inst_ptr, model_ptr, scratch.jac_buf.as_mut_ptr()); }
            for (idx, &(n1, n2)) in m.resist_jac_pairs.iter().enumerate() {
                let s1 = m.node_map.get(n1 as usize).copied().unwrap_or(-1);
                let s2 = m.node_map.get(n2 as usize).copied().unwrap_or(-1);
                if s1 >= 0 && s2 >= 0 {
                    let o1 = m.slot_to_out[s1 as usize];
                    let o2 = m.slot_to_out[s2 as usize];
                    if o1 >= 0 && o2 >= 0 {
                        cond[o1 as usize * num_all_nodes + o2 as usize] += scratch.jac_buf[idx];
                    }
                }
            }
        }
    }

    // Reactive extraction
    if m.num_react_jac > 0 {
        if let Some(lr) = m.load_residual_react {
            scratch.node_buf.clear();
            scratch.node_buf.resize(num_slots, 0.0);
            unsafe { lr(inst_ptr, model_ptr, scratch.node_buf.as_mut_ptr()); }
            for slot in 0..num_slots {
                let out = m.slot_to_out[slot];
                if out >= 0 {
                    chg[out as usize] = scratch.node_buf[slot];
                }
            }
        }
        if !residual_only {
            if let Some(wj) = m.write_jacobian_react {
                scratch.jac_buf.clear();
                scratch.jac_buf.resize(m.num_react_jac as usize, 0.0);
                unsafe { wj(inst_ptr, model_ptr, scratch.jac_buf.as_mut_ptr()); }
                for (idx, &(n1, n2)) in m.react_jac_pairs.iter().enumerate() {
                    let s1 = m.node_map.get(n1 as usize).copied().unwrap_or(-1);
                    let s2 = m.node_map.get(n2 as usize).copied().unwrap_or(-1);
                    if s1 >= 0 && s2 >= 0 {
                        let o1 = m.slot_to_out[s1 as usize];
                        let o2 = m.slot_to_out[s2 as usize];
                        if o1 >= 0 && o2 >= 0 {
                            cap[o1 as usize * num_all_nodes + o2 as usize] += scratch.jac_buf[idx];
                        }
                    }
                }
            }
        }
    }
}

/// Stateless eval: re-run setup every call using thread-local scratch.
/// Used by the legacy `batched_osdi_eval_ffi` / `batched_osdi_residual_eval_ffi`
/// paths. For Newton inner loops where params are fixed, prefer the handle API
/// which runs setup once and reuses.
fn eval_one_device(
    m: &LoadedOsdi,
    vol: &[f64],
    param: &[f64],
    cur: &mut [f64],
    cond: &mut [f64],
    chg: &mut [f64],
    cap: &mut [f64],
    residual_only: bool,
) {
    SCRATCH.with(|s| {
        let mut scratch = s.borrow_mut();
        scratch.model_data.clear();
        scratch.model_data.resize(m.model_size, 0);
        scratch.inst_data.clear();
        scratch.inst_data.resize(m.instance_size, 0);

        let EvalScratch { model_data, inst_data, small } = &mut *scratch;
        setup_device(m, param, model_data, inst_data);
        eval_device_from_setup(
            m, model_data, inst_data, vol, cur, cond, chg, cap, residual_only, small,
        );
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. PHASE 2: BATCHED FFI ENTRY POINT (called from C++ XLA handler)
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn batched_osdi_eval_ffi(
    model_id:         u32,
    num_devices:      usize,
    num_pins:         usize,   // = num_nodes (terminals + internal); Python passes meta.num_nodes
    num_params:       usize,
    num_states:       usize,
    voltages_ptr:     *const f64,
    params_ptr:       *const f64,
    _old_state_ptr:   *const f64,
    currents_ptr:     *mut f64,
    conductances_ptr: *mut f64,
    charges_ptr:      *mut f64,
    capacitances_ptr: *mut f64,
    new_state_ptr:    *mut f64,
) {
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = registry.get(&model_id).expect("Unknown OSDI model ID");

    // jac_size always matches Python's allocation: num_pins^2.
    let jac_size = num_pins * num_pins;

    let voltages = unsafe { std::slice::from_raw_parts(voltages_ptr, num_devices * num_pins) };
    let params   = unsafe { std::slice::from_raw_parts(params_ptr,   num_devices * num_params) };

    let currents     = unsafe { std::slice::from_raw_parts_mut(currents_ptr,     num_devices * num_pins) };
    let conductances = unsafe { std::slice::from_raw_parts_mut(conductances_ptr, num_devices * jac_size) };
    let charges      = unsafe { std::slice::from_raw_parts_mut(charges_ptr,      num_devices * num_pins) };
    let capacitances = unsafe { std::slice::from_raw_parts_mut(capacitances_ptr, num_devices * jac_size) };

    // State slices (num_states=0 for the resistor — avoid par_chunks_exact(0) panic).
    if num_states == 0 {
        // Adaptive parallelism: small ``num_devices`` runs sequentially to
        // avoid Rayon's per-task scheduling overhead — see the
        // ``RAYON_DEVICE_THRESHOLD_DEFAULT`` comment for measurements.
        if use_parallel(num_devices) {
            currents.par_chunks_exact_mut(num_pins)
                .zip(conductances.par_chunks_exact_mut(jac_size))
                .zip(charges.par_chunks_exact_mut(num_pins))
                .zip(capacitances.par_chunks_exact_mut(jac_size))
                .zip(voltages.par_chunks_exact(num_pins))
                .zip(params.par_chunks_exact(num_params))
                .for_each(|(((((cur, cond), chg), cap), vol), param)| {
                    cur.fill(0.0);
                    cond.fill(0.0);
                    chg.fill(0.0);
                    cap.fill(0.0);
                    eval_one_device(m, vol, param, cur, cond, chg, cap, false);
                });
        } else {
            currents.chunks_exact_mut(num_pins)
                .zip(conductances.chunks_exact_mut(jac_size))
                .zip(charges.chunks_exact_mut(num_pins))
                .zip(capacitances.chunks_exact_mut(jac_size))
                .zip(voltages.chunks_exact(num_pins))
                .zip(params.chunks_exact(num_params))
                .for_each(|(((((cur, cond), chg), cap), vol), param)| {
                    cur.fill(0.0);
                    cond.fill(0.0);
                    chg.fill(0.0);
                    cap.fill(0.0);
                    eval_one_device(m, vol, param, cur, cond, chg, cap, false);
                });
        }
    } else {
        // Stateful models: initialise state to zero and evaluate.
        // States (NQS charge-partition variables in BSIM3v3/4, etc.) are not yet
        // carried across Newton steps — they start at zero each call.  This gives
        // correct resistive (DC / low-frequency) behaviour and is sufficient for
        // operating-point finding and ring-oscillator frequency benchmarking.
        // Reactive (capacitive) accuracy requires state propagation; that is a
        // future enhancement.
        let new_state = unsafe {
            std::slice::from_raw_parts_mut(new_state_ptr, num_devices * num_states)
        };
        new_state.fill(0.0);

        if use_parallel(num_devices) {
            currents.par_chunks_exact_mut(num_pins)
                .zip(conductances.par_chunks_exact_mut(jac_size))
                .zip(charges.par_chunks_exact_mut(num_pins))
                .zip(capacitances.par_chunks_exact_mut(jac_size))
                .zip(voltages.par_chunks_exact(num_pins))
                .zip(params.par_chunks_exact(num_params))
                .for_each(|(((((cur, cond), chg), cap), vol), param)| {
                    cur.fill(0.0);
                    cond.fill(0.0);
                    chg.fill(0.0);
                    cap.fill(0.0);
                    eval_one_device(m, vol, param, cur, cond, chg, cap, false);
                });
        } else {
            currents.chunks_exact_mut(num_pins)
                .zip(conductances.chunks_exact_mut(jac_size))
                .zip(charges.chunks_exact_mut(num_pins))
                .zip(capacitances.chunks_exact_mut(jac_size))
                .zip(voltages.chunks_exact(num_pins))
                .zip(params.chunks_exact(num_params))
                .for_each(|(((((cur, cond), chg), cap), vol), param)| {
                    cur.fill(0.0);
                    cond.fill(0.0);
                    chg.fill(0.0);
                    cap.fill(0.0);
                    eval_one_device(m, vol, param, cur, cond, chg, cap, false);
                });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8b. PHASE 2 (residual-only): skip Jacobian flags + write_jacobian_* calls
// ─────────────────────────────────────────────────────────────────────────────
//
// Intended for Newton inner iterations with a frozen Jacobian: we still need
// the residual (cur) and charge (chg) to evaluate F(x), but the conductance
// and capacitance stamps from the first iter can be reused. For BSIM4/PSP103-
// sized models, the ∂/∂V pass is roughly half of per-device OSDI work.

#[no_mangle]
pub extern "C" fn batched_osdi_residual_eval_ffi(
    model_id:         u32,
    num_devices:      usize,
    num_pins:         usize,
    num_params:       usize,
    num_states:       usize,
    voltages_ptr:     *const f64,
    params_ptr:       *const f64,
    _old_state_ptr:   *const f64,
    currents_ptr:     *mut f64,
    charges_ptr:      *mut f64,
    new_state_ptr:    *mut f64,
) {
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = registry.get(&model_id).expect("Unknown OSDI model ID");

    let voltages = unsafe { std::slice::from_raw_parts(voltages_ptr, num_devices * num_pins) };
    let params   = unsafe { std::slice::from_raw_parts(params_ptr,   num_devices * num_params) };

    let currents = unsafe { std::slice::from_raw_parts_mut(currents_ptr, num_devices * num_pins) };
    let charges  = unsafe { std::slice::from_raw_parts_mut(charges_ptr,  num_devices * num_pins) };

    if num_states == 0 {
        // See ``use_parallel`` rationale at top of file.
        if use_parallel(num_devices) {
            currents.par_chunks_exact_mut(num_pins)
                .zip(charges.par_chunks_exact_mut(num_pins))
                .zip(voltages.par_chunks_exact(num_pins))
                .zip(params.par_chunks_exact(num_params))
                .for_each(|(((cur, chg), vol), param)| {
                    cur.fill(0.0);
                    chg.fill(0.0);
                    // Dummy Jacobian buffers — eval_one_device ignores them when
                    // residual_only = true (neither the CALC_*_JACOBIAN flag nor
                    // the write_jacobian_* calls fire).
                    let mut cond_ignored: [f64; 0] = [];
                    let mut cap_ignored:  [f64; 0] = [];
                    eval_one_device(m, vol, param, cur, &mut cond_ignored, chg, &mut cap_ignored, true);
                });
        } else {
            currents.chunks_exact_mut(num_pins)
                .zip(charges.chunks_exact_mut(num_pins))
                .zip(voltages.chunks_exact(num_pins))
                .zip(params.chunks_exact(num_params))
                .for_each(|(((cur, chg), vol), param)| {
                    cur.fill(0.0);
                    chg.fill(0.0);
                    let mut cond_ignored: [f64; 0] = [];
                    let mut cap_ignored:  [f64; 0] = [];
                    eval_one_device(m, vol, param, cur, &mut cond_ignored, chg, &mut cap_ignored, true);
                });
        }
    } else {
        let new_state = unsafe {
            std::slice::from_raw_parts_mut(new_state_ptr, num_devices * num_states)
        };
        new_state.fill(0.0);
        if use_parallel(num_devices) {
            currents.par_chunks_exact_mut(num_pins)
                .zip(charges.par_chunks_exact_mut(num_pins))
                .zip(voltages.par_chunks_exact(num_pins))
                .zip(params.par_chunks_exact(num_params))
                .for_each(|(((cur, chg), vol), param)| {
                    cur.fill(0.0);
                    chg.fill(0.0);
                    let mut cond_ignored: [f64; 0] = [];
                    let mut cap_ignored:  [f64; 0] = [];
                    eval_one_device(m, vol, param, cur, &mut cond_ignored, chg, &mut cap_ignored, true);
                });
        } else {
            currents.chunks_exact_mut(num_pins)
                .zip(charges.chunks_exact_mut(num_pins))
                .zip(voltages.chunks_exact(num_pins))
                .zip(params.chunks_exact(num_params))
                .for_each(|(((cur, chg), vol), param)| {
                    cur.fill(0.0);
                    chg.fill(0.0);
                    let mut cond_ignored: [f64; 0] = [];
                    let mut cap_ignored:  [f64; 0] = [];
                    eval_one_device(m, vol, param, cur, &mut cond_ignored, chg, &mut cap_ignored, true);
                });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8c. BATCH HANDLE API — pay setup_model + setup_instance once per param change
// ─────────────────────────────────────────────────────────────────────────────
//
// Inside a Newton inner loop the parameters are fixed; only voltages change
// per iter. The stateless path (`batched_osdi_eval_ffi`) re-runs setup_model +
// setup_instance every call — for BSIM4/PSP103-sized models that's tens of
// microseconds of wasted setup per device per iter.
//
// The handle API splits the workflow:
//   1. `osdi_setup_batch_ffi(model_id, params)` runs setup once for all
//      devices in parallel, stores a flat (model_data ⧺ inst_data) snapshot
//      in HANDLES, returns a handle id.
//   2. `batched_osdi_eval_handle_ffi(handle_id, voltages, ...)` copies the
//      snapshot into thread-local scratch per device and runs eval — no
//      setup_instance, no access() pointer-walks, just the eval + extract.
//   3. `batched_osdi_residual_eval_handle_ffi` is the same but residual-only.
//   4. `osdi_free_handle_ffi(handle_id)` drops the entry. Callers (Python
//      `OsdiBatchHandle.__del__`) are responsible for lifetime.

struct OsdiBatchHandle {
    model_id:        u32,
    num_devices:     usize,
    /// Flat: num_devices * m.model_size bytes (one model snapshot per device).
    model_data_flat: Vec<u8>,
    /// Flat: num_devices * m.instance_size bytes (one inst snapshot per device).
    inst_data_flat:  Vec<u8>,
}

lazy_static::lazy_static! {
    static ref HANDLES: RwLock<HashMap<u64, OsdiBatchHandle>> =
        RwLock::new(HashMap::new());
    static ref NEXT_HANDLE_ID: RwLock<u64> = RwLock::new(1);
}

/// Runs setup_model + setup_instance in parallel across `num_devices` and
/// stores the post-setup (model_data, inst_data) snapshots in the handle
/// registry. Returns the new handle id, or 0 if the model id is unknown.
#[no_mangle]
pub extern "C" fn osdi_setup_batch_ffi(
    model_id:    u32,
    num_devices: usize,
    num_params:  usize,
    params_ptr:  *const f64,
) -> u64 {
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = match registry.get(&model_id) {
        Some(m) => m,
        None => { eprintln!("osdi_setup_batch: unknown model_id={model_id}"); return 0; }
    };

    let params = unsafe { std::slice::from_raw_parts(params_ptr, num_devices * num_params) };

    let mut model_data_flat = vec![0u8; num_devices * m.model_size];
    let mut inst_data_flat  = vec![0u8; num_devices * m.instance_size];

    if use_parallel(num_devices) {
        model_data_flat.par_chunks_exact_mut(m.model_size)
            .zip(inst_data_flat.par_chunks_exact_mut(m.instance_size))
            .zip(params.par_chunks_exact(num_params))
            .for_each(|((model_data, inst_data), param)| {
                setup_device(m, param, model_data, inst_data);
            });
    } else {
        model_data_flat.chunks_exact_mut(m.model_size)
            .zip(inst_data_flat.chunks_exact_mut(m.instance_size))
            .zip(params.chunks_exact(num_params))
            .for_each(|((model_data, inst_data), param)| {
                setup_device(m, param, model_data, inst_data);
            });
    }

    drop(registry);

    let handle_id = {
        let mut id = NEXT_HANDLE_ID.write().unwrap();
        let cur = *id;
        *id += 1;
        cur
    };

    HANDLES.write().unwrap().insert(handle_id, OsdiBatchHandle {
        model_id,
        num_devices,
        model_data_flat,
        inst_data_flat,
    });

    handle_id
}

/// Removes the handle from the registry, freeing its memory. Safe to call with
/// an unknown id (silent no-op). Python's OsdiBatchHandle.__del__ invokes this.
#[no_mangle]
pub extern "C" fn osdi_free_handle_ffi(handle_id: u64) {
    HANDLES.write().unwrap().remove(&handle_id);
}

/// Returns the num_devices stored in the handle, or 0 if unknown. Used by
/// Python to validate output-buffer sizes before a handle-based eval.
#[no_mangle]
pub extern "C" fn osdi_handle_num_devices(handle_id: u64) -> usize {
    HANDLES.read().unwrap()
        .get(&handle_id)
        .map(|h| h.num_devices)
        .unwrap_or(0)
}

/// Full-stamp batched eval reusing the handle's pre-setup state.
/// `num_devices` is the number of devices the CALLER wants evaluated (derived
/// from the voltages buffer). If it's greater than `h.num_devices`, the
/// handle's snapshots are tiled modulo — this is how `jax.vmap` replicates a
/// single-device-group handle across its replica axis.
#[no_mangle]
pub extern "C" fn batched_osdi_eval_handle_ffi(
    handle_id:        u64,
    num_devices:      usize,
    num_pins:         usize,
    num_states:       usize,
    voltages_ptr:     *const f64,
    _old_state_ptr:   *const f64,
    currents_ptr:     *mut f64,
    conductances_ptr: *mut f64,
    charges_ptr:      *mut f64,
    capacitances_ptr: *mut f64,
    new_state_ptr:    *mut f64,
) {
    run_handle_eval(
        handle_id, num_devices, num_pins, num_states,
        voltages_ptr,
        currents_ptr, Some(conductances_ptr), charges_ptr, Some(capacitances_ptr),
        new_state_ptr,
        false,
    );
}

/// Residual-only batched eval reusing the handle's pre-setup state.
#[no_mangle]
pub extern "C" fn batched_osdi_residual_eval_handle_ffi(
    handle_id:        u64,
    num_devices:      usize,
    num_pins:         usize,
    num_states:       usize,
    voltages_ptr:     *const f64,
    _old_state_ptr:   *const f64,
    currents_ptr:     *mut f64,
    charges_ptr:      *mut f64,
    new_state_ptr:    *mut f64,
) {
    run_handle_eval(
        handle_id, num_devices, num_pins, num_states,
        voltages_ptr,
        currents_ptr, None, charges_ptr, None,
        new_state_ptr,
        true,
    );
}

/// Shared dispatcher for the handle-based eval entry points. When
/// `conductances_ptr` / `capacitances_ptr` are `None`, the caller has opted
/// into residual-only mode — per-device `cond` / `cap` are exposed to
/// `eval_device_from_setup` as empty slices that it never writes.
#[allow(clippy::too_many_arguments)]
fn run_handle_eval(
    handle_id:        u64,
    num_devices:      usize,   // caller-requested device count (≥ h.num_devices)
    num_pins:         usize,
    num_states:       usize,
    voltages_ptr:     *const f64,
    currents_ptr:     *mut f64,
    conductances_ptr: Option<*mut f64>,
    charges_ptr:      *mut f64,
    capacitances_ptr: Option<*mut f64>,
    new_state_ptr:    *mut f64,
    residual_only:    bool,
) {
    let handles = HANDLES.read().unwrap();
    let h = match handles.get(&handle_id) {
        Some(h) => h,
        None => { eprintln!("osdi handle-eval: unknown handle_id={handle_id}"); return; }
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = match registry.get(&h.model_id) {
        Some(m) => m,
        None => { eprintln!("osdi handle-eval: model {} gone", h.model_id); return; }
    };

    let jac_size        = num_pins * num_pins;
    let model_size      = m.model_size;
    let inst_size       = m.instance_size;
    let handle_n_devs   = h.num_devices;

    if handle_n_devs == 0 || num_devices % handle_n_devs != 0 {
        eprintln!(
            "osdi handle-eval: num_devices={} must be a positive multiple of \
             handle.num_devices={}",
            num_devices, handle_n_devs
        );
        return;
    }

    let voltages = unsafe { std::slice::from_raw_parts(voltages_ptr, num_devices * num_pins) };
    let currents = unsafe { std::slice::from_raw_parts_mut(currents_ptr, num_devices * num_pins) };
    let charges  = unsafe { std::slice::from_raw_parts_mut(charges_ptr,  num_devices * num_pins) };

    // Stateful models: states initialised to zero, continue with normal eval.
    if num_states != 0 {
        let new_state = unsafe {
            std::slice::from_raw_parts_mut(new_state_ptr, num_devices * num_states)
        };
        new_state.fill(0.0);
    }

    // Per-device snapshot indexing (inlined below): device k uses snapshot
    // (k % handle_n_devs). When num_devices == handle_n_devs the modulo is a
    // no-op; when vmap has flattened B replicas on top, this broadcasts the
    // handle's snapshots across the replica axis.

    // Per-device work bodies are inlined into each branch (sequential vs
    // parallel) — extracting them into a closure measurably hurt parallel
    // throughput at N=54 (rayon's task-stealing scheduler doesn't always
    // inline closures cleanly, leaving an indirect call per work item).
    if residual_only {
        if use_parallel(num_devices) {
            currents.par_chunks_exact_mut(num_pins)
                .zip(charges.par_chunks_exact_mut(num_pins))
                .zip(voltages.par_chunks_exact(num_pins))
                .enumerate()
                .for_each(|(device_idx, ((cur, chg), vol))| {
                    cur.fill(0.0);
                    chg.fill(0.0);
                    let snap_idx = device_idx % handle_n_devs;
                    let model_snap = &h.model_data_flat[snap_idx * model_size .. (snap_idx + 1) * model_size];
                    let inst_snap  = &h.inst_data_flat [snap_idx * inst_size  .. (snap_idx + 1) * inst_size ];
                    SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.model_data.clear();
                        scratch.model_data.extend_from_slice(model_snap);
                        scratch.inst_data.clear();
                        scratch.inst_data.extend_from_slice(inst_snap);
                        let EvalScratch { model_data, inst_data, small } = &mut *scratch;
                        let mut cond_ignored: [f64; 0] = [];
                        let mut cap_ignored:  [f64; 0] = [];
                        eval_device_from_setup(
                            m, model_data, inst_data, vol,
                            cur, &mut cond_ignored, chg, &mut cap_ignored,
                            true, small,
                        );
                    });
                });
        } else {
            currents.chunks_exact_mut(num_pins)
                .zip(charges.chunks_exact_mut(num_pins))
                .zip(voltages.chunks_exact(num_pins))
                .enumerate()
                .for_each(|(device_idx, ((cur, chg), vol))| {
                    cur.fill(0.0);
                    chg.fill(0.0);
                    let snap_idx = device_idx % handle_n_devs;
                    let model_snap = &h.model_data_flat[snap_idx * model_size .. (snap_idx + 1) * model_size];
                    let inst_snap  = &h.inst_data_flat [snap_idx * inst_size  .. (snap_idx + 1) * inst_size ];
                    SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.model_data.clear();
                        scratch.model_data.extend_from_slice(model_snap);
                        scratch.inst_data.clear();
                        scratch.inst_data.extend_from_slice(inst_snap);
                        let EvalScratch { model_data, inst_data, small } = &mut *scratch;
                        let mut cond_ignored: [f64; 0] = [];
                        let mut cap_ignored:  [f64; 0] = [];
                        eval_device_from_setup(
                            m, model_data, inst_data, vol,
                            cur, &mut cond_ignored, chg, &mut cap_ignored,
                            true, small,
                        );
                    });
                });
        }
    } else {
        let conductances = unsafe {
            std::slice::from_raw_parts_mut(conductances_ptr.unwrap(), num_devices * jac_size)
        };
        let capacitances = unsafe {
            std::slice::from_raw_parts_mut(capacitances_ptr.unwrap(), num_devices * jac_size)
        };

        if use_parallel(num_devices) {
            currents.par_chunks_exact_mut(num_pins)
                .zip(conductances.par_chunks_exact_mut(jac_size))
                .zip(charges.par_chunks_exact_mut(num_pins))
                .zip(capacitances.par_chunks_exact_mut(jac_size))
                .zip(voltages.par_chunks_exact(num_pins))
                .enumerate()
                .for_each(|(device_idx, ((((cur, cond), chg), cap), vol))| {
                    cur.fill(0.0);
                    cond.fill(0.0);
                    chg.fill(0.0);
                    cap.fill(0.0);
                    let snap_idx = device_idx % handle_n_devs;
                    let model_snap = &h.model_data_flat[snap_idx * model_size .. (snap_idx + 1) * model_size];
                    let inst_snap  = &h.inst_data_flat [snap_idx * inst_size  .. (snap_idx + 1) * inst_size ];
                    SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.model_data.clear();
                        scratch.model_data.extend_from_slice(model_snap);
                        scratch.inst_data.clear();
                        scratch.inst_data.extend_from_slice(inst_snap);
                        let EvalScratch { model_data, inst_data, small } = &mut *scratch;
                        eval_device_from_setup(
                            m, model_data, inst_data, vol,
                            cur, cond, chg, cap,
                            false, small,
                        );
                    });
                });
        } else {
            currents.chunks_exact_mut(num_pins)
                .zip(conductances.chunks_exact_mut(jac_size))
                .zip(charges.chunks_exact_mut(num_pins))
                .zip(capacitances.chunks_exact_mut(jac_size))
                .zip(voltages.chunks_exact(num_pins))
                .enumerate()
                .for_each(|(device_idx, ((((cur, cond), chg), cap), vol))| {
                    cur.fill(0.0);
                    cond.fill(0.0);
                    chg.fill(0.0);
                    cap.fill(0.0);
                    let snap_idx = device_idx % handle_n_devs;
                    let model_snap = &h.model_data_flat[snap_idx * model_size .. (snap_idx + 1) * model_size];
                    let inst_snap  = &h.inst_data_flat [snap_idx * inst_size  .. (snap_idx + 1) * inst_size ];
                    SCRATCH.with(|s| {
                        let mut scratch = s.borrow_mut();
                        scratch.model_data.clear();
                        scratch.model_data.extend_from_slice(model_snap);
                        scratch.inst_data.clear();
                        scratch.inst_data.extend_from_slice(inst_snap);
                        let EvalScratch { model_data, inst_data, small } = &mut *scratch;
                        eval_device_from_setup(
                            m, model_data, inst_data, vol,
                            cur, cond, chg, cap,
                            false, small,
                        );
                    });
                });
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. DIAGNOSTIC — dump model internals for debugging
// ─────────────────────────────────────────────────────────────────────────────

/// Print a detailed report of param flags, Jacobian entry pairs, and function
/// pointer addresses for a loaded model.  Called from Python via nanobind.
#[no_mangle]
pub extern "C" fn dump_model_info(model_id: u32) {
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = match registry.get(&model_id) {
        Some(m) => m,
        None => { eprintln!("dump_model_info: unknown model_id={model_id}"); return; }
    };

    println!("=== Model {} ===", model_id);
    println!("  num_nodes={} num_terminals={}", m.num_nodes, m.num_terminals);
    println!("  instance_size={} model_size={}", m.instance_size, m.model_size);
    println!("  node_map_off={}", m.node_map_off);
    println!("  num_resist_jac={} num_react_jac={}", m.num_resist_jac, m.num_react_jac);

    println!("\n  param_flags (first 20 of {}):", m.param_flags.len());
    for (i, &f) in m.param_flags.iter().enumerate().take(20) {
        let kind = (f >> 30) & 3;
        let ty   = f & 3;
        let kind_s = ["MODEL","INST","OPVAR","?"][kind as usize];
        let ty_s   = ["REAL","INT","STR","?"][ty as usize];
        println!("    [{i:3}] flags=0x{f:08x}  kind={kind_s}  ty={ty_s}");
    }
    if m.param_flags.len() > 20 {
        // count kinds across all params
        let (n_model, n_inst, n_opvar) = m.param_flags.iter().fold((0usize,0,0), |(a,b,c), &f| {
            let k = (f >> 30) & 3;
            match k { 0 => (a+1,b,c), 1 => (a,b+1,c), 2 => (a,b,c+1), _ => (a,b,c) }
        });
        println!("    ... ({} total: {} MODEL, {} INST, {} OPVAR)",
                 m.param_flags.len(), n_model, n_inst, n_opvar);
    }

    // Build node_map as eval_one_device would
    let mut dbg_node_map: Vec<i32> = (0..m.num_nodes as i32).collect();
    for &(n1, n2) in &m.collapsible_pairs {
        let (n1, n2) = (n1 as usize, n2 as usize);
        if n1 < m.num_nodes as usize && n2 < m.num_nodes as usize {
            let s1 = dbg_node_map[n1]; let s2 = dbg_node_map[n2];
            let merged = s1.min(s2); let higher = s1.max(s2);
            for s in dbg_node_map.iter_mut() { if *s == higher { *s = merged; } }
        }
    }
    let dbg_num_slots = dbg_node_map.iter().max().map(|&m| m as usize + 1)
        .unwrap_or(m.num_nodes as usize);
    println!("\n  node_map after collapsing: {:?}", dbg_node_map);
    println!("  num_slots={}", dbg_num_slots);

    println!("\n  resist_jac_pairs (first 16 of {}):", m.resist_jac_pairs.len());
    for (i, &(n1, n2)) in m.resist_jac_pairs.iter().enumerate().take(16) {
        let is_terminal = n1 < m.num_terminals && n2 < m.num_terminals;
        println!("    [{i:2}] ({n1}, {n2}){}",
                 if is_terminal { " ← terminal-terminal" } else { "" });
    }

    // Count and list terminal-terminal pairs AFTER collapsing
    let nt = m.num_terminals as usize;
    let tt_resist: Vec<_> = m.resist_jac_pairs.iter().enumerate().filter_map(|(idx, &(n1,n2))| {
        let s1 = dbg_node_map.get(n1 as usize).copied().unwrap_or(-1);
        let s2 = dbg_node_map.get(n2 as usize).copied().unwrap_or(-1);
        if s1 >= 0 && s2 >= 0 && (s1 as usize) < nt && (s2 as usize) < nt {
            Some((idx, n1, n2, s1, s2))
        } else { None }
    }).collect();
    let tt_react: Vec<_> = m.react_jac_pairs.iter().enumerate().filter_map(|(idx, &(n1,n2))| {
        let s1 = dbg_node_map.get(n1 as usize).copied().unwrap_or(-1);
        let s2 = dbg_node_map.get(n2 as usize).copied().unwrap_or(-1);
        if s1 >= 0 && s2 >= 0 && (s1 as usize) < nt && (s2 as usize) < nt {
            Some((idx, n1, n2, s1, s2))
        } else { None }
    }).collect();
    println!("\n  Terminal-terminal pairs after collapsing: {} resistive, {} reactive",
             tt_resist.len(), tt_react.len());
    for (idx, n1, n2, s1, s2) in &tt_resist {
        println!("    jac[{idx}] ({n1},{n2}) → slot ({s1},{s2})");
    }
    for (idx, n1, n2, s1, s2) in &tt_react {
        println!("    react_jac[{idx}] ({n1},{n2}) → slot ({s1},{s2})");
    }

    println!("\n  collapsible_pairs ({}):", m.collapsible_pairs.len());
    for (i, &(a, b)) in m.collapsible_pairs.iter().enumerate() {
        let a_term = if a < m.num_terminals { "terminal" } else { "internal" };
        let b_term = if b < m.num_terminals { "terminal" } else { "internal" };
        println!("    [{i}] ({a} {a_term}, {b} {b_term})");
    }
}
