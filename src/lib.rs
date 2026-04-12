use libloading::{Library, Symbol};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::sync::RwLock;

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
    // OsdiJacobianEntry is 16 bytes: {node_1: u32, node_2: u32, field2: u32, field3: u32}.
    // Ordering: first num_resist_jac entries are resistive, then num_react_jac reactive.
    for i in 0..num_jac_entries as usize {
        let node_1 = unsafe { read_u32(jac_entries_ptr.add(i * 16), 0) };
        let node_2 = unsafe { read_u32(jac_entries_ptr.add(i * 16), 4) };
        let react = if i < num_resist_jac as usize { 0u32 } else { 1u32 };
        if react == 0 {
            resist_jac_pairs.push((node_1, node_2));
        } else {
            react_jac_pairs.push((node_1, node_2));
        }
    }

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

    // ── num_all_nodes + resistive_mask: one pass after collapsing ───────────────
    // Run the same deterministic collapse algorithm as eval_one_device, then build
    // the slot_to_out map to produce num_all_nodes and resistive_mask together.
    // Phantom slots (numbered but not mapped to by any node, e.g. PSP103's gap
    // between slots 3 and 12) are skipped in slot_to_out and never appear in the mask.
    let (num_all_nodes, resistive_mask) = {
        let num_terms = num_terminals as usize;
        let num_n     = num_nodes    as usize;

        // Collapse
        let mut nm: Vec<i32> = (0..num_n as i32).collect();
        for &(n1, n2) in &collapsible_pairs {
            let (n1, n2) = (n1 as usize, n2 as usize);
            if n1 >= num_n || n2 >= num_n { continue; }
            let s1 = nm[n1]; let s2 = nm[n2];
            let merged = s1.min(s2); let higher = s1.max(s2);
            for s in nm.iter_mut() { if *s == higher { *s = merged; } }
        }
        let num_slots = nm.iter().map(|&s| s as usize + 1).max().unwrap_or(num_terms);

        // Occupied slots
        let mut slot_occupied = vec![false; num_slots];
        for &s in &nm { slot_occupied[s as usize] = true; }

        // slot → output index (terminal slots first, then internal)
        let mut slot_to_out = vec![-1i32; num_slots];
        for t in 0..num_terms {
            slot_to_out[nm[t] as usize] = t as i32;
        }
        let mut next_out = num_terms;
        for slot in 0..num_slots {
            if slot_occupied[slot] && slot_to_out[slot] < 0 {
                slot_to_out[slot] = next_out as i32;
                next_out += 1;
            }
        }
        let num_all = next_out;

        // resistive_mask[out_idx] = true iff that output node appears as a row
        // in at least one resist_jac_pair (G[row, :] can be non-zero).
        let mut mask = vec![false; num_all];
        for &(n1, _) in &resist_jac_pairs {
            let n1 = n1 as usize;
            if n1 < nm.len() {
                let slot = nm[n1] as usize;
                if slot < slot_to_out.len() {
                    let out = slot_to_out[slot];
                    if out >= 0 { mask[out as usize] = true; }
                }
            }
        }

        (num_all, mask)
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
// 7. PHASE 2: SINGLE-DEVICE EVALUATION
// ─────────────────────────────────────────────────────────────────────────────
//
// OSDI 0.4 evaluation protocol:
//
//  Parameters are written to the model/instance structs via the descriptor's
//  access() function, using the param_opvar flags cached in LoadedOsdi::param_flags
//  to determine each parameter's kind (model vs. instance) and type (real/int/str).
//  The given_flag_model() / given_flag_inst() functions mark each written parameter
//  as user-provided so that setup_model/setup_instance applies model defaults only
//  to parameters not supplied by the caller.
//
//  Call sequence per device:
//    1. Write all params via access() + given_flag_*()
//    2. setup_model()     — applies defaults, validates model params
//    3. setup_instance()  — derives instance quantities from model params
//    4. eval()            — computes residuals at the given operating point
//    5. load_residual / write_jacobian — extract currents and conductances

fn eval_one_device(
    m: &LoadedOsdi,
    vol:   &[f64],       // voltages for this device (num_nodes elements: terminals first, then internal)
    param: &[f64],       // params for this device   (num_params elements)
    cur:   &mut [f64],   // output currents          (num_nodes elements, zeroed by caller)
    cond:  &mut [f64],   // output conductances      (num_nodes^2 elements, zeroed by caller)
    chg:   &mut [f64],   // output charges           (num_nodes elements, zeroed by caller)
    cap:   &mut [f64],   // output capacitances      (num_nodes^2 elements, zeroed by caller)
) {
    // ── allocate opaque model and instance data blocks ────────────────────────
    let mut model_data = vec![0u8; m.model_size];
    let mut inst_data  = vec![0u8; m.instance_size];

    // ── set node mapping with OSDI collapsing ────────────────────────────────
    // Start with identity: node i → slot i.
    let mut node_map: Vec<i32> = (0..m.num_nodes as i32).collect();

    // Apply collapsible pairs: when coupling element is zero the two nodes share
    // a slot.  We always collapse — bosdi evaluates at a single point and doesn't
    // know the coupling element values.  Collapsing ensures internal nodes (e.g.
    // PSP103's Dint/Sint) inherit the voltage of their terminal counterparts so
    // the model sees the correct V_DS and can produce non-zero drain current.
    //
    // Strategy: iterate pairs; whichever node has the lower slot gets to "own"
    // the merged slot, so terminals (low indices 0..num_terminals) tend to win.
    for &(n1, n2) in &m.collapsible_pairs {
        let (n1, n2) = (n1 as usize, n2 as usize);
        if n1 >= m.num_nodes as usize || n2 >= m.num_nodes as usize { continue; }
        // Follow any existing chain (in case of multi-hop collapsing)
        let slot1 = node_map[n1];
        let slot2 = node_map[n2];
        let merged = slot1.min(slot2);
        // Map both to the lower slot; also redirect any other node already
        // pointing at the higher slot.
        let higher = slot1.max(slot2);
        for slot in node_map.iter_mut() {
            if *slot == higher { *slot = merged; }
        }
    }

    // Write the final node_map into the instance data block.
    for (i, &slot) in node_map.iter().enumerate() {
        unsafe {
            *(inst_data.as_mut_ptr().add(m.node_map_off + i * 4) as *mut i32) = slot;
        }
    }

    // Effective number of distinct slots used (after collapsing).
    let num_slots = node_map.iter().max().map(|&m| m as usize + 1)
                    .unwrap_or(m.num_nodes as usize);

    let num_terms = m.num_terminals as usize;

    // Map each slot → output index.
    // Terminal slots get output indices 0..num_terms.
    // Non-terminal OCCUPIED slots get indices num_terms, num_terms+1, …
    // Phantom slots (numbered but not mapped to by any node, e.g. gaps in PSP103's
    // collapsing) stay at -1 and are skipped in all scatter operations.
    let mut slot_occupied = vec![false; num_slots];
    for &s in &node_map { slot_occupied[s as usize] = true; }

    let mut slot_to_out: Vec<i32> = vec![-1; num_slots];
    for t in 0..num_terms {
        let slot = node_map[t] as usize;
        slot_to_out[slot] = t as i32;
    }
    let mut next_out = num_terms;
    for slot in 0..num_slots {
        if slot_occupied[slot] && slot_to_out[slot] < 0 {
            slot_to_out[slot] = next_out as i32;
            next_out += 1;
        }
    }
    let num_all_nodes = next_out;  // == num_terminals + num_non_collapsed_internal

    let model_ptr = model_data.as_mut_ptr() as *mut c_void;
    let inst_ptr  = inst_data.as_mut_ptr() as *mut c_void;

    // ── write model params, then setup_model, then write instance params ────
    // Params are written in two passes so that:
    //   (a) NaN values are silently skipped → Verilog-A default is used instead.
    //   (b) Instance params are written AFTER setup_model and BEFORE
    //       setup_instance, because setup_instance may reinitialise the inst
    //       block, overwriting values we pre-wrote.
    //
    // Call sequence:
    //   1. write MODEL params + given_flag_model()
    //   2. setup_model()     — fills MODEL defaults for unset params
    //   3. write INSTANCE params + given_flag_inst()
    //   4. setup_instance()  — fills INST defaults for unset params

    // Helper closure: write one param via access() + given_flag_*().
    // Returns false if the param should be skipped (NaN, OPVAR, STR, null ptr).
    let write_param = |i: usize, val: f64, kind: u32, ty: u32| {
        if val.is_nan()           { return; }   // NaN → use Verilog-A default
        if kind == PARA_KIND_OPVAR { return; }  // output-only, never written
        if ty   == PARA_TY_STR    { return; }   // can't map str from f64 array

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

    // Pass 1: model params only
    for (i, &val) in param.iter().enumerate() {
        let flags = match m.param_flags.get(i) { Some(&f) => f, None => break };
        let kind = flags & PARA_KIND_MASK;
        let ty   = flags & PARA_TY_MASK;
        if kind != PARA_KIND_MODEL { continue; }
        write_param(i, val, kind, ty);
    }

    // ── setup_model: applies defaults for unset params, validates ────────────
    let mut init1 = OsdiInitInfo::default();
    unsafe {
        (m.setup_model)(std::ptr::null_mut(), model_ptr, std::ptr::null_mut(), &mut init1);
    }

    // Pass 2: instance params only (after setup_model, before setup_instance)
    for (i, &val) in param.iter().enumerate() {
        let flags = match m.param_flags.get(i) { Some(&f) => f, None => break };
        let kind = flags & PARA_KIND_MASK;
        let ty   = flags & PARA_TY_MASK;
        if kind != PARA_KIND_INST { continue; }
        write_param(i, val, kind, ty);
    }

    // ── setup_instance: precomputes derived quantities from model params ──────
    let mut init2 = OsdiInitInfo::default();
    unsafe {
        (m.setup_instance)(
            std::ptr::null_mut(), inst_ptr, model_ptr,
            300.0, m.num_terminals, std::ptr::null_mut(), &mut init2,
        );
    }

    // ── eval: compute resistive and/or reactive residuals ────────────────────
    // OSDI 0.4: use the typed OsdiSimInfo struct (layout verified by disassembly).
    // For V05: use layout-driven raw buffer (see AbiLayout::sim_info_*).
    let mut flags = 0u32;
    if m.num_resist_jac > 0 {
        flags |= m.layout.flag_calc_resist_residual;
        flags |= m.layout.flag_calc_resist_jacobian;
    }
    if m.num_react_jac > 0 {
        flags |= m.layout.flag_calc_react_residual;
        flags |= m.layout.flag_calc_react_jacobian;
    }

    // ── voltage buffer: num_slots entries, indexed by node_map slot ─────────
    // eval() reads prev_solve[node_map[i]] for each node i.
    // vol[0..num_terms]         → terminal voltages (from circuit node solution)
    // vol[num_terms..num_nodes] → internal node voltages from previous Newton iterate
    let mut vol_buf = vec![0.0f64; num_slots];
    for slot in 0..num_slots {
        let out_idx = slot_to_out[slot];
        if out_idx >= 0 && (out_idx as usize) < vol.len() {
            vol_buf[slot] = vol[out_idx as usize];
        }
    }

    // ── sim paras: provide a null-sentinel names array ────────────────────────
    // Some models (e.g. the compiled OpenVAF diode) dereference sim_paras->names
    // before checking if the pointer is null, crashing when we pass null.
    // Providing a valid pointer to a single null entry tells the model "no
    // simulator parameters available" without a null-pointer fault.
    let mut names_sentinel: *mut i8 = std::ptr::null_mut();
    let sim_paras = unsafe { OsdiSimParas::with_null_sentinel(&mut names_sentinel) };

    let mut sim_info = OsdiSimInfo {
        paras:      sim_paras,
        abstime:    0.0,
        prev_solve: vol_buf.as_mut_ptr(),
        prev_state: std::ptr::null_mut(),
        next_state: std::ptr::null_mut(),
        flags,
        _pad:       0,
    };
    unsafe {
        (m.eval)(
            std::ptr::null_mut(), inst_ptr, model_ptr,
            &mut sim_info as *mut _ as *mut c_void,
        );
    }

    // ── extract resistive outputs ─────────────────────────────────────────────
    // load_residual writes to dst[node_map[i]] for each node i, accumulating
    // contributions from collapsed nodes into their shared terminal slot.
    // The dst buffer must cover all slots (num_slots).
    if m.num_resist_jac > 0 {
        let mut cur_all = vec![0.0f64; num_slots];
        unsafe { (m.load_residual)(inst_ptr, model_ptr, cur_all.as_mut_ptr()); }
        // Copy all active slots to their output index (terminals and internal nodes).
        for slot in 0..num_slots {
            let out = slot_to_out[slot];
            if out >= 0 {
                cur[out as usize] = cur_all[slot];
            }
        }

        // write_jacobian writes num_resist_jac values. Scatter all pairs into cond
        // using slot_to_out for both axes — no terminal-only filter.
        let mut jac_buf = vec![0.0f64; m.num_resist_jac as usize];
        unsafe { (m.write_jacobian)(inst_ptr, model_ptr, jac_buf.as_mut_ptr()); }
        for (idx, &(n1, n2)) in m.resist_jac_pairs.iter().enumerate() {
            let s1 = node_map.get(n1 as usize).copied().unwrap_or(-1);
            let s2 = node_map.get(n2 as usize).copied().unwrap_or(-1);
            if s1 >= 0 && s2 >= 0 {
                let o1 = slot_to_out[s1 as usize];
                let o2 = slot_to_out[s2 as usize];
                if o1 >= 0 && o2 >= 0 {
                    cond[o1 as usize * num_all_nodes + o2 as usize] += jac_buf[idx];
                }
            }
        }
    }

    // ── extract reactive outputs ──────────────────────────────────────────────
    if m.num_react_jac > 0 {
        if let Some(lr) = m.load_residual_react {
            let mut chg_all = vec![0.0f64; num_slots];
            unsafe { lr(inst_ptr, model_ptr, chg_all.as_mut_ptr()); }
            for slot in 0..num_slots {
                let out = slot_to_out[slot];
                if out >= 0 {
                    chg[out as usize] = chg_all[slot];
                }
            }
        }
        if let Some(wj) = m.write_jacobian_react {
            let mut jac_buf = vec![0.0f64; m.num_react_jac as usize];
            unsafe { wj(inst_ptr, model_ptr, jac_buf.as_mut_ptr()); }
            for (idx, &(n1, n2)) in m.react_jac_pairs.iter().enumerate() {
                let s1 = node_map.get(n1 as usize).copied().unwrap_or(-1);
                let s2 = node_map.get(n2 as usize).copied().unwrap_or(-1);
                if s1 >= 0 && s2 >= 0 {
                    let o1 = slot_to_out[s1 as usize];
                    let o2 = slot_to_out[s2 as usize];
                    if o1 >= 0 && o2 >= 0 {
                        cap[o1 as usize * num_all_nodes + o2 as usize] += jac_buf[idx];
                    }
                }
            }
        }
    }
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
                eval_one_device(m, vol, param, cur, cond, chg, cap);
            });
    } else {
        // Stateful model support is not yet implemented.
        // Zero all output buffers so callers receive defined (zero) values rather
        // than uninitialized memory.  The XLA runtime does NOT guarantee that
        // output buffers are zeroed before calling the FFI handler.
        eprintln!("OSDI: stateful models not yet supported (num_states={})", num_states);
        currents.fill(0.0);
        conductances.fill(0.0);
        charges.fill(0.0);
        capacitances.fill(0.0);
        let new_state = unsafe {
            std::slice::from_raw_parts_mut(new_state_ptr, num_devices * num_states)
        };
        new_state.fill(0.0);
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
