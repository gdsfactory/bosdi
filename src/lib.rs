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
    desc_num_resist_jac:       usize,
    desc_fn_write_jac_resist:  usize,
    // OsdiSimInfo shape (for layout-driven construction in future versions)
    sim_info_prev_solve_off:   usize,
    sim_info_flags_off:        usize,
    sim_info_total_size:       usize,
    // Simulation flags
    flag_calc_resist_residual: u32,
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
            desc_num_resist_jac:       256,
            desc_fn_write_jac_resist:  264,
            sim_info_prev_solve_off:   40,
            sim_info_flags_off:        64,
            sim_info_total_size:       72,
            flag_calc_resist_residual: 1,
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
    _lib:                 Library,
    layout:               AbiLayout,
    pub num_terminals:    u32,
    pub num_nodes:        u32,
    pub num_resist_jac:   u32,
    pub instance_size:    usize,
    pub model_size:       usize,
    /// Byte offset within inst where the u32 node-index array begins.
    pub node_map_off:     usize,
    // Functions with NULL descriptor slots — looked up by name:
    pub setup_model:      SetupModelFn,
    pub setup_instance:   SetupInstanceFn,
    pub eval:             EvalFn,
    // Functions read from descriptor (relocated by dynamic linker):
    pub load_residual:    LoadResidualFn,
    pub write_jacobian:   WriteJacobianFn,
}
unsafe impl Send for LoadedOsdi {}
unsafe impl Sync for LoadedOsdi {}

/// Metadata returned to C++ and then to Python.
#[repr(C)]
pub struct ModelMetadata {
    pub model_id:      u32,
    pub num_pins:      usize,
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
        model_id: 0, num_pins: 0, num_params: 0,
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

    // ── function pointers with NULL descriptor slots — look up by name ────────
    // OpenVAF exports these as `fname_0` (index 0 = first model in the binary).
    let setup_model:    SetupModelFn    = sym!(lib, b"setup_model_0\0",    SetupModelFn);
    let setup_instance: SetupInstanceFn = sym!(lib, b"setup_instance_0\0", SetupInstanceFn);
    let eval:           EvalFn          = sym!(lib, b"eval_0\0",           EvalFn);

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
        instance_size,
        model_size,
        node_map_off,
        setup_model,
        setup_instance,
        eval,
        load_residual,
        write_jacobian,
    });

    ModelMetadata {
        model_id,
        num_pins:     num_terminals as usize,
        num_params:   num_params as usize,
        num_states:   num_states as usize,
        osdi_version: version,
        success:      true,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. PHASE 2: SINGLE-DEVICE EVALUATION
// ─────────────────────────────────────────────────────────────────────────────
//
// OSDI 0.4 evaluation protocol (confirmed by disassembling the resistor binary):
//
//  Instance struct (104 bytes for resistor):
//    inst[node_map_off + i*4] = u32 node index for terminal i
//    inst[64]                 = m (multiplicity instance param)      ← given flag bit 0 at inst[0]
//    inst[72]                 = G  = m/R  (set by setup_instance)
//    inst[80]                 = -G        (set by setup_instance)
//    inst[88]                 = I_A       (set by eval)
//    inst[96]                 = I_B       (set by eval)
//
//  Model struct (24 bytes for resistor):
//    model[8]  = R (resistance param)          ← given flag bit 0 at model[0]
//    model[16] = secondary model param (tnom)  ← uses default if flag bit 1 not set
//
//  Param mapping from Python params array [R, m] (for resistor):
//    params[0] = R → model[8]
//    params[1] = m → inst[64]
//
// TODO: For general OSDI models, derive param→offset mapping from param_opvar.

fn eval_one_device(
    m: &LoadedOsdi,
    vol:   &[f64],       // voltages for this device (num_pins elements)
    param: &[f64],       // params for this device   (num_params elements)
    cur:   &mut [f64],   // output currents          (num_pins elements, zeroed by caller)
    cond:  &mut [f64],   // output conductances      (num_resist_jac elements)
) {
    // ── allocate opaque model and instance data blocks ────────────────────────
    let mut model_data = vec![0u8; m.model_size];
    let mut inst_data  = vec![0u8; m.instance_size];

    // ── write model param: params[0] = R → model[8], set given flag ──────────
    if param.len() > 0 {
        unsafe {
            *(model_data.as_mut_ptr().add(8) as *mut f64) = param[0];
        }
        model_data[0] |= 0x01; // bit 0 = "R given"
    }

    // ── write instance param: params[1] = m → inst[64], set given flag ───────
    if param.len() > 1 {
        unsafe {
            *(inst_data.as_mut_ptr().add(64) as *mut f64) = param[1];
        }
        inst_data[0] |= 0x01; // bit 0 = "m given"
    }

    // ── set node mapping: terminal i → node index i ───────────────────────────
    for i in 0..m.num_terminals as usize {
        unsafe {
            *(inst_data.as_mut_ptr().add(m.node_map_off + i * 4) as *mut i32) = i as i32;
        }
    }

    let model_ptr = model_data.as_mut_ptr() as *mut c_void;
    let inst_ptr  = inst_data.as_mut_ptr() as *mut c_void;

    // ── setup_model: validates R and applies defaults ─────────────────────────
    let mut init1 = OsdiInitInfo::default();
    unsafe {
        (m.setup_model)(std::ptr::null_mut(), model_ptr, std::ptr::null_mut(), &mut init1);
    }

    // ── setup_instance: precomputes G = m/R into inst[72], -G into inst[80] ──
    let mut init2 = OsdiInitInfo::default();
    unsafe {
        (m.setup_instance)(
            std::ptr::null_mut(), inst_ptr, model_ptr,
            300.0, m.num_terminals, std::ptr::null_mut(), &mut init2,
        );
    }

    // ── eval: computes I_A → inst[88], I_B → inst[96] ────────────────────────
    // OSDI 0.4: use the typed OsdiSimInfo struct (layout verified by disassembly).
    // For V05: use layout-driven raw buffer (see AbiLayout::sim_info_*).
    let mut vol_buf: Vec<f64> = vol.to_vec();
    let mut sim_info = OsdiSimInfo {
        paras:      OsdiSimParas::null(),
        abstime:    0.0,
        prev_solve: vol_buf.as_mut_ptr(),
        prev_state: std::ptr::null_mut(),
        next_state: std::ptr::null_mut(),
        flags:      m.layout.flag_calc_resist_residual,
        _pad:       0,
    };
    unsafe {
        (m.eval)(
            std::ptr::null_mut(), inst_ptr, model_ptr,
            &mut sim_info as *mut _ as *mut c_void,
        );
    }

    // ── extract currents via load_residual_resist ─────────────────────────────
    unsafe { (m.load_residual)(inst_ptr, model_ptr, cur.as_mut_ptr()); }

    // ── extract conductances via write_jacobian_array_resist ──────────────────
    unsafe { (m.write_jacobian)(inst_ptr, model_ptr, cond.as_mut_ptr()); }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. PHASE 2: BATCHED FFI ENTRY POINT (called from C++ XLA handler)
// ─────────────────────────────────────────────────────────────────────────────

#[no_mangle]
pub extern "C" fn batched_osdi_eval_ffi(
    model_id:         u32,
    num_devices:      usize,
    num_pins:         usize,
    num_params:       usize,
    num_states:       usize,
    voltages_ptr:     *const f64,
    params_ptr:       *const f64,
    _old_state_ptr:   *const f64,
    currents_ptr:     *mut f64,
    conductances_ptr: *mut f64,
    charges_ptr:      *mut f64,
    capacitances_ptr: *mut f64,
    _new_state_ptr:   *mut f64,
) {
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = registry.get(&model_id).expect("Unknown OSDI model ID");

    let jac_size = m.num_resist_jac as usize;

    let voltages = unsafe { std::slice::from_raw_parts(voltages_ptr, num_devices * num_pins) };
    let params   = unsafe { std::slice::from_raw_parts(params_ptr,   num_devices * num_params) };

    // Reactive outputs are zero for resistive-only models.
    let charges      = unsafe { std::slice::from_raw_parts_mut(charges_ptr,      num_devices * num_pins) };
    let capacitances = unsafe { std::slice::from_raw_parts_mut(capacitances_ptr, num_devices * jac_size) };
    charges.fill(0.0);
    capacitances.fill(0.0);

    let currents     = unsafe { std::slice::from_raw_parts_mut(currents_ptr,     num_devices * num_pins) };
    let conductances = unsafe { std::slice::from_raw_parts_mut(conductances_ptr, num_devices * jac_size) };

    // State slices (num_states=0 for the resistor — avoid par_chunks_exact(0) panic).
    if num_states == 0 {
        currents.par_chunks_exact_mut(num_pins)
            .zip(conductances.par_chunks_exact_mut(jac_size))
            .zip(voltages.par_chunks_exact(num_pins))
            .zip(params.par_chunks_exact(num_params))
            .for_each(|(((cur, cond), vol), param)| {
                cur.fill(0.0);
                eval_one_device(m, vol, param, cur, cond);
            });
    } else {
        // TODO: stateful model support
        eprintln!("OSDI: stateful models not yet supported (num_states={})", num_states);
    }
}
