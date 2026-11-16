//! Handle execution configuration and caller-owned buffer bindings.
use super::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::{Arc, Mutex, OnceLock};

pub(super) struct Configuration {
    pub pool: Arc<ThreadPool>,
    pub min_lanes: usize,
    pub min_chunk: usize,
}
impl Configuration {
    pub(super) fn parallel(&self, count: usize) -> bool {
        self.pool.current_num_threads() > 1 && count >= self.min_lanes && count > self.min_chunk
    }
}
fn env_usize(name: &str, fallback: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(fallback)
}
fn defaults() -> (usize, usize, usize) {
    static DEFAULTS: OnceLock<(usize, usize, usize)> = OnceLock::new();
    *DEFAULTS.get_or_init(|| {
        (
            env_usize(
                "RAYON_NUM_THREADS",
                std::thread::available_parallelism().map_or(1, usize::from),
            ),
            env_usize("BOSDI_RAYON_THRESHOLD", 1024),
            env_usize("BOSDI_RAYON_MIN_CHUNK", 256),
        )
    })
}
fn settings() -> &'static RwLock<Arc<Configuration>> {
    static SETTINGS: OnceLock<RwLock<Arc<Configuration>>> = OnceLock::new();
    SETTINGS.get_or_init(|| {
        let (threads, min_lanes, min_chunk) = defaults();
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("bosdi-{i}"))
            .build()
            .expect("BOSDI worker pool");
        RwLock::new(Arc::new(Configuration {
            pool: Arc::new(pool),
            min_lanes,
            min_chunk,
        }))
    })
}
pub(super) fn configuration() -> Arc<Configuration> {
    settings().read().unwrap().clone()
}
pub(super) fn use_parallel(count: usize) -> bool {
    configuration().parallel(count)
}
pub(super) fn minimum_chunk() -> usize {
    configuration().min_chunk
}

/// Replaces only the BOSDI-owned pool; existing calls retain their old pool.
/// Zero restores the once-read environment default. Never configures Rayon globally.
#[no_mangle]
pub extern "C" fn bosdi_set_num_threads(threads: usize) -> bool {
    let threads = if threads == 0 { defaults().0 } else { threads };
    let Ok(pool) = ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("bosdi-{i}"))
        .build()
    else {
        return false;
    };
    let mut config = settings().write().unwrap();
    let replacement = Arc::new(Configuration {
        pool: Arc::new(pool),
        min_lanes: config.min_lanes,
        min_chunk: config.min_chunk,
    });
    let old = std::mem::replace(&mut *config, replacement);
    drop(config);
    drop(old);
    true
}
#[no_mangle]
pub extern "C" fn bosdi_num_threads() -> usize {
    configuration().pool.current_num_threads()
}
#[no_mangle]
pub extern "C" fn bosdi_set_parallel_policy(min_lanes: usize, min_chunk: usize) -> bool {
    if min_lanes == 0 || min_chunk == 0 {
        return false;
    }
    let mut config = settings().write().unwrap();
    *config = Arc::new(Configuration {
        pool: config.pool.clone(),
        min_lanes,
        min_chunk,
    });
    true
}

pub(super) fn configuration_for_handle(h: &OsdiBatchHandle, base: &Configuration) -> Configuration {
    let (min_lanes, min_chunk) = h
        .parallel_policy
        .read()
        .unwrap()
        .unwrap_or((base.min_lanes, base.min_chunk));
    Configuration {
        pool: base.pool.clone(),
        min_lanes,
        min_chunk,
    }
}
/// Both zero reset the per-handle override; otherwise both must be positive.
#[no_mangle]
pub extern "C" fn bosdi_set_handle_parallel_policy(
    handle: u64,
    min_lanes: usize,
    min_chunk: usize,
) -> bool {
    if (min_lanes == 0) != (min_chunk == 0) {
        return false;
    }
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&handle) else {
        return false;
    };
    *h.parallel_policy.write().unwrap() = if min_lanes == 0 {
        None
    } else {
        Some((min_lanes, min_chunk))
    };
    true
}

#[derive(Clone, Copy)]
pub(super) struct Buffers {
    pub voltages: usize,
    pub currents: usize,
    pub conductances: usize,
    pub charges: usize,
    pub capacitances: usize,
}
// Safety: callers supply distinct, properly sized arrays. Each lane below gets
// exactly one disjoint subrange, including zero-width Jacobians. No reference to
// a complete output array is created while lane references are alive.
unsafe fn lane_mut<'a>(addr: usize, lane: usize, width: usize) -> &'a mut [f64] {
    if width == 0 {
        &mut []
    } else {
        std::slice::from_raw_parts_mut((addr as *mut f64).add(lane * width), width)
    }
}
unsafe fn lane_input<'a>(addr: usize, lane: usize, width: usize) -> &'a [f64] {
    if width == 0 {
        &[]
    } else {
        std::slice::from_raw_parts((addr as *const f64).add(lane * width), width)
    }
}
fn widths(m: &LoadedOsdi, residual: bool, sparse: bool) -> (usize, usize) {
    if residual {
        (0, 0)
    } else if sparse {
        (m.num_resist_jac as usize, m.num_react_jac as usize)
    } else {
        (
            m.num_all_nodes * m.num_all_nodes,
            m.num_all_nodes * m.num_all_nodes,
        )
    }
}

pub(super) fn evaluate_qualified(
    lanes: &mut [Lane],
    plans: &[LanePlan],
    m: &LoadedOsdi,
    count: usize,
    buffers: Buffers,
    residual: bool,
    sparse: bool,
    config: &Configuration,
) {
    let chunk = if config.parallel(count) {
        config.min_chunk
    } else {
        count.max(1)
    };
    let work = |(task, lanes): (usize, &mut [Lane])| {
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (g, c) = widths(m, residual, sparse);
            for (offset, lane) in lanes.iter_mut().enumerate() {
                let index = task * chunk + offset;
                unsafe {
                    let cur = lane_mut(buffers.currents, index, m.num_all_nodes);
                    let chg = lane_mut(buffers.charges, index, m.num_all_nodes);
                    let cond = lane_mut(buffers.conductances, index, g);
                    let cap = lane_mut(buffers.capacitances, index, c);
                    cur.fill(0.0);
                    chg.fill(0.0);
                    cond.fill(0.0);
                    cap.fill(0.0);
                    eval_device_from_setup(
                        m,
                        lane.model.bytes(m.model_size),
                        lane.instance.bytes(m.instance_size),
                        lane_input(buffers.voltages, index, m.num_all_nodes),
                        cur,
                        cond,
                        chg,
                        cap,
                        residual,
                        sparse,
                        &mut scratch.small,
                        Some(&plans[index]),
                        true,
                    );
                }
            }
        })
    };
    if config.parallel(count) {
        lanes.par_chunks_mut(chunk).enumerate().for_each(work);
    } else {
        lanes.chunks_mut(chunk).enumerate().for_each(work);
    }
}

pub(super) fn evaluate_snapshots(
    h: &OsdiBatchHandle,
    m: &LoadedOsdi,
    count: usize,
    buffers: Buffers,
    residual: bool,
    sparse: bool,
    config: &Configuration,
) {
    let (g, c) = widths(m, residual, sparse);
    // Borrow TLS once per task, rather than once per lane. The task size also
    // bounds scheduling overhead; serial calls use a single borrow.
    let chunk = if config.parallel(count) {
        config.min_chunk
    } else {
        count.max(1)
    };
    let work = |task: usize| {
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            for lane in task * chunk..((task + 1) * chunk).min(count) {
                let snapshot = lane % h.num_devices;
                scratch.model_data.clear();
                scratch.model_data.extend_from_slice(
                    &h.model_data_flat[snapshot * m.model_size..(snapshot + 1) * m.model_size],
                );
                scratch.inst_data.clear();
                scratch.inst_data.extend_from_slice(
                    &h.inst_data_flat[snapshot * m.instance_size..(snapshot + 1) * m.instance_size],
                );
                let EvalScratch {
                    model_data,
                    inst_data,
                    small,
                } = &mut *scratch;
                unsafe {
                    let cur = lane_mut(buffers.currents, lane, m.num_all_nodes);
                    let chg = lane_mut(buffers.charges, lane, m.num_all_nodes);
                    let cond = lane_mut(buffers.conductances, lane, g);
                    let cap = lane_mut(buffers.capacitances, lane, c);
                    cur.fill(0.0);
                    chg.fill(0.0);
                    cond.fill(0.0);
                    cap.fill(0.0);
                    eval_device_from_setup(
                        m,
                        model_data,
                        inst_data,
                        lane_input(buffers.voltages, lane, m.num_all_nodes),
                        cur,
                        cond,
                        chg,
                        cap,
                        residual,
                        sparse,
                        small,
                        Some(&h.plans[snapshot]),
                        false,
                    );
                }
            }
        })
    };
    let tasks = count.div_ceil(chunk);
    if config.parallel(count) {
        (0..tasks).into_par_iter().for_each(work);
    } else {
        (0..tasks).for_each(work);
    }
}

fn dense_target(row: i32, col: i32, width: usize) -> Option<usize> {
    if row < 0 || col < 0 {
        None
    } else {
        Some(row as usize * width + col as usize)
    }
}

pub(super) struct LanePlan {
    pub node_map: Vec<i32>,
    pub resist_targets: Vec<Option<usize>>,
    pub react_targets: Vec<Option<usize>>,
    pub direct_voltage: bool,
    pub equalities: Vec<(usize, usize)>,
}
impl LanePlan {
    pub(super) fn new(m: &LoadedOsdi, instance: &[u8]) -> Self {
        let node_map: Vec<i32> = (0..m.num_all_nodes)
            .map(|i| unsafe {
                (instance.as_ptr().add(m.node_map_off + i * 4) as *const i32).read_unaligned()
            })
            .collect();
        let targets = |pairs: &[(u32, u32)]| {
            pairs
                .iter()
                .map(|&(r, c)| {
                    let r = *node_map.get(r as usize)?;
                    let c = *node_map.get(c as usize)?;
                    if r < 0 || c < 0 {
                        return None;
                    }
                    let row = *m.slot_to_out.get(r as usize)?;
                    let col = *m.slot_to_out.get(c as usize)?;
                    dense_target(row, col, m.num_all_nodes)
                })
                .collect()
        };
        Self {
            resist_targets: targets(&m.resist_jac_pairs),
            react_targets: targets(&m.react_jac_pairs),
            direct_voltage: node_map
                .iter()
                .all(|&s| s >= 0 && (s as usize) < m.num_all_nodes),
            equalities: node_map
                .iter()
                .enumerate()
                .filter_map(|(node, &survivor)| {
                    (node != survivor as usize).then_some((node, survivor as usize))
                })
                .collect(),
            node_map,
        }
    }
}

// Mutable provider workspaces are owned by a binding, never derived by casting
// shared handle slices to mutable pointers. u64 backing guarantees the OSDI
// scalar/pointer alignment even when descriptor byte sizes are not multiples of 8.
struct Memory {
    words: Box<[u64]>,
}
impl Memory {
    fn new(snapshot: &[u8]) -> Self {
        let mut mem = Self {
            words: vec![0; snapshot.len().div_ceil(8)].into_boxed_slice(),
        };
        mem.bytes(snapshot.len()).copy_from_slice(snapshot);
        mem
    }
    fn bytes(&mut self, len: usize) -> &mut [u8] {
        debug_assert!(len <= self.words.len() * 8);
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast(), len) }
    }
}
pub(super) struct Lane {
    model: Memory,
    instance: Memory,
}
struct Binding {
    handle_id: u64,
    generation: u64,
    buffers: Buffers,
    lanes: Vec<Lane>,
    qualified: bool,
    active: bool,
}
lazy_static::lazy_static! {
    static ref BINDINGS: RwLock<HashMap<u64, Arc<Mutex<Binding>>>> = RwLock::new(HashMap::new());
}
static NEXT_GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
static NEXT_BINDING: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// All lengths count doubles. F/Q include a final private ground slot per lane;
/// their stride is num_nodes+1. G/C are raw-entry arrays, without equality rows.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BosdiBufferBinding {
    pub voltages: *const f64,
    pub voltages_len: usize,
    pub currents: *mut f64,
    pub currents_len: usize,
    pub conductances: *mut f64,
    pub conductances_len: usize,
    pub charges: *mut f64,
    pub charges_len: usize,
    pub capacitances: *mut f64,
    pub capacitances_len: usize,
}
impl BosdiBufferBinding {
    fn buffers(self) -> Buffers {
        Buffers {
            voltages: self.voltages as usize,
            currents: self.currents as usize,
            conductances: self.conductances as usize,
            charges: self.charges as usize,
            capacitances: self.capacitances as usize,
        }
    }
    fn regions(self) -> [(usize, usize); 5] {
        [
            (self.voltages as usize, self.voltages_len),
            (self.currents as usize, self.currents_len),
            (self.conductances as usize, self.conductances_len),
            (self.charges as usize, self.charges_len),
            (self.capacitances as usize, self.capacitances_len),
        ]
    }
    fn valid(self, h: &OsdiBatchHandle, m: &LoadedOsdi) -> bool {
        let widths = [
            m.num_all_nodes,
            m.num_slots,
            m.num_resist_jac as usize,
            m.num_slots,
            m.num_react_jac as usize,
        ];
        h.num_devices > 0
            && self
                .regions()
                .iter()
                .zip(widths)
                .all(|(&(_, len), width)| h.num_devices.checked_mul(width) == Some(len))
            && disjoint_regions(&self.regions(), 8)
    }
}
fn disjoint_regions(regions: &[(usize, usize)], element_size: usize) -> bool {
    let mut ranges = Vec::with_capacity(regions.len());
    for &(addr, len) in regions {
        if len == 0 {
            continue;
        }
        let Some(bytes) = len
            .checked_mul(element_size)
            .filter(|&n| n <= isize::MAX as usize)
        else {
            return false;
        };
        let Some(end) = addr.checked_add(bytes) else {
            return false;
        };
        if addr == 0 || addr % element_size != 0 {
            return false;
        }
        if ranges
            .iter()
            .any(|&(start, stop)| addr < stop && start < end)
        {
            return false;
        }
        ranges.push((addr, end));
    }
    true
}

/// Explicit qualification for the shared JAX/C handle path. Mutable workspaces
/// are exclusively borrowed under a per-handle lock and split into disjoint
/// lanes. Tiled calls always retain post-setup snapshot restoration.
#[no_mangle]
pub extern "C" fn bosdi_qualify_handle(id: u64, pure_recomputing_eval: bool) -> bool {
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&id) else {
        return false;
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&h.model_id) else {
        return false;
    };
    if pure_recomputing_eval && m.num_states != 0 {
        return false;
    }
    *h.qualified_work.lock().unwrap() = pure_recomputing_eval.then(|| {
        (0..h.num_devices)
            .map(|i| Lane {
                model: Memory::new(&h.model_data_flat[i * m.model_size..(i + 1) * m.model_size]),
                instance: Memory::new(
                    &h.inst_data_flat[i * m.instance_size..(i + 1) * m.instance_size],
                ),
            })
            .collect()
    });
    true
}

/// Registers caller-owned buffers. The caller keeps every buffer alive and
/// exclusively available to BOSDI through unbind/rebind (including host threads).
/// This API never tiles a handle. Returns zero on invalid handle/layout/aliasing.
#[no_mangle]
pub unsafe extern "C" fn bosdi_bind_buffers(
    handle_id: u64,
    desc: *const BosdiBufferBinding,
) -> u64 {
    if desc.is_null() {
        return 0;
    }
    let desc = *desc;
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&handle_id) else {
        return 0;
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&h.model_id) else {
        return 0;
    };
    if !desc.valid(h, m) {
        return 0;
    }
    let lanes = (0..h.num_devices)
        .map(|i| Lane {
            model: Memory::new(&h.model_data_flat[i * m.model_size..(i + 1) * m.model_size]),
            instance: Memory::new(
                &h.inst_data_flat[i * m.instance_size..(i + 1) * m.instance_size],
            ),
        })
        .collect();
    let id = NEXT_BINDING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    BINDINGS.write().unwrap().insert(
        id,
        Arc::new(Mutex::new(Binding {
            handle_id,
            generation: next_generation(),
            buffers: desc.buffers(),
            lanes,
            qualified: false,
            active: true,
        })),
    );
    id
}
fn binding(id: u64) -> Option<Arc<Mutex<Binding>>> {
    BINDINGS.read().unwrap().get(&id).cloned()
}

/// Successful rebinding advances the generation, even if an address is unchanged.
/// Failed rebind preserves the binding.
#[no_mangle]
pub unsafe extern "C" fn bosdi_rebind_buffers(id: u64, desc: *const BosdiBufferBinding) -> u64 {
    if desc.is_null() {
        return 0;
    }
    let Some(b) = binding(id) else {
        return 0;
    };
    let mut b = b.lock().unwrap();
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&b.handle_id) else {
        return 0;
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&h.model_id) else {
        return 0;
    };
    if !(*desc).valid(h, m) {
        return 0;
    }
    b.buffers = (*desc).buffers();
    b.generation = next_generation();
    b.generation
}
#[no_mangle]
pub extern "C" fn bosdi_unbind_buffers(id: u64) {
    if let Some(b) = BINDINGS.write().unwrap().remove(&id) {
        // Wait for any in-flight call before allowing the owner to release buffers.
        let mut guard = b.lock().unwrap();
        guard.active = false;
    }
}

/// Caller certifies this provider fully overwrites all eval intermediates and
/// reads no history. This is a qualification assertion, not inferred metadata.
/// Models exposing ABI state slots cannot qualify. History-dependent providers
/// cannot qualify even when they expose no slots; this is a provider assertion.
#[no_mangle]
pub extern "C" fn bosdi_qualify_binding(id: u64, pure_recomputing_eval: bool) -> bool {
    let Some(b) = binding(id) else {
        return false;
    };
    let mut b = b.lock().unwrap();
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&b.handle_id) else {
        return false;
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&h.model_id) else {
        return false;
    };
    if pure_recomputing_eval && m.num_states != 0 {
        return false;
    }
    b.qualified = pure_recomputing_eval;
    // A qualification change invalidates prior evaluation assumptions.
    b.generation = next_generation();
    true
}
#[no_mangle]
pub extern "C" fn bosdi_binding_generation(id: u64) -> u64 {
    binding(id).map_or(0, |b| b.lock().unwrap().generation)
}

/// Exact bound evaluation with optional output finiteness checks.
/// Returns 0 on success, -1 for stale/unbound handles, and -4 for nonfinite outputs.
#[no_mangle]
pub unsafe extern "C" fn bosdi_eval_bound(
    id: u64,
    generation: u64,
    residual_only: bool,
    check_finite: bool,
) -> i32 {
    let Some(b) = binding(id) else {
        return -1;
    };
    let mut b = b.lock().unwrap();
    if !b.active || generation != b.generation {
        return -1;
    }
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&b.handle_id) else {
        return -1;
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&h.model_id) else {
        return -1;
    };
    let config = configuration_for_handle(h, &configuration());
    let buffers = b.buffers;
    let restore = !b.qualified || m.num_states != 0;
    let b_qualified = b.qualified;
    let chunk = if config.parallel(h.num_devices) {
        config.min_chunk
    } else {
        h.num_devices.max(1)
    };
    let work = |(task, lanes): (usize, &mut [Lane])| -> bool {
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let mut direct_info = OsdiSimInfo {
                paras: OsdiSimParas {
                    names: m.simparam_name_ptrs.as_ptr() as *mut *mut i8,
                    vals: m.simparam_values.as_ptr() as *mut f64,
                    names_str: std::ptr::null_mut(),
                    vals_str: std::ptr::null_mut(),
                },
                abstime: 0.0,
                prev_solve: std::ptr::null_mut(),
                prev_state: std::ptr::null_mut(),
                next_state: std::ptr::null_mut(),
                flags: evaluation_flags(m, residual_only),
                _pad: 0,
            };
            let mut finite = true;
            for (offset, lane) in lanes.iter_mut().enumerate() {
                let index = task * chunk + offset;
                let vol = lane_input(buffers.voltages, index, m.num_all_nodes);
                if restore {
                    lane.model.bytes(m.model_size).copy_from_slice(
                        &h.model_data_flat[index * m.model_size..(index + 1) * m.model_size],
                    );
                    lane.instance.bytes(m.instance_size).copy_from_slice(
                        &h.inst_data_flat[index * m.instance_size..(index + 1) * m.instance_size],
                    );
                }
                let cur = lane_mut(buffers.currents, index, m.num_slots);
                let chg = lane_mut(buffers.charges, index, m.num_slots);
                let (g, c) = widths(m, residual_only, true);
                let cond = lane_mut(buffers.conductances, index, g);
                let cap = lane_mut(buffers.capacitances, index, c);
                cur.fill(0.0);
                chg.fill(0.0);
                // Overwrite callbacks need no zero fill; missing/DC reactive
                // outputs still need defined zeros. NaN audits check coverage.
                if check_finite {
                    cond.fill(f64::NAN);
                } else if !b_qualified {
                    cond.fill(0.0);
                }
                if m.analysis == 0 || m.write_jacobian_react.is_none() {
                    cap.fill(0.0);
                } else if check_finite {
                    cap.fill(f64::NAN);
                } else if !b_qualified {
                    cap.fill(0.0);
                }
                let plan = &h.plans[index];
                if b_qualified && plan.direct_voltage {
                    direct_info.prev_solve = vol.as_ptr() as *mut f64;
                    eval_bound_direct(
                        m,
                        lane,
                        plan,
                        &mut direct_info,
                        vol,
                        cur,
                        cond,
                        chg,
                        cap,
                        residual_only,
                    );
                } else {
                    eval_device_from_setup(
                        m,
                        lane.model.bytes(m.model_size),
                        lane.instance.bytes(m.instance_size),
                        vol,
                        cur,
                        cond,
                        chg,
                        cap,
                        residual_only,
                        true,
                        &mut scratch.small,
                        Some(plan),
                        false,
                    );
                }
                if check_finite {
                    finite &= cur[..m.num_all_nodes]
                        .iter()
                        .chain(&chg[..m.num_all_nodes])
                        .chain(cond.iter())
                        .chain(cap.iter())
                        .all(|x| x.is_finite());
                }
            }
            finite
        })
    };
    let lanes = &mut b.lanes[..];
    let mut execute = || {
        if config.parallel(h.num_devices) {
            lanes
                .par_chunks_mut(chunk)
                .enumerate()
                .map(work)
                .reduce(|| true, combine)
        } else {
            lanes
                .chunks_mut(chunk)
                .enumerate()
                .map(work)
                .fold(true, combine)
        }
    };
    let finite = if config.parallel(h.num_devices) {
        config.pool.install(execute)
    } else {
        execute()
    };
    if finite {
        0
    } else {
        -4
    }
}
#[inline]
fn eval_bound_direct(
    m: &LoadedOsdi,
    lane: &mut Lane,
    plan: &LanePlan,
    info: &mut OsdiSimInfo,
    vol: &[f64],
    cur: &mut [f64],
    cond: &mut [f64],
    chg: &mut [f64],
    cap: &mut [f64],
    residual: bool,
) {
    let inst = lane.instance.bytes(m.instance_size).as_mut_ptr().cast();
    let model = lane.model.bytes(m.model_size).as_mut_ptr().cast();
    // Caller qualification ensures no history, no state slots, and read-only
    // prev_solve. All pointers refer to this lane's exclusively owned regions.
    unsafe {
        (m.eval)(
            std::ptr::null_mut(),
            inst,
            model,
            info as *mut _ as *mut c_void,
        );
        if m.num_resist_jac > 0 {
            (m.load_residual)(inst, model, cur.as_mut_ptr());
            if !residual {
                (m.write_jacobian)(inst, model, cond.as_mut_ptr());
            }
        }
        if m.analysis != 0 && m.num_react_jac > 0 {
            if let Some(load) = m.load_residual_react {
                load(inst, model, chg.as_mut_ptr());
            }
            if !residual {
                if let Some(write) = m.write_jacobian_react {
                    write(inst, model, cap.as_mut_ptr());
                }
            }
        }
    }
    for &(node, survivor) in &plan.equalities {
        cur[node] = if survivor == m.num_all_nodes {
            vol[node]
        } else {
            vol[node] - vol[survivor]
        };
        chg[node] = 0.0;
    }
}

fn combine(a: bool, b: bool) -> bool {
    a && b
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ground_targets_and_workspace_alignment() {
        assert_eq!(dense_target(-1, 0, 13), None);
        assert_eq!(dense_target(0, -1, 13), None);
        assert_eq!(dense_target(1, 2, 13), Some(15));
        let mut memory = Memory::new(&[7; 9]);
        assert_eq!(memory.bytes(9).as_ptr() as usize % 8, 0);
        assert_eq!(memory.bytes(9), &[7; 9]);
    }
    #[test]
    fn owned_pool_and_comparison_contract() {
        assert!(bosdi_set_num_threads(2));
        let config = configuration();
        assert_eq!(config.pool.install(rayon::current_num_threads), 2);
        assert!(bosdi_set_num_threads(3));
        assert_eq!(bosdi_num_threads(), 3);
        assert_eq!(config.pool.install(rayon::current_num_threads), 2);
        // Our configuration has not initialized or modified the global pool.
        assert!(ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .is_ok());
        assert_eq!(rayon::current_num_threads(), 1);
    }
}
