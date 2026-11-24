//! Additive checked delay observation and transactional lane replacement.
use super::*;
use checked_runtime::{BosdiCallError, BosdiEvalContext, LogCallback};
use handle_runtime::BosdiBufferBinding;

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Delay {
    y_node: u32,
    z_node: u32,
    pub(super) td_offset: u32,
    maxdelay_offset: u32,
}
impl Delay {
    pub(super) fn duration(&self, instance: &[u8]) -> f64 {
        unsafe {
            instance
                .as_ptr()
                .add(self.td_offset as usize)
                .cast::<f64>()
                .read_unaligned()
        }
    }
}

pub(super) unsafe fn read_metadata(
    desc: *const u8,
    nodes: u32,
    terminals: u32,
    size: usize,
) -> (Vec<Delay>, Option<String>) {
    let count = read_u32(desc, 348) as usize;
    let ptr = *(desc.add(352) as *const *const Delay);
    if count > nodes as usize / 2 || (count > 0 && ptr.is_null()) {
        return (Vec::new(), Some("invalid OSDI delay table".into()));
    }
    let delays = if count == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(ptr, count).to_vec()
    };
    let mut seen = vec![false; nodes as usize];
    for d in &delays {
        for node in [d.y_node, d.z_node] {
            if node < terminals || node >= nodes || seen[node as usize] {
                return (
                    delays,
                    Some("overlapping, terminal or invalid delay nodes".into()),
                );
            }
            seen[node as usize] = true;
        }
        for (index, offset) in [d.td_offset, d.maxdelay_offset].into_iter().enumerate() {
            if index == 1 && offset == u32::MAX {
                continue;
            }
            if (offset as usize)
                .checked_add(8)
                .is_none_or(|end| end > size)
            {
                return (delays, Some("invalid delay storage offset".into()));
            }
        }
    }
    (delays, None)
}

#[repr(C)]
pub struct BosdiDelayMetadata {
    y_node: u32,
    z_node: u32,
    has_maxdelay: u32,
}
/// Enabled registry-v1 feature bits (CG2). Bit 32 is reserved, not advertised.
#[no_mangle]
pub extern "C" fn bosdi_runtime_capabilities() -> u64 {
    1 | 2 | 4 | 8 | 16
}

/// Query count with NULL/0, then copy exactly count metadata records.
#[no_mangle]
pub unsafe extern "C" fn bosdi_model_delays(
    model: u32,
    out: *mut BosdiDelayMetadata,
    len: usize,
    count: *mut usize,
) -> i32 {
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&model) else {
        return -1;
    };
    if count.is_null() {
        return -2;
    }
    *count = m.delays.len();
    if m.delay_error.is_some() {
        return -7;
    }
    if out.is_null() && len == 0 {
        return 0;
    }
    if out.is_null() || len != m.delays.len() {
        return -2;
    }
    for (i, d) in m.delays.iter().enumerate() {
        *out.add(i) = BosdiDelayMetadata {
            y_node: d.y_node,
            z_node: d.z_node,
            has_maxdelay: u32::from(d.maxdelay_offset != u32::MAX),
        };
    }
    0
}
#[no_mangle]
pub unsafe extern "C" fn bosdi_handle_prepared_delays(
    handle: u64,
    out: *mut f64,
    len: usize,
) -> i32 {
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&handle) else {
        return -1;
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = &registry[&h.model_id];
    if m.delay_error.is_some() {
        return -7;
    }
    if len != h.num_devices * m.delays.len()
        || (len > 0 && (out.is_null() || out as usize % 8 != 0))
    {
        return -2;
    }
    for lane in 0..h.num_devices {
        for (i, d) in m.delays.iter().enumerate() {
            *out.add(lane * m.delays.len() + i) =
                d.duration(&h.inst_data_flat[lane * m.instance_size..(lane + 1) * m.instance_size]);
        }
    }
    0
}
#[no_mangle]
pub extern "C" fn bosdi_handle_observed_workers(handle: u64) -> usize {
    HANDLES.read().unwrap().get(&handle).map_or(0, |h| {
        h.observed_workers
            .load(std::sync::atomic::Ordering::Relaxed)
    })
}
#[no_mangle]
pub unsafe extern "C" fn bosdi_eval_handle_delays_checked(
    handle: u64,
    buffers: *const BosdiBufferBinding,
    context: *const BosdiEvalContext,
    residual: bool,
    check_finite: bool,
    delays: *mut f64,
    len: usize,
    error: *mut BosdiCallError,
    callback: LogCallback,
    user: *mut c_void,
) -> i32 {
    let bad = |status| BosdiCallError {
        status,
        ..Default::default()
    };
    if buffers.is_null() || context.is_null() {
        return checked_runtime::finish(bad(-2), error);
    }
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&handle) else {
        return checked_runtime::finish(bad(-1), error);
    };
    h.observed_workers
        .store(0, std::sync::atomic::Ordering::Relaxed);
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = &registry[&h.model_id];
    let b = *buffers;
    let expected = h.num_devices.checked_mul(m.delays.len());
    let regions = [
        (delays as usize, len),
        (b.voltages as usize, b.voltages_len),
        (b.currents as usize, b.currents_len),
        (b.charges as usize, b.charges_len),
        (b.conductances as usize, b.conductances_len),
        (b.capacitances as usize, b.capacitances_len),
    ];
    if !((delays.is_null() && len == 0) || expected == Some(len))
        || !handle_runtime::disjoint_regions(&regions, 8)
    {
        return checked_runtime::finish(bad(-2), error);
    }
    checked_runtime::finish(
        checked_runtime::evaluate_delays(
            h,
            m,
            b,
            *context,
            residual,
            check_finite,
            callback,
            user as usize,
            delays as usize,
        ),
        error,
    )
}

/// Fresh setup, validate at caller voltages, then publish all or none.
/// Nonblocking global handle write acquisition refuses in-flight evaluations.
#[no_mangle]
pub unsafe extern "C" fn bosdi_update_lanes_checked(
    handle: u64,
    count: usize,
    lanes: *const usize,
    num_params: usize,
    params: *const f64,
    voltages: *const f64,
    voltages_len: usize,
    context: *const BosdiEvalContext,
    error: *mut BosdiCallError,
    callback: LogCallback,
    user: *mut c_void,
) -> i32 {
    staged_runtime::update(
        handle,
        count,
        lanes,
        num_params,
        params,
        voltages,
        voltages_len,
        context,
        error,
        callback,
        user,
        false,
    )
}

/// Setup plus an explicit preparation eval for compilers which only write td in eval.
#[no_mangle]
pub unsafe extern "C" fn bosdi_setup_batch_delays_checked(
    model: u32,
    count: usize,
    num_params: usize,
    params: *const f64,
    voltages: *const f64,
    voltages_len: usize,
    context: *const BosdiEvalContext,
    out: *mut u64,
    error: *mut BosdiCallError,
    callback: LogCallback,
    user: *mut c_void,
) -> i32 {
    let bad = |status| BosdiCallError {
        status,
        ..Default::default()
    };
    if out.is_null() {
        return checked_runtime::finish(bad(-2), error);
    }
    *out = 0;
    if context.is_null() || voltages.is_null() {
        return checked_runtime::finish(bad(-2), error);
    }
    {
        let registry = OSDI_REGISTRY.read().unwrap();
        let Some(m) = registry.get(&model) else {
            return checked_runtime::finish(bad(-1), error);
        };
        if count.checked_mul(m.num_all_nodes) != Some(voltages_len)
            || [
                m.num_slots,
                m.num_resist_jac as usize,
                m.num_react_jac as usize,
                m.delays.len(),
            ]
            .into_iter()
            .any(|n| {
                count
                    .checked_mul(n)
                    .and_then(|n| n.checked_mul(8))
                    .is_none_or(|n| n > isize::MAX as usize)
            })
        {
            return checked_runtime::finish(bad(-2), error);
        }
    }
    let mut id = 0;
    let status = checked_runtime::bosdi_setup_batch_checked(
        model, count, num_params, params, &mut id, error, callback, user,
    );
    if status != 0 {
        return status;
    }
    let mut handles = HANDLES.write().unwrap();
    let h = handles.get_mut(&id).unwrap();
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = &registry[&model];
    let mut f = vec![0.0; count * m.num_slots];
    let mut q = vec![0.0; count * m.num_slots];
    let mut g = vec![0.0; count * m.num_resist_jac as usize];
    let mut c = vec![0.0; count * m.num_react_jac as usize];
    let mut td = vec![0.0_f64; count * m.delays.len()];
    let b = BosdiBufferBinding {
        voltages,
        voltages_len,
        currents: f.as_mut_ptr(),
        currents_len: f.len(),
        charges: q.as_mut_ptr(),
        charges_len: q.len(),
        conductances: g.as_mut_ptr(),
        conductances_len: g.len(),
        capacitances: c.as_mut_ptr(),
        capacitances_len: c.len(),
    };
    let result = checked_runtime::evaluate_delays(
        h,
        m,
        b,
        *context,
        false,
        true,
        callback,
        user as usize,
        td.as_mut_ptr() as usize,
    );
    if result.status != 0 {
        handles.remove(&id);
        return checked_runtime::finish(result, error);
    }
    for lane in 0..count {
        for (i, d) in m.delays.iter().enumerate() {
            h.inst_data_flat[lane * m.instance_size + d.td_offset as usize
                ..lane * m.instance_size + d.td_offset as usize + 8]
                .copy_from_slice(&td[lane * m.delays.len() + i].to_ne_bytes());
        }
    }
    h.observed_workers
        .store(0, std::sync::atomic::Ordering::Relaxed);
    *out = id;
    checked_runtime::finish(result, error)
}

pub(super) fn nodes_preserved(m: &LoadedOsdi, plan: &handle_runtime::LanePlan) -> bool {
    m.delays.iter().all(|d| {
        [d.y_node, d.z_node]
            .into_iter()
            .all(|n| plan.node_map[n as usize] == n as i32)
    })
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    struct CountingAllocator;
    thread_local! {
        static TRACK: Cell<bool> = const { Cell::new(false) };
        static ALLOCS: Cell<usize> = const { Cell::new(0) };
    }
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if TRACK.try_with(Cell::get).unwrap_or(false) {
                ALLOCS.with(|n| n.set(n.get() + 1));
            }
            System.alloc(layout)
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout);
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            if TRACK.try_with(Cell::get).unwrap_or(false) {
                ALLOCS.with(|n| n.set(n.get() + 1));
            }
            System.realloc(ptr, layout, size)
        }
    }
    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    #[test]
    fn checked_serial_allocation_free_after_warmup() {
        let _guard = RUNTIME_TEST_LOCK.lock().unwrap();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let target = std::env::temp_dir().join(format!("bosdi-alloc-{}.osdi", std::process::id()));
        assert!(std::process::Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(format!("-I{}", root.join("tests/headers").display()))
            .arg(root.join("tests/fixtures/embedder.c"))
            .arg("-o")
            .arg(&target)
            .status()
            .unwrap()
            .success());
        let path = CString::new(target.to_str().unwrap()).unwrap();
        let md = load_osdi_library_with_analysis(path.as_ptr(), 5, 300.0, 0);
        assert!(md.success);
        unsafe {
            let mut h = 0;
            let mut error = BosdiCallError::default();
            let params = [1000.0, 1e-9];
            assert_eq!(
                checked_runtime::bosdi_setup_batch_checked(
                    md.model_id,
                    1,
                    2,
                    params.as_ptr(),
                    &mut h,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            assert!(handle_runtime::bosdi_set_handle_parallel_policy(
                h,
                usize::MAX,
                1
            ));
            let v = [0.2, 0.0, 0.0];
            let mut f = [0.0; 4];
            let mut q = [0.0; 4];
            let mut g = [0.0; 4];
            let mut c = [0.0; 4];
            let b = BosdiBufferBinding {
                voltages: v.as_ptr(),
                voltages_len: 3,
                currents: f.as_mut_ptr(),
                currents_len: 4,
                charges: q.as_mut_ptr(),
                charges_len: 4,
                conductances: g.as_mut_ptr(),
                conductances_len: 4,
                capacitances: c.as_mut_ptr(),
                capacitances_len: 4,
            };
            let context = BosdiEvalContext {
                analysis: 2,
                options: 3,
                abstime: 0.0,
            };
            for _ in 0..10 {
                assert_eq!(
                    checked_runtime::bosdi_eval_handle_checked(
                        h,
                        &b,
                        &context,
                        false,
                        true,
                        &mut error,
                        None,
                        std::ptr::null_mut()
                    ),
                    0
                );
            }
            ALLOCS.with(|n| n.set(0));
            TRACK.with(|n| n.set(true));
            let status = checked_runtime::bosdi_eval_handle_checked(
                h,
                &b,
                &context,
                false,
                true,
                &mut error,
                None,
                std::ptr::null_mut(),
            );
            TRACK.with(|n| n.set(false));
            assert_eq!(status, 0);
            assert_eq!(ALLOCS.with(Cell::get), 0);
            osdi_free_handle_ffi(h);
        }
        assert!(checked_runtime::bosdi_free_model(md.model_id));
        std::fs::remove_file(target).unwrap();
    }
}
