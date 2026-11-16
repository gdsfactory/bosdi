//! Versioned, opt-in embedder calls. Provider diagnostics are scoped to a lane
//! on the executing thread, never to a process-wide counter or mutable model.
use super::*;
use handle_runtime::{BosdiBufferBinding, Buffers};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BosdiEvalContext {
    pub analysis: u32,
    pub options: u32,
    pub abstime: f64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BosdiCallError {
    pub status: i32,
    pub lane: usize,
    pub message: [c_char; 256],
}
impl Default for BosdiCallError {
    fn default() -> Self {
        Self {
            status: 0,
            lane: usize::MAX,
            message: [0; 256],
        }
    }
}
pub type LogCallback = Option<unsafe extern "C" fn(*mut c_void, usize, *const c_char, u32)>;
#[derive(Clone, Copy)]
struct Call {
    context: Option<BosdiEvalContext>,
    error: BosdiCallError,
    phase: i32,
    callback: LogCallback,
    user: usize,
}
thread_local! { static CALL: RefCell<Option<Call>> = const { RefCell::new(None) }; }
fn scoped(
    context: Option<BosdiEvalContext>,
    lane: usize,
    callback: LogCallback,
    user: usize,
    work: impl FnOnce(),
) -> BosdiCallError {
    struct Restore(Option<Call>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CALL.with(|c| *c.borrow_mut() = self.0);
        }
    }
    let old = CALL.with(|c| {
        c.replace(Some(Call {
            context,
            error: BosdiCallError {
                lane,
                ..Default::default()
            },
            phase: -5,
            callback,
            user,
        }))
    });
    let _restore = Restore(old);
    work();
    CALL.with(|c| {
        let error = c.borrow().as_ref().unwrap().error;
        if error.status == 0 {
            BosdiCallError::default()
        } else {
            error
        }
    })
}
pub(super) fn context(default_analysis: u32) -> (u32, u32, f64) {
    CALL.with(|c| {
        c.borrow()
            .as_ref()
            .and_then(|c| c.context)
            .map_or((default_analysis, 0, 0.0), |c| {
                (c.analysis, c.options, c.abstime)
            })
    })
}
pub(super) fn phase(phase: i32) {
    CALL.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut() {
            c.phase = phase;
        }
    });
}
fn fail(message: &[u8]) {
    CALL.with(|c| {
        if let Some(c) = c.borrow_mut().as_mut() {
            if c.error.status == 0 {
                c.error.status = c.phase;
                for (dst, &src) in c.error.message.iter_mut().take(255).zip(message) {
                    *dst = src as c_char;
                }
            }
        }
    });
}
fn failed() -> bool {
    CALL.with(|c| c.borrow().as_ref().is_some_and(|c| c.error.status != 0))
}
pub(super) unsafe fn log(msg: *const c_char, level: u32) {
    if level & 7 >= 4 {
        fail(if msg.is_null() {
            b"OSDI error log"
        } else {
            CStr::from_ptr(msg).to_bytes()
        });
    }
    // Release the TLS borrow before invoking caller code.
    let call = CALL.with(|c| *c.borrow());
    if let Some(c) = call {
        if let Some(callback) = c.callback {
            callback(c.user as *mut c_void, c.error.lane, msg, level);
        }
    }
}
pub(super) fn setup_failed(init: &OsdiInitInfo) -> bool {
    if init.flags != 0 || init.num_errors != 0 {
        fail(b"OSDI setup failed");
    }
    failed()
}
pub(super) fn eval_failed(status: u32) -> bool {
    // LIM is an informational convergence bit; fatal/finish/stop are failures
    // for a numerical host. Preserve unrecognized bits as failures too.
    if status & !1 != 0 {
        fail(b"OSDI eval returned failure status");
    }
    failed()
}
pub(super) unsafe fn finish(error: BosdiCallError, out: *mut BosdiCallError) -> i32 {
    if !out.is_null() {
        *out = error;
    }
    error.status
}
fn invalid() -> BosdiCallError {
    BosdiCallError {
        status: -2,
        ..Default::default()
    }
}
#[no_mangle]
pub extern "C" fn bosdi_runtime_abi_version() -> u32 {
    1
}

/// Reject live handles; registry readers keep the library alive during calls.
#[no_mangle]
pub extern "C" fn bosdi_free_model(id: u32) -> bool {
    // Never wait while holding either lock. Evaluators and setup publication
    // acquire handles before the registry.
    let Ok(handles) = HANDLES.try_write() else {
        return false;
    };
    if handles.values().any(|h| h.model_id == id) {
        return false;
    }
    let Ok(mut registry) = OSDI_REGISTRY.try_write() else {
        return false;
    };
    registry.remove(&id).is_some()
}
#[no_mangle]
pub unsafe extern "C" fn bosdi_handle_node_maps(
    handle: u64,
    count: usize,
    nodes: usize,
    out: *mut i32,
) -> i32 {
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&handle) else {
        return -1;
    };
    if count != h.num_devices || out.is_null() {
        return -2;
    }
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&h.model_id) else {
        return -1;
    };
    if nodes != m.num_all_nodes {
        return -2;
    }
    for (lane, plan) in h.plans.iter().enumerate() {
        for (node, &slot) in plan.node_map.iter().enumerate() {
            *out.add(lane * nodes + node) = if slot == nodes as i32 { -1 } else { slot };
        }
    }
    0
}
#[no_mangle]
pub unsafe extern "C" fn bosdi_setup_batch_checked(
    model_id: u32,
    count: usize,
    num_params: usize,
    params: *const f64,
    out_handle: *mut u64,
    error: *mut BosdiCallError,
    callback: LogCallback,
    user: *mut c_void,
) -> i32 {
    if !out_handle.is_null() {
        *out_handle = 0;
    }
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&model_id) else {
        return finish(
            BosdiCallError {
                status: -1,
                ..Default::default()
            },
            error,
        );
    };
    if out_handle.is_null()
        || count == 0
        || num_params != m.param_flags.len()
        || count.checked_mul(num_params).is_none()
        || count.checked_mul(m.model_size).is_none()
        || count.checked_mul(m.instance_size).is_none()
        || (num_params > 0 && params.is_null())
    {
        return finish(invalid(), error);
    }
    let params = if num_params == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(params, count * num_params)
    };
    let mut models = vec![0; count * m.model_size];
    let mut instances = vec![0; count * m.instance_size];
    let user = user as usize;
    let config = configuration();
    let work = |(lane, (model, instance)): (usize, (&mut [u8], &mut [u8]))| {
        scoped(None, lane, callback, user, || {
            setup_device(
                m,
                &params[lane * num_params..(lane + 1) * num_params],
                model,
                instance,
            )
        })
    };
    let errors: Vec<_> = config.pool.install(|| {
        if config.parallel(count) {
            models
                .par_chunks_mut(m.model_size)
                .zip(instances.par_chunks_mut(m.instance_size))
                .enumerate()
                .with_min_len(config.min_chunk)
                .map(work)
                .collect()
        } else {
            models
                .chunks_mut(m.model_size)
                .zip(instances.chunks_mut(m.instance_size))
                .enumerate()
                .map(work)
                .collect()
        }
    });
    if let Some(e) = errors.into_iter().find(|e| e.status != 0) {
        return finish(e, error);
    }
    let plans = instances
        .chunks(m.instance_size)
        .map(|i| handle_runtime::LanePlan::new(m, i))
        .collect();
    drop(registry);
    let mut handles = HANDLES.write().unwrap();
    if !OSDI_REGISTRY.read().unwrap().contains_key(&model_id) {
        return finish(
            BosdiCallError {
                status: -1,
                ..Default::default()
            },
            error,
        );
    }
    let mut next = NEXT_HANDLE_ID.write().unwrap();
    let id = *next;
    *next += 1;
    handles.insert(
        id,
        OsdiBatchHandle {
            model_id,
            num_devices: count,
            model_data_flat: models,
            inst_data_flat: instances,
            parallel_policy: RwLock::new(None),
            qualified_work: std::sync::Mutex::new(None),
            plans,
        },
    );
    *out_handle = id;
    finish(BosdiCallError::default(), error)
}

/// Shared checked path uses restored snapshots even for qualified handles.
/// This makes context changes independent of any cached provider workspace.
pub(super) unsafe fn evaluate(
    h: &OsdiBatchHandle,
    m: &LoadedOsdi,
    buffers: BosdiBufferBinding,
    context: BosdiEvalContext,
    residual: bool,
    check_finite: bool,
    callback: LogCallback,
    user: usize,
) -> BosdiCallError {
    if context.analysis > 2
        || context.options & !3 != 0
        || !context.abstime.is_finite()
        || context.abstime < 0.0
        || !buffers.valid(h, m)
    {
        return invalid();
    }
    let b: Buffers = buffers.buffers();
    let config = handle_runtime::configuration_for_handle(h, &configuration());
    let work = |lane: usize| {
        scoped(Some(context), lane, callback, user, || {
            SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                scratch.model_data.clear();
                scratch.model_data.extend_from_slice(
                    &h.model_data_flat[lane * m.model_size..(lane + 1) * m.model_size],
                );
                scratch.inst_data.clear();
                scratch.inst_data.extend_from_slice(
                    &h.inst_data_flat[lane * m.instance_size..(lane + 1) * m.instance_size],
                );
                let EvalScratch {
                    model_data,
                    inst_data,
                    small,
                } = &mut *scratch;
                let cur = handle_runtime::lane_mut(b.currents, lane, m.num_slots);
                let chg = handle_runtime::lane_mut(b.charges, lane, m.num_slots);
                let g = if residual {
                    0
                } else {
                    m.num_resist_jac as usize
                };
                let c = if residual {
                    0
                } else {
                    m.num_react_jac as usize
                };
                let cond = handle_runtime::lane_mut(b.conductances, lane, g);
                let cap = handle_runtime::lane_mut(b.capacitances, lane, c);
                cur.fill(0.0);
                chg.fill(0.0);
                cond.fill(0.0);
                cap.fill(0.0);
                eval_device_from_setup(
                    m,
                    model_data,
                    inst_data,
                    handle_runtime::lane_input(b.voltages, lane, m.num_all_nodes),
                    cur,
                    cond,
                    chg,
                    cap,
                    residual,
                    true,
                    small,
                    Some(&h.plans[lane]),
                    false,
                );
                if check_finite
                    && !cur
                        .iter()
                        .chain(chg.iter())
                        .chain(cond.iter())
                        .chain(cap.iter())
                        .all(|x| x.is_finite())
                {
                    phase(-4);
                    fail(b"non-finite OSDI output");
                }
            });
        })
    };
    let errors: Vec<_> = if config.parallel(h.num_devices) {
        config.pool.install(|| {
            (0..h.num_devices)
                .into_par_iter()
                .with_min_len(config.min_chunk)
                .map(work)
                .collect()
        })
    } else {
        (0..h.num_devices).map(work).collect()
    };
    errors
        .into_iter()
        .find(|e| e.status != 0)
        .unwrap_or_default()
}
#[no_mangle]
pub unsafe extern "C" fn bosdi_eval_handle_checked(
    handle: u64,
    buffers: *const BosdiBufferBinding,
    context: *const BosdiEvalContext,
    residual: bool,
    check_finite: bool,
    error: *mut BosdiCallError,
    callback: LogCallback,
    user: *mut c_void,
) -> i32 {
    if buffers.is_null() || context.is_null() {
        return finish(invalid(), error);
    }
    let handles = HANDLES.read().unwrap();
    let Some(h) = handles.get(&handle) else {
        return finish(
            BosdiCallError {
                status: -1,
                ..Default::default()
            },
            error,
        );
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let Some(m) = registry.get(&h.model_id) else {
        return finish(invalid(), error);
    };
    finish(
        evaluate(
            h,
            m,
            *buffers,
            *context,
            residual,
            check_finite,
            callback,
            user as usize,
        ),
        error,
    )
}
