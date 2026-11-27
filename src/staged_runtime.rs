//! Caller-owned staged lane candidates. Lock order: HANDLES -> registry -> STAGES.
//! Tokens hold owned bytes/indices, never a lock or a provider library reference.
use super::*;
use checked_runtime::{finish, BosdiCallError, BosdiEvalContext, LogCallback};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

struct Prepared {
    lanes: Vec<usize>,
    models: Vec<u8>,
    instances: Vec<u8>,
    plans: Vec<Option<handle_runtime::LanePlan>>,
    td: Vec<f64>,
    model_size: usize,
    instance_size: usize,
}
struct Stage {
    handle: u64,
    revision: u64,
    prepared: Option<Prepared>,
}
lazy_static::lazy_static! { static ref STAGES: Mutex<HashMap<u64, Stage>> = Mutex::new(HashMap::new()); }
static NEXT_STAGE: AtomicU64 = AtomicU64::new(1);

/// Caller must hold the handle write lock. Stale tokens retain only a tombstone.
pub(super) fn invalidate_stages(handle: u64) {
    invalidate(&mut STAGES.lock().unwrap(), handle);
}
fn invalidate(stages: &mut HashMap<u64, Stage>, handle: u64) {
    for stage in stages.values_mut().filter(|s| s.handle == handle) {
        stage.prepared = None;
    }
}
unsafe fn prepare(
    h: &OsdiBatchHandle,
    m: &LoadedOsdi,
    count: usize,
    lanes: *const usize,
    num_params: usize,
    params: *const f64,
    voltages: *const f64,
    voltages_len: usize,
    context: *const BosdiEvalContext,
    callback: LogCallback,
    user: *mut c_void,
    allow_defaults: bool,
) -> Result<Prepared, BosdiCallError> {
    let bad = |status| BosdiCallError {
        status,
        ..Default::default()
    };
    if count == 0
        || count > h.num_devices
        || lanes.is_null()
        || context.is_null()
        || num_params != m.param_flags.len()
        || count.checked_mul(num_params).is_none()
        || count.checked_mul(m.num_all_nodes) != Some(voltages_len)
        || voltages.is_null()
        || (num_params > 0 && params.is_null())
        || (*context).analysis > 2
        || (*context).options & !(3 | checked_runtime::AC_WITHOUT_STATIC) != 0
        || ((*context).options & checked_runtime::AC_WITHOUT_STATIC != 0
            && (*context).analysis != 1)
        || !(*context).abstime.is_finite()
        || (*context).abstime < 0.0
    {
        return Err(bad(-2));
    }
    let lanes = std::slice::from_raw_parts(lanes, count);
    let params = if num_params == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(params, count * num_params)
    };
    let voltages = std::slice::from_raw_parts(voltages, voltages_len);
    let mut seen = vec![false; h.num_devices];
    let mut nonfinite = BosdiCallError::default();
    for (row, &lane) in lanes.iter().enumerate() {
        if lane >= h.num_devices || seen[lane] {
            return Err(bad(-2));
        }
        seen[lane] = true;
        if !params[row * num_params..(row + 1) * num_params]
            .iter()
            .all(|x| x.is_finite() || (allow_defaults && x.is_nan()))
            || !voltages[row * m.num_all_nodes..(row + 1) * m.num_all_nodes]
                .iter()
                .all(|x| x.is_finite())
        {
            nonfinite = checked_runtime::lowest(nonfinite, BosdiCallError { lane, ..bad(-4) });
        }
    }
    if nonfinite.status != 0 {
        return Err(nonfinite);
    }
    if m.delay_error.is_some() {
        return Err(bad(-7));
    }
    let mut models = vec![0; count * m.model_size];
    let mut instances = vec![0; count * m.instance_size];
    let mut plans: Vec<Option<handle_runtime::LanePlan>> = (0..count).map(|_| None).collect();
    let context = *context;
    let user = user as usize;
    let config = handle_runtime::configuration_for_handle(h, &configuration());
    let parallel = config.parallel(count);
    let chunk = if parallel { config.min_chunk } else { count };
    let work = |(task, ((models, instances), plans)): (
        usize,
        (
            (&mut [u8], &mut [u8]),
            &mut [Option<handle_runtime::LanePlan>],
        ),
    )| {
        // Allocate validation outputs once per task, never once per lane.
        let mut f = vec![0.0; m.num_slots];
        let mut q = vec![0.0; m.num_slots];
        let mut g = vec![0.0; m.num_resist_jac as usize];
        let mut c = vec![0.0; m.num_react_jac as usize];
        let mut failure = BosdiCallError::default();
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            for (offset, ((model, instance), plan_out)) in models
                .chunks_mut(m.model_size)
                .zip(instances.chunks_mut(m.instance_size))
                .zip(plans.iter_mut())
                .enumerate()
            {
                let row = task * chunk + offset;
                let lane = lanes[row];
                let result = checked_runtime::scoped(Some(context), lane, callback, user, || {
                    setup_device(
                        m,
                        &params[row * num_params..(row + 1) * num_params],
                        model,
                        instance,
                    );
                    if checked_runtime::failed() {
                        return;
                    }
                    let plan = handle_runtime::LanePlan::new(m, instance);
                    if !delay_runtime::nodes_preserved(m, &plan)
                        || plan.node_map != h.plans[lane].node_map
                        || instance[m.collapsed_off..m.collapsed_off + m.collapsible_pairs.len()]
                            != h.inst_data_flat[lane * m.instance_size + m.collapsed_off
                                ..lane * m.instance_size
                                    + m.collapsed_off
                                    + m.collapsible_pairs.len()]
                    {
                        checked_runtime::phase(-7);
                        checked_runtime::fail(b"staged topology changed");
                        return;
                    }
                    scratch.model_data.clear();
                    scratch.model_data.extend_from_slice(model);
                    scratch.inst_data.clear();
                    scratch.inst_data.extend_from_slice(instance);
                    let EvalScratch {
                        model_data,
                        inst_data,
                        small,
                    } = &mut *scratch;
                    f.fill(0.0);
                    q.fill(0.0);
                    g.fill(0.0);
                    c.fill(0.0);
                    eval_device_from_setup(
                        m,
                        model_data,
                        inst_data,
                        &voltages[row * m.num_all_nodes..(row + 1) * m.num_all_nodes],
                        &mut f,
                        &mut g,
                        &mut q,
                        &mut c,
                        false,
                        true,
                        small,
                        Some(&plan),
                        false,
                    );
                    if !f
                        .iter()
                        .chain(&q)
                        .chain(&g)
                        .chain(&c)
                        .all(|x| x.is_finite())
                        || m.delays.iter().any(|d| !d.duration(inst_data).is_finite())
                    {
                        checked_runtime::phase(-4);
                        checked_runtime::fail(b"non-finite staged output");
                    }
                    for d in &m.delays {
                        let start = d.td_offset as usize;
                        instance[start..start + 8].copy_from_slice(&inst_data[start..start + 8]);
                    }
                    *plan_out = Some(plan);
                });
                if result.status != 0 {
                    failure = checked_runtime::lowest(failure, result);
                }
            }
        });
        failure
    };
    let result = if parallel {
        config.pool.install(|| {
            models
                .par_chunks_mut(chunk * m.model_size)
                .zip(instances.par_chunks_mut(chunk * m.instance_size))
                .zip(plans.par_chunks_mut(chunk))
                .enumerate()
                .map(work)
                .reduce(BosdiCallError::default, checked_runtime::lowest)
        })
    } else {
        work((0, ((&mut models, &mut instances), &mut plans)))
    };
    if result.status != 0 {
        return Err(result);
    }

    let td = instances
        .chunks(m.instance_size)
        .flat_map(|instance| m.delays.iter().map(move |d| d.duration(instance)))
        .collect();
    Ok(Prepared {
        lanes: lanes.to_vec(),
        models,
        instances,
        plans,
        td,
        model_size: m.model_size,
        instance_size: m.instance_size,
    })
}
fn publish(h: &mut OsdiBatchHandle, prepared: Prepared) {
    let Prepared {
        lanes,
        models,
        instances,
        mut plans,
        model_size,
        instance_size,
        ..
    } = prepared;
    if lanes.len() == h.num_devices && lanes.iter().copied().eq(0..lanes.len()) {
        // All-lane identity update transfers the staged snapshots without copying.
        h.model_data_flat = models;
        h.inst_data_flat = instances;
        for (lane, plan) in plans.into_iter().enumerate() {
            h.plans[lane] = plan.unwrap();
        }
    } else {
        for (row, &lane) in lanes.iter().enumerate() {
            h.model_data_flat[lane * model_size..(lane + 1) * model_size]
                .copy_from_slice(&models[row * model_size..(row + 1) * model_size]);
            h.inst_data_flat[lane * instance_size..(lane + 1) * instance_size]
                .copy_from_slice(&instances[row * instance_size..(row + 1) * instance_size]);
            h.plans[lane] = plans[row].take().unwrap();
        }
    }
    *h.qualified_work.lock().unwrap() = None;
    h.revision += 1;
    h.observed_workers
        .store(0, std::sync::atomic::Ordering::Relaxed);
}

#[no_mangle]
pub unsafe extern "C" fn bosdi_stage_lanes_checked(
    handle: u64,
    count: usize,
    lanes: *const usize,
    num_params: usize,
    params: *const f64,
    voltages: *const f64,
    voltages_len: usize,
    context: *const BosdiEvalContext,
    out_stage: *mut u64,
    error: *mut BosdiCallError,
    callback: LogCallback,
    user: *mut c_void,
) -> i32 {
    if out_stage.is_null() {
        return finish(checked_runtime::diagnostic(-2, b"null stage output"), error);
    }
    *out_stage = 0;
    // Read-only staging is allowed alongside accepted-handle evaluations.
    let Ok(handles) = HANDLES.try_read() else {
        return finish(checked_runtime::diagnostic(-8, b"handles busy"), error);
    };
    let Some(h) = handles.get(&handle) else {
        return finish(checked_runtime::diagnostic(-1, b"unknown handle"), error);
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = &registry[&h.model_id];
    let prepared = match prepare(
        h,
        m,
        count,
        lanes,
        num_params,
        params,
        voltages,
        voltages_len,
        context,
        callback,
        user,
        true,
    ) {
        Ok(p) => p,
        Err(e) => return finish(e, error),
    };
    let id = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
    STAGES.lock().unwrap().insert(
        id,
        Stage {
            handle,
            revision: h.revision,
            prepared: Some(prepared),
        },
    );
    *out_stage = id;
    finish(BosdiCallError::default(), error)
}

#[no_mangle]
pub unsafe extern "C" fn bosdi_stage_prepared_delays(id: u64, out: *mut f64, len: usize) -> i32 {
    let handles = HANDLES.read().unwrap();
    let stages = STAGES.lock().unwrap();
    let Some(stage) = stages.get(&id) else {
        return -1;
    };
    if !handles
        .get(&stage.handle)
        .is_some_and(|h| h.revision == stage.revision)
    {
        return -9;
    }
    let Some(p) = stage.prepared.as_ref() else {
        return -9;
    };
    if len != p.td.len() || (len > 0 && (out.is_null() || out as usize % 8 != 0)) {
        return -2;
    }
    if len > 0 {
        std::ptr::copy_nonoverlapping(p.td.as_ptr(), out, len);
    }
    0
}
#[no_mangle]
pub extern "C" fn bosdi_commit_stage(id: u64) -> i32 {
    // No token is consumed on busy or stale. Publication performs no setup/eval/allocation.
    let Ok(mut handles) = HANDLES.try_write() else {
        return -8;
    };
    let mut stages = STAGES.lock().unwrap();
    let Some(stage) = stages.get(&id) else {
        return -1;
    };
    let Some(h) = handles.get_mut(&stage.handle) else {
        return -9;
    };
    if h.revision != stage.revision || stage.prepared.is_none() {
        return -9;
    }
    let stage = stages.remove(&id).unwrap();
    publish(h, stage.prepared.unwrap());
    invalidate(&mut stages, stage.handle);
    0
}
#[no_mangle]
pub extern "C" fn bosdi_discard_stage(id: u64) {
    STAGES.lock().unwrap().remove(&id);
}

pub(super) unsafe fn update(
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
    allow_defaults: bool,
) -> i32 {
    // Preserve the original immediate call's exclusion throughout setup/publication.
    let Ok(mut handles) = HANDLES.try_write() else {
        return finish(
            BosdiCallError {
                status: -8,
                ..Default::default()
            },
            error,
        );
    };
    let Some(h) = handles.get_mut(&handle) else {
        return finish(
            BosdiCallError {
                status: -1,
                ..Default::default()
            },
            error,
        );
    };
    let registry = OSDI_REGISTRY.read().unwrap();
    let m = &registry[&h.model_id];
    let prepared = match prepare(
        h,
        m,
        count,
        lanes,
        num_params,
        params,
        voltages,
        voltages_len,
        context,
        callback,
        user,
        allow_defaults,
    ) {
        Ok(p) => p,
        Err(e) => return finish(e, error),
    };
    publish(h, prepared);
    invalidate_stages(handle);
    finish(BosdiCallError::default(), error)
}
#[no_mangle]
pub unsafe extern "C" fn bosdi_update_lanes_defaults_checked(
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
    update(
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
        true,
    )
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use handle_runtime::BosdiBufferBinding;

    fn snapshot(id: u64) -> (Vec<u8>, Vec<u8>, Vec<Vec<i32>>, u64, usize, Option<usize>) {
        let handles = HANDLES.read().unwrap();
        let h = &handles[&id];
        let result = (
            h.model_data_flat.clone(),
            h.inst_data_flat.clone(),
            h.plans.iter().map(|p| p.node_map.clone()).collect(),
            h.revision,
            h.observed_workers.load(Ordering::Relaxed),
            h.qualified_work
                .lock()
                .unwrap()
                .as_ref()
                .map(|w| w.as_ptr() as usize),
        );
        result
    }
    #[test]
    fn staged_snapshots_owned_and_released() {
        let _guard = RUNTIME_TEST_LOCK.lock().unwrap();
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let target = std::env::temp_dir().join(format!("bosdi-stage-{}.osdi", std::process::id()));
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
        let params = [2000.0, 2e-9, 2000.0, 2e-9, 2000.0, 2e-9];
        let defaults = [f64::NAN; 6];
        let lanes = [0, 1, 2];
        let x = [0.25; 9];
        let mut f = [0.0; 12];
        let mut q = [0.0; 12];
        let mut g = [0.0; 12];
        let mut c = [0.0; 12];
        let buffers = BosdiBufferBinding {
            voltages: x.as_ptr(),
            voltages_len: 9,
            currents: f.as_mut_ptr(),
            currents_len: 12,
            charges: q.as_mut_ptr(),
            charges_len: 12,
            conductances: g.as_mut_ptr(),
            conductances_len: 12,
            capacitances: c.as_mut_ptr(),
            capacitances_len: 12,
        };
        let context = BosdiEvalContext {
            analysis: 2,
            options: 3,
            abstime: 0.0,
        };
        let mut error = BosdiCallError::default();
        let mut h = 0;
        let mut fresh = 0;
        unsafe {
            assert_eq!(
                delay_runtime::bosdi_setup_batch_delays_checked(
                    md.model_id,
                    3,
                    2,
                    params.as_ptr(),
                    x.as_ptr(),
                    9,
                    &context,
                    &mut h,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            assert!(handle_runtime::bosdi_qualify_handle(h, true));
            assert_eq!(
                checked_runtime::bosdi_eval_handle_checked(
                    h,
                    &buffers,
                    &context,
                    false,
                    true,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            let binding = handle_runtime::bosdi_bind_buffers(h, &buffers);
            assert_ne!(binding, 0);
            let generation = handle_runtime::bosdi_binding_generation(binding);
            let before = snapshot(h);
            assert!(before.5.is_some());
            assert_eq!(before.4, 1);
            let initial = STAGES.lock().unwrap().len();
            for _ in 0..100 {
                let mut token = 0;
                assert_eq!(
                    bosdi_stage_lanes_checked(
                        h,
                        3,
                        lanes.as_ptr(),
                        2,
                        defaults.as_ptr(),
                        x.as_ptr(),
                        9,
                        &context,
                        &mut token,
                        &mut error,
                        None,
                        std::ptr::null_mut()
                    ),
                    0
                );
                assert_eq!(snapshot(h), before);
                assert_eq!(
                    handle_runtime::bosdi_binding_generation(binding),
                    generation
                );
                bosdi_discard_stage(token);
                bosdi_discard_stage(token);
                assert_eq!(STAGES.lock().unwrap().len(), initial);
            }
            // NaN setup, stage+commit, and defaults-aware update produce identical raw snapshots.
            assert_eq!(
                delay_runtime::bosdi_setup_batch_delays_checked(
                    md.model_id,
                    3,
                    2,
                    defaults.as_ptr(),
                    x.as_ptr(),
                    9,
                    &context,
                    &mut fresh,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            let mut token = 0;
            assert_eq!(
                bosdi_stage_lanes_checked(
                    h,
                    3,
                    lanes.as_ptr(),
                    2,
                    defaults.as_ptr(),
                    x.as_ptr(),
                    9,
                    &context,
                    &mut token,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(bosdi_commit_stage(token), 0);
            let accepted = snapshot(h);
            let expected = snapshot(fresh);
            assert_eq!(accepted.0, expected.0);
            assert_eq!(accepted.1, expected.1);
            assert_eq!(accepted.2, expected.2);
            assert_eq!(accepted.5, None);
            assert_eq!(
                bosdi_update_lanes_defaults_checked(
                    h,
                    3,
                    lanes.as_ptr(),
                    2,
                    defaults.as_ptr(),
                    x.as_ptr(),
                    9,
                    &context,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(snapshot(h).0, expected.0);
            assert_eq!(snapshot(h).1, expected.1);
            // Finite immediate legacy update agrees byte-for-byte with stage+commit.
            assert_eq!(
                update(
                    h,
                    3,
                    lanes.as_ptr(),
                    2,
                    params.as_ptr(),
                    x.as_ptr(),
                    9,
                    &context,
                    &mut error,
                    None,
                    std::ptr::null_mut(),
                    false
                ),
                0
            );
            assert_eq!(
                bosdi_stage_lanes_checked(
                    fresh,
                    3,
                    lanes.as_ptr(),
                    2,
                    params.as_ptr(),
                    x.as_ptr(),
                    9,
                    &context,
                    &mut token,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(bosdi_commit_stage(token), 0);
            assert_eq!(snapshot(h).0, snapshot(fresh).0);
            assert_eq!(snapshot(h).1, snapshot(fresh).1);
            assert_eq!(
                bosdi_stage_lanes_checked(
                    h,
                    3,
                    lanes.as_ptr(),
                    2,
                    defaults.as_ptr(),
                    x.as_ptr(),
                    9,
                    &context,
                    &mut token,
                    &mut error,
                    None,
                    std::ptr::null_mut()
                ),
                0
            );
            osdi_free_handle_ffi(h);
            assert!(STAGES.lock().unwrap()[&token].prepared.is_none());
            assert_eq!(bosdi_commit_stage(token), -9);
            bosdi_discard_stage(token);
            assert_eq!(STAGES.lock().unwrap().len(), initial);
            handle_runtime::bosdi_unbind_buffers(binding);
            osdi_free_handle_ffi(fresh);
        }
        assert!(checked_runtime::bosdi_free_model(md.model_id));
        std::fs::remove_file(target).unwrap();
    }
    #[test]
    fn compiled_default_snapshot_parity() {
        let Some(dir) = std::env::var_os("BOSDI_STAGE_FIXTURES") else {
            return;
        };
        let _guard = RUNTIME_TEST_LOCK.lock().unwrap();
        let dir = std::path::PathBuf::from(dir);
        // Compiled OpenVAF artifacts resolve math symbols from their native host.
        // Python supplies libm; this standalone Rust test supplies it explicitly.
        let _math = unsafe {
            libloading::os::unix::Library::open(
                Some("libm.so.6"),
                libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
            )
        }
        .unwrap();
        for (name, version) in [
            ("waveguide.osdi", 5),
            ("resistor.osdi", 4),
            ("psp103v4_psp103.osdi", 4),
        ] {
            let path = CString::new(dir.join(name).to_str().unwrap()).unwrap();
            let md = load_osdi_library_with_analysis(path.as_ptr(), version, 300.0, 0);
            assert!(md.success, "{name}");
            let n = 3;
            let defaults = vec![f64::NAN; n * md.num_params];
            let mut explicit = defaults.clone();
            let (index, nodes) = {
                let registry = OSDI_REGISTRY.read().unwrap();
                let m = &registry[&md.model_id];
                (
                    m.param_names.iter().position(|n| n == "$mfactor").unwrap(),
                    m.num_all_nodes,
                )
            };
            for row in 0..n {
                explicit[row * md.num_params + index] = 2.0;
            }
            let x = vec![0.25; n * nodes];
            let lanes = [0, 1, 2];
            let context = BosdiEvalContext {
                analysis: 2,
                options: 3,
                abstime: 0.0,
            };
            let mut h = 0;
            let mut fresh = 0;
            let mut error = BosdiCallError::default();
            let mut token = 0;
            unsafe {
                assert_eq!(
                    delay_runtime::bosdi_setup_batch_delays_checked(
                        md.model_id,
                        n,
                        md.num_params,
                        explicit.as_ptr(),
                        x.as_ptr(),
                        x.len(),
                        &context,
                        &mut h,
                        &mut error,
                        None,
                        std::ptr::null_mut()
                    ),
                    0,
                    "{name}"
                );
                assert_eq!(
                    delay_runtime::bosdi_setup_batch_delays_checked(
                        md.model_id,
                        n,
                        md.num_params,
                        defaults.as_ptr(),
                        x.as_ptr(),
                        x.len(),
                        &context,
                        &mut fresh,
                        &mut error,
                        None,
                        std::ptr::null_mut()
                    ),
                    0,
                    "{name}"
                );
                let before = snapshot(h);
                for _ in 0..20 {
                    assert_eq!(
                        bosdi_stage_lanes_checked(
                            h,
                            n,
                            lanes.as_ptr(),
                            md.num_params,
                            defaults.as_ptr(),
                            x.as_ptr(),
                            x.len(),
                            &context,
                            &mut token,
                            &mut error,
                            None,
                            std::ptr::null_mut()
                        ),
                        0,
                        "{name}"
                    );
                    assert_eq!(snapshot(h), before);
                    bosdi_discard_stage(token);
                }
                assert_eq!(
                    bosdi_stage_lanes_checked(
                        h,
                        n,
                        lanes.as_ptr(),
                        md.num_params,
                        defaults.as_ptr(),
                        x.as_ptr(),
                        x.len(),
                        &context,
                        &mut token,
                        &mut error,
                        None,
                        std::ptr::null_mut()
                    ),
                    0,
                    "{name}"
                );
                assert_eq!(bosdi_commit_stage(token), 0);
                let got = snapshot(h);
                let want = snapshot(fresh);
                assert_eq!(got.0, want.0, "model bytes: {name}");
                assert_eq!(got.1, want.1, "instance bytes: {name}");
                assert_eq!(got.2, want.2, "maps: {name}");
                osdi_free_handle_ffi(h);
                osdi_free_handle_ffi(fresh);
            }
            assert!(checked_runtime::bosdi_free_model(md.model_id));
        }
    }
}
