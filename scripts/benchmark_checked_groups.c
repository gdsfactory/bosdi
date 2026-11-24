#include "bosdi_runtime.h"
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
static double seconds(clockid_t id) {
  struct timespec t;
  clock_gettime(id, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
typedef int32_t (*eval_fn)(uint64_t, const BosdiBufferBinding *,
                           const BosdiEvalContext *, bool, bool,
                           BosdiCallError *, BosdiLogCallback, void *);
int time_checked(eval_fn fn, uint64_t h, const BosdiBufferBinding *b, size_t n,
                 double duration, double *out) {
  BosdiEvalContext c = {2, 0, 0};
  BosdiCallError e;
  size_t calls = 0;
  for (int i = 0; i < 50; i++)
    if (fn(h, b, &c, true, false, &e, 0, 0))
      return e.status;
  double cpu = seconds(CLOCK_PROCESS_CPUTIME_ID),
         start = seconds(CLOCK_MONOTONIC), elapsed;
  do {
    for (int k = 0; k < 100; k++) {
      if (fn(h, b, &c, true, false, &e, 0, 0))
        return e.status;
      calls++;
    }
    elapsed = seconds(CLOCK_MONOTONIC) - start;
  } while (elapsed < duration);
  out[0] = elapsed * 1e9 / (calls * n);
  out[1] = (seconds(CLOCK_PROCESS_CPUTIME_ID) - cpu) / elapsed;
  return 0;
}
typedef int32_t (*setup_fn)(uint32_t, size_t, size_t, const double *,
                            const double *, size_t, const BosdiEvalContext *,
                            uint64_t *, BosdiCallError *, BosdiLogCallback,
                            void *);
typedef int32_t (*update_fn)(uint64_t, size_t, const size_t *, size_t,
                             const double *, const double *, size_t,
                             const BosdiEvalContext *, BosdiCallError *,
                             BosdiLogCallback, void *);
typedef void (*free_fn)(uint64_t);
int time_prepare(setup_fn setup, update_fn update, free_fn release,
                 uint32_t model, uint64_t handle, const double *params,
                 size_t nparams, const double *x, size_t xlen, size_t count,
                 bool replace, double duration, double *out) {
  size_t *lanes = malloc(count * sizeof(size_t));
  if (!lanes)
    return -2;
  for (size_t i = 0; i < count; i++)
    lanes[i] = i;
  BosdiEvalContext context = {2, 3, 0};
  BosdiCallError error;
  size_t calls = 0;
  double start = seconds(CLOCK_MONOTONIC), elapsed;
  do {
    uint64_t h = 0;
    int status = replace ? update(handle, count, lanes, nparams, params, x,
                                  xlen, &context, &error, 0, 0)
                         : setup(model, count, nparams, params, x, xlen,
                                 &context, &h, &error, 0, 0);
    if (status) {
      free(lanes);
      return status;
    }
    if (!replace)
      release(h);
    calls++;
    elapsed = seconds(CLOCK_MONOTONIC) - start;
  } while (elapsed < duration);
  out[0] = elapsed * 1e9 / (calls * count);
  free(lanes);
  return 0;
}

typedef int32_t (*stage_fn)(uint64_t, size_t, const size_t *, size_t,
                            const double *, const double *, size_t,
                            const BosdiEvalContext *, uint64_t *,
                            BosdiCallError *, BosdiLogCallback, void *);
int time_stage_discard(stage_fn stage, free_fn discard, uint64_t handle,
                       const double *params, size_t nparams, const double *x,
                       size_t xlen, size_t count, double duration,
                       double *out) {
  size_t *lanes = malloc(count * sizeof(size_t));
  if (!lanes)
    return -2;
  for (size_t i = 0; i < count; i++)
    lanes[i] = i;
  BosdiEvalContext context = {2, 3, 0};
  BosdiCallError error;
  size_t calls = 0;
  double start = seconds(CLOCK_MONOTONIC), elapsed;
  do {
    uint64_t token = 0;
    int status = stage(handle, count, lanes, nparams, params, x, xlen, &context,
                       &token, &error, 0, 0);
    if (status) {
      free(lanes);
      return status;
    }
    discard(token);
    calls++;
    elapsed = seconds(CLOCK_MONOTONIC) - start;
  } while (elapsed < duration);
  out[0] = elapsed * 1e9 / (calls * count);
  free(lanes);
  return 0;
}

typedef int32_t (*bound_fn)(uint64_t, uint64_t, const BosdiEvalContext *, bool,
                            bool, BosdiCallError *, BosdiLogCallback, void *);
int time_explicit_bound(eval_fn explicit_eval, bound_fn bound_eval,
                        uint64_t handle, const BosdiBufferBinding *buffers,
                        uint64_t binding, uint64_t generation, size_t count,
                        bool bound, bool residual, double duration,
                        double *out) {
  (void)count; /* This helper reports ns/call. */
  BosdiEvalContext context = {2, 0, 0};
  BosdiCallError error;
  size_t calls = 0;
  double start = seconds(CLOCK_MONOTONIC), elapsed;
  do {
    for (int k = 0; k < 100; k++) {
      int status = bound ? bound_eval(binding, generation, &context, residual,
                                      false, &error, 0, 0)
                         : explicit_eval(handle, buffers, &context, residual,
                                         false, &error, 0, 0);
      if (status)
        return status;
      calls++;
    }
    elapsed = seconds(CLOCK_MONOTONIC) - start;
  } while (elapsed < duration);
  out[0] = elapsed * 1e9 / calls;
  return 0;
}
