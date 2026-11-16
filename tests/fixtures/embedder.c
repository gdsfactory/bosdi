/* Synthetic embedder contract fixture based on osdi_0_5.c. */
#include "osdi_0_5.h"
#include <stddef.h>

typedef struct { double c; } Model;
typedef struct { uint32_t map[3]; double r, current, charge; uint8_t collapsed; } Instance;

void (*osdi_log)(void *, const char *, uint32_t);
#ifdef ZERO_MODEL
#define CAP(m) 1e-9
#else
#define CAP(m) (((Model *)(m))->c)
#endif
uint32_t OSDI_VERSION_MAJOR = 0;
uint32_t OSDI_VERSION_MINOR = 5;
uint32_t OSDI_NUM_DESCRIPTORS = 1;

static void *access_param(void *i, void *m, uint32_t id, uint32_t flags) {
    (void)flags;
#ifdef ZERO_MODEL
    return id == 0 ? (void *)&((Instance *)i)->r : NULL;
#else
    return id == 0 ? (void *)&((Instance *)i)->r : (void *)&CAP(m);
#endif
}
static void setup_model(void *h, void *m, OsdiSimParas *p, OsdiInitInfo *res) {
    (void)h; (void)p; (void)res;
#ifndef ZERO_MODEL
    if (CAP(m) == -1) { res->flags = 1; return; }
    if (CAP(m) == -2) { res->num_errors = 1; return; }
    if (CAP(m) == -3) { osdi_log(h, "model error", LOG_LVL_ERR); return; }
    if (CAP(m) == 0) ((Model *)m)->c = 1e-9;
#endif
}
static void setup_instance(void *h, void *i, void *m, double t, uint32_t n,
                           OsdiSimParas *p, OsdiInitInfo *res) {
    (void)h; (void)m; (void)t; (void)n; (void)p; (void)res;
    Instance *inst = i;
    if (inst->r == -1) { res->flags = 1; return; }
    if (inst->r == -2) { osdi_log(h, "instance error", LOG_LVL_ERR); return; }
    if (inst->r == -6) { res->num_errors = 1; return; }
    if (inst->r == 0) inst->r = 1000;
    inst->collapsed = inst->r == 42;
}
static uint32_t eval(void *h, void *i, void *m, OsdiSimInfo *info) {
    (void)h;
    Instance *inst = i;
    if (inst->r == -3) return EVAL_RET_FLAG_FATAL;
    if (inst->r == -4) { osdi_log(h, "eval error", LOG_LVL_FATAL | LOG_FMT_ERR); return 0; }
    if (inst->r == -7) return EVAL_RET_FLAG_LIM;
    if (inst->r == -8) return EVAL_RET_FLAG_STOP;
    double v = info->prev_solve[inst->map[0]] - info->prev_solve[inst->map[1]];
    inst->current = info->flags & CALC_RESIST_RESIDUAL ? (v / inst->r + info->abstime) : 0;
    if (inst->r == -5) inst->current = 0.0 / 0.0;
    inst->charge = info->flags & CALC_REACT_RESIDUAL ? v * CAP(m) : 0;
    return 0;
}
static void residual(void *i, void *m, double *dst) {
    (void)m;
    Instance *inst = i;
    dst[inst->map[0]] += inst->current;
    dst[inst->map[1]] -= inst->current;
}
static void reactive_residual(void *i, void *m, double *dst) {
    (void)m;
    Instance *inst = i;
    dst[inst->map[0]] += inst->charge;
    dst[inst->map[1]] -= inst->charge;
}
static void jacobian(void *i, void *m, double *dst) {
    (void)m;
#ifdef ZERO_RESIST
    return;
#endif
    double g = 1 / ((Instance *)i)->r;
    dst[0] = g; dst[1] = -g; dst[2] = -g; dst[3] = g;
}
static void reactive_jacobian(void *i, void *m, double *dst) {
    (void)i;
    double c = CAP(m);
    dst[0] = c; dst[1] = -c; dst[2] = -c; dst[3] = c;
}
static char *r_names[] = {"r"}, *c_names[] = {"c"};
static OsdiParamOpvar params[] = {
    {.name=r_names, .flags=PARA_KIND_INST | PARA_TY_REAL},
    {.name=c_names, .flags=PARA_KIND_MODEL | PARA_TY_REAL},
};
#ifdef ZERO_RESIST
#define ENTRY_FLAGS JACOBIAN_ENTRY_REACT
#else
#define ENTRY_FLAGS (JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT)
#endif
static OsdiJacobianEntry entries[] = {
    {{0,0}, 0, ENTRY_FLAGS},
    {{0,1}, 0, ENTRY_FLAGS},
    {{1,0}, 0, ENTRY_FLAGS},
    {{1,1}, 0, ENTRY_FLAGS},
};
static OsdiNodePair collapsible[] = {{0, 2}};
OsdiDescriptor OSDI_DESCRIPTORS[] = {{
    .name="header_rc", .num_nodes=3, .num_terminals=2,
    .num_collapsible=1, .collapsible=collapsible, .collapsed_offset=offsetof(Instance, collapsed),
    .num_jacobian_entries=4, .jacobian_entries=entries,
    .num_params=2, .num_instance_params=1, .param_opvar=params,
    .node_mapping_offset=offsetof(Instance, map),
    .instance_size=sizeof(Instance), .model_size=
#ifdef ZERO_MODEL
    0,
#else
    sizeof(Model),
#endif
    .access=access_param, .setup_model=setup_model, .setup_instance=setup_instance,
    .eval=eval, .load_residual_resist=residual, .load_residual_react=reactive_residual,
    .num_resistive_jacobian_entries=
#ifdef ZERO_RESIST
    0,
#else
    4,
#endif
    .num_reactive_jacobian_entries=4,
    .write_jacobian_array_resist=jacobian, .write_jacobian_array_react=reactive_jacobian,
    .module_flags=0, .absdelay_count=0, .absdelays=NULL,
}};
