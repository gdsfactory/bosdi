/* ABI fixture for the mb-experimental osdi_0_5.h. */
#include "osdi_0_5.h"
#include <stddef.h>

/* All offsets consumed by AbiLayout, independently checked by the C compiler. */
#define OFFSET(field, value) _Static_assert(offsetof(OsdiDescriptor, field) == value, #field)
OFFSET(num_nodes, 8);
OFFSET(num_terminals, 12);
OFFSET(num_params, 76);
OFFSET(node_mapping_offset, 96);
OFFSET(num_states, 104);
OFFSET(instance_size, 116);
OFFSET(model_size, 120);
OFFSET(load_residual_resist, 168);
OFFSET(load_residual_react, 176);
OFFSET(num_resistive_jacobian_entries, 256);
OFFSET(num_reactive_jacobian_entries, 260);
OFFSET(write_jacobian_array_resist, 264);
OFFSET(write_jacobian_array_react, 272);
OFFSET(num_collapsible, 40);
OFFSET(collapsible, 48);
OFFSET(collapsed_offset, 56);
OFFSET(num_jacobian_entries, 24);
OFFSET(jacobian_entries, 32);
OFFSET(param_opvar, 88);
OFFSET(access, 128);
OFFSET(setup_model, 136);
OFFSET(setup_instance, 144);
OFFSET(eval, 152);
OFFSET(given_flag_model, 240);
OFFSET(given_flag_instance, 248);
_Static_assert(sizeof(OsdiSimInfo) == 72, "OsdiSimInfo");
_Static_assert(offsetof(OsdiSimInfo, prev_solve) == 40, "prev_solve");
_Static_assert(offsetof(OsdiSimInfo, flags) == 64, "flags");
_Static_assert(sizeof(OsdiParamOpvar) == 40, "OsdiParamOpvar");
_Static_assert(sizeof(OsdiJacobianEntry) == 16, "OsdiJacobianEntry");

typedef struct { double c; } Model;
typedef struct { uint32_t map[2]; double r, current, charge; } Instance;

uint32_t OSDI_VERSION_MAJOR = 0;
uint32_t OSDI_VERSION_MINOR = 5;
uint32_t OSDI_NUM_DESCRIPTORS = 1;

static void *access_param(void *i, void *m, uint32_t id, uint32_t flags) {
    (void)flags;
    return id == 0 ? (void *)&((Instance *)i)->r : (void *)&((Model *)m)->c;
}
static void setup_model(void *h, void *m, OsdiSimParas *p, OsdiInitInfo *res) {
    (void)h; (void)p; (void)res;
    if (((Model *)m)->c == 0) ((Model *)m)->c = 1e-9;
}
static void setup_instance(void *h, void *i, void *m, double t, uint32_t n,
                           OsdiSimParas *p, OsdiInitInfo *res) {
    (void)h; (void)m; (void)t; (void)n; (void)p; (void)res;
    if (((Instance *)i)->r == 0) ((Instance *)i)->r = 1000;
}
static uint32_t eval(void *h, void *i, void *m, OsdiSimInfo *info) {
    (void)h;
    Instance *inst = i;
    double v = info->prev_solve[inst->map[0]] - info->prev_solve[inst->map[1]];
    inst->current = info->flags & CALC_RESIST_RESIDUAL ? v / inst->r : 0;
    inst->charge = info->flags & CALC_REACT_RESIDUAL ? v * ((Model *)m)->c : 0;
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
    double g = 1 / ((Instance *)i)->r;
    dst[0] = g; dst[1] = -g; dst[2] = -g; dst[3] = g;
}
static void reactive_jacobian(void *i, void *m, double *dst) {
    (void)i;
    double c = ((Model *)m)->c;
    dst[0] = c; dst[1] = -c; dst[2] = -c; dst[3] = c;
}
static char *r_names[] = {"r"}, *c_names[] = {"c"};
static OsdiParamOpvar params[] = {
    {.name=r_names, .flags=PARA_KIND_INST | PARA_TY_REAL},
    {.name=c_names, .flags=PARA_KIND_MODEL | PARA_TY_REAL},
};
static OsdiJacobianEntry entries[] = {
    {{0,0}, 0, JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT},
    {{0,1}, 0, JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT},
    {{1,0}, 0, JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT},
    {{1,1}, 0, JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT},
};
OsdiDescriptor OSDI_DESCRIPTORS[] = {{
    .name="header_rc", .num_nodes=2, .num_terminals=2,
    .num_jacobian_entries=4, .jacobian_entries=entries,
    .num_params=2, .num_instance_params=1, .param_opvar=params,
    .node_mapping_offset=offsetof(Instance, map),
    .instance_size=sizeof(Instance), .model_size=sizeof(Model),
    .access=access_param, .setup_model=setup_model, .setup_instance=setup_instance,
    .eval=eval, .load_residual_resist=residual, .load_residual_react=reactive_residual,
    .num_resistive_jacobian_entries=4, .num_reactive_jacobian_entries=4,
    .write_jacobian_array_resist=jacobian, .write_jacobian_array_react=reactive_jacobian,
    .module_flags=0, .absdelay_count=0, .absdelays=NULL,
}};
