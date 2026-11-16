/* Native-only timing loops. No Python, JAX dispatch, or setup inside timing. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../tests/headers/osdi_0_5.h"
#include "../include/bosdi_runtime.h"

typedef struct ScalarLane { void *model, *instance; double *voltage; } ScalarLane;
typedef struct Scalar {
    const OsdiDescriptor *d;
    size_t count, slots;
    ScalarLane *lanes;
    double *f, *g, *q, *c, *state;
    OsdiSimInfo info;
    OsdiSimParas paras;
    char *names[1];
} Scalar;
static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void write_params(Scalar *s, ScalarLane *l, const double *params, uint32_t kind) {
    for (uint32_t i=0; i<s->d->num_params; ++i) {
        uint32_t flags = s->d->param_opvar[i].flags;
        if ((flags & PARA_KIND_MASK) != kind || isnan(params[i]) || (flags & PARA_TY_MASK) == PARA_TY_STR) continue;
        void *ptr = s->d->access(l->instance, l->model, i, ACCESS_FLAG_SET | (kind == PARA_KIND_INST ? ACCESS_FLAG_INSTANCE : 0));
        if (!ptr) continue;
        if ((flags & PARA_TY_MASK) == PARA_TY_INT) *(int32_t*)ptr = (int32_t)params[i];
        else *(double*)ptr = params[i];
        if (kind == PARA_KIND_INST) s->d->given_flag_instance(l->instance,i);
        else s->d->given_flag_model(l->model,i);
    }
}
void *scalar_create(const OsdiDescriptor *d, size_t count, const double *params, const double *voltage, uint32_t analysis) {
    Scalar *s = calloc(1,sizeof(*s));
    s->d=d; s->count=count; s->slots=d->num_nodes+1;
    s->lanes=calloc(count,sizeof(ScalarLane));
    s->f=calloc(count*s->slots,sizeof(double)); s->q=calloc(count*s->slots,sizeof(double));
    s->g=calloc(count*d->num_resistive_jacobian_entries+1,sizeof(double));
    s->c=calloc(count*d->num_reactive_jacobian_entries+1,sizeof(double));
    s->state=calloc(2*d->num_states+1,sizeof(double));
    s->paras.names=s->names;
    s->info.paras=s->paras;
    s->info.flags=analysis == 0 ? ANALYSIS_DC|ANALYSIS_STATIC : ANALYSIS_TRAN;
    if (d->num_resistive_jacobian_entries) s->info.flags |= CALC_RESIST_RESIDUAL|CALC_RESIST_JACOBIAN;
    if (analysis && d->num_reactive_jacobian_entries) s->info.flags |= CALC_REACT_RESIDUAL|CALC_REACT_JACOBIAN;
    s->info.prev_state=s->state; s->info.next_state=s->state+d->num_states;
    for (size_t k=0;k<count;++k) {
        ScalarLane *l=&s->lanes[k];
        l->model=calloc(d->model_size+1,1); l->instance=calloc(d->instance_size+1,1);
        l->voltage=calloc(s->slots,sizeof(double));
        memcpy(l->voltage,voltage+k*d->num_nodes,d->num_nodes*sizeof(double));
        int32_t *map=(int32_t*)((char*)l->instance+d->node_mapping_offset);
        for (uint32_t n=0;n<d->num_nodes;++n) map[n]=n;
        write_params(s,l,params+k*d->num_params,PARA_KIND_MODEL);
        OsdiInitInfo init={0};
        d->setup_model(NULL,l->model,&s->paras,&init);
        write_params(s,l,params+k*d->num_params,PARA_KIND_INST);
        memset(&init,0,sizeof(init));
        d->setup_instance(NULL,l->instance,l->model,300.,d->num_terminals,&s->paras,&init);
        bool *collapsed=(bool*)((char*)l->instance+d->collapsed_offset);
        for (uint32_t pair=0;pair<d->num_collapsible;++pair) {
            if (!collapsed[pair]) continue;
            uint32_t n1=d->collapsible[pair].node_1,n2=d->collapsible[pair].node_2;
            int32_t a=n1 == UINT32_MAX ? -1 : map[n1],b=n2 == UINT32_MAX ? -1 : map[n2];
            int32_t lower=a<b?a:b,upper=a>b?a:b;
            for (uint32_t n=0;n<d->num_nodes;++n) if (map[n]==upper) map[n]=lower;
        }
        for (uint32_t n=0;n<d->num_nodes;++n) if (map[n]<0) map[n]=d->num_nodes;
    }
    return s;
}
void scalar_run(void *opaque) {
    Scalar *s=opaque; const OsdiDescriptor *d=s->d;
    for (size_t k=0;k<s->count;++k) {
        ScalarLane *l=&s->lanes[k];
        double *f=s->f+k*s->slots,*q=s->q+k*s->slots;
        memset(f,0,s->slots*sizeof(double)); memset(q,0,s->slots*sizeof(double));
        s->info.prev_solve=l->voltage;
        memset(s->state,0,2*d->num_states*sizeof(double));
        d->eval(NULL,l->instance,l->model,&s->info);
        if (d->num_resistive_jacobian_entries) {
            d->load_residual_resist(l->instance,l->model,f);
            d->write_jacobian_array_resist(l->instance,l->model,s->g+k*d->num_resistive_jacobian_entries);
        }
        if (s->info.flags & CALC_REACT_RESIDUAL) {
            d->load_residual_react(l->instance,l->model,q);
            d->write_jacobian_array_react(l->instance,l->model,s->c+k*d->num_reactive_jacobian_entries);
        }
        int32_t *map=(int32_t*)((char*)l->instance+d->node_mapping_offset);
        for (uint32_t n=0;n<d->num_nodes;++n) {
            if (map[n] == (int32_t)d->num_nodes) { f[n]=l->voltage[n]; q[n]=0.; }
            else if (map[n] != (int32_t)n) { f[n]=l->voltage[n]-l->voltage[map[n]]; q[n]=0.; }
        }
    }
}
double *scalar_output(void *opaque, unsigned which) {
    Scalar *s=opaque;
    switch(which) { case 0:return s->f; case 1:return s->g; case 2:return s->q; default:return s->c; }
}
void scalar_free(void *opaque) {
    Scalar *s=opaque;
    for (size_t k=0;k<s->count;++k) { free(s->lanes[k].model);free(s->lanes[k].instance);free(s->lanes[k].voltage); }
    free(s->lanes);free(s->f);free(s->q);free(s->g);free(s->c);free(s->state);free(s);
}
double time_scalar(void *s, size_t lanes, double seconds) {
    size_t repeats=0; double start=now(),end;
    do { scalar_run(s); ++repeats; end=now(); } while(end-start<seconds);
    return (end-start)*1e9/repeats/lanes;
}
typedef void (*Eval)(uint64_t,size_t,size_t,size_t,const double*,const double*,double*,double*,double*,double*,double*);
double time_handle(Eval eval,uint64_t handle,size_t count,size_t nodes,size_t states,
    const double *v,double *f,double *g,double *q,double *c,double *state,double seconds) {
    size_t repeats=0;double start=now(),end;
    do { eval(handle,count,nodes,states,v,state,f,g,q,c,state); ++repeats;end=now(); } while(end-start<seconds);
    return (end-start)*1e9/repeats/count;
}
typedef int32_t (*BoundEval)(uint64_t,uint64_t,bool,bool);
double time_bound(BoundEval eval,uint64_t binding,uint64_t generation,size_t count,double seconds) {
    size_t repeats=0;double start=now(),end;
    do { if(eval(binding,generation,false,false)) return -1.; ++repeats;end=now(); } while(end-start<seconds);
    return (end-start)*1e9/repeats/count;
}
typedef int32_t (*BypassEval)(uint64_t,uint64_t,bool,bool,const BosdiBypass*,BosdiEvalStats*);
double time_bound_bypass(BypassEval eval,uint64_t binding,uint64_t generation,size_t count,
    const BosdiBypass *bypass,double seconds) {
    size_t repeats=0;double start=now(),end;
    do { if(eval(binding,generation,false,false,bypass,NULL)) return -1.; ++repeats;end=now(); } while(end-start<seconds);
    return (end-start)*1e9/repeats/count;
}
