/*
 * serpens_hotloop.c
 *
 * Drop-in C replacement for SerpensSimulation.advance_integrate().
 * Implements:
 *   - Magnetic dipole Lorentz force composed with REBOUNDx radiation forces
 *   - Threaded split-integrate-merge of test particles via pthreads
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "rebound.h"
#include "reboundx.h"

/* ========================================================================
 *  Lorentz force configuration  (set from Python before calling integrate)
 * ======================================================================== */

typedef struct {
    int    enabled;
    int    central_index;       /* index of the central body in sim->particles */
    double moment[3];           /* magnetic dipole moment [A m^2] */
    double mag_tilt_rad;        /* tilt in radians */
    double mag_rotation[3];     /* rotation axis vector */
    double mag_rotation_norm;   /* |mag_rotation|, precomputed */
    double softening;           /* ignore r < softening [m] */
} lorentz_config_t;

static lorentz_config_t g_lorentz = {0};

/* Exported: Python sets these before each integrate call */
void serpens_set_lorentz_config(
    int enabled,
    int central_index,
    double mx, double my, double mz,
    double mag_tilt_rad,
    double rotx, double roty, double rotz,
    double softening)
{
    g_lorentz.enabled        = enabled;
    g_lorentz.central_index  = central_index;
    g_lorentz.moment[0]      = mx;
    g_lorentz.moment[1]      = my;
    g_lorentz.moment[2]      = mz;
    g_lorentz.mag_tilt_rad   = mag_tilt_rad;
    g_lorentz.mag_rotation[0]= rotx;
    g_lorentz.mag_rotation[1]= roty;
    g_lorentz.mag_rotation[2]= rotz;
    g_lorentz.mag_rotation_norm = sqrt(rotx*rotx + roty*roty + rotz*rotz);
    g_lorentz.softening      = softening;
}

/* ========================================================================
 *  Dipole B-field in SI
 * ======================================================================== */

static inline void dipole_B(const double m_vec[3], const double r_vec[3], double B_out[3])
{
    double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
    if (r2 == 0.0) { B_out[0] = B_out[1] = B_out[2] = 0.0; return; }

    double r     = sqrt(r2);
    double inv_r = 1.0 / r;
    double rhat[3] = { r_vec[0]*inv_r, r_vec[1]*inv_r, r_vec[2]*inv_r };

    double mu0_4pi = 1e-7;
    double factor  = mu0_4pi / (r * r2);   /* mu0/(4pi * r^3) */

    double m_dot_rhat = m_vec[0]*rhat[0] + m_vec[1]*rhat[1] + m_vec[2]*rhat[2];

    B_out[0] = factor * (3.0 * rhat[0] * m_dot_rhat - m_vec[0]);
    B_out[1] = factor * (3.0 * rhat[1] * m_dot_rhat - m_vec[1]);
    B_out[2] = factor * (3.0 * rhat[2] * m_dot_rhat - m_vec[2]);
}

/* ========================================================================
 *  REBOUNDx velocity-dependent callback — Lorentz force
 * ======================================================================== */

static void lorentz_force(struct reb_simulation* sim, struct rebx_force* force,
                          struct reb_particle* particles, const int N)
{
    (void)force;
    if (!g_lorentz.enabled) return;

    int ci = g_lorentz.central_index;
    if (ci < 0 || ci >= N) return;

    struct reb_particle* central = &particles[ci];

    /* Time-dependent dipole orientation */
    double tilt = g_lorentz.mag_tilt_rad;
    double omega_t = g_lorentz.mag_rotation_norm * sim->t;

    double m_vec[3];
    m_vec[0] = g_lorentz.moment[0] * sin(tilt) * cos(omega_t);
    m_vec[1] = g_lorentz.moment[1] * sin(tilt) * sin(omega_t);
    m_vec[2] = g_lorentz.moment[2] * cos(tilt);

    double soft2 = g_lorentz.softening * g_lorentz.softening;

    struct rebx_extras* rebx = sim->extras;  /* rebx attached to this sim */

    for (int i = sim->N_active; i < N; i++) {
        struct reb_particle* p = &particles[i];

        /* Get q/m from REBOUNDx params */
        double* q_over_m_ptr = rebx_get_param(rebx, p->ap, "q_over_m");
        if (q_over_m_ptr == NULL) continue;
        double qm = *q_over_m_ptr;
        if (qm == 0.0) continue;

        double r_rel[3] = {
            p->x - central->x,
            p->y - central->y,
            p->z - central->z
        };

        if (soft2 > 0.0) {
            double rr = r_rel[0]*r_rel[0] + r_rel[1]*r_rel[1] + r_rel[2]*r_rel[2];
            if (rr < soft2) continue;
        }

        double v_rel[3] = {
            p->vx - central->vx,
            p->vy - central->vy,
            p->vz - central->vz
        };

        double B[3];
        dipole_B(m_vec, r_rel, B);


        /* v_eff = v_rel - (omega x r_rel)   (corotation correction) */
        double* rot = g_lorentz.mag_rotation;
        double omega_cross_r[3] = {
            rot[1]*r_rel[2] - rot[2]*r_rel[1],
            rot[2]*r_rel[0] - rot[0]*r_rel[2],
            rot[0]*r_rel[1] - rot[1]*r_rel[0]
        };
        double v_eff[3] = {
            v_rel[0] - omega_cross_r[0],
            v_rel[1] - omega_cross_r[1],
            v_rel[2] - omega_cross_r[2]
        };

        /* a = (q/m) * (v_eff x B) */
        p->ax += qm * (v_eff[1]*B[2] - v_eff[2]*B[1]);
        p->ay += qm * (v_eff[2]*B[0] - v_eff[0]*B[2]);
        p->az += qm * (v_eff[0]*B[1] - v_eff[1]*B[0]);
    }
}

/* ========================================================================
 *  Thread worker for parallel integration
 * ======================================================================== */

typedef struct {
    struct reb_simulation*  sim;
    struct rebx_extras*     rebx;
    double                  target_time;
    double                  max_dt;
    int                     status;
    int                     started;
} worker_arg_t;

/* Grain impacts remove only the tracer; other pairs retain historical merging. */
static enum REB_COLLISION_RESOLVE_OUTCOME resolve_collision(struct reb_simulation* sim, struct reb_collision collision)
{
    struct rebx_extras* rebx = sim->extras;
    int* kind1 = rebx_get_param(rebx, sim->particles[collision.p1].ap, "particle_kind");
    int* kind2 = rebx_get_param(rebx, sim->particles[collision.p2].ap, "particle_kind");
    if (collision.p1 >= sim->N_active && kind1 && *kind1 == 1 && collision.p2 < sim->N_active)
        return 1;
    if (collision.p2 >= sim->N_active && kind2 && *kind2 == 1 && collision.p1 < sim->N_active)
        return 2;
    if ((kind1 && *kind1 == 1) || (kind2 && *kind2 == 1)) return 0;
    return reb_collision_resolve_merge(sim, collision);
}

static void* worker_thread(void* arg)
{
    worker_arg_t* w = (worker_arg_t*)arg;

    while (w->sim->t < w->target_time) {
        double end = w->target_time;
        if (w->max_dt > 0.0) end = fmin(end, w->sim->t + w->max_dt);
        if (end <= w->sim->t) {
            w->status = REB_STATUS_GENERIC_ERROR;
            break;
        }
        w->status = reb_simulation_integrate(w->sim, end);
        if (w->status != REB_STATUS_SUCCESS) break;
    }
    return NULL;
}


/* ========================================================================
 *  Main exported function: serpens_advance_integrate
 *
 *  Called from Python. Receives flat arrays describing the simulation state,
 *  performs the split-integrate-merge, and writes results back.
 *
 *  Arguments:
 *    n_active        – number of active (gravitating) particles
 *    n_total         – total number of particles
 *    state_in        – flat array [n_total * 7]: m, x, y, z, vx, vy, vz per particle
 *    hashes_in       – uint32 array [n_total]
 *    beta_values     – double array [n_total]  (radiation_forces beta, 0 for active)
 *    qm_values       – double array [n_total]  (q/m values, 0 for active)
 *    rad_source_flags– int array [n_total]     (1 if radiation source, 0 otherwise)
 *    target_time     – integration target time (absolute)
 *    G_value         – gravitational constant
 *    min_dt          – minimum timestep for IAS15
 *    n_threads       – number of threads to use
 *    max_dt          – optional maximum collision-search interval (0 disables)
 *    state_out       – output array, same layout as state_in (preallocated)
 *    n_out           – output: actual number of particles after merges/collisions
 *    sim_time_out    – output: final simulation time
 * ======================================================================== */

int serpens_advance_integrate(
    int n_active,
    int n_total,
    const double* state_in,
    const uint32_t* hashes_in,
    const double* beta_values,
    const double* qm_values,
    const int* rad_source_flags,
    const uint32_t* source_primary_hashes,
    const double* radii_in,
    const int* particle_kinds,
    double target_time,
    double G_value,
    double min_dt,
    double epsilon,
    double sim_t0,
    double initial_dt,
    double max_dt,
    double radiation_c,
    int n_threads,
    int force_is_velocity_dependent,
    /* outputs */
    double* state_out,
    uint32_t* hashes_out,
    double* radii_out,
    int* n_out,
    int* n_active_out,
    double* sim_time_out)
{
    int has_grains = 0;
    for (int i = n_active; i < n_total; i++) {
        if (particle_kinds[i] == 1) has_grains = 1;
    }
    if (n_threads < 1) n_threads = 1;
    int n_test = n_total - n_active;
    if (n_threads > n_test && n_test > 0) n_threads = n_test;
    if (n_test == 0) n_threads = 1;

    /* --- Allocate worker data --- */
    worker_arg_t* workers = (worker_arg_t*)calloc(n_threads, sizeof(worker_arg_t));
    pthread_t*    threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
    if (!workers || !threads) {
        free(workers);
        free(threads);
        return REB_STATUS_GENERIC_ERROR;
    }

    /* Compute how many test particles per thread */
    int base_count = n_test / n_threads;
    int remainder  = n_test % n_threads;

    int test_offset = 0;
    for (int t = 0; t < n_threads; t++) {
        int my_test_count = base_count + (t < remainder ? 1 : 0);

        /* Create simulation */
        struct reb_simulation* sim = reb_simulation_create();
        sim->G = G_value;
        sim->integrator = REB_INTEGRATOR_IAS15;
        sim->ri_ias15.min_dt = min_dt;
        sim->ri_ias15.epsilon = epsilon;
        sim->collision = has_grains ? REB_COLLISION_LINE : REB_COLLISION_DIRECT;
        sim->collision_resolve = resolve_collision;
        sim->collision_resolve_keep_sorted = 1;
        sim->exact_finish_time = 1;
        sim->dt = initial_dt;
        sim->t = sim_t0;

        /* Add active particles */
        for (int i = 0; i < n_active; i++) {
            struct reb_particle p = {0};
            p.m  = state_in[i*7 + 0];
            p.x  = state_in[i*7 + 1];
            p.y  = state_in[i*7 + 2];
            p.z  = state_in[i*7 + 3];
            p.vx = state_in[i*7 + 4];
            p.vy = state_in[i*7 + 5];
            p.vz = state_in[i*7 + 6];
            p.hash = hashes_in[i];
            p.r = radii_in[i];
            reb_simulation_add(sim, p);
        }

        /* Add this thread's slice of test particles */
        for (int j = 0; j < my_test_count; j++) {
            int gi = n_active + test_offset + j;  /* global index */
            struct reb_particle p = {0};
            p.m  = state_in[gi*7 + 0];
            p.x  = state_in[gi*7 + 1];
            p.y  = state_in[gi*7 + 2];
            p.z  = state_in[gi*7 + 3];
            p.vx = state_in[gi*7 + 4];
            p.vy = state_in[gi*7 + 5];
            p.vz = state_in[gi*7 + 6];
            p.hash = hashes_in[gi];
            p.r = radii_in[gi];
            reb_simulation_add(sim, p);
        }

        sim->N_active = n_active;

        /* Attach REBOUNDx and set per-particle parameters */
        struct rebx_extras* rebx = rebx_attach(sim);

        /* Register custom parameters */
        rebx_register_param(rebx, "q_over_m", REBX_TYPE_DOUBLE);
        rebx_register_param(rebx, "source_primary", REBX_TYPE_UINT32);
        rebx_register_param(rebx, "particle_kind", REBX_TYPE_INT);

        /* Radiation forces */
        struct rebx_force* rf = rebx_load_force(rebx, "radiation_forces");
        rebx_add_force(rebx, rf);
        rebx_set_param_double(rebx, &rf->ap, "c", radiation_c);
        if (g_lorentz.enabled) {
            struct rebx_force* lf = rebx_create_force(rebx, "lorentz_force");
            lf->force_type = REBX_FORCE_VEL;
            lf->update_accelerations = lorentz_force;
            rebx_add_force(rebx, lf);
        }
        sim->force_is_velocity_dependent |= force_is_velocity_dependent;

        /* Active particles: set radiation_source flag and source_primary */
        for (int i = 0; i < n_active; i++) {
            if (rad_source_flags[i]) {
                rebx_set_param_int(rebx, (struct rebx_node**)&sim->particles[i].ap, "radiation_source", 1);
            }
            /* beta and q/m for active particles (usually 0) */
            if (beta_values[i] != 0.0) {
                rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[i].ap, "beta", beta_values[i]);
            }
            if (qm_values[i] != 0.0) {
                rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[i].ap, "q_over_m", qm_values[i]);
            }
            /* Preserve source identity without modifying its orbit in workers. */
            if (source_primary_hashes[i] != 0) {
                rebx_set_param_uint32(rebx, (struct rebx_node**)&sim->particles[i].ap, "source_primary", source_primary_hashes[i]);
            }
        }

        /* Test particles: set beta and q_over_m */
        for (int j = 0; j < my_test_count; j++) {
            int gi = n_active + test_offset + j;
            int li = n_active + j;  /* local index in this sim */
            rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[li].ap, "beta", beta_values[gi]);
            rebx_set_param_double(rebx, (struct rebx_node**)&sim->particles[li].ap, "q_over_m", qm_values[gi]);
            rebx_set_param_int(rebx, (struct rebx_node**)&sim->particles[li].ap, "particle_kind", particle_kinds[gi]);
            if (rad_source_flags[gi]) {
                rebx_set_param_int(rebx, (struct rebx_node**)&sim->particles[li].ap, "radiation_source", 1);
            }
        }

        workers[t].sim          = sim;
        workers[t].rebx         = rebx;
        workers[t].target_time  = target_time;
        workers[t].max_dt       = max_dt;

        test_offset += my_test_count;
    }

    /* --- Launch threads --- */
    for (int t = 0; t < n_threads; t++) {
        workers[t].started = pthread_create(&threads[t], NULL, worker_thread, &workers[t]) == 0;
        if (!workers[t].started) worker_thread(&workers[t]);
    }

    /* --- Join threads --- */
    int status = REB_STATUS_SUCCESS;
    for (int t = 0; t < n_threads; t++) {
        if (workers[t].started) pthread_join(threads[t], NULL);
        if (workers[t].status != REB_STATUS_SUCCESS) status = workers[t].status;
    }

    /* --- Merge results back --- */
    /* Active particles come from thread 0 (they should all agree) */
    int out_idx = 0;

    for (int i = 0; i < workers[0].sim->N_active; i++) {
        struct reb_particle* p = &workers[0].sim->particles[i];
        state_out[out_idx*7 + 0] = p->m;
        state_out[out_idx*7 + 1] = p->x;
        state_out[out_idx*7 + 2] = p->y;
        state_out[out_idx*7 + 3] = p->z;
        state_out[out_idx*7 + 4] = p->vx;
        state_out[out_idx*7 + 5] = p->vy;
        state_out[out_idx*7 + 6] = p->vz;
        hashes_out[out_idx] = p->hash;
        radii_out[out_idx] = p->r;
        out_idx++;
    }

    /* Test particles from each thread */
    for (int t = 0; t < n_threads; t++) {
        struct reb_simulation* sim = workers[t].sim;
        for (uint32_t i = sim->N_active; i < sim->N; i++) {
            struct reb_particle* p = &sim->particles[i];
            state_out[out_idx*7 + 0] = p->m;
            state_out[out_idx*7 + 1] = p->x;
            state_out[out_idx*7 + 2] = p->y;
            state_out[out_idx*7 + 3] = p->z;
            state_out[out_idx*7 + 4] = p->vx;
            state_out[out_idx*7 + 5] = p->vy;
            state_out[out_idx*7 + 6] = p->vz;
            hashes_out[out_idx] = p->hash;
            radii_out[out_idx] = p->r;
            out_idx++;
        }
    }

    *n_out = out_idx;
    *n_active_out = workers[0].sim->N_active;
    *sim_time_out = workers[0].sim->t;

    /* --- Cleanup --- */
    for (int t = 0; t < n_threads; t++) {
        struct reb_simulation* sim = workers[t].sim;
        struct rebx_extras* rebx = workers[t].rebx;

        if (sim) {
            reb_simulation_free(sim);
            workers[t].sim = NULL;
        }
        if (rebx) rebx_free(rebx);
    }
    free(workers);
    free(threads);
    return status;
}

