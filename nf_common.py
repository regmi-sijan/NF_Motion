"""
nf_common.py - shared parameters, physics and plots for the NF_Motion scripts
=============================================================================

All three models use the functions in this file, so they share exactly the
same physics:

    nf_constant_force.py     NF + crossbridge bonds, pulled by a CONSTANT force F
    motor_variable_force.py  stochastic kinesin/dynein motors pulling on a held NF
                             -> the VARIABLE force the motors produce
    nf_stochastic_motors.py  NF + crossbridge bonds, pulled by the stochastic
                             motors (two-way coupled)
    run_parallel.py          many runs of any of the three models on all CPU cores

Change a physical parameter HERE (section 0) and it changes in every model.
Each script keeps its own run settings (force, motor numbers, time ...) at its top.
Everything can also be overridden on the command line (python <script> --help).

Physics summary
---------------
Crossbridge bonds ("instant" model):
    N bonds, each breaks at rate a + b*|f|, re-forms at rate c, spring k.
    Bond forces may point either way (the motor force can be negative).

Motors, two stochastic binding steps:
    motor <-> NF : binds at rate k_on (per NF), leaves at rate k_off
                   -> number on the NF is Poisson with mean k_on/k_off
    motor <-> MT : binds at rate pi0; unbinds at
                   kinesin eps0*exp(F/Fd)          (slip bond)
                   dynein  eps0*exp(min(F,Fs)/Fd)  (saturates above stall)
    a motor bound to the MT walks with v = v0(1 - F/Fs) (vB(1 - F/Fs) above
    stall), stretching its linker (stiffness km): force builds towards Fs.

NF motion every sub-step dt_int (force balance, two-way coupling):
    bonds intact : dx = (F - sum f_bond) / (n*k + K_m)
    no bonds     : dx = F*dt_int / (gamma_eff + K_m*dt_int),  gamma_eff = gamma*1e3
    K_m = stiffness of loaded motor linkers (0 for a constant force).

Sign convention: + anterograde (towards axon terminal), - retrograde.

Parameter sources: kinesin Fs 6 pN (Svoboda & Block 1994), Fd 3 pN, eps0 1/s,
vB 6 nm/s (Müller et al. PNAS 2008), pi0 4.7/s (Leduc et al. PNAS 2004);
dynein-dynactin Fs 4.3 pN (Belyy et al. NCB 2016), eps0 0.27/s, pi0 1.6/s,
vB 72 nm/s (Müller 2008), dynein saturating detachment (arXiv 1205.5858).
ASSUMED: dynein Fd 1.5 pN, motor<->NF k_off 0.1/s.
"""

import argparse
import os
import time

import numpy as np

# =============================================================================
# 0. SHARED PHYSICAL PARAMETERS (defaults for every model)
# =============================================================================
# ---- NF crossbridges ----
N_BONDS = 100          # number of crossbridge bonds (sidearms)
A = 0.2                # force-independent bond breaking rate (1/s)
B = 0.1                # force-dependent bond breaking rate (1/(s*pN))
C = 0.05               # bond re-forming rate (1/s)
K_BOND = 0.1           # bond spring constant (pN/nm)
GAMMA = 5.7e-5         # friction constant (kg/s); drag = GAMMA*1e3 pN*s/nm

# ---- motors on the NF ----
KOFF_NF = 0.1          # rate a motor leaves the NF (1/s)  [assumed]
MAX_MOTORS = 12        # max motors of each type that can sit on one NF

# ---- kinesin-1 (pulls +, anterograde) ----
FS_K = 6.0             # stall force (pN)
FD_K = 3.0             # detachment force (pN)
EPS0_K = 1.0           # MT unbinding rate at zero load (1/s)
PI0_K = 4.7            # MT binding rate (1/s)
V0_K = 800.0           # zero-load velocity (nm/s)
VB_K = 6.0             # super-stall backward speed scale (nm/s)
KM_K = 0.3             # linker stiffness (pN/nm)

# ---- dynein-dynactin-adaptor (pulls -, retrograde) ----
FS_D = 4.3             # stall force (pN)
FD_D = 1.5             # detachment force (pN) [assumed]
EPS0_D = 0.27          # MT unbinding rate at zero load (1/s)
PI0_D = 1.6            # MT binding rate (1/s)
V0_D = 700.0           # zero-load velocity (nm/s)
VB_D = 72.0            # super-stall backward speed scale (nm/s)
KM_D = 0.3             # linker stiffness (pN/nm)

# ---- time ----
DT = 0.01              # output / bond time step (s)
DT_INT = 1e-4          # motor sub-step (s)  (5e-4 gives the same results, ~4x faster)

# ---- output ----
OUTPUT_DIR = "results"
PAUSE_THRESHOLD = 0.05  # |displacement| per step (nm) below which the NF counts as paused
# =============================================================================

MODELS = ("constant", "motors", "coupled")
_MODE = {"constant": 0, "motors": 1, "coupled": 2}
KINESIN, DYNEIN = 0, 1


# -----------------------------------------------------------------------------
# numba (optional): compiles the physics below, ~100x faster
# -----------------------------------------------------------------------------
try:
    from numba import njit
    jit = njit(cache=True)
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    def jit(f):
        return f
    HAVE_NUMBA = False


# =============================================================================
# 1. MOTOR PHYSICS
# =============================================================================
@jit
def seed_random(seed):
    """Seed the random generator used inside the compiled functions."""
    np.random.seed(seed)


@jit
def motor_velocity(F, v0, vB, Fs):
    """Linear force-velocity relation (nm/s). F = load on the motor (pN, >= 0)."""
    if F <= 0.0:
        return v0
    if F <= Fs:
        return v0 * (1.0 - F / Fs)
    return vB * (1.0 - F / Fs)          # negative: slow backward walking


@jit
def mt_unbinding_rate(F, kind, eps0, Fd, Fs):
    """Load-dependent MT detachment rate (1/s)."""
    if kind == KINESIN:
        return eps0 * np.exp(F / Fd)                 # slip bond
    return eps0 * np.exp(min(F, Fs) / Fd)            # dynein: saturates above stall


@jit
def motor_substep(kind, on_nf, on_mt, stretch, p_arrive, p_leave, p_bind,
                  eps0, v0, vB, Fs, Fd, km, dt_int):
    """Advance all motors of one type by dt_int (NF binding, MT binding, walking).
    Returns (force along the motor's own direction, stiffness of loaded linkers)."""
    n_slots = on_nf.shape[0]
    # a new motor arrives on the NF (Poisson arrivals)
    if np.random.random() < p_arrive:
        for j in range(n_slots):
            if on_nf[j] == 0:
                on_nf[j] = 1
                on_mt[j] = 0
                stretch[j] = 0.0
                break
    force = 0.0
    stiff = 0.0
    for j in range(n_slots):
        if on_nf[j] == 0:
            continue
        # the motor leaves the NF (releases any force it carried)
        if np.random.random() < p_leave:
            on_nf[j] = 0
            on_mt[j] = 0
            stretch[j] = 0.0
            continue
        if on_mt[j] == 1:
            F = km * stretch[j] if stretch[j] > 0.0 else 0.0
            stretch[j] += motor_velocity(F, v0, vB, Fs) * dt_int      # walk
            rate = mt_unbinding_rate(F, kind, eps0, Fd, Fs)
            if np.random.random() < 1.0 - np.exp(-rate * dt_int):   # let go of the MT
                on_mt[j] = 0
                stretch[j] = 0.0
        elif np.random.random() < p_bind:                            # grab the MT
            on_mt[j] = 1
            stretch[j] = 0.0
        if on_mt[j] == 1 and stretch[j] > 0.0:
            force += km * stretch[j]
            stiff += km
    return force, stiff


@jit
def relax_linkers(on_mt, stretch, dx, direction, km):
    """NF moved by dx: linkers of motors pulling in +direction slacken.
    Returns the force these motors exert after the move."""
    f = 0.0
    for j in range(on_mt.shape[0]):
        if on_mt[j] == 1:
            stretch[j] -= direction * dx
            if stretch[j] > 0.0:
                f += km * stretch[j]
    return f


# =============================================================================
# 2. NF / CROSSBRIDGE PHYSICS
# =============================================================================
@jit
def update_bonds(state, f_bond, a, b, c, dt):
    """Bonds break (rate a + b|f|) or re-form (rate c). Returns (sum f, n intact)."""
    S = 0.0
    n = 0
    for j in range(state.shape[0]):
        fj = f_bond[j]
        R = np.random.random()
        if state[j] == 1:
            new = 1 if R >= (a + b * abs(fj)) * dt else 0
        else:
            new = 1 if R <= c * dt else 0
        state[j] = new
        if new == 0:
            fj = 0.0
        f_bond[j] = fj
        S += fj
        n += new
    return S, n


@jit
def nf_step(F, K_m, S, n, k, gamma_eff, dt_int):
    """NF displacement (nm) to the new force balance during one sub-step."""
    if n > 0:
        f_net = F - S
        if abs(f_net) < 1e-12:
            f_net = 0.0
        return f_net / (n * k + K_m)
    return F * dt_int / (gamma_eff + K_m * dt_int)


# =============================================================================
# 3. TIME LOOP (one kernel for all three models)
# =============================================================================
@jit
def run_kernel(mode, state, f_bond, kin_nf, kin_mt, kin_x, dyn_nf, dyn_mt, dyn_x,
               kpar, dpar, F_const, v_nf, a, b, c, k, gamma_eff, dt, dt_int, n_sub,
               o_F, o_Fk, o_Fd, o_nk_nf, o_nd_nf, o_nk_mt, o_nd_mt, o_bonds, o_disp):
    """mode 0 = constant force, 1 = motors on a held NF, 2 = motors + bonds.
    kpar/dpar = [p_arrive, p_leave, p_bind, eps0, v0, vB, Fs, Fd, km]"""
    for i in range(o_disp.shape[0]):
        S = 0.0
        n = 0
        if mode != 1:
            S, n = update_bonds(state, f_bond, a, b, c, dt)
        shift = 0.0
        disp = 0.0
        sum_F = 0.0
        sum_fk = 0.0
        sum_fd = 0.0
        for s in range(n_sub):
            if mode == 0:
                F = F_const
                K_m = 0.0
            else:
                fk, Kk = motor_substep(KINESIN, kin_nf, kin_mt, kin_x,
                                       kpar[0], kpar[1], kpar[2], kpar[3], kpar[4],
                                       kpar[5], kpar[6], kpar[7], kpar[8], dt_int)
                fd, Kd = motor_substep(DYNEIN, dyn_nf, dyn_mt, dyn_x,
                                       dpar[0], dpar[1], dpar[2], dpar[3], dpar[4],
                                       dpar[5], dpar[6], dpar[7], dpar[8], dt_int)
                F = fk - fd
                K_m = Kk + Kd
            if mode == 1:
                dx = v_nf * dt_int                    # NF held (or moved at a set speed)
            else:
                dx = nf_step(F, K_m, S, n, k, gamma_eff, dt_int)
                if n > 0:
                    S += n * k * dx
                    shift += k * dx
            if mode == 0:
                sum_F += F
            else:
                fk_after = relax_linkers(kin_mt, kin_x, dx, 1.0, kpar[8])
                fd_after = relax_linkers(dyn_mt, dyn_x, dx, -1.0, dpar[8])
                sum_fk += fk_after
                sum_fd += fd_after
                sum_F += fk_after - fd_after
            disp += dx
        if n > 0:
            for j in range(state.shape[0]):
                if state[j] == 1:
                    f_bond[j] += shift
        o_F[i] = sum_F / n_sub
        o_Fk[i] = sum_fk / n_sub
        o_Fd[i] = -sum_fd / n_sub
        o_nk_nf[i] = kin_nf.sum()
        o_nd_nf[i] = dyn_nf.sum()
        o_nk_mt[i] = kin_mt.sum()
        o_nd_mt[i] = dyn_mt.sum()
        o_bonds[i] = n
        o_disp[i] = disp


# =============================================================================
# 4. SIMULATION DRIVER
# =============================================================================
RESULT_KEYS = ("F_net", "F_kinesin", "F_dynein", "kin_on_nf", "dyn_on_nf",
               "kin_on_mt", "dyn_on_mt", "intact_bonds", "displacement")


def simulate(p):
    """Run one simulation for parameters p (from parse_args). Returns a dict of
    arrays, one value per dt: time, position and RESULT_KEYS.
    If p.seed is None a random seed is chosen and stored in p.seed_used."""
    mode = _MODE[p.model]
    dt_int = p.dt if mode == 0 else p.dt_int          # constant force: no sub-steps needed
    n_sub = int(round(p.dt / dt_int))
    if n_sub < 1 or abs(n_sub * dt_int - p.dt) > 1e-9 * p.dt:
        raise SystemExit("--dt must be a whole multiple of --dt-int")
    n_steps = int(round(p.time / p.dt))

    seed = p.seed if p.seed is not None else int(np.random.SeedSequence().entropy % 2**31)
    p.seed_used = seed
    seed_random(seed)
    rng = np.random.default_rng(seed)

    M = getattr(p, "max_motors", MAX_MOTORS)
    kin_nf = np.zeros(M, dtype=np.int64)
    dyn_nf = np.zeros(M, dtype=np.int64)
    if mode != 0:  # start from the steady state: Poisson number of motors on the NF
        kin_nf[:min(M, rng.poisson(p.mean_kinesin))] = 1
        dyn_nf[:min(M, rng.poisson(p.mean_dynein))] = 1
    kin_mt = np.zeros(M, dtype=np.int64)
    dyn_mt = np.zeros(M, dtype=np.int64)
    kin_x = np.zeros(M)
    dyn_x = np.zeros(M)

    N = getattr(p, "N", 1) if mode != 1 else 1
    state = np.ones(N, dtype=np.int64)
    # constant force: bonds start sharing F equally
    f_bond = np.full(N, p.F / N if mode == 0 else 0.0)

    def prob(rate):
        return 1.0 - np.exp(-rate * dt_int)

    if mode == 0:
        kpar = dpar = np.zeros(9)
    else:
        kpar = np.array([prob(p.mean_kinesin * p.koff_nf), prob(p.koff_nf), prob(p.pi0_k),
                         p.eps0_k, p.v0_k, p.vB_k, p.Fs_k, p.Fd_k, p.km_k])
        dpar = np.array([prob(p.mean_dynein * p.koff_nf), prob(p.koff_nf), prob(p.pi0_d),
                         p.eps0_d, p.v0_d, p.vB_d, p.Fs_d, p.Fd_d, p.km_d])

    out = {key: np.zeros(n_steps) for key in RESULT_KEYS}
    run_kernel(mode, state, f_bond, kin_nf, kin_mt, kin_x, dyn_nf, dyn_mt, dyn_x,
               kpar, dpar, float(getattr(p, "F", 0.0)), float(getattr(p, "nf_velocity", 0.0)),
               float(getattr(p, "a", A)), float(getattr(p, "b", B)), float(getattr(p, "c", C)),
               float(getattr(p, "k", K_BOND)), float(getattr(p, "gamma", GAMMA) * 1e3),
               float(p.dt), float(dt_int), n_sub, *(out[key] for key in RESULT_KEYS))
    out["time"] = (np.arange(n_steps) + 1) * p.dt
    out["position"] = np.cumsum(out["displacement"])
    return out


def summary_stats(res, p):
    """Numbers describing one run (used by the scripts and run_parallel.py)."""
    d = res["displacement"]
    T = res["time"][-1]
    thr = PAUSE_THRESHOLD
    s = {
        "velocity_nm_s": res["position"][-1] / T,
        "distance_nm": res["position"][-1],
        "mean_F_net_pN": res["F_net"].mean(),
        "std_F_net_pN": res["F_net"].std(),
        "frac_forward": float(np.mean(d > thr)),
        "frac_backward": float(np.mean(d < -thr)),
        "frac_paused": float(np.mean(np.abs(d) <= thr)),
    }
    if p.model != "motors":
        s["mean_intact_bonds"] = res["intact_bonds"].mean()
    if p.model != "constant":
        for key in ("kin_on_nf", "dyn_on_nf", "kin_on_mt", "dyn_on_mt"):
            s["mean_" + key] = res[key].mean()
        s["frac_F_positive"] = float(np.mean(res["F_net"] > 0.5))
        s["frac_F_negative"] = float(np.mean(res["F_net"] < -0.5))
    return s


def print_summary(res, p, elapsed):
    s = summary_stats(res, p)
    print(f"done in {elapsed:.1f} s   (seed {p.seed_used}; pass --seed {p.seed_used} to repeat)")
    if p.model != "constant":
        for name, key_nf, key_mt in (("kinesin", "kin_on_nf", "kin_on_mt"),
                                     ("dynein", "dyn_on_nf", "dyn_on_mt")):
            cnt = res[key_nf].astype(int)
            dist = np.bincount(cnt, minlength=5)[:5] / len(cnt)
            print(f"  {name:7s} on NF   : mean {cnt.mean():.2f} (0/1/2/3/4: "
                  + " / ".join(f"{x:.0%}" for x in dist)
                  + f"),  pulling on MT: mean {res[key_mt].mean():.2f}")
        print(f"  net motor force : mean {s['mean_F_net_pN']:+.3g} pN, std {s['std_F_net_pN']:.3g} pN;"
              f"  pulling + / - : {s['frac_F_positive']:.0%} / {s['frac_F_negative']:.0%} of the time")
    if p.model != "motors":
        v = s["velocity_nm_s"]
        print(f"  net distance    : {s['distance_nm']:+.4g} nm  (velocity {v:+.3g} nm/s = "
              f"{v * 0.0864:+.3g} mm/day)")
        print(f"  moving + / - / paused : {s['frac_forward']:.1%} / {s['frac_backward']:.1%} / "
              f"{s['frac_paused']:.1%}")
        print(f"  mean intact bonds     : {s['mean_intact_bonds']:.3g} / {p.N}")


def save_results(res, path):
    cols = ("time",) + RESULT_KEYS + ("position",)
    np.savetxt(path, np.column_stack([res[c] for c in cols]), fmt="%.8g",
               delimiter=", ", header=", ".join(cols))


def run_name(p):
    """Descriptive file name stem for a run."""
    if p.model == "constant":
        core = f"constant force (F {p.F:g} pN, N {p.N}"
    elif p.model == "motors":
        core = f"motor force (K {p.mean_kinesin:g}, D {p.mean_dynein:g}"
    else:
        core = f"motor driven NF (K {p.mean_kinesin:g}, D {p.mean_dynein:g}, N {p.N}"
    return f"{core}, {p.time:g} sec, seed {p.seed_used})"


# =============================================================================
# 5. PLOTS - MOTORS AND FORCE
# =============================================================================
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def _plt(show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _window(t, start, length):
    t0 = start if start < t[-1] else 0.0
    return t0, t0 + length, (t >= t0) & (t <= t0 + length)


def plot_motor_force(res, p, path=None, show=False):
    """Motors and the variable force they produce.
    (a) forces in a zoom window, (b) motors on the NF / pulling on the MT,
    (c) how often 0, 1, 2, ... motors are on the NF, (d) net force distribution."""
    plt = _plt(show)
    t = res["time"]
    t0, t1, w = _window(t, p.plot_start, p.plot_window)
    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1.1, 1.3], hspace=0.6, wspace=0.28)
    ax_f = fig.add_subplot(gs[0, :])
    ax_n = fig.add_subplot(gs[1, :], sharex=ax_f)
    ax_h = fig.add_subplot(gs[2, 0])
    ax_F = fig.add_subplot(gs[2, 1])

    ax_f.plot(t[w], res["F_kinesin"][w], color=BLUE, lw=1.2, label="kinesin (+)")
    ax_f.plot(t[w], res["F_dynein"][w], color=ORANGE, lw=1.2, label="dynein (−)")
    ax_f.plot(t[w], res["F_net"][w], color=INK, lw=1.5, label="net force on NF")
    ax_f.axhline(0, color=INK2, lw=0.8)
    ax_f.set_ylabel("Force (pN)", fontsize=12, color=INK)
    held = "  (NF held in place)" if p.model == "motors" else ""
    ax_f.set_title(f"Motor force, {t0:g}–{t1:g} s  (mean on NF: {p.mean_kinesin:g} kinesin, "
                   f"{p.mean_dynein:g} dynein){held}", loc="left", fontsize=13, color=INK, pad=24)
    ax_f.legend(frameon=False, fontsize=10, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)

    for key_nf, key_mt, color, name in (("kin_on_nf", "kin_on_mt", BLUE, "kinesin"),
                                        ("dyn_on_nf", "dyn_on_mt", ORANGE, "dynein")):
        ax_n.step(t[w], res[key_nf][w], where="post", color=color, lw=1.0, alpha=0.45,
                  label=f"{name} on NF")
        ax_n.step(t[w], res[key_mt][w], where="post", color=color, lw=1.6,
                  label=f"{name} pulling (on MT)")
    ax_n.set_ylabel("Motors", fontsize=12, color=INK)
    ax_n.set_xlabel("Time (s)", fontsize=12, color=INK)
    ax_n.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4)

    top = int(max(res["kin_on_nf"].max(), res["dyn_on_nf"].max()))
    bins = np.arange(top + 2) - 0.5
    ks = np.arange(top + 1)
    hk, _ = np.histogram(res["kin_on_nf"], bins=bins)
    hd, _ = np.histogram(res["dyn_on_nf"], bins=bins)
    ax_h.bar(ks - 0.2, hk / hk.sum(), width=0.38, color=BLUE, label="kinesin")
    ax_h.bar(ks + 0.2, hd / hd.sum(), width=0.38, color=ORANGE, label="dynein")
    ax_h.set_xticks(ks)
    ax_h.set_xlabel("Motors on the NF", fontsize=12, color=INK)
    ax_h.set_ylabel("Fraction of time", fontsize=12, color=INK)
    ax_h.legend(frameon=False, fontsize=9)

    F = res["F_net"]
    ax_F.hist(F, bins=80, color=INK2, edgecolor="white", linewidth=0.4)
    ax_F.axvline(F.mean(), color=INK, lw=1.5, ls="--")
    ax_F.text(F.mean(), ax_F.get_ylim()[1] * 0.95, f"  mean {F.mean():+.2f} pN",
              va="top", fontsize=10, color=INK)
    ax_F.set_xlabel("Net force on NF (pN), whole run", fontsize=12, color=INK)
    ax_F.set_ylabel("Count", fontsize=12, color=INK)

    for ax in (ax_f, ax_n, ax_h, ax_F):
        _style(ax)
    return _finish(fig, plt, path, show)


# =============================================================================
# 6. PLOTS - NF DISPLACEMENT
# =============================================================================
def plot_displacement(res, p, path=None, show=False):
    """NF movement.
    (a) distance vs time, whole run, (b) zoom window, (c) intact bonds in the
    same window, (d) distribution of the displacement per step."""
    plt = _plt(show)
    t = res["time"]
    pos = res["position"]
    t0, t1, w = _window(t, p.plot_start, p.plot_window)
    thin = max(1, len(t) // 200_000)
    v = pos[-1] / t[-1]

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.3, 1.1], hspace=0.6, wspace=0.28)
    ax_all = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[2, 0], sharex=ax_zoom)
    ax_d = fig.add_subplot(gs[1:, 1])

    unit, ulab = (1e-3, "µm") if np.abs(pos).max() > 5000 else (1.0, "nm")
    ax_all.plot(t[::thin], pos[::thin] * unit, color=INK, lw=1.4)
    ax_all.axhline(0, color=INK2, lw=0.8)
    ax_all.axvspan(t0, t1, color=BLUE, alpha=0.12, lw=0)
    ax_all.set_ylabel(f"Distance ({ulab})", fontsize=12, color=INK)
    ax_all.set_xlabel("Time (s)", fontsize=12, color=INK)
    what = (f"constant force {p.F:g} pN" if p.model == "constant"
            else f"motors ({p.mean_kinesin:g} kinesin, {p.mean_dynein:g} dynein)")
    ax_all.set_title(f"NF displacement, {what}, N = {p.N} bonds\n"
                     f"net velocity {v:+.3g} nm/s = {v * 0.0864:+.3g} mm/day",
                     loc="left", fontsize=13, color=INK)

    ax_zoom.plot(t[w], pos[w], color=INK, lw=1.3)
    ax_zoom.set_ylabel("Distance (nm)", fontsize=12, color=INK)
    ax_zoom.set_title(f"Zoom {t0:g}–{t1:g} s (shaded above)", loc="left",
                      fontsize=11, color=INK2)

    ax_b.step(t[w], res["intact_bonds"][w], where="post", color=INK2, lw=1.2)
    ax_b.set_ylabel("Intact\nbonds", fontsize=12, color=INK)
    ax_b.set_xlabel("Time (s)", fontsize=12, color=INK)

    d = res["displacement"]
    lim = np.percentile(np.abs(d), 99.5) or 1.0
    ax_d.hist(np.clip(d, -lim, lim), bins=81, color=INK2, edgecolor="white", linewidth=0.4)
    ax_d.set_yscale("log")
    ax_d.axvline(0, color=INK, lw=0.8)
    ax_d.set_xlabel(f"Displacement per step of {p.dt:g} s (nm)", fontsize=12, color=INK)
    ax_d.set_ylabel("Count (log)", fontsize=12, color=INK)
    ax_d.set_title(f"+ forward   − backward   (edge bars: beyond ±{lim:.2g} nm)",
                   loc="left", fontsize=11, color=INK2)

    for ax in (ax_all, ax_zoom, ax_b, ax_d):
        _style(ax)
    return _finish(fig, plt, path, show)


def _finish(fig, plt, path, show):
    if path:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


# =============================================================================
# 7. COMMAND LINE (shared by all scripts)
# =============================================================================
def build_parser(model, defaults):
    """Argument parser for one model. `defaults` = the script's run settings
    (F, mean_kinesin, mean_dynein, time, plot_start, plot_window ...)."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    ap = argparse.ArgumentParser(
        description={"constant": "NF movement under a constant force.",
                     "motors": "Variable force from stochastic kinesin/dynein motors "
                               "pulling on a held NF.",
                     "coupled": "NF movement driven by stochastic kinesin/dynein motors."}[model],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    uses_bonds = model != "motors"
    uses_motors = model != "constant"

    if model == "constant":
        g = ap.add_argument_group("force")
        g.add_argument("-F", "--F", "--force", dest="F", type=float, default=defaults.get("F", 20.0),
                       help="constant pulling force (pN, + anterograde)")
    if uses_motors:
        g = ap.add_argument_group("motors on the NF")
        g.add_argument("--mean-kinesin", type=float, default=defaults.get("mean_kinesin", 2.0),
                       help="average kinesins on NF")
        g.add_argument("--mean-dynein", type=float, default=defaults.get("mean_dynein", 2.0),
                       help="average dyneins on NF")
        g.add_argument("--koff-nf", type=float, default=KOFF_NF, help="motor leaves NF rate (1/s)")
        g.add_argument("--max-motors", type=int, default=MAX_MOTORS, help="max motors per type on NF")
        if model == "motors":
            g.add_argument("--nf-velocity", type=float, default=defaults.get("nf_velocity", 0.0),
                           help="NF speed (nm/s); 0 = NF held in place")
        for tag, name, vals in (("k", "kinesin", (FS_K, FD_K, EPS0_K, PI0_K, V0_K, VB_K, KM_K)),
                                ("d", "dynein", (FS_D, FD_D, EPS0_D, PI0_D, V0_D, VB_D, KM_D))):
            g = ap.add_argument_group(name)
            for (opt, desc), val in zip((("Fs", "stall force (pN)"), ("Fd", "detachment force (pN)"),
                                         ("eps0", "MT unbinding rate (1/s)"),
                                         ("pi0", "MT binding rate (1/s)"), ("v0", "velocity (nm/s)"),
                                         ("vB", "super-stall speed (nm/s)"),
                                         ("km", "linker stiffness (pN/nm)")), vals):
                g.add_argument(f"--{opt}-{tag}", type=float, default=val, help=desc)
    if uses_bonds:
        g = ap.add_argument_group("NF crossbridges")
        g.add_argument("-N", "--N", dest="N", type=int, default=defaults.get("N", N_BONDS),
                       help="number of bonds")
        g.add_argument("--a", type=float, default=A, help="bond breaking rate (1/s)")
        g.add_argument("--b", type=float, default=B, help="force-dependent breaking (1/(s*pN))")
        g.add_argument("--c", type=float, default=C, help="bond re-forming rate (1/s)")
        g.add_argument("--k", type=float, default=K_BOND, help="bond spring constant (pN/nm)")
        g.add_argument("--gamma", type=float, default=GAMMA, help="friction constant (kg/s)")

    g = ap.add_argument_group("time and output")
    g.add_argument("-T", "--time", type=float, default=defaults.get("time", 1000.0),
                   help="simulated time (s)")
    g.add_argument("--dt", type=float, default=DT, help="output / bond step (s)")
    if uses_motors:
        g.add_argument("--dt-int", type=float, default=DT_INT, help="motor sub-step (s)")
    g.add_argument("--seed", type=int, default=defaults.get("seed"), help="random seed")
    g.add_argument("--outdir", default=OUTPUT_DIR, help="folder for results")
    g.add_argument("--no-save", action="store_true", help="do not write the data file")
    g.add_argument("--no-plot", action="store_true", help="do not make the plots")
    g.add_argument("--show", action="store_true", help="open the plots in a window")
    g.add_argument("--plot-start", type=float, default=defaults.get("plot_start", 100.0),
                   help="zoom window start (s)")
    g.add_argument("--plot-window", type=float, default=defaults.get("plot_window", 20.0),
                   help="zoom window length (s)")
    ap.set_defaults(model=model)
    return ap


def parse_args(model, defaults=None, argv=None):
    ap = build_parser(model, defaults or {})
    p = ap.parse_args(argv)
    if p.time <= 0 or p.dt <= 0 or getattr(p, "dt_int", 1.0) <= 0:
        ap.error("time, dt and dt-int must be > 0")
    if getattr(p, "N", 1) < 1 or getattr(p, "max_motors", 1) < 1:
        ap.error("N and max-motors must be >= 1")
    if getattr(p, "mean_kinesin", 0) < 0 or getattr(p, "mean_dynein", 0) < 0 \
            or getattr(p, "koff_nf", 1) <= 0:
        ap.error("mean motor numbers must be >= 0 and koff-nf > 0")
    return p


def main(model, defaults, argv=None):
    """Run one model from the command line: simulate, summarise, save, plot."""
    p = parse_args(model, defaults, argv)
    if model == "constant":
        head = f"Constant force {p.F:g} pN, N = {p.N} bonds"
    elif model == "motors":
        head = (f"Motor force: mean {p.mean_kinesin:g} kinesin + {p.mean_dynein:g} dynein on a "
                + ("held NF" if p.nf_velocity == 0 else f"NF moving at {p.nf_velocity:g} nm/s"))
    else:
        head = (f"Motor-driven NF: mean {p.mean_kinesin:g} kinesin + {p.mean_dynein:g} dynein, "
                f"N = {p.N} bonds")
    print(f"{head}, T = {p.time:g} s")
    if not HAVE_NUMBA:
        print("  (numba not installed - this will be slow; pip install numba)")
    t0 = time.time()
    res = simulate(p)
    print_summary(res, p, time.time() - t0)

    stem = os.path.join(p.outdir, run_name(p))
    if not (p.no_save and p.no_plot):
        os.makedirs(p.outdir, exist_ok=True)
    if not p.no_save:
        save_results(res, stem + ".txt")
        print(f"  data   : {stem}.txt")
    if not p.no_plot:
        if model != "constant":
            print(f"  motors : {plot_motor_force(res, p, stem + ' motors.png', show=p.show)}")
        if model != "motors":
            print(f"  motion : {plot_displacement(res, p, stem + ' displacement.png', show=p.show)}")
    return res
