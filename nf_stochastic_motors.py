"""
Model 3 - NF movement driven by stochastic kinesin and dynein motors
===================================================================

Combines the two other models: the NF with its crossbridge bonds
(nf_constant_force.py) is pulled by the stochastic motors
(motor_variable_force.py) instead of a constant force.

Coupling works both ways. The motor force moves the NF against its bonds, and
the NF motion loosens the motors that pull it and loads the ones that resist
it, so motors cannot drag the NF faster than they walk.

Shared physics and plots: nf_common.py (edit physical parameters there).

Usage
-----
    python nf_stochastic_motors.py                          # 2 kinesin + 2 dynein, N = 100
    python nf_stochastic_motors.py -T 2000 --seed 1
    python nf_stochastic_motors.py --mean-kinesin 3 --mean-dynein 1 -N 300
    python nf_stochastic_motors.py --plot-start 300 --plot-window 30 --show
    python nf_stochastic_motors.py --help                   # every parameter

Output (in results/): data file + "... motors.png" + "... displacement.png"
"""

import nf_common as nf

# =============================================================================
# RUN SETTINGS (defaults - override on the command line)
# =============================================================================
MEAN_KINESIN = 2.0   # average number of kinesins on the NF
MEAN_DYNEIN = 2.0    # average number of dyneins on the NF
N = 100              # number of crossbridge bonds
TIME = 1000.0        # simulated time (s)
SEED = None          # integer for a reproducible run
PLOT_START = 100.0   # start of the zoomed window in the plots (s)
PLOT_WINDOW = 20.0   # length of the zoomed window (s)
# =============================================================================

DEFAULTS = dict(mean_kinesin=MEAN_KINESIN, mean_dynein=MEAN_DYNEIN, N=N,
                time=TIME, seed=SEED, plot_start=PLOT_START, plot_window=PLOT_WINDOW)


def main(argv=None):
    return nf.main("coupled", DEFAULTS, argv)


if __name__ == "__main__":
    main()
