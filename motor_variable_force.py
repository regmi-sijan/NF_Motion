"""
Model 2 - VARIABLE force from stochastic kinesin and dynein motors
=================================================================

Motors bind to and leave the neurofilament at random (on average MEAN_KINESIN
kinesins and MEAN_DYNEIN dyneins, but sometimes more or fewer). Motors on the
NF grab and release the microtubule under load, walk, and stretch their
linkers. The NF is held in place (or moved at a set speed with --nf-velocity),
so this script shows only the force the motors produce: + kinesin
(anterograde), - dynein (retrograde).

Shared physics and plots: nf_common.py (edit motor parameters there).

Usage
-----
    python motor_variable_force.py                          # 2 kinesin + 2 dynein
    python motor_variable_force.py --mean-kinesin 3 --mean-dynein 1 -T 500
    python motor_variable_force.py --koff-nf 0.05 --Fs-d 1.1
    python motor_variable_force.py --help                   # every parameter

Output (in results/): data file (time, F_net, F_kinesin, F_dynein, motor
counts, ...) + "... motors.png"
"""

import nf_common as nf

# =============================================================================
# RUN SETTINGS (defaults - override on the command line)
# =============================================================================
MEAN_KINESIN = 2.0   # average number of kinesins on the NF
MEAN_DYNEIN = 2.0    # average number of dyneins on the NF
NF_VELOCITY = 0.0    # NF speed (nm/s); 0 = NF held in place
TIME = 500.0         # simulated time (s)
SEED = None          # integer for a reproducible run
PLOT_START = 100.0   # start of the zoomed window in the plot (s)
PLOT_WINDOW = 20.0   # length of the zoomed window (s)
# =============================================================================

DEFAULTS = dict(mean_kinesin=MEAN_KINESIN, mean_dynein=MEAN_DYNEIN,
                nf_velocity=NF_VELOCITY, time=TIME, seed=SEED,
                plot_start=PLOT_START, plot_window=PLOT_WINDOW)


def main(argv=None):
    return nf.main("motors", DEFAULTS, argv)


if __name__ == "__main__":
    main()
