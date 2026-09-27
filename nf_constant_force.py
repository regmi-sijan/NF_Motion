"""
Model 1 - NF movement under a CONSTANT force
============================================

A neurofilament held by N crossbridge bonds (sidearms) is pulled by a constant
force F. Bonds break at rate a + b*|f|, re-form at rate c, and act as springs
(k). When bonds break, their load is shared by the rest and the NF steps
forward (the "instant" crossbridge model).

Shared physics and plots: nf_common.py (edit physical parameters there).

Usage
-----
    python nf_constant_force.py                      # defaults below
    python nf_constant_force.py -F 25 -N 300 -T 5000
    python nf_constant_force.py -F -10               # negative = retrograde pull
    python nf_constant_force.py --help               # every parameter

Output (in results/): data file + "... displacement.png"
"""

import nf_common as nf

# =============================================================================
# RUN SETTINGS (defaults - override on the command line)
# =============================================================================
F = 20.0             # constant pulling force (pN, + anterograde)
N = 100              # number of crossbridge bonds
TIME = 1000.0        # simulated time (s)
SEED = None          # integer for a reproducible run
PLOT_START = 100.0   # start of the zoomed window in the plot (s)
PLOT_WINDOW = 20.0   # length of the zoomed window (s)
# =============================================================================

DEFAULTS = dict(F=F, N=N, time=TIME, seed=SEED,
                plot_start=PLOT_START, plot_window=PLOT_WINDOW)


def main(argv=None):
    return nf.main("constant", DEFAULTS, argv)


if __name__ == "__main__":
    main()
