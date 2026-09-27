# NF_Motion

Stochastic simulations of **neurofilament (NF) movement in axons**.

A neurofilament is held in place by crossbridges (sidearm bonds) that break and re-form. It is pulled either by a constant force or by teams of molecular motors: **kinesin-1** pulls toward the axon terminal (+) and **dynein** pulls toward the cell body (−). The motors bind to and leave the NF at random, grab and release the microtubule under load, and walk against elastic linkers. This produces a fluctuating, tug-of-war force that moves the NF forward and backward, with pauses in between.

## Models

| File | What it does | Plots |
|---|---|---|
| `nf_constant_force.py` | NF + crossbridge bonds pulled by a **constant force** F | displacement |
| `motor_variable_force.py` | stochastic kinesin/dynein motors (2 + 2 on average) pulling on a held NF: the **variable force** | motors |
| `nf_stochastic_motors.py` | NF + bonds **driven by the stochastic motors** (two-way coupled) | motors + displacement |
| `run_parallel.py` | many runs of any model (seeds and parameter sweeps) on all CPU cores | summary |
| `nf_common.py` | shared parameters, physics, plotting and command-line options | — |

## Installation

Requires Python 3.9+.

    git clone https://github.com/regmi-sijan/NF_Motion.git
    cd NF_Motion
    pip install -r requirements.txt

`numba` is optional but strongly recommended: it makes the simulations about 100× faster.

## Usage

    python nf_constant_force.py -F 25 -N 300 -T 5000
    python motor_variable_force.py --mean-kinesin 2 --mean-dynein 2 -T 500
    python nf_stochastic_motors.py -T 2000 --seed 1 --show

    python run_parallel.py --model constant --runs 8 --sweep F=10,20,30
    python run_parallel.py --model motors   --runs 4 --sweep mean-dynein=1,2,3
    python run_parallel.py --model coupled  --runs 8 -T 2000 --dt-int 5e-4

Every parameter can be set on the command line; `python <script> --help` lists them all.
Results go to `results/`: a data file with a header row, plus PNG plots.

**Where to change defaults**
- Physical parameters (bond rates, motor stall forces, rates, time steps) are at the top of `nf_common.py`. A change there applies to all three models.
- Run settings (F, average motor numbers, N, time, plot window) are at the top of each script.

## Model summary

- **Crossbridges:** N bonds, each breaks at rate a + b·|f|, re-forms at rate c, spring constant k.
- **Motor ↔ NF:** motors bind at rate k_on and leave at rate k_off, so the number on the NF is Poisson with mean k_on/k_off.
- **Motor ↔ microtubule:**
  - binds at rate π₀;
  - kinesin unbinds at ε₀·exp(F/F_d) (slip bond);
  - dynein unbinds at ε₀·exp(min(F, F_s)/F_d) (saturates above stall).
- **Walking:** v = v₀(1 − F/F_s), which stretches the motor linker until the motor stalls.
- **NF motion:** at every sub-step the NF moves to the new force balance between motors and bonds. Moving the NF loosens the motors that pull it and loads the ones that resist it.

Default motor parameters are taken from the literature:
- kinesin stall force 6 pN (Svoboda & Block 1994);
- dynein-dynactin stall force 4.3 pN (Belyy et al., Nat. Cell Biol. 2016);
- rates from the tug-of-war model of Müller, Klumpp & Lipowsky (PNAS 2008);
- kinesin rebinding rate from Leduc et al. (PNAS 2004).

Two values are assumptions, marked in `nf_common.py`: dynein F_d = 1.5 pN and motor–NF k_off = 0.1/s.

## Checks

- With the NF held in place, no motor exceeds its stall force.
- Compiled (numba) and plain-Python runs give identical results for the same seed.
- The constant-force model reproduces the statistics of the original "instant" crossbridge model.
- `--dt-int 5e-4` gives the same results as the default 1e-4 within noise, and is about 4× faster.

## License

MIT, see [LICENSE](LICENSE).
