# Memristive CPG for robotic tuna tail control

This repository contains the Python source code for the numerical experiments and analyses in *Central Pattern Generation for Robotic Tuna Tail Control under Memristive Programming Constraints*. The scripts process memristor measurements, simulate the oscillator network and tail actuator, evaluate control strategies, and regenerate figures and tables.

## Running the code

Keep the original directory structure, including `scripts/`, `src/`, and `tests/`. From the repository root, you can view the available options with Python 3.13:

```bash
python3.13 -m scripts.reproduce_all --help
```

A full reproduction requires the following files from the complete project archive. Copy them into the repository root and preserve their paths:

- `data/`, including the raw measurements, processed inputs, reference files, and input inventory.
- `configs/`, containing the parameters and protocols for the numerical experiments.
- The C++ kernels under `src/memristive_cpg/protocols/`: `control/numerical_kernel.cpp`, `stochastic/guard_kernel.cpp`, and `sensitivity/drift_kernel.cpp`.

The full run also requires a C++17-capable `clang++` on your `PATH`. The scripts compile the kernels automatically. The workflow was tested on macOS arm64 with Python 3.13.9.

From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5 scipy==1.16.3 pandas==3.0.1 matplotlib==3.10.7
python -m scripts.reproduce_all --workers 4
```

The pipeline runs the tests and simulations, writes figures to `results/figures/`, exports tables to `results/tables/`, and saves the numerical comparison in `results/validation/comparison.csv`. Execution logs are written to `results/logs/`.

If you already have the full `results/` directory, you can repeat the validation and regenerate the figures without rerunning the simulations:

```bash
python -m scripts.reproduce_all --verify-existing --workers 4
```
