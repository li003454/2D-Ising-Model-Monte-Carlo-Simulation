# 2D Ising Model Monte Carlo Simulation Project

This project implements a Monte Carlo simulation of the 2D Ising Model using the Metropolis-Hastings algorithm to study phase transition phenomena.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Theoretical Background](#theoretical-background)
3. [Project Structure](#project-structure)
4. [Dependencies](#dependencies)
5. [Usage](#usage)
6. [Stage 1: Basic Simulation](#stage-1-basic-simulation)
7. [Stage 2: Order Parameter and Susceptibility](#stage-2-order-parameter-and-susceptibility)
8. [Stage 3: Spin Configuration Visualization](#stage-3-spin-configuration-visualization)
9. [Results Analysis](#results-analysis)
10. [References](#references)

## Project Overview

The 2D Ising model is a classic model in statistical physics for studying phase transitions. This project uses the Monte Carlo method to simulate the 2D Ising model at different temperatures, calculate physical quantities such as magnetization and susceptibility, and investigate the system's phase transition behavior.

Main objectives:
1. Implement the Metropolis algorithm to simulate the 2D Ising model
2. Calculate and plot the magnetization (M) and susceptibility (χ) as functions of inverse temperature (β)
3. Visualize typical spin configurations at different temperatures
4. Determine the critical temperature βc

## Theoretical Background

The Ising model describes a lattice system composed of spins (±1), where each spin interacts with its nearest neighbors. The Hamiltonian of the model is:

H = -J Σ<i,j> s_i s_j

where J > 0 represents ferromagnetic coupling. In a two-dimensional system, the theoretically predicted critical inverse temperature is:

βc = ln(1 + √2) / 2 ≈ 0.44068...

Near the critical point, the system transitions from a disordered state (high temperature) to an ordered state (low temperature).

## Project Structure

The project consists primarily of a Python file `ising_model.py`, which contains the following logical modules:

1. **LatticeSetup**: Responsible for creating and initializing the lattice, handling periodic boundary conditions
2. **EnergyCalculator**: Calculates total energy and energy changes
3. **Observables**: Calculates measurable quantities such as magnetization
4. **MetropolisStep**: Implements the Metropolis update step
5. **SimulationRunner**: Controls the entire simulation process
6. **Visualization Functions**: For result display and analysis

## Dependencies

This project requires the following Python libraries:

```bash
numpy      # Numerical computation
matplotlib # Result visualization
```

You can install them using pip:

```bash
pip install numpy matplotlib
```

## Usage

To run the complete simulation:

```bash
python ising_model.py
```

This will execute all three stages of calculation: basic simulation, β scanning, and spin configuration visualization.

To modify parameters, you can edit the `__main__` block in the script:
- `param_L`: Sets the lattice size (N=2L)
- `param_eq_sweeps`: Number of equilibration sweeps
- `param_meas_sweeps`: Number of measurement sweeps
- `beta_values`: Range of β values

## Stage 1: Basic Simulation

Stage 1 implements the basic Monte Carlo simulation functionality for the 2D Ising model:

1. **Initialization**: Create an N×N lattice with randomly initialized spin states
2. **Periodic Boundary Conditions**: Correctly handle lattice boundaries
3. **Energy Calculation**: Calculate the total system energy and the energy change of a single spin flip
4. **Metropolis Step**: Randomly select spins and decide whether to flip them based on energy change and temperature

Key functions:
- `initialize_lattice(N, state)`: Create and initialize the lattice
- `get_neighbor_indices(N, i, j)`: Get neighbor indices under periodic boundary conditions
- `calculate_energy_change(lattice, N, i, j)`: Calculate the energy change of flipping the spin at position (i,j)
- `calculate_total_energy(lattice, N)`: Calculate the energy of the entire system
- `metropolis_step(lattice, N, beta)`: Perform one Metropolis update

## Stage 2: Order Parameter and Susceptibility

Stage 2 extends the functionality of Stage 1, implementing:

1. **Equilibration Phase**: Allow the system to reach thermal equilibrium
2. **Measurement Phase**: Collect physical quantity measurements after equilibration
3. **Order Parameter Calculation**: Calculate average magnetization <|M|>
4. **Susceptibility Calculation**: Calculate magnetic susceptibility χ
5. **β Scanning**: Run simulations across a series of β values to study phase transitions

Key functions:
- `calculate_magnetization(lattice, N)`: Calculate magnetization
- `run_simulation(L, beta, num_equilibration_sweeps, num_measurement_sweeps, ...)`: Execute complete simulation with equilibration and measurement phases

The main program of Stage 2 scans a series of β values and plots curves of <|M|> and χ versus β, used to study phase transitions and determine the critical point βc.

## Stage 3: Spin Configuration Visualization

Stage 3 focuses on the characteristics of spin configurations at different temperatures:

1. **Representative Temperature Selection**: Select three temperature points below, near, and above βc
2. **Snapshot Collection**: Run simulations at each temperature point and save multiple spin configurations
3. **Visual Comparison**: Create side-by-side comparison figures showing configuration features at different temperatures

Key functions:
- `save_lattice_snapshot(lattice, L, beta, snapshot_id, ...)`: Save spin configuration snapshots
- `run_simulation_with_snapshots(...)`: Run simulation and collect configuration snapshots
- `visualize_snapshots_grid(...)`: Create grid visualization of multiple snapshots

This stage clearly demonstrates:
- **Low Temperature**: Highly ordered configurations (large regions of identical spins)
- **Critical Temperature**: Characteristic scale-free spin clusters of various sizes
- **High Temperature**: Random disordered spin distribution

## Results Analysis

The simulation results generate the following output files:

1. **Phase Transition Curves**: `ising_phase_transition_L{L}_Eq{eq}_Me{meas}.png`
   - Top plot: <|M|> vs β
   - Bottom plot: χ vs β
   - Vertical red line marks the theoretical critical point βc

2. **Spin Configuration Comparison**: `ising_configurations_comparison_L{L}.png`
   - Shows representative spin configurations at different temperatures

Through these analyses, you can:
- Observe how <|M|> transitions from ~0 at high temperature to ~1 at low temperature
- Confirm that χ exhibits a peak near βc
- Intuitively understand spin arrangement characteristics at different temperatures

## References

1. Metropolis, N., et al. (1953). *Equation of State Calculations by Fast Computing Machines*. The Journal of Chemical Physics, 21(6), 1087–1092.
2. Onsager, L. (1944). *Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition*. Physical Review, 65(3-4), 117–149.
3. Newman, M. E. J., & Barkema, G. T. (1999). *Monte Carlo Methods in Statistical Physics*. Oxford University Press. 