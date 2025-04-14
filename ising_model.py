import numpy as np
import random
import time # Added for timing
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------
# Module: LatticeSetup
# ---------------------------
def initialize_lattice(N: int, state: str = 'random') -> np.ndarray:
    """
    Initializes an NxN lattice.
    (Corresponds to LatticeSetup.create_lattice in the plan)

    Args:
        N: The dimension of the square lattice (N x N).
        state: The initial state ('random', 'up', 'down'). Defaults to 'random'.

    Returns:
        An NxN NumPy array representing the lattice with spins (+1 or -1).
    """
    if state == 'random':
        lattice = np.random.choice([-1, 1], size=(N, N))
    elif state == 'up':
        lattice = np.ones((N, N), dtype=int)
    elif state == 'down':
        lattice = -np.ones((N, N), dtype=int)
    else:
        raise ValueError("Invalid state. Choose 'random', 'up', or 'down'.")
    return lattice

def get_neighbor_indices(N: int, i: int, j: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Gets the indices of the four nearest neighbors for site (i, j)
    on an NxN lattice with periodic boundary conditions.
    (Corresponds to LatticeSetup.get_neighbors in the plan)

    Args:
        N: The dimension of the square lattice (N x N).
        i: The row index of the site.
        j: The column index of the site.

    Returns:
        A tuple containing the (row, col) indices of the four neighbors:
        (up, down, left, right).
    """
    up = ((i - 1 + N) % N, j)
    down = ((i + 1) % N, j)
    left = (i, (j - 1 + N) % N)
    right = (i, (j + 1) % N)
    return up, down, left, right

# ---------------------------
# Module: EnergyCalculator
# ---------------------------
def _sum_neighbor_spins(lattice: np.ndarray, N: int, i: int, j: int) -> int:
    """Helper function to sum the spins of the four neighbors of site (i, j)."""
    up, down, left, right = get_neighbor_indices(N, i, j)
    neighbor_sum = (
        lattice[up] +
        lattice[down] +
        lattice[left] +
        lattice[right]
    )
    return neighbor_sum

def calculate_energy_change(lattice: np.ndarray, N: int, i: int, j: int) -> int:
    """
    Calculates the change in energy if the spin at (i, j) is flipped.
    Assumes the standard ferromagnetic Ising Hamiltonian H = - Sum_{<ij>} s_i s_j.
    (Corresponds to EnergyCalculator.calculate_energy_change in the plan)
    Delta_H = H_new - H_old = 2 * s_ij * sum(neighbors).

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.
        i: The row index of the spin to flip.
        j: The column index of the spin to flip.

    Returns:
        The change in energy (Delta H).
    """
    s_ij = lattice[i, j]
    neighbor_sum = _sum_neighbor_spins(lattice, N, i, j)
    delta_H = 2 * s_ij * neighbor_sum
    return delta_H

def calculate_total_energy(lattice: np.ndarray, N: int) -> int:
    """
    Calculates the total energy of the lattice configuration.
    Assumes the standard ferromagnetic Ising Hamiltonian H = - Sum_{<ij>} s_i s_j.
    (Corresponds to EnergyCalculator.calculate_total_energy in the plan)
    Iterates through each site and sums the interaction with right and down neighbors
    to avoid double counting.

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.

    Returns:
        The total energy H.
    """
    total_energy = 0
    for i in range(N):
        for j in range(N):
            s_ij = lattice[i,j]
            right_neighbor_idx = (i, (j + 1) % N)
            bottom_neighbor_idx = ((i + 1) % N, j)
            total_energy += -s_ij * lattice[right_neighbor_idx]
            total_energy += -s_ij * lattice[bottom_neighbor_idx]
    return total_energy

# ---------------------------
# Module: Observables (for Stage 2)
# ---------------------------
def calculate_magnetization(lattice: np.ndarray, N: int) -> float:
    """
    Calculates the average magnetization per spin for the lattice.
    (Corresponds to Observables.calculate_magnetization in the plan)
    M = (1/N^2) * Sum_{i,j} s_ij

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.

    Returns:
        The average magnetization.
    """
    total_spin = np.sum(lattice)
    return total_spin / (N * N)

# ---------------------------
# Module: MetropolisStep
# ---------------------------
def metropolis_step(lattice: np.ndarray, N: int, beta: float) -> bool:
    """
    Performs a single Metropolis update step:
    1. Randomly selects a site (i, j).
    2. Calculates energy change for flipping spin at (i, j).
    3. Accepts/rejects the flip based on Metropolis criteria.
    (Corresponds to MetropolisStep.metropolis_step in the plan)

    Args:
        lattice: The current NxN lattice configuration (modified in place if accepted).
        N: The dimension of the lattice.
        beta: The inverse temperature (1 / kT).

    Returns:
        True if the flip was accepted, False otherwise.
    """
    # 1. Select a random spin site
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)

    # 2. Calculate energy change
    delta_H = calculate_energy_change(lattice, N, i, j)

    # 3. Metropolis acceptance criteria
    if delta_H <= 0:
        lattice[i, j] *= -1
        return True
    else:
        acceptance_prob = np.exp(-beta * delta_H)
        if random.random() < acceptance_prob:
            lattice[i, j] *= -1
            return True
        else:
            return False

# ---------------------------
# Module: SimulationRunner (Core Logic)
# ---------------------------
def run_simulation(L: int, beta: float, num_equilibration_sweeps: int, num_measurement_sweeps: int, initial_state: str = 'random', visualize_final_state: bool = False):
    """
    Runs the 2D Ising model simulation with equilibration and measurement phases.
    Calculates average magnetization and susceptibility.

    Args:
        L: Linear dimension of the lattice (total size N=2L).
        beta: Inverse temperature.
        num_equilibration_sweeps: Number of MC sweeps for equilibration.
        num_measurement_sweeps: Number of MC sweeps for measurement.
        initial_state: Initial lattice configuration ('random', 'up', 'down').
        visualize_final_state: Whether to plot the final lattice state.

    Returns:
        A tuple containing:
        - beta (float): The inverse temperature used for this run.
        - mean_magnetization (float): Average magnetization per spin <M>.
        - susceptibility (float): Magnetic susceptibility per spin chi.
    """
    N = 2 * L
    equilibration_steps = num_equilibration_sweeps * N * N
    measurement_steps = num_measurement_sweeps * N * N
    total_steps = equilibration_steps + measurement_steps

    print(f"--- Running Ising Simulation for beta={beta:.4f} ---")
    print(f" L={L}, N={N}x{N}, EqSweeps={num_equilibration_sweeps}, MeasSweeps={num_measurement_sweeps}, Initial='{initial_state}'")

    # Initialization
    start_time = time.time()
    lattice = initialize_lattice(N, state=initial_state)

    # --- Equilibration Phase ---
    print(f" Starting Equilibration ({equilibration_steps} steps)...")
    for step in range(equilibration_steps):
        metropolis_step(lattice, N, beta)
        # Optional: Print progress within equilibration if needed
        # if (step + 1) % (N * N * max(1, num_equilibration_sweeps // 10)) == 0:
        #     print(f"  Eq sweep {(step + 1) // (N * N)}/{num_equilibration_sweeps} done.")

    print(" Equilibration finished.")

    # --- Measurement Phase ---
    print(f" Starting Measurement ({measurement_steps} steps)...")
    magnetization_measurements = []
    accepted_flips_measurement = 0

    for step in range(measurement_steps):
        accepted = metropolis_step(lattice, N, beta)
        if accepted:
            accepted_flips_measurement += 1

        # Measure magnetization after each SWEEP
        if (step + 1) % (N * N) == 0:
            current_M = calculate_magnetization(lattice, N)
            magnetization_measurements.append(current_M)
            # Optional: Print progress within measurement
            # current_meas_sweep = (step + 1) // (N * N)
            # if current_meas_sweep % max(1, num_measurement_sweeps // 10) == 0:
            #    print(f"  Meas sweep {current_meas_sweep}/{num_measurement_sweeps} done.")

    print(" Measurement finished.")

    # --- Analysis of Measurements ---
    if not magnetization_measurements:
        print(" Warning: No measurements collected.")
        mean_M = np.nan
        mean_M_sq = np.nan
        susceptibility = np.nan
    else:
        measurements_array = np.array(magnetization_measurements)
        # Use absolute value for mean magnetization for ferromagnetic case
        # Often we look at |<M>| as symmetry can be broken to +M or -M
        mean_M = np.mean(np.abs(measurements_array))
        mean_M_sq = np.mean(measurements_array**2)
        # Susceptibility calculation (check definition if needed!)
        # chi = beta * N^2 * (<M^2> - <M>^2)
        susceptibility = beta * (N**2) * (mean_M_sq - np.mean(measurements_array)**2)
        # Note: Using mean(M)^2 here, not mean(|M|)^2, for susceptibility calc.
        # Variance should be calculated based on the raw magnetization values.

    end_time = time.time()
    total_time = end_time - start_time
    measurement_acceptance_rate = accepted_flips_measurement / measurement_steps if measurement_steps > 0 else 0

    print(f" Beta={beta:.4f} finished in {total_time:.2f}s. <|M|>: {mean_M:.4f}, Chi: {susceptibility:.4f}, Acceptance (meas): {measurement_acceptance_rate:.4f}")

    # Visualization (Optional - only the final state)
    if visualize_final_state and MATPLOTLIB_AVAILABLE:
        final_energy = calculate_total_energy(lattice, N) # Calculate if needed for title
        plt.figure(figsize=(6, 6))
        plt.imshow(lattice, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
        plt.title(f"Final State (L={L}, beta={beta:.3f}, EqS={num_equilibration_sweeps}, MeS={num_measurement_sweeps})")
        plt.xticks([])
        plt.yticks([])
        filename = f'ising_final_state_L{L}_beta{beta:.3f}_Eq{num_equilibration_sweeps}_Me{num_measurement_sweeps}.png'
        plt.savefig(filename)
        print(f" Final lattice state saved to {filename}")
    elif visualize_final_state and not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not found. Cannot visualize the final state.")

    # Return calculated observables and the final lattice
    return beta, mean_M, susceptibility, lattice

# ---------------------------
# Module: Stage 3 - Snapshot Visualization 
# ---------------------------
def save_lattice_snapshot(lattice, L, beta, snapshot_id, base_dir='snapshots'):
    """
    Saves the lattice configuration to a file and generates a visualization.
    
    Args:
        lattice: The current lattice configuration.
        L: Linear dimension parameter.
        beta: Inverse temperature.
        snapshot_id: Identifier for this snapshot.
        base_dir: Directory to save snapshots.
    """
    import os
    N = 2 * L
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Save lattice to .npy file
    npy_filename = f'{base_dir}/lattice_L{L}_beta{beta:.3f}_id{snapshot_id}.npy'
    np.save(npy_filename, lattice)
    
    # Save visualization if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(6, 6))
        plt.imshow(lattice, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
        plt.title(f"L={L}, N={N}x{N}, beta={beta:.3f}, ID={snapshot_id}")
        plt.xticks([])
        plt.yticks([])
        png_filename = f'{base_dir}/lattice_L{L}_beta{beta:.3f}_id{snapshot_id}.png'
        plt.savefig(png_filename)
        plt.close()
        return npy_filename, png_filename
    
    return npy_filename, None

def run_simulation_with_snapshots(L, beta, num_equilibration_sweeps, num_measurement_sweeps, 
                                 num_snapshots=5, initial_state='random'):
    """
    Runs the simulation and takes multiple snapshots during the measurement phase.
    
    Args:
        L: Linear dimension parameter.
        beta: Inverse temperature.
        num_equilibration_sweeps: Number of MC sweeps for equilibration.
        num_measurement_sweeps: Number of MC sweeps for measurement.
        num_snapshots: Number of snapshots to take during measurement.
        initial_state: Initial lattice configuration.
        
    Returns:
        Tuple of (beta, mean_M, susceptibility, snapshot_filenames)
    """
    N = 2 * L
    equilibration_steps = num_equilibration_sweeps * N * N
    measurement_steps = num_measurement_sweeps * N * N
    
    print(f"--- Running Simulation with Snapshots for beta={beta:.4f} ---")
    
    # Initialization
    lattice = initialize_lattice(N, state=initial_state)
    
    # Equilibration Phase
    print(f" Running {num_equilibration_sweeps} equilibration sweeps...")
    for step in range(equilibration_steps):
        metropolis_step(lattice, N, beta)
    
    # Measurement Phase with Snapshots
    print(f" Running {num_measurement_sweeps} measurement sweeps with {num_snapshots} snapshots...")
    magnetization_measurements = []
    snapshot_filenames = []
    
    # Calculate snapshot intervals
    snapshot_interval_sweeps = max(1, num_measurement_sweeps // num_snapshots)
    snapshot_interval_steps = snapshot_interval_sweeps * N * N
    
    for step in range(measurement_steps):
        metropolis_step(lattice, N, beta)
        
        # Measure magnetization after each sweep
        if (step + 1) % (N * N) == 0:
            current_M = calculate_magnetization(lattice, N)
            magnetization_measurements.append(current_M)
            
            # Take snapshot at intervals
            current_sweep = (step + 1) // (N * N)
            if current_sweep % snapshot_interval_sweeps == 0:
                snapshot_id = current_sweep // snapshot_interval_sweeps
                print(f"  Taking snapshot {snapshot_id} at sweep {current_sweep}")
                npy_file, png_file = save_lattice_snapshot(lattice, L, beta, snapshot_id)
                snapshot_filenames.append((npy_file, png_file))
    
    # Calculate results
    measurements_array = np.array(magnetization_measurements)
    mean_M = np.mean(np.abs(measurements_array))
    mean_M_sq = np.mean(measurements_array**2)
    susceptibility = beta * (N**2) * (mean_M_sq - np.mean(measurements_array)**2)
    
    print(f" Simulation complete. <|M|>: {mean_M:.4f}, Chi: {susceptibility:.4f}")
    return beta, mean_M, susceptibility, snapshot_filenames

def visualize_snapshots_grid(snapshot_files, title, output_filename):
    """
    Creates a grid visualization of multiple snapshots.
    
    Args:
        snapshot_files: List of snapshot .npy filenames.
        title: Title for the overall grid.
        output_filename: Filename to save the grid plot.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create grid visualization.")
        return
    
    n_snapshots = len(snapshot_files)
    cols = min(3, n_snapshots)
    rows = (n_snapshots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (npy_file, _) in enumerate(snapshot_files):
        if i < len(axes):
            # Load the snapshot
            lattice = np.load(npy_file)
            
            # Extract beta from filename
            import re
            beta_match = re.search(r'beta([\d\.]+)', npy_file)
            beta_str = beta_match.group(1) if beta_match else "unknown"
            
            # Plot on the grid
            axes[i].imshow(lattice, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
            axes[i].set_title(f"β = {beta_str}")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
    
    # Hide any unused subplots
    for i in range(n_snapshots, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    plt.savefig(output_filename)
    print(f"Grid visualization saved to {output_filename}")
    plt.close()

# ---------------------------
# Main Execution Block
# ---------------------------
if __name__ == '__main__':
    # --- Parameters for Stage 2: Scanning Beta ---
    param_L = 10 # Use a smaller L for quicker testing, change back to 50 for final run
    param_eq_sweeps = 500   # Sweeps for equilibration
    param_meas_sweeps = 1000 # Sweeps for measurement
    param_initial_state = 'random'

    # Define the range of beta values to scan
    # beta_values = np.linspace(0.1, 1.0, 19) # Example: Coarse scan
    beta_values = np.linspace(0.3, 0.6, 31) # Example: Finer scan around expected Tc (~0.44)
    print(f"Scanning {len(beta_values)} beta values from {beta_values[0]:.3f} to {beta_values[-1]:.3f}")
    print(f"Using L={param_L}, EqSweeps={param_eq_sweeps}, MeasSweeps={param_meas_sweeps}")

    results_beta = []
    results_M = []
    results_chi = []

    # --- Loop over Beta Values ---
    total_start_time = time.time()
    for beta in beta_values:
        # Run simulation for the current beta value
        # Set visualize_final_state=False during scan to avoid many plots
        b, m, chi, _ = run_simulation(
            L=param_L,
            beta=beta,
            num_equilibration_sweeps=param_eq_sweeps,
            num_measurement_sweeps=param_meas_sweeps,
            initial_state=param_initial_state,
            visualize_final_state=False # Avoid plotting individual final states
        )
        results_beta.append(b)
        results_M.append(m)
        results_chi.append(chi)
        print("-" * 20) # Separator between beta runs

    total_end_time = time.time()
    print(f"\n--- Scan Finished --- Total time: {total_end_time - total_start_time:.2f} seconds ---")

    # --- Plotting Results --- (Requires Matplotlib)
    if MATPLOTLIB_AVAILABLE:
        print("Plotting results...")
        fig, axes = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

        # Plot <|M|> vs Beta
        ax1 = axes[0]
        ax1.plot(results_beta, results_M, 'o-', markersize=4)
        ax1.set_ylabel("Average Magnetization <|M|>")
        ax1.set_title(f"Ising Model Phase Transition (L={param_L}, Eq={param_eq_sweeps}, Meas={param_meas_sweeps})")
        ax1.grid(True)
        # Add theoretical critical beta line
        beta_c_theory = np.log(1 + np.sqrt(2)) / 2
        ax1.axvline(beta_c_theory, color='r', linestyle='--', label=f'Theoretical βc ≈ {beta_c_theory:.4f}')
        ax1.legend()

        # Plot Chi vs Beta
        ax2 = axes[1]
        ax2.plot(results_beta, results_chi, 'o-', markersize=4)
        ax2.set_xlabel("Inverse Temperature (β = 1/kT)")
        ax2.set_ylabel("Susceptibility χ")
        ax2.grid(True)
        ax2.axvline(beta_c_theory, color='r', linestyle='--', label=f'Theoretical βc ≈ {beta_c_theory:.4f}')
        ax2.legend()

        plt.tight_layout()
        plot_filename = f'ising_phase_transition_L{param_L}_Eq{param_eq_sweeps}_Me{param_meas_sweeps}.png'
        plt.savefig(plot_filename)
        print(f"Phase transition plots saved to {plot_filename}")
        # plt.show() # Uncomment to display plot interactively
    else:
        print("\nMatplotlib not found. Cannot plot the results.")
        # Optionally print results to console if no plot
        print("Beta\t<|M|>\tChi")
        for b, m, chi in zip(results_beta, results_M, results_chi):
            print(f"{b:.4f}\t{m:.4f}\t{chi:.4f}")
            
    # --- Stage 3: Take snapshots at representative beta values ---
    print("\n--- Stage 3: Representative Configurations ---")
    # Based on theoretical critical point
    beta_values_for_snapshots = [0.3, 0.44, 0.6]  # Below, at, above βc
    
    snapshot_results = []
    for beta in beta_values_for_snapshots:
        print(f"\nRunning simulation with snapshots at beta={beta:.4f}")
        _, _, _, snapshot_files = run_simulation_with_snapshots(
            L=param_L,
            beta=beta,
            num_equilibration_sweeps=param_eq_sweeps,
            num_measurement_sweeps=param_meas_sweeps,
            num_snapshots=5
        )
        snapshot_results.append((beta, snapshot_files))
        
    # Create grid visualization comparing different beta values
    if MATPLOTLIB_AVAILABLE and snapshot_results:
        # Get one representative snapshot from each beta
        representative_snapshots = []
        for beta, snapshot_files in snapshot_results:
            if snapshot_files:  # Use the last snapshot (most equilibrated)
                representative_snapshots.append((beta, snapshot_files[-1]))
                
        if representative_snapshots:
            # Create combined visualization
            visualize_snapshots_grid(
                [files for _, files in representative_snapshots],
                f"2D Ising Model Configurations (L={param_L})",
                f"ising_configurations_comparison_L{param_L}.png"
            )

    print("\n--- Main script finished --- ") 