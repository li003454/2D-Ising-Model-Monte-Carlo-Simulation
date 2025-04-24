import numpy as np
import random
import time # Added for timing
import numba # Added for JIT compilation
from numba import prange # Added for parallel loops
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Add check for imageio
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

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
@numba.jit(nopython=True) # Standard JIT compilation
def _sum_neighbor_spins(lattice: np.ndarray, N: int, i: int, j: int) -> int:
    """
    Calculates the sum of spins of the four nearest neighbors for site (i, j).
    Uses periodic boundary conditions implicitly via modulo arithmetic.
    (Helper function)

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.
        i: The row index.
        j: The column index.

    Returns:
        Sum of the four neighbor spins.
    """
    # Calculate neighbor indices with periodic boundary conditions
    top    = lattice[(i - 1 + N) % N, j]
    bottom = lattice[(i + 1) % N, j]
    left   = lattice[i, (j - 1 + N) % N]
    right  = lattice[i, (j + 1) % N]
    return top + bottom + left + right

@numba.jit(nopython=True) # Standard JIT compilation
def calculate_energy_change(lattice: np.ndarray, N: int, i: int, j: int) -> int:
    """
    Calculates the energy change if the spin at (i, j) were flipped.
    Delta H = H_final - H_initial = - Sum_{k neighbors} s_new s_k - (- Sum_{k neighbors} s_old s_k)
            = - (-s_old) Sum s_k + (-s_old) Sum s_k = 2 * s_old * Sum_{k neighbors} s_k

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

@numba.jit(nopython=True, parallel=True) # This function benefits from parallelization
def calculate_total_energy(lattice: np.ndarray, N: int) -> int:
    """
    Calculates the total energy of the lattice configuration.
    Assumes the standard ferromagnetic Ising Hamiltonian H = - Sum_{<ij>} s_i s_j.
    Iterates through each site and sums the interaction with right and down neighbors
    to avoid double counting. Uses parallel loops for better performance.

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.

    Returns:
        The total energy H.
    """
    total_energy = 0
    # Use prange for parallelization when possible
    for i in prange(N):
        row_energy = 0
        for j in range(N):
            s_ij = lattice[i,j]
            # Only sum right and bottom neighbors to avoid double count
            right_neighbor = lattice[i, (j + 1) % N]
            bottom_neighbor = lattice[(i + 1) % N, j]
            row_energy += -s_ij * right_neighbor
            row_energy += -s_ij * bottom_neighbor
        total_energy += row_energy
    return total_energy

# ---------------------------
# Module: Observables (for Stage 2)
# ---------------------------
@numba.jit(nopython=True) # Standard JIT compilation
def calculate_magnetization(lattice: np.ndarray, N: int) -> float:
    """
    Calculates the average magnetization per spin for the lattice.
    M = (1/N^2) * Sum_{i,j} s_ij

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.

    Returns:
        The average magnetization (float, between -1 and 1).
    """
    total_spin = np.sum(lattice) # Numba handles np.sum
    return total_spin / (N * N)

# ---------------------------
# Module: MetropolisStep
# ---------------------------
@numba.jit(nopython=True) # Standard JIT compilation
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
    # 1. Select a random spin site (Numba compatible random choice)
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)

    # 2. Calculate energy change
    delta_H = calculate_energy_change(lattice, N, i, j)

    # 3. Metropolis acceptance criteria
    accept = False
    if delta_H <= 0:
        accept = True
    else:
        acceptance_prob = np.exp(-beta * delta_H)
        if random.random() < acceptance_prob:
            accept = True

    if accept:
        lattice[i, j] *= -1
        return True
    else:
        return False

# Optimized version for performing many metropolis steps at once
@numba.jit(nopython=True) # Standard JIT compilation
def metropolis_sweep(lattice: np.ndarray, N: int, beta: float, num_steps: int) -> int:
    """
    Performs multiple Metropolis steps and returns the number of accepted flips.
    This is more efficient than calling metropolis_step in a Python loop
    since the entire loop is JIT compiled.
    
    Args:
        lattice: The current NxN lattice configuration (modified in place).
        N: The dimension of the lattice.
        beta: The inverse temperature (1 / kT).
        num_steps: Number of individual steps to perform.
        
    Returns:
        Number of accepted spin flips.
    """
    accepted_count = 0
    for _ in range(num_steps):
        if metropolis_step(lattice, N, beta):
            accepted_count += 1
    return accepted_count

# ---------------------------
# Module: SimulationRunner (Core Logic)
# ---------------------------
def run_simulation(L: int, beta: float, num_equilibration_sweeps: int, num_measurement_sweeps: int, 
                  initial_state: str = 'random', visualize_final_state: bool = False, 
                  use_annealing: bool = False, bin_size: int = 10):
    """
    Runs the 2D Ising model simulation with equilibration and measurement phases.
    Calculates average magnetization, susceptibility, and energy per site.

    Args:
        L: Linear dimension of the lattice (total size N=2L).
        beta: Inverse temperature.
        num_equilibration_sweeps: Number of MC sweeps for equilibration.
        num_measurement_sweeps: Number of MC sweeps for measurement.
        initial_state: Initial lattice configuration ('random', 'up', 'down').
        visualize_final_state: Whether to plot the final lattice state.
        use_annealing: Whether to use simulated annealing before equilibration.
        bin_size: Number of measurements to group together for binning analysis.

    Returns:
        A tuple containing:
        - beta (float): The inverse temperature used for this run.
        - mean_magnetization (float): Average absolute magnetization per spin <|M|>.
        - susceptibility (float): Magnetic susceptibility per spin chi.
        - mean_energy_per_site (float): Average energy per site <E> / N^2.
        - lattice (np.ndarray): The final lattice configuration.
    """
    N = 2 * L
    T = 1.0 / beta  # Current temperature
    equilibration_steps = num_equilibration_sweeps * N * N
    measurement_steps = num_measurement_sweeps * N * N
    total_steps = equilibration_steps + measurement_steps

    print(f"--- Running Ising Simulation for beta={beta:.4f} (T={T:.4f}) ---")
    print(f" L={L}, N={N}x{N}, EqSweeps={num_equilibration_sweeps}, MeasSweeps={num_measurement_sweeps}, Initial='{initial_state}'")

    # Initialization
    start_time = time.time()
    lattice = initialize_lattice(N, state=initial_state)

    # --- Simulated Annealing Phase (if enabled) ---
    if use_annealing:
        annealing_steps = int(0.5 * equilibration_steps)  # Use 50% of equilibration steps for annealing
        print(f" Starting Simulated Annealing ({annealing_steps} steps)...")
        
        # Start from higher temperature (2T) and gradually cool down to target T
        start_T = min(5.0, 2.0 * T)  # Cap starting T at 5.0 to avoid too high temperatures
        start_beta = 1.0 / start_T
        
        cooling_schedule = np.linspace(start_beta, beta, 10)  # 10 temperature steps
        steps_per_temp = annealing_steps // len(cooling_schedule)
        
        for annealing_beta in cooling_schedule:
            # Use optimized sweep function instead of individual steps
            accepted = metropolis_sweep(lattice, N, annealing_beta, steps_per_temp * N * N)
            current_T = 1.0 / annealing_beta
            print(f"  Annealing at T={current_T:.4f} (beta={annealing_beta:.4f}) completed.")
        
        print(" Annealing finished.")
        
        # Adjust remaining equilibration steps
        equilibration_steps -= annealing_steps

    # --- Equilibration Phase ---
    print(f" Starting Equilibration ({equilibration_steps} steps)...")
    # Use optimized sweep function instead of step-by-step loop
    eq_accepted = metropolis_sweep(lattice, N, beta, equilibration_steps)
    eq_acceptance_rate = eq_accepted / equilibration_steps if equilibration_steps > 0 else 0
    print(f" Equilibration finished. Acceptance rate: {eq_acceptance_rate:.4f}")

    # --- Measurement Phase with improved strategy ---
    print(f" Starting Measurement ({measurement_steps} steps)...")
    magnetization_measurements = []
    energy_measurements = []
    accepted_flips_measurement = 0
    
    # Increase interval between measurements to reduce correlations
    # Take fewer measurements but with larger interval between them
    measure_interval = 5 * N * N  # Every 5 sweeps instead of every sweep
    num_actual_measurements = measurement_steps // measure_interval
    
    # Perform measurements with intervals
    remaining_steps = 0
    for m in range(num_actual_measurements):
        # Run simulation for measure_interval steps
        accepted = metropolis_sweep(lattice, N, beta, measure_interval)
        accepted_flips_measurement += accepted
        
        # Take measurements
        current_M = calculate_magnetization(lattice, N)
        current_E = calculate_total_energy(lattice, N)
        magnetization_measurements.append(current_M)
        energy_measurements.append(current_E)
    
    # Run any remaining steps
    remaining_steps = measurement_steps - (num_actual_measurements * measure_interval)
    if remaining_steps > 0:
        accepted = metropolis_sweep(lattice, N, beta, remaining_steps)
        accepted_flips_measurement += accepted
    
    print(f" Took {len(magnetization_measurements)} measurements with {measure_interval//(N*N)} sweeps between each.")
    print(" Measurement finished.")

    # --- Analysis of Measurements with binning ---
    if not magnetization_measurements:
        print(" Warning: No measurements collected.")
        mean_M = np.nan
        mean_M_sq = np.nan
        susceptibility = np.nan
        mean_E_per_site = np.nan
    else:
        measurements_array = np.array(magnetization_measurements)
        energy_array = np.array(energy_measurements)
        
        # Perform binning to reduce correlations (if we have enough measurements)
        if len(measurements_array) >= bin_size:
            num_bins = len(measurements_array) // bin_size
            # Reshape the arrays to group measurements into bins
            binned_M = measurements_array[:num_bins*bin_size].reshape(num_bins, bin_size)
            binned_E = energy_array[:num_bins*bin_size].reshape(num_bins, bin_size)
            
            # Calculate bin averages
            bin_M_avgs = np.mean(binned_M, axis=1)
            bin_E_avgs = np.mean(binned_E, axis=1)
            
            # Use bin averages for final calculations
            mean_M = np.mean(np.abs(bin_M_avgs))
            mean_M_sq = np.mean(bin_M_avgs**2)
            mean_E_per_site = np.mean(bin_E_avgs) / (N * N)
        else:
            # Fall back to regular calculation if not enough measurements for binning
            mean_M = np.mean(np.abs(measurements_array))
            mean_M_sq = np.mean(measurements_array**2)
            mean_E_per_site = np.mean(energy_array) / (N * N)
        
        # Regular susceptibility calculation
        susceptibility = beta * (N**2) * (mean_M_sq - np.mean(measurements_array)**2)

    end_time = time.time()
    total_time = end_time - start_time
    measurement_acceptance_rate = accepted_flips_measurement / measurement_steps if measurement_steps > 0 else 0

    # Updated print statement
    print(f" Beta={beta:.4f} finished in {total_time:.2f}s. <|M|>: {mean_M:.4f}, Chi: {susceptibility:.4f}, <E/N^2>: {mean_E_per_site:.4f}, Acceptance (meas): {measurement_acceptance_rate:.4f}")

    # Visualization (Optional - only the final state)
    if visualize_final_state and MATPLOTLIB_AVAILABLE:
        final_energy = calculate_total_energy(lattice, N)
        plt.figure(figsize=(6, 6))
        plt.imshow(lattice, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
        plt.title(f"Final State (L={L}, beta={beta:.3f}, EqS={num_equilibration_sweeps}, MeS={num_measurement_sweeps}, E/N^2={final_energy/(N*N):.3f})")
        plt.xticks([])
        plt.yticks([])
        filename = f'ising_final_state_L{L}_beta{beta:.3f}_Eq{num_equilibration_sweeps}_Me{num_measurement_sweeps}.png'
        plt.savefig(filename)
        print(f" Final lattice state saved to {filename}")
    elif visualize_final_state and not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not found. Cannot visualize the final state.")

    # Return calculated observables and the final lattice
    return beta, mean_M, susceptibility, mean_E_per_site, lattice

# ---------------------------
# Module: Animation (New functionality)
# ---------------------------
def create_ising_animation(L: int, beta: float, num_frames: int, interval_sweeps: int, 
                         initial_state: str = 'random', equilibrate_sweeps: int = 100):
    """
    Creates an animation of the Ising model evolving over time.
    
    Args:
        L: Linear dimension of the lattice (N=2L).
        beta: Inverse temperature.
        num_frames: Number of frames in the animation.
        interval_sweeps: Number of sweeps between frames.
        initial_state: Initial lattice state ('random', 'up', 'down').
        equilibrate_sweeps: Number of sweeps to run before starting animation.
        
    Returns:
        Animation object if matplotlib is available, None otherwise.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib and/or animation module not available. Cannot create animation.")
        return None
    
    N = 2 * L
    
    # Initialize lattice
    lattice = initialize_lattice(N, state=initial_state)
    
    # Equilibrate first using optimized sweep function
    print(f"Equilibrating for {equilibrate_sweeps} sweeps...")
    equilibrate_steps = equilibration_sweeps = equilibrate_sweeps * N * N
    eq_accepted = metropolis_sweep(lattice, N, beta, equilibrate_steps)
    eq_acceptance_rate = eq_accepted / equilibrate_steps if equilibrate_steps > 0 else 0
    print(f"Equilibration finished. Acceptance rate: {eq_acceptance_rate:.4f}")
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.close()  # Prevents the empty figure from being displayed
    
    # Create a custom colormap with white for -1 and black for +1
    cmap = ListedColormap(['white', 'black'])
    
    # Initial plot
    im = ax.imshow((lattice + 1) // 2, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"2D Ising Model (L={L}, β={beta:.3f}, T={1/beta:.3f})")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create a text annotation for magnetization
    mag_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, 
                      color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    frames = []
    
    # Update function for animation
    def update(frame):
        # Run simulation for interval_sweeps sweeps using optimized sweep function
        steps = interval_sweeps * N * N
        metropolis_sweep(lattice, N, beta, steps)
        
        # Update the plot
        im.set_array((lattice + 1) // 2)
        
        # Calculate and display magnetization
        mag = calculate_magnetization(lattice, N)
        mag_text.set_text(f"Frame: {frame+1}/{num_frames}\n|M|: {abs(mag):.3f}")
        
        # Return the artists that have been modified
        return [im, mag_text]
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)
    
    # Save the animation
    filename = f"ising_animation_L{L}_beta{beta:.3f}.gif"
    print(f"Creating animation with {num_frames} frames...")
    ani.save(filename, writer='pillow', fps=5)
    print(f"Animation saved to {filename}")
    
    return ani

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
    # --- Parameters for 100x100 grid (L=50) ---
    param_L = 50  # Half the linear size of the grid (N=2L=100)
    param_eq_sweeps = 5000 # Increased from 500 to 5000 for better equilibration
    param_meas_sweeps = 10000 # Increased from 1000 to 10000 for better statistics
    param_bin_size = 10  # Bin size for measurement binning

    # --- Scan over Temperature T with denser sampling near Tc ~ 2.269 ---
    # Define Temperature points with concentration around Tc
    T_low = np.linspace(0.5, 2.1, 15, endpoint=False) # 15 points below 2.1
    T_crit = np.linspace(2.1, 2.4, 25) # 25 points in the critical region [2.1, 2.4]
    T_high = np.linspace(2.4, 5.0, 15, endpoint=True)[1:] # 14 points above 2.4 (start from index 1 to avoid duplicate 2.4)
    temp_values = np.unique(np.concatenate((T_low, T_crit, T_high))) # Combine and remove duplicates
    num_temp_points = len(temp_values) # Get the actual number of points

    # Avoid T=0 directly, as beta -> infinity
    beta_values = 1.0 / temp_values # Calculate corresponding beta values

    print(f"Scanning {num_temp_points} Temperature values from {temp_values[0]:.3f} to {temp_values[-1]:.3f}")
    print(f"Concentrated sampling around Tc ~ 2.269")
    print(f"(Corresponding beta range: {beta_values[-1]:.3f} to {beta_values[0]:.3f})")
    print(f"Using L={param_L}, EqSweeps={param_eq_sweeps}, MeasSweeps={param_meas_sweeps}")
    print(f"Using data binning with bin_size={param_bin_size}")
    print(f"Using temperature-dependent initialization WITHOUT annealing")

    # --- Run Simulations ---
    scan_start_time = time.time()
    results_T = []
    results_M = [] # Store <|M|>
    results_Chi = [] # Store susceptibility
    results_E = [] # Store <E>/N^2

    for T, beta in zip(temp_values, beta_values):
        # Temperature-dependent initialization strategy
        if T < 2.0:  # Low temperature (ordered phase)
            initial_state = 'up'  # Start from ordered state
        elif T < 2.5:  # Near critical (T_c ≈ 2.269)
            initial_state = 'random'  # Random state
        else:  # High temperature (disordered phase)
            initial_state = 'random'  # Random state is good
        
        # Call simulation function with improved strategies but NO annealing
        res_beta, res_M, res_Chi, res_E, _ = run_simulation(
            L=param_L,
            beta=beta,
            num_equilibration_sweeps=param_eq_sweeps,
            num_measurement_sweeps=param_meas_sweeps,
            initial_state=initial_state,
            use_annealing=False,  # Disable annealing for all points
            bin_size=param_bin_size
        )
        # Store results
        results_T.append(T)
        results_M.append(res_M)
        results_Chi.append(res_Chi)
        results_E.append(res_E)
        print("--------------------")

    scan_end_time = time.time()
    print(f"\n--- Scan Finished --- Total time: {scan_end_time - scan_start_time:.2f} seconds ---")

    # --- Plotting Results vs Temperature ---
    if MATPLOTLIB_AVAILABLE and results_T:
        print("Plotting results...")
        fig, axs = plt.subplots(1, 3, figsize=(21, 6)) # 1 row, 3 columns

        # Theoretical critical temperature for 2D Ising model
        Tc_exact = 2.0 / np.log(1 + np.sqrt(2)) # Approx 2.269

        # Plot 1: Energy vs Temperature
        axs[0].plot(results_T, results_E, 'o', linestyle='-', color='indianred', label='Simulation Data')
        axs[0].axvline(Tc_exact, color='gray', linestyle='--', label=f'Exact $T_c \\approx {Tc_exact:.3f}$')
        axs[0].set_xlabel("Temperature (T)")
        axs[0].set_ylabel("Energy per Site $\\langle E \\rangle / N^2$")
        axs[0].set_title(f"Energy vs Temperature (L={param_L})")
        axs[0].legend()
        axs[0].grid(True, linestyle=':', alpha=0.7)

        # Plot 2: Magnetization vs Temperature
        axs[1].plot(results_T, results_M, 'o', linestyle='-', color='royalblue', label='Simulation Data $\\langle |M| \\rangle$')
        axs[1].axvline(Tc_exact, color='gray', linestyle='--', label=f'Exact $T_c \\approx {Tc_exact:.3f}$')
        axs[1].set_xlabel("Temperature (T)")
        axs[1].set_ylabel("Magnetization $\\langle |M| \\rangle$")
        axs[1].set_title(f"Magnetization vs Temperature (L={param_L})")
        axs[1].legend()
        axs[1].grid(True, linestyle=':', alpha=0.7)

        # Plot 3: Susceptibility vs Temperature
        axs[2].plot(results_T, results_Chi, 'o', linestyle='-', color='mediumseagreen', label='Simulation Data $\\chi$')
        axs[2].axvline(Tc_exact, color='gray', linestyle='--', label=f'Exact $T_c \\approx {Tc_exact:.3f}$')
        axs[2].set_xlabel("Temperature (T)")
        axs[2].set_ylabel("Susceptibility $\\chi$")
        axs[2].set_title(f"Susceptibility vs Temperature (L={param_L})")
        axs[2].legend()
        axs[2].grid(True, linestyle=':', alpha=0.7)

        fig.tight_layout(pad=3.0)
        # Update plot filename to reflect the enhanced methodology
        plot_filename = f'ising_E_M_Chi_vs_T_L{param_L}_Eq{param_eq_sweeps}_Me{param_meas_sweeps}_enhanced.png'
        plt.savefig(plot_filename)
        print(f"Energy, Magnetization and Susceptibility plots saved to {plot_filename}")
    elif not MATPLOTLIB_AVAILABLE:
         print("\nMatplotlib not found. Cannot create plots.")
    else:
         print("\nNo results to plot.")

    # --- Generate Animations at different temperatures ---
    if MATPLOTLIB_AVAILABLE and IMAGEIO_AVAILABLE:
        print("\n--- Generating Animations ---")
        
        # Animation parameters
        animation_frames = 40
        animation_interval_sweeps = 10
        animation_equilibrate_sweeps = 100
        
        # 1. Low Temperature (T ≈ 1.67, beta = 0.6)
        low_beta = 0.6
        low_temp = 1 / low_beta
        print(f"\nGenerating animation for low temperature (T={low_temp:.4f}, beta={low_beta:.4f})...")
        create_ising_animation(
            L=param_L, 
            beta=low_beta, 
            num_frames=animation_frames, 
            interval_sweeps=animation_interval_sweeps,
            initial_state='up',  # Ordered state for low temp
            equilibrate_sweeps=animation_equilibrate_sweeps
        )
        
        # 2. Critical Temperature (T ≈ 2.269)
        crit_temp = Tc_exact
        crit_beta = 1.0 / crit_temp
        print(f"\nGenerating animation for critical temperature (T={crit_temp:.4f}, beta={crit_beta:.4f})...")
        create_ising_animation(
            L=param_L, 
            beta=crit_beta, 
            num_frames=animation_frames, 
            interval_sweeps=animation_interval_sweeps,
            initial_state='random',  # Random state for critical point
            equilibrate_sweeps=animation_equilibrate_sweeps
        )
        
        # 3. High Temperature (T = 5.0)
        high_temp = 5.0
        high_beta = 1.0 / high_temp
        print(f"\nGenerating animation for high temperature (T={high_temp}, beta={high_beta:.4f})...")
        create_ising_animation(
            L=param_L, 
            beta=high_beta, 
            num_frames=animation_frames, 
            interval_sweeps=animation_interval_sweeps,
            initial_state='random',  # Random state for high temp
            equilibrate_sweeps=animation_equilibrate_sweeps
        )
        
        print("\nAnimations generated successfully!")
    else:
        print("\nMatplotlib or imageio not available. Cannot create animations.")

    print("\n--- Main script finished ---") 