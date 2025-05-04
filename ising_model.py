import numpy as np
import random
import time # Added for timing
import numba # Added for JIT compilation
from numba import prange # Added for parallel loops
import os # Import the os module
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
def calculate_total_energy(lattice: np.ndarray, N: int, H: float = 0.0) -> int:
    """
    Calculates the total energy of the lattice configuration.
    Assumes the standard ferromagnetic Ising Hamiltonian H = - Sum_{<ij>} s_i s_j.
    Iterates through each site and sums the interaction with right and down neighbors
    to avoid double counting. Uses parallel loops for better performance.

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.
        H: External magnetic field strength (default is 0).

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
def metropolis_sweep(lattice: np.ndarray, N: int, beta: float, H: float, num_steps: int) -> int:
    """
    Performs multiple Metropolis steps and returns the number of accepted flips.
    This is more efficient than calling metropolis_step in a Python loop
    since the entire loop is JIT compiled.
    
    Args:
        lattice: The current NxN lattice configuration (modified in place).
        N: The dimension of the lattice.
        beta: The inverse temperature (1 / kT).
        H: External magnetic field strength (default is 0).
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
                   bin_size: int = 10, H: float = 0.0):
    """
    Runs the 2D Ising model simulation using Metropolis with equilibration and measurement phases.
    Calculates average magnetization, susceptibility, and energy per site.

    Args:
        L: Linear dimension of the lattice (total size N=2L).
        beta: Inverse temperature.
        num_equilibration_sweeps: Number of MC sweeps for equilibration.
        num_measurement_sweeps: Number of MC sweeps for measurement.
        initial_state: Initial lattice configuration ('random', 'up', 'down').
        visualize_final_state: Whether to plot the final lattice state.
        bin_size: Number of measurements to group together for binning analysis.
        H: External magnetic field strength (default is 0).

    Returns:
        A tuple containing:
        - beta (float): The inverse temperature used for this run.
        - mean_magnetization (float): Average absolute magnetization per spin <|M|> (if H=0) or <M> (if H!=0).
        - susceptibility (float): Magnetic susceptibility per spin chi.
        - mean_energy_per_site (float): Average energy per site <E> / N^2.
        - lattice (np.ndarray): The final lattice configuration.
    """
    N = 2 * L
    T = 1.0 / beta  # Current temperature
    equilibration_steps = num_equilibration_sweeps * N * N
    measurement_steps = num_measurement_sweeps * N * N
    total_steps = equilibration_steps + measurement_steps
    
    field_str = f", H={H:.3f}" if H != 0 else ""
    print(f"--- Running Ising Simulation for beta={beta:.4f} (T={T:.4f}{field_str}) ---")
    print(f" L={L}, N={N}x{N}, EqSweeps={num_equilibration_sweeps}, MeasSweeps={num_measurement_sweeps}, Initial='{initial_state}'")

    # Initialization
    start_time = time.time()
    lattice = initialize_lattice(N, state=initial_state)

    # --- Equilibration Phase ---
    print(f" Starting Equilibration ({equilibration_steps} steps)...")
    # Use optimized sweep function instead of step-by-step loop
    eq_flipped = metropolis_sweep(lattice, N, beta, H, equilibration_steps)
    eq_flip_rate = eq_flipped / equilibration_steps if equilibration_steps > 0 else 0
    print(f" Equilibration finished. Flip rate: {eq_flip_rate:.4f}")

    # --- Measurement Phase with improved strategy ---
    print(f" Starting Measurement ({measurement_steps} steps)...")
    magnetization_measurements = []
    energy_measurements = []
    flipped_measurement = 0
    
    # Increase interval between measurements to reduce correlations
    measure_interval = 5 * N * N  # Every 5 sweeps instead of every sweep
    num_actual_measurements = measurement_steps // measure_interval
    
    # Perform measurements with intervals
    remaining_steps = 0
    for m in range(num_actual_measurements):
        # Run simulation for measure_interval steps
        flipped = metropolis_sweep(lattice, N, beta, H, measure_interval)
        flipped_measurement += flipped
        
        # Take measurements
        current_M = calculate_magnetization(lattice, N)
        current_E = calculate_total_energy(lattice, N, H)
        magnetization_measurements.append(current_M)
        energy_measurements.append(current_E)
    
    # Run any remaining steps
    remaining_steps = measurement_steps - (num_actual_measurements * measure_interval)
    if remaining_steps > 0:
        flipped = metropolis_sweep(lattice, N, beta, H, remaining_steps)
        flipped_measurement += flipped
    
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
            if H == 0:
                mean_M = np.mean(np.abs(bin_M_avgs))
                mean_M_sq = np.mean(bin_M_avgs**2)
            else:
                mean_M = np.mean(bin_M_avgs)
                mean_M_sq = np.mean(bin_M_avgs**2)
            mean_E_per_site = np.mean(bin_E_avgs) / (N * N)
        else:
            # Fall back to regular calculation if not enough measurements for binning
            if H == 0:
                mean_M = np.mean(np.abs(measurements_array))
                mean_M_sq = np.mean(measurements_array**2)
            else:
                mean_M = np.mean(measurements_array)
                mean_M_sq = np.mean(measurements_array**2)
            mean_E_per_site = np.mean(energy_array) / (N * N)
        
        # Susceptibility calculation
        if H == 0:
            # For H=0, chi = beta * N^2 * (<M^2> - <|M|>^2) is often used, 
            # or sometimes beta * N^2 * <M^2> if assuming <M>=0 above Tc
            # We will use <M^2> - <|M|>^2 here for consistency with spontaneous mag.
             susceptibility = beta * (N**2) * (mean_M_sq - mean_M**2) 
        else:
            # For H!=0, use the standard definition chi = beta * N^2 * (<M^2> - <M>^2)
            susceptibility = beta * (N**2) * (mean_M_sq - mean_M**2)

    end_time = time.time()
    total_time = end_time - start_time
    measurement_flip_rate = flipped_measurement / measurement_steps if measurement_steps > 0 else 0

    print(f" Beta={beta:.4f}{field_str} finished in {total_time:.2f}s. <|M| or M>: {mean_M:.4f}, Chi: {susceptibility:.4f}, <E/N^2>: {mean_E_per_site:.4f}, Flip rate (meas): {measurement_flip_rate:.4f}")

    # Visualization (Optional - only the final state)
    if visualize_final_state and MATPLOTLIB_AVAILABLE:
        # Create directory if it doesn't exist
        output_dir = 'image_metropolis'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        final_energy = calculate_total_energy(lattice, N, H)
        plt.figure(figsize=(6, 6))
        plt.imshow(lattice, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
        title_str = f"Final State (L={L}, T={T:.3f}{field_str}, E/N^2={final_energy/(N*N):.3f})"
        plt.title(title_str)
        plt.xticks([])
        plt.yticks([])
        field_tag = f"_H{H:.3f}" if H != 0 else ""
        filename = f'{output_dir}/ising_final_state_L{L}_beta{beta:.3f}{field_tag}.png'
        plt.savefig(filename)
        print(f" Final lattice state saved to {filename}")
        plt.close()
    elif visualize_final_state and not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not found. Cannot visualize the final state.")

    # Return calculated observables and the final lattice
    return beta, mean_M, susceptibility, mean_E_per_site, lattice

# ---------------------------
# Animation Function
# ---------------------------
def create_ising_animation(L: int, beta: float, num_frames: int, interval_sweeps: int, 
                           initial_state: str = 'random', equilibrate_sweeps: int = 100, H: float = 0.0):
    """
    Creates an animation of the Ising model evolving over time.
    
    Args:
        L: Linear dimension of the lattice (N=2L).
        beta: Inverse temperature.
        num_frames: Number of frames in the animation.
        interval_sweeps: Number of sweeps between frames.
        initial_state: Initial lattice state ('random', 'up', 'down').
        equilibrate_sweeps: Number of sweeps to run before starting animation.
        H: External magnetic field strength (default is 0).
        
    Returns:
        Animation object if matplotlib is available, None otherwise.
    """
    if not MATPLOTLIB_AVAILABLE or not hasattr(animation, 'FuncAnimation'):
        print("Matplotlib and/or animation module not available. Cannot create animation.")
        return None
    
    N = 2 * L
    
    # Create directory if it doesn't exist
    output_dir = 'image_metropolis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize lattice
    lattice = initialize_lattice(N, state=initial_state)
    
    # Equilibrate first
    print(f"Equilibrating for {equilibrate_sweeps} sweeps...")
    equilibration_steps = equilibrate_sweeps * N * N
    metropolis_sweep(lattice, N, beta, H, equilibration_steps)
    print("Equilibration finished.")
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.close() # Prevents the empty figure from being displayed
    
    # Create a custom colormap with white for -1 and black for +1
    cmap = ListedColormap(['white', 'black'])
    
    # Initial plot
    im = ax.imshow((lattice + 1) // 2, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
    field_str = f", H={H:.3f}" if H != 0 else ""
    ax.set_title(f"2D Ising Model (L={L}, β={beta:.3f}, T={1/beta:.3f}{field_str})")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create a text annotation for magnetization
    mag_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, 
                       color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Update function for animation
    def update(frame):
        # Run simulation for interval_sweeps sweeps
        steps = interval_sweeps * N * N
        metropolis_sweep(lattice, N, beta, H, steps)
        
        # Update the plot
        im.set_array((lattice + 1) // 2)
        
        # Update magnetization text
        mag = calculate_magnetization(lattice, N)
        mag_text.set_text(f"Frame: {frame+1}/{num_frames}\n|M|: {abs(mag):.3f}")
        
        # Return the artists that have been modified
        return [im, mag_text]

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)
    
    # Save the animation
    field_tag = f"_H{H:.3f}" if H != 0 else ""
    filename = f'{output_dir}/ising_animation_L{L}{field_tag}_beta{beta:.3f}.gif'
    print(f"Creating animation with {num_frames} frames...")
    try:
        # Use the Pillow writer for better compatibility and smaller file sizes
        ani.save(filename, writer='pillow', fps=5)
        print(f"Animation saved to {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Attempting fallback writer...")
        try:
            ani.save(filename, fps=5) # Try default writer
            print(f"Animation saved to {filename} using fallback writer.")
        except Exception as e2:
            print(f"Fallback writer also failed: {e2}")
            # Try saving the last frame as a static image
            print("Saving a static image of the final state instead...")
            plt.figure(figsize=(8, 8))
            plt.imshow((lattice + 1) // 2, cmap=cmap, interpolation='nearest')
            plt.title(f"Final State (L={L}, T={1/beta:.3f}{field_str})")
            plt.axis('off')
            static_filename = f'{output_dir}/ising_final_state_L{L}{field_tag}_beta{beta:.3f}.png'
            plt.savefig(static_filename)
            print(f"Static image saved to {static_filename}")
            plt.close()
            return None # Indicate animation saving failed
    
    return ani

# ---------------------------
# Main Execution Block
# ---------------------------
if __name__ == '__main__':
    # --- Parameters for 100x100 grid (L=50) ---
    param_L = 50 # Half the linear size of the grid (N=2L=100)
    param_eq_sweeps = 5000 # Equilibration sweeps
    param_meas_sweeps = 10000 # Measurement sweeps
    param_bin_size = 10 # Bin size for measurement binning
    param_H = 0.0 # Set external field H (default 0 for standard model)

    # --- Scan over Temperature T with denser sampling near Tc ~ 2.269 ---
    # Define Temperature points with concentration around Tc
    T_low = np.linspace(0.5, 2.1, 15, endpoint=False) # 15 points below 2.1
    T_crit = np.linspace(2.1, 2.4, 25) # 25 points in the critical region [2.1, 2.4]
    T_high = np.linspace(2.4, 5.0, 15, endpoint=True)[1:] # 14 points above 2.4
    temp_values = np.unique(np.concatenate((T_low, T_crit, T_high))) # Combine and remove duplicates
    num_temp_points = len(temp_values) # Get the actual number of points

    # Avoid T=0 directly, as beta -> infinity
    beta_values = 1.0 / temp_values # Calculate corresponding beta values

    print(f"Scanning {num_temp_points} Temperature values from {temp_values[0]:.3f} to {temp_values[-1]:.3f}")
    print(f"Concentrated sampling around Tc ~ 2.269")
    print(f"(Corresponding beta range: {beta_values[-1]:.3f} to {beta_values[0]:.3f})")
    print(f"Using L={param_L}, EqSweeps={param_eq_sweeps}, MeasSweeps={param_meas_sweeps}")
    print(f"Using data binning with bin_size={param_bin_size}")
    if param_H != 0:
        print(f"Using external field H = {param_H:.3f}")
    else:
        print(f"Using H = 0")

    # Create directory for output
    output_dir = 'image_metropolis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Run Simulations --- 
    scan_start_time = time.time()
    results_T = []
    results_M = [] # Store <|M|> or <M>
    results_Chi = [] # Store susceptibility
    results_E = [] # Store <E>/N^2

    for T, beta in zip(temp_values, beta_values):
        # ALWAYS use random initial state
        initial_state = 'random'
        
        # Call simulation function
        res_beta, res_M, res_Chi, res_E, _ = run_simulation(
            L=param_L,
            beta=beta,
            num_equilibration_sweeps=param_eq_sweeps,
            num_measurement_sweeps=param_meas_sweeps,
            initial_state=initial_state,
            bin_size=param_bin_size,
            H=param_H
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
        
        # Determine plot labels based on H
        mag_label = "|M|" if param_H == 0 else "M"
        plot_title_suffix = f" (L={param_L})" if param_H == 0 else f" (L={param_L}, H={param_H:.2f})"
        filename_suffix = f"_L{param_L}" if param_H == 0 else f"_L{param_L}_H{param_H:.2f}"
        sweep_info = f"_Eq{param_eq_sweeps}_Me{param_meas_sweeps}"

        # Plot 1: Energy vs Temperature
        axs[0].plot(results_T, results_E, 'o-', markersize=6, color='indianred', label='Simulation Data')
        if param_H == 0:
            axs[0].axvline(Tc_exact, color='gray', linestyle='--', label=f'Exact Tc ≈ {Tc_exact:.3f}')
        axs[0].set_xlabel("Temperature (T)", fontsize=12)
        axs[0].set_ylabel("Energy per Site <E>/N²", fontsize=12)
        axs[0].set_title("Energy vs Temperature" + plot_title_suffix, fontsize=14)
        axs[0].legend(fontsize=11)
        axs[0].grid(True, linestyle=':', alpha=0.7)
        axs[0].tick_params(axis='both', which='major', labelsize=11)

        # Plot 2: Magnetization vs Temperature
        axs[1].plot(results_T, results_M, 'o-', markersize=6, color='royalblue', label=f'<{mag_label}>')
        if param_H == 0:
            axs[1].axvline(Tc_exact, color='gray', linestyle='--', label=f'Exact Tc ≈ {Tc_exact:.3f}')
        axs[1].set_xlabel("Temperature (T)", fontsize=12)
        axs[1].set_ylabel(f"Magnetization <{mag_label}>", fontsize=12)
        axs[1].set_title("Magnetization vs Temperature" + plot_title_suffix, fontsize=14)
        axs[1].legend(fontsize=11)
        axs[1].grid(True, linestyle=':', alpha=0.7)
        axs[1].tick_params(axis='both', which='major', labelsize=11)

        # Plot 3: Susceptibility vs Temperature
        axs[2].plot(results_T, results_Chi, 'o-', markersize=6, color='mediumseagreen', label='Susceptibility χ')
        if param_H == 0:
            axs[2].axvline(Tc_exact, color='gray', linestyle='--', label=f'Exact Tc ≈ {Tc_exact:.3f}')
        axs[2].set_xlabel("Temperature (T)", fontsize=12)
        axs[2].set_ylabel("Susceptibility χ", fontsize=12)
        axs[2].set_title("Susceptibility vs Temperature" + plot_title_suffix, fontsize=14)
        axs[2].legend(fontsize=11)
        axs[2].grid(True, linestyle=':', alpha=0.7)
        axs[2].tick_params(axis='both', which='major', labelsize=11)

        # Enhance the plot aesthetics
        fig.tight_layout(pad=3.0)
        plot_filename = f'{output_dir}/ising_E_M_Chi_vs_T{filename_suffix}{sweep_info}_enhanced.png'
        plt.savefig(plot_filename, dpi=150)
        print(f"Energy, Magnetization and Susceptibility plots saved to {plot_filename}")
        plt.close()
    elif not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not found. Cannot create plots.")
    else:
        print("\nNo results to plot.")

    # --- Generate Animations at different temperatures (only if H=0 for now) ---
    if MATPLOTLIB_AVAILABLE and param_H == 0.0:
        print("\n--- Generating Animations ---")
        
        # Animation parameters
        animation_frames = 50 # Reduced frames for faster generation
        animation_interval_sweeps = 10 # Sweeps between frames
        animation_equilibrate_sweeps = 500 # Equilibration sweeps for animation
        
        # Define specific temperatures for animation
        anim_temps = {
            "low": 1.0, # Low temperature
            "crit": Tc_exact, # Critical temperature
            "high": 5.0 # High temperature
        }
        anim_betas = {k: 1.0/v for k, v in anim_temps.items()}
        
        # 1. Low Temperature
        print(f"\nGenerating animation for low temperature (T={anim_temps['low']:.4f}, beta={anim_betas['low']:.4f})...")
        create_ising_animation(
            L=param_L, 
            beta=anim_betas['low'], 
            num_frames=animation_frames, 
            interval_sweeps=animation_interval_sweeps,
            initial_state='random', # Always start random
            equilibrate_sweeps=animation_equilibrate_sweeps,
            H=param_H
        )
        
        # 2. Critical Temperature
        print(f"\nGenerating animation for critical temperature (T={anim_temps['crit']:.4f}, beta={anim_betas['crit']:.4f})...")
        create_ising_animation(
            L=param_L, 
            beta=anim_betas['crit'], 
            num_frames=animation_frames, 
            interval_sweeps=animation_interval_sweeps,
            initial_state='random', # Always start random
            equilibrate_sweeps=animation_equilibrate_sweeps,
            H=param_H
        )
        
        # 3. High Temperature
        print(f"\nGenerating animation for high temperature (T={anim_temps['high']:.4f}, beta={anim_betas['high']:.4f})...")
        create_ising_animation(
            L=param_L, 
            beta=anim_betas['high'], 
            num_frames=animation_frames, 
            interval_sweeps=animation_interval_sweeps,
            initial_state='random', # Always start random
            equilibrate_sweeps=animation_equilibrate_sweeps,
            H=param_H
        )
        
        print("\nAnimations generated successfully!")
    elif param_H != 0.0:
        print("\nSkipping animation generation because H is non-zero in this script run.")
    else:
        print("\nMatplotlib not available. Cannot create animations.")

    print("\n--- Simulation complete ---") 