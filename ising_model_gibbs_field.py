import numpy as np
import random
import time
import numba
from numba import prange
import os

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

# ---------------------------
# Module: EnergyCalculator with External Field
# ---------------------------
@numba.jit(nopython=True)
def _sum_neighbor_spins(lattice: np.ndarray, N: int, i: int, j: int) -> int:
    """
    Calculates the sum of spins of the four nearest neighbors for site (i, j).
    Uses periodic boundary conditions implicitly via modulo arithmetic.
    
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

@numba.jit(nopython=True, parallel=True)
def calculate_total_energy(lattice: np.ndarray, N: int, h: float = 0.0) -> float:
    """
    Calculates the total energy of the lattice configuration with an external field.
    H = - Sum_{<ij>} s_i s_j - h * Sum_{i} s_i

    Args:
        lattice: The current NxN lattice configuration.
        N: The dimension of the lattice.
        h: External magnetic field strength.

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
            # Add external field contribution for this site
            row_energy += -h * s_ij
        total_energy += row_energy
    return total_energy

@numba.jit(nopython=True)
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
    total_spin = np.sum(lattice)
    return total_spin / (N * N)

# ---------------------------
# Module: GibbsSampling with External Field
# ---------------------------
@numba.jit(nopython=True)
def gibbs_step_field(lattice: np.ndarray, N: int, beta: float, h: float) -> bool:
    """
    Performs a single Gibbs sampling step with external field:
    1. Randomly selects a site (i, j).
    2. Calculates the conditional probability of spin being up or down given its neighbors and field.
    3. Sets the spin according to this probability.

    Args:
        lattice: The current NxN lattice configuration (modified in place).
        N: The dimension of the lattice.
        beta: The inverse temperature (1 / kT).
        h: External magnetic field strength.

    Returns:
        True if the spin was flipped, False otherwise.
    """
    # 1. Select a random spin site
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    
    # Get current spin value
    current_spin = lattice[i, j]
    
    # 2. Calculate the sum of neighboring spins
    neighbor_sum = _sum_neighbor_spins(lattice, N, i, j)
    
    # 3. Calculate conditional probabilities using Boltzmann distribution
    # Energy for spin up = -neighbor_sum * (+1) - h * (+1)
    # Energy for spin down = -neighbor_sum * (-1) - h * (-1) = neighbor_sum + h
    energy_up = -neighbor_sum - h
    energy_down = neighbor_sum + h
    
    # Calculate conditional probabilities
    # p(s = +1 | neighbors) ∝ exp(-β * energy_up)
    # p(s = -1 | neighbors) ∝ exp(-β * energy_down)
    weight_up = np.exp(-beta * energy_up)
    weight_down = np.exp(-beta * energy_down)
    
    # Normalize to get probability of spin being up
    prob_up = weight_up / (weight_up + weight_down)
    
    # 4. Sample the new spin value
    new_spin = 1 if random.random() < prob_up else -1
    
    # Check if spin was flipped
    was_flipped = new_spin != current_spin
    
    # Update the lattice
    lattice[i, j] = new_spin
    
    return was_flipped

@numba.jit(nopython=True)
def gibbs_sweep_field(lattice: np.ndarray, N: int, beta: float, h: float, num_steps: int) -> int:
    """
    Performs multiple Gibbs sampling steps with external field and returns the number of flipped spins.
    
    Args:
        lattice: The current NxN lattice configuration (modified in place).
        N: The dimension of the lattice.
        beta: The inverse temperature (1 / kT).
        h: External magnetic field strength.
        num_steps: Number of individual steps to perform.
        
    Returns:
        Number of flipped spins.
    """
    flipped_count = 0
    for _ in range(num_steps):
        if gibbs_step_field(lattice, N, beta, h):
            flipped_count += 1
    return flipped_count

# ---------------------------
# Module: SimulationRunner (Using Gibbs Sampling with External Field)
# ---------------------------
def run_gibbs_field_simulation(L: int, beta: float, h: float, num_equilibration_sweeps: int, 
                      num_measurement_sweeps: int, initial_state: str = 'random', 
                      visualize_final_state: bool = False, bin_size: int = 10):
    """
    Runs the 2D Ising model simulation with external field using Gibbs sampling.
    Calculates average magnetization, susceptibility, and energy per site.

    Args:
        L: Linear dimension of the lattice (total size N=2L).
        beta: Inverse temperature.
        h: External magnetic field strength.
        num_equilibration_sweeps: Number of MC sweeps for equilibration.
        num_measurement_sweeps: Number of MC sweeps for measurement.
        initial_state: Initial lattice configuration ('random', 'up', 'down').
        visualize_final_state: Whether to plot the final lattice state.
        bin_size: Number of measurements to group together for binning analysis.

    Returns:
        A tuple containing:
        - beta (float): The inverse temperature used for this run.
        - h (float): The external field strength used.
        - mean_magnetization (float): Average absolute magnetization per spin <|M|>.
        - mean_magnetization_signed (float): Average signed magnetization per spin <M>.
        - susceptibility (float): Magnetic susceptibility per spin chi.
        - mean_energy_per_site (float): Average energy per site <E> / N^2.
        - lattice (np.ndarray): The final lattice configuration.
    """
    N = 2 * L
    T = 1.0 / beta  # Current temperature
    equilibration_steps = num_equilibration_sweeps * N * N
    measurement_steps = num_measurement_sweeps * N * N
    total_steps = equilibration_steps + measurement_steps

    print(f"--- Running Ising Simulation (Gibbs+Field) for beta={beta:.4f} (T={T:.4f}), h={h:.4f} ---")
    print(f" L={L}, N={N}x{N}, EqSweeps={num_equilibration_sweeps}, MeasSweeps={num_measurement_sweeps}, Initial='{initial_state}'")

    # Initialization
    start_time = time.time()
    lattice = initialize_lattice(N, state=initial_state)

    # --- Equilibration Phase ---
    print(f" Starting Equilibration ({equilibration_steps} steps)...")
    # Use optimized sweep function for field case
    eq_flipped = gibbs_sweep_field(lattice, N, beta, h, equilibration_steps)
    eq_flip_rate = eq_flipped / equilibration_steps if equilibration_steps > 0 else 0
    print(f" Equilibration finished. Flip rate: {eq_flip_rate:.4f}")

    # --- Measurement Phase with improved strategy ---
    print(f" Starting Measurement ({measurement_steps} steps)...")
    magnetization_measurements = []
    magnetization_signed_measurements = [] # Track signed magnetization for external field
    energy_measurements = []
    flipped_measurement = 0
    
    # Increase interval between measurements to reduce correlations
    measure_interval = 5 * N * N  # Every 5 sweeps instead of every sweep
    num_actual_measurements = measurement_steps // measure_interval
    
    # Perform measurements with intervals
    remaining_steps = 0
    for m in range(num_actual_measurements):
        # Run simulation for measure_interval steps
        flipped = gibbs_sweep_field(lattice, N, beta, h, measure_interval)
        flipped_measurement += flipped
        
        # Take measurements
        current_M = calculate_magnetization(lattice, N)
        current_E = calculate_total_energy(lattice, N, h)
        magnetization_measurements.append(abs(current_M))  # Store absolute magnetization
        magnetization_signed_measurements.append(current_M)  # Also store signed magnetization
        energy_measurements.append(current_E)
    
    # Run any remaining steps
    remaining_steps = measurement_steps - (num_actual_measurements * measure_interval)
    if remaining_steps > 0:
        flipped = gibbs_sweep_field(lattice, N, beta, h, remaining_steps)
        flipped_measurement += flipped
    
    print(f" Took {len(magnetization_measurements)} measurements with {measure_interval//(N*N)} sweeps between each.")
    print(" Measurement finished.")

    # --- Analysis of Measurements with binning ---
    if not magnetization_measurements:
        print(" Warning: No measurements collected.")
        mean_M = np.nan
        mean_M_signed = np.nan
        mean_M_sq = np.nan
        susceptibility = np.nan
        mean_E_per_site = np.nan
    else:
        measurements_array = np.array(magnetization_measurements)
        measurements_signed_array = np.array(magnetization_signed_measurements)
        energy_array = np.array(energy_measurements)
        
        # Perform binning to reduce correlations (if we have enough measurements)
        if len(measurements_array) >= bin_size:
            num_bins = len(measurements_array) // bin_size
            # Reshape the arrays to group measurements into bins
            binned_M = measurements_array[:num_bins*bin_size].reshape(num_bins, bin_size)
            binned_M_signed = measurements_signed_array[:num_bins*bin_size].reshape(num_bins, bin_size)
            binned_E = energy_array[:num_bins*bin_size].reshape(num_bins, bin_size)
            
            # Calculate bin averages
            bin_M_avgs = np.mean(binned_M, axis=1)
            bin_M_signed_avgs = np.mean(binned_M_signed, axis=1)
            bin_E_avgs = np.mean(binned_E, axis=1)
            
            # Use bin averages for final calculations
            mean_M = np.mean(bin_M_avgs)
            mean_M_signed = np.mean(bin_M_signed_avgs)
            mean_M_sq = np.mean(bin_M_signed_avgs**2)
            mean_E_per_site = np.mean(bin_E_avgs) / (N * N)
        else:
            # Fall back to regular calculation if not enough measurements for binning
            mean_M = np.mean(measurements_array)
            mean_M_signed = np.mean(measurements_signed_array)
            mean_M_sq = np.mean(measurements_signed_array**2)
            mean_E_per_site = np.mean(energy_array) / (N * N)
        
        # Susceptibility calculation (using *signed* magnetization)
        susceptibility = beta * (N**2) * (mean_M_sq - mean_M_signed**2)

    end_time = time.time()
    total_time = end_time - start_time
    measurement_flip_rate = flipped_measurement / measurement_steps if measurement_steps > 0 else 0

    print(f" Beta={beta:.4f}, h={h:.4f} finished in {total_time:.2f}s.")
    print(f" <|M|>: {mean_M:.4f}, <M>: {mean_M_signed:.4f}, Chi: {susceptibility:.4f}")
    print(f" <E/N^2>: {mean_E_per_site:.4f}, Flip rate (meas): {measurement_flip_rate:.4f}")

    # Visualization (Optional - only the final state)
    if visualize_final_state and MATPLOTLIB_AVAILABLE:
        # Create directory if it doesn't exist
        if not os.path.exists('image_field'):
            os.makedirs('image_field')
            
        final_energy = calculate_total_energy(lattice, N, h)
        plt.figure(figsize=(6, 6))
        plt.imshow(lattice, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
        plt.title(f"Final State (Field h={h:.3f}, T={T:.3f}, E/N^2={final_energy/(N*N):.3f})")
        plt.xticks([])
        plt.yticks([])
        filename = f'image_field/ising_field_final_state_L{L}_beta{beta:.3f}_h{h:.3f}.png'
        plt.savefig(filename)
        print(f" Final lattice state saved to {filename}")
        plt.close()
    elif visualize_final_state and not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not found. Cannot visualize the final state.")

    # Return calculated observables and the final lattice
    return beta, h, mean_M, mean_M_signed, susceptibility, mean_E_per_site, lattice

# ---------------------------
# Animation Function for Field Model
# ---------------------------
def create_ising_field_animation(L: int, beta: float, h: float, num_frames: int, interval_sweeps: int, 
                         initial_state: str = 'random', equilibrate_sweeps: int = 100):
    """
    Creates an animation of the Ising model with external field evolving over time.
    
    Args:
        L: Linear dimension of the lattice (N=2L).
        beta: Inverse temperature.
        h: External magnetic field strength.
        num_frames: Number of frames in the animation.
        interval_sweeps: Number of sweeps between frames.
        initial_state: Initial lattice state ('random', 'up', 'down').
        equilibrate_sweeps: Number of sweeps to run before starting animation.
        
    Returns:
        True if animation was created successfully, False otherwise.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot create animation.")
        return False
    
    if not IMAGEIO_AVAILABLE:
        print("Imageio not available. Cannot create animation.")
        return False
    
    N = 2 * L
    
    # Create directories if they don't exist
    if not os.path.exists('image_field'):
        os.makedirs('image_field')
    
    temp_dir = 'image_field/temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Initialize lattice
    lattice = initialize_lattice(N, state=initial_state)
    
    # Equilibrate first using Gibbs sampling with field
    print(f"Equilibrating for {equilibrate_sweeps} sweeps using Gibbs sampling with h={h:.4f}...")
    equilibrate_steps = equilibrate_sweeps * N * N
    eq_flipped = gibbs_sweep_field(lattice, N, beta, h, equilibrate_steps)
    eq_flip_rate = eq_flipped / equilibrate_steps if equilibrate_steps > 0 else 0
    print(f"Equilibration finished. Flip rate: {eq_flip_rate:.4f}")
    
    # Generate and save individual frames
    print(f"Generating {num_frames} frames...")
    frame_paths = []
    
    for frame in range(num_frames):
        # Run simulation for interval_sweeps sweeps
        steps = interval_sweeps * N * N
        gibbs_sweep_field(lattice, N, beta, h, steps)
        
        # Calculate magnetization
        mag = calculate_magnetization(lattice, N)
        
        # Save this frame as a separate image
        plt.figure(figsize=(8, 8))
        plt.imshow((lattice + 1) // 2, cmap='binary', interpolation='nearest')
        plt.title(f"2D Ising Model - With Field h={h:.3f} (β={beta:.3f}, T={1/beta:.3f})")
        plt.text(0.02, 0.95, f"Frame: {frame+1}/{num_frames}\nM: {mag:.3f}", 
                transform=plt.gca().transAxes, color='red', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
        plt.axis('off')
        
        # Save the frame
        frame_path = f"{temp_dir}/frame_{frame:04d}.png"
        plt.savefig(frame_path)
        plt.close()
        frame_paths.append(frame_path)
        
        # Progress indicator
        if (frame + 1) % 10 == 0 or frame == num_frames - 1:
            print(f"Generated {frame + 1}/{num_frames} frames")
    
    # Create the GIF from the individual frames
    gif_path = f"image_field/ising_field_animation_L{L}_beta{beta:.3f}_h{h:.3f}.gif"
    print(f"Creating animation from {len(frame_paths)} frames...")
    
    try:
        # Use imageio to create the GIF
        images = []
        for frame_path in frame_paths:
            images.append(imageio.imread(frame_path))
        
        # Save as GIF with 200ms duration per frame (5 FPS)
        imageio.mimsave(gif_path, images, duration=200)
        print(f"Animation saved to {gif_path}")
        
        # Clean up temporary files
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        
        return True
    except Exception as e:
        print(f"Error creating animation: {e}")
        # Save the last frame as a static image
        static_filename = f"image_field/ising_field_final_L{L}_beta{beta:.3f}_h{h:.3f}.png"
        if frame_paths:
            # Just copy the last frame
            try:
                import shutil
                shutil.copy(frame_paths[-1], static_filename)
                print(f"Saved final state to {static_filename}")
            except:
                print("Failed to save final state image")
        return False

# ---------------------------
# Main Execution Block
# ---------------------------
if __name__ == '__main__':
    # --- Parameters for 100x100 grid (L=50) ---
    param_L = 50  # Half the linear size of the grid (N=2L=100)
    param_eq_sweeps = 2000  # Equilibration sweeps (reduced for external field)
    param_meas_sweeps = 4000  # Measurement sweeps (reduced for external field)
    param_bin_size = 10  # Bin size for measurement binning
    
    # --- Create directory for field model output ---
    if not os.path.exists('image_field'):
        os.makedirs('image_field')
        
    # --- Define the external field values to test ---
    field_values = [0.0, 0.1, 0.3, 0.5, 1.0]
    
    # --- Scan over Temperature T and field h ---
    # Define a smaller set of temperatures to run faster with multiple fields
    T_values = np.linspace(1.0, 4.0, 16)  # 16 temperatures from 1.0 to 4.0
    beta_values = 1.0 / T_values

    print(f"Running field simulation with {len(T_values)} temperature points")
    print(f"Testing {len(field_values)} external field values: {field_values}")
    
    # For each field value, run a temperature scan
    for h in field_values:
        print(f"\n=== Running temperature scan with external field h = {h:.2f} ===")
        
        # Store results for this field value
        results_T = []
        results_M = []  # Store <|M|>
        results_M_signed = []  # Store <M> with sign
        results_Chi = []  # Store susceptibility
        results_E = []  # Store <E>/N^2
        
        scan_start_time = time.time()
        
        for T, beta in zip(T_values, beta_values):
            # Temperature-dependent initialization strategy
            if T < 2.0:  # Low temperature (ordered phase)
                initial_state = 'up'  # Start from ordered state
            else:  # Near critical or high temperature
                initial_state = 'random'  # Random state
            
            # Call simulation function with Gibbs sampling and field
            res_beta, res_h, res_M, res_M_signed, res_Chi, res_E, _ = run_gibbs_field_simulation(
                L=param_L,
                beta=beta,
                h=h,
                num_equilibration_sweeps=param_eq_sweeps,
                num_measurement_sweeps=param_meas_sweeps,
                initial_state=initial_state,
                bin_size=param_bin_size
            )
            
            # Store results
            results_T.append(T)
            results_M.append(res_M)
            results_M_signed.append(res_M_signed)
            results_Chi.append(res_Chi)
            results_E.append(res_E)
            print("--------------------")
        
        scan_end_time = time.time()
        print(f"\n--- Scan Finished for h={h:.2f} --- Total time: {scan_end_time - scan_start_time:.2f} seconds ---")
        
        # --- Plotting Results vs Temperature for this field value ---
        if MATPLOTLIB_AVAILABLE and results_T:
            print(f"Plotting results for h={h:.2f}...")
            fig, axs = plt.subplots(1, 3, figsize=(21, 6))  # 1 row, 3 columns
            
            # Theoretical critical temperature for 2D Ising model (only valid for h=0)
            Tc_exact = 2.0 / np.log(1 + np.sqrt(2))  # Approx 2.269
            
            # Plot 1: Energy vs Temperature
            axs[0].plot(results_T, results_E, 'o-', markersize=6, color='indianred', 
                        label=f'Field h={h:.2f}')
            if h == 0.0:
                axs[0].axvline(Tc_exact, color='gray', linestyle='--', 
                            label=f'Exact $T_c \\approx {Tc_exact:.3f}$ (h=0)')
            axs[0].set_xlabel("Temperature (T)", fontsize=12)
            axs[0].set_ylabel("Energy per Site $\\langle E \\rangle / N^2$", fontsize=12)
            axs[0].set_title(f"Energy vs Temperature - Field h={h:.2f} (L={param_L})", fontsize=14)
            axs[0].legend(fontsize=11)
            axs[0].grid(True, linestyle=':', alpha=0.7)
            axs[0].tick_params(axis='both', which='major', labelsize=11)
            
            # Plot 2: Magnetization vs Temperature
            axs[1].plot(results_T, results_M, 'o-', markersize=6, color='royalblue',
                        label=f'$\\langle |M| \\rangle$, h={h:.2f}')
            axs[1].plot(results_T, results_M_signed, 's--', markersize=6, color='forestgreen',
                        label=f'$\\langle M \\rangle$, h={h:.2f}')
            if h == 0.0:
                axs[1].axvline(Tc_exact, color='gray', linestyle='--', 
                            label=f'Exact $T_c \\approx {Tc_exact:.3f}$ (h=0)')
            axs[1].set_xlabel("Temperature (T)", fontsize=12)
            axs[1].set_ylabel("Magnetization", fontsize=12)
            axs[1].set_title(f"Magnetization vs Temperature - Field h={h:.2f} (L={param_L})", fontsize=14)
            axs[1].legend(fontsize=11)
            axs[1].grid(True, linestyle=':', alpha=0.7)
            axs[1].tick_params(axis='both', which='major', labelsize=11)
            
            # Plot 3: Susceptibility vs Temperature
            axs[2].plot(results_T, results_Chi, 'o-', markersize=6, color='mediumseagreen',
                        label=f'Field h={h:.2f}')
            if h == 0.0:
                axs[2].axvline(Tc_exact, color='gray', linestyle='--', 
                            label=f'Exact $T_c \\approx {Tc_exact:.3f}$ (h=0)')
            axs[2].set_xlabel("Temperature (T)", fontsize=12)
            axs[2].set_ylabel("Susceptibility $\\chi$", fontsize=12)
            axs[2].set_title(f"Susceptibility vs Temperature - Field h={h:.2f} (L={param_L})", fontsize=14)
            axs[2].legend(fontsize=11)
            axs[2].grid(True, linestyle=':', alpha=0.7)
            axs[2].tick_params(axis='both', which='major', labelsize=11)
            
            # Enhance the plot aesthetics
            fig.tight_layout(pad=3.0)
            plot_filename = f'image_field/ising_field_E_M_Chi_vs_T_L{param_L}_h{h:.2f}.png'
            plt.savefig(plot_filename, dpi=150)
            print(f"Plots saved to {plot_filename}")
            plt.close()
    
    # --- Generate animations for selected field values and temperatures ---
    if MATPLOTLIB_AVAILABLE and IMAGEIO_AVAILABLE:
        print("\n--- Generating Animations with External Field ---")
        
        # Animation parameters
        animation_frames = 50
        animation_interval_sweeps = 10
        animation_equilibrate_sweeps = 500
        
        # Generate animations for 3 field values at 2 temperatures
        selected_fields = [0.0, 0.3, 1.0]  # Selected field values 
        selected_temps = [1.5, 2.5]  # Selected temperatures (below and above Tc)
        
        for h in selected_fields:
            for T in selected_temps:
                beta = 1.0 / T
                print(f"\nGenerating animation for field h={h:.2f}, temperature T={T:.2f}...")
                create_ising_field_animation(
                    L=param_L, 
                    beta=beta, 
                    h=h,
                    num_frames=animation_frames, 
                    interval_sweeps=animation_interval_sweeps,
                    initial_state='random',
                    equilibrate_sweeps=animation_equilibrate_sweeps
                )
        
        print("\nField animations generated successfully!")
        
    # --- Generate comparison plot of M vs h for different temperatures ---
    if MATPLOTLIB_AVAILABLE and field_values:
        print("\n--- Generating M vs h plot for different temperatures ---")
        
        # Selected temperatures for M vs h plot
        plot_temps = [1.5, 2.2, 3.0, 4.0]
        
        # Collect results for each temperature
        mh_results = {}
        
        # Structure for storing results
        for T in plot_temps:
            mh_results[T] = {"h": [], "M": []}
        
        # For each temperature, run a field scan
        for T in plot_temps:
            beta = 1.0 / T
            print(f"\nRunning field scan at temperature T={T:.2f}...")
            
            for h in field_values:
                # Choose appropriate initial state
                initial_state = 'up' if T < 2.0 else 'random'
                
                # Run simulation with this field and temperature
                _, _, _, res_M_signed, _, _, _ = run_gibbs_field_simulation(
                    L=param_L,
                    beta=beta,
                    h=h,
                    num_equilibration_sweeps=param_eq_sweeps,
                    num_measurement_sweeps=param_meas_sweeps,
                    initial_state=initial_state,
                    bin_size=param_bin_size
                )
                
                # Store results
                mh_results[T]["h"].append(h)
                mh_results[T]["M"].append(res_M_signed)
                print("--------------------")
        
        # Create M vs h plot
        plt.figure(figsize=(10, 8))
        
        for T in plot_temps:
            plt.plot(mh_results[T]["h"], mh_results[T]["M"], 'o-', markersize=8, 
                    label=f'T={T:.2f}')
        
        plt.xlabel("External Field (h)", fontsize=14)
        plt.ylabel("Magnetization $\\langle M \\rangle$", fontsize=14)
        plt.title("Magnetization vs External Field at Different Temperatures", fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Save the plot
        mh_plot_filename = f'image_field/ising_field_M_vs_h_L{param_L}.png'
        plt.savefig(mh_plot_filename, dpi=150)
        print(f"M vs h plot saved to {mh_plot_filename}")
        plt.close()
    
    print("\n--- Field Ising model simulation complete ---") 