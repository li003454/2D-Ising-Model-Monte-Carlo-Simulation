import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import prange
import imageio
import os
import time

# --- LatticeSetup Module ---
@numba.jit(nopython=True)
def initialize_lattice(L, init_type='random', T=None):
    """Initializes the lattice.

    Args:
        L (int): Linear size of the lattice (NxN where N=2L).
        init_type (str): 'random' or 'ordered'. If 'auto', uses T to decide.
        T (float, optional): Temperature, used if init_type is 'auto'.

    Returns:
        np.ndarray: Initialized lattice (N x N).
    """
    N = 2 * L
    if init_type == 'auto' and T is not None:
        if T < 2.0:  # Below Tc, start ordered
            init_type = 'ordered'
        else:        # Near or above Tc, start random
            init_type = 'random'

    if init_type == 'ordered':
        lattice = np.ones((N, N), dtype=np.int8)
    else: # Default to random
        lattice = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(N, N))
    return lattice

# --- EnergyCalculator Module ---
@numba.jit(nopython=True)
def calculate_energy_change(lattice, i, j, L, H):
    """Calculates the energy change if spin (i, j) is flipped, including external field H.

    Args:
        lattice (np.ndarray): The current lattice configuration.
        i (int): Row index of the spin.
        j (int): Column index of the spin.
        L (int): Linear size parameter (N=2L).
        H (float): External magnetic field strength.

    Returns:
        float: Change in energy (Delta E) if the spin were flipped.
    """
    N = 2 * L
    s = lattice[i, j]
    # Sum of nearest neighbors using periodic boundary conditions
    neighbor_sum = (lattice[(i + 1) % N, j] +
                    lattice[(i - 1 + N) % N, j] +
                    lattice[i, (j + 1) % N] +
                    lattice[i, (j - 1 + N) % N])
    # Delta E = 2*J*s*neighbor_sum + 2*H*s (with J=1)
    delta_E = 2 * s * neighbor_sum + 2 * H * s
    return delta_E

@numba.jit(nopython=True, parallel=True)
def calculate_total_energy(lattice, L, H):
    """Calculates the total energy of the lattice, including external field H.

    Args:
        lattice (np.ndarray): The lattice configuration.
        L (int): Linear size parameter (N=2L).
        H (float): External magnetic field strength.

    Returns:
        float: Total energy of the system.
    """
    N = 2 * L
    total_energy_interaction = 0.0
    total_energy_field = 0.0
    # Use prange for parallel loop execution
    # Cannot directly sum into a shared variable in parallel with H term easily
    # Calculate interaction energy in parallel
    for i in prange(N):
        for j in range(N):
            s = lattice[i, j]
            # Sum neighbors - only right and down to avoid double counting
            neighbor_sum = (lattice[(i + 1) % N, j] + lattice[i, (j + 1) % N])
            total_energy_interaction -= s * neighbor_sum # J=1 assumed

    # Calculate field energy (can be done sequentially or in parallel reduction)
    # For simplicity, sequential sum here. Could parallelize if needed.
    for i in range(N):
        for j in range(N):
             total_energy_field -= H * lattice[i, j]

    return total_energy_interaction + total_energy_field

# --- Observables Module ---
@numba.jit(nopython=True)
def calculate_magnetization(lattice):
    """Calculates the total magnetization of the lattice.

    Args:
        lattice (np.ndarray): The lattice configuration.

    Returns:
        int: Total magnetization M.
    """
    return np.sum(lattice)

# --- MetropolisStep Module ---
@numba.jit(nopython=True)
def metropolis_step(lattice, beta, L):
    """Performs one Metropolis update step on a randomly chosen spin.

    Args:
        lattice (np.ndarray): The lattice configuration (modified in-place).
        beta (float): Inverse temperature (1/T).
        L (int): Linear size parameter (N=2L).
    """
    N = 2 * L
    # Choose a random spin
    i, j = np.random.randint(0, N, size=2)

    # Calculate energy change if this spin is flipped
    delta_E = calculate_energy_change(lattice, i, j, L, 0.0)

    # Metropolis acceptance criterion
    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
        lattice[i, j] *= -1 # Flip the spin

# --- Optimized Sweep ---
@numba.jit(nopython=True)
def metropolis_sweep(lattice, beta, L, H):
    """Performs N*N Metropolis steps (one full sweep) efficiently, including field H.

    Args:
        lattice (np.ndarray): The lattice configuration (modified in-place).
        beta (float): Inverse temperature (1/T).
        L (int): Linear size parameter (N=2L).
        H (float): External magnetic field strength.
    """
    N = 2 * L
    for _ in range(N * N):
        # Choose a random spin
        i, j = np.random.randint(0, N, size=2)
        s = lattice[i, j]
        # Sum neighbors using periodic boundary conditions
        neighbor_sum = (lattice[(i + 1) % N, j] +
                        lattice[(i - 1 + N) % N, j] +
                        lattice[i, (j + 1) % N] +
                        lattice[i, (j - 1 + N) % N])
        # delta_E = 2*J*s*neighbor_sum + 2*H*s (J=1)
        delta_E = 2 * s * neighbor_sum + 2 * H * s

        # Metropolis acceptance
        if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
            lattice[i, j] *= -1

# --- SimulationRunner Module ---
def run_simulation(L, T, H, eq_sweeps, meas_sweeps, init_state='auto'):
    """Runs the Ising model simulation for a given temperature and field H.

    Args:
        L (int): Linear size parameter.
        T (float): Temperature.
        H (float): External magnetic field strength.
        eq_sweeps (int): Number of equilibration sweeps.
        meas_sweeps (int): Number of measurement sweeps.
        init_state (str): Initial state ('random', 'ordered', 'auto').

    Returns:
        tuple: (average_energy_per_site, average_magnetization_per_site, susceptibility)
    """
    N = 2 * L
    N2 = N * N
    beta = 1.0 / T
    lattice = initialize_lattice(L, init_type=init_state, T=T)

    print(f"  Equilibrating at T={T:.3f}, H={H:.2f} ({eq_sweeps} sweeps)...")
    # Equilibration phase
    for sweep in range(eq_sweeps):
        metropolis_sweep(lattice, beta, L, H)
        # Optional: print progress
        # if (sweep + 1) % (eq_sweeps // 10) == 0:
        #     print(f"    Equilibration sweep {sweep+1}/{eq_sweeps}")


    print(f"  Measuring at T={T:.3f}, H={H:.2f} ({meas_sweeps} sweeps)...")
    # Measurement phase
    energies = []
    magnetizations = []
    measurement_interval = 5 # Measure every 5 sweeps

    for sweep in range(meas_sweeps):
        metropolis_sweep(lattice, beta, L, H)
        if (sweep + 1) % measurement_interval == 0:
            energy = calculate_total_energy(lattice, L, H)
            magnetization = calculate_magnetization(lattice)
            energies.append(energy)
            magnetizations.append(magnetization)
        # Optional: print progress
        # if (sweep + 1) % (meas_sweeps // 10) == 0:
        #     print(f"    Measurement sweep {sweep+1}/{meas_sweeps}")

    energies = np.array(energies)
    magnetizations = np.array(magnetizations)

    # Calculate observables
    avg_E = np.mean(energies)
    avg_M = np.mean(magnetizations) # Use average M directly for H!=0
    avg_M2 = np.mean(magnetizations**2)
    # avg_abs_M = np.mean(np.abs(magnetizations))

    # Susceptibility: chi = beta * N^2 * (<M^2> - <M>^2) for H!=0
    susceptibility = beta * N2 * (avg_M2 - avg_M**2)

    avg_E_per_site = avg_E / N2
    avg_M_per_site = avg_M / N2 # Return average M per site

    return avg_E_per_site, avg_M_per_site, susceptibility


# --- Animation Function ---
def create_ising_animation(L, T, H, num_frames=100, steps_per_frame=10, filename="ising_animation.gif", init_state='auto'):
    """Creates a GIF animation of the Ising model evolution with field H.

    Args:
        L (int): Linear size parameter.
        T (float): Temperature.
        H (float): External magnetic field strength.
        num_frames (int): Number of frames in the GIF.
        steps_per_frame (int): Number of sweeps between frames.
        filename (str): Output filename for the GIF.
        init_state (str): Initial state type.
    """
    N = 2 * L
    beta = 1.0 / T
    lattice = initialize_lattice(L, init_type=init_state, T=T)
    frames = []

    print(f"Generating animation for T={T:.3f}, H={H:.2f}...")
    start_time = time.time()
    for frame in range(num_frames):
        for _ in range(steps_per_frame):
            metropolis_sweep(lattice, beta, L, H)

        # Create an image representation: map -1 to white (255), +1 to black (0)
        img_array = np.zeros((N, N, 3), dtype=np.uint8)
        img_array[lattice == -1] = [255, 255, 255] # White for spin down
        img_array[lattice == 1] = [0, 0, 0]       # Black for spin up
        frames.append(img_array)
        if (frame + 1) % (num_frames // 10) == 0:
             elapsed = time.time() - start_time
             print(f"  Frame {frame+1}/{num_frames} generated... ({(frame+1)/elapsed:.1f} frames/sec)")


    print(f"Saving animation to {filename}...")
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    imageio.mimsave(filename, frames, duration=100) # Use duration in ms (100ms = 10 fps)
    print("Animation saved.")


# --- Main Block ---
if __name__ == "__main__":
    start_total_time = time.time()

    # Simulation Parameters (Enhanced)
    param_L = 50  # Linear size L (N=2L) -> 100x100 lattice
    param_H = 0.1 # External magnetic field strength
    param_eq_sweeps = 5000 # Equilibration sweeps (increased)
    param_meas_sweeps = 10000 # Measurement sweeps (increased)

    # Refined Temperature Sampling
    temps_low = np.linspace(0.5, 2.09, 15)  # Below Tc
    temps_crit = np.linspace(2.1, 2.4, 25)   # Around Tc (dense)
    temps_high = np.linspace(2.41, 4.0, 14) # Above Tc
    temp_values = np.concatenate((temps_low, temps_crit, temps_high))
    temp_values = np.unique(temp_values) # Ensure no duplicates if overlap
    num_temps = len(temp_values)
    print(f"Running simulation for L={param_L} (N={2*param_L}), H={param_H:.2f}")
    print(f"Equilibration sweeps: {param_eq_sweeps}")
    print(f"Measurement sweeps: {param_meas_sweeps}")
    print(f"Number of temperature points: {num_temps}")

    # Data storage
    results_E = np.zeros(num_temps)
    results_M = np.zeros(num_temps)
    results_Chi = np.zeros(num_temps)

    # Run simulation across temperatures
    print("Starting temperature scan...")
    for i, T in enumerate(temp_values):
        print(f"--- Processing Temperature {i+1}/{num_temps} (T={T:.3f}, H={param_H:.2f}) ---")
        start_temp_time = time.time()
        # Use 'auto' initial state based on temperature
        E_site, M_site, Chi = run_simulation(param_L, T, param_H, param_eq_sweeps, param_meas_sweeps, init_state='auto')
        results_E[i] = E_site
        results_M[i] = M_site # Already normalized in run_simulation
        results_Chi[i] = Chi
        end_temp_time = time.time()
        print(f"--- Finished T={T:.3f} in {end_temp_time - start_temp_time:.2f} seconds ---")
        print(f"    Results: <E>/N^2={E_site:.4f}, <M>/N^2={M_site:.4f}, Chi={Chi:.4f}")


    print("\nTemperature scan complete.")
    end_total_time = time.time()
    print(f"Total simulation time: {(end_total_time - start_total_time) / 60:.2f} minutes")


    # --- Plotting Results ---
    print("Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    Tc_theor = 2 / np.log(1 + np.sqrt(2)) # Theoretical Tc for J=1, H=0

    fig.suptitle(f'Ising Model Simulation Results (L={param_L}, H={param_H:.2f})', fontsize=16)

    # Energy Plot
    axes[0].plot(temp_values, results_E, 'o-', label='Simulated')
    axes[0].axvline(Tc_theor, color='r', linestyle='--', label=rf'$T_c(H=0) \approx {Tc_theor:.3f}$')
    axes[0].set_xlabel('Temperature (T)')
    axes[0].set_ylabel(r'Average Energy per Site $\langle E \rangle / N^2$')
    axes[0].set_title('Energy vs Temperature')
    axes[0].legend()
    axes[0].grid(True)

    # Magnetization Plot
    axes[1].plot(temp_values, results_M, 'o-', label=r'Simulated $\langle M \rangle / N^2$')
    axes[1].axvline(Tc_theor, color='r', linestyle='--', label=rf'$T_c(H=0) \approx {Tc_theor:.3f}$')
    axes[1].set_xlabel('Temperature (T)')
    axes[1].set_ylabel(r'Average Magnetization per Site $\langle M \rangle / N^2$')
    axes[1].set_title('Magnetization vs Temperature')
    axes[1].legend()
    axes[1].grid(True)

    # Susceptibility Plot
    axes[2].plot(temp_values, results_Chi, 'o-', label=r'Simulated $\chi$')
    axes[2].axvline(Tc_theor, color='r', linestyle='--', label=rf'$T_c(H=0) \approx {Tc_theor:.3f}$')
    axes[2].set_xlabel('Temperature (T)')
    axes[2].set_ylabel(r'Magnetic Susceptibility $\chi$')
    axes[2].set_title('Susceptibility vs Temperature')
    # axes[2].set_ylim(bottom=0) # Might need adjustment for H!=0
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    # Ensure the output directory exists
    plot_filename = f"image/ising_E_M_Chi_vs_T_L{param_L}_H{param_H:.2f}_Eq{param_eq_sweeps}_Me{param_meas_sweeps}.png"
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show() # Optionally display plot interactively


    # --- Generate Animations at Specific Temperatures ---
    print("\nGenerating animations...")
    anim_temps = {
        'low': 1.0 / 0.600, # T approx 1.67 (beta=0.6)
        'crit': Tc_theor,   # T approx 2.269 (beta approx 0.441)
        'high': 1.0 / 0.200  # T = 5.0 (beta=0.2)
    }
    animation_sweeps = 2000 # Number of sweeps to simulate for animation frames
    animation_frames = 200  # Number of frames in the GIF
    steps_p_frame = animation_sweeps // animation_frames

    for name, T_anim in anim_temps.items():
        beta_anim = 1.0 / T_anim
        anim_filename = f"image/ising_animation_L{param_L}_H{param_H:.2f}_beta{beta_anim:.3f}.gif"
        # For animation, use 'auto' to get appropriate starting state
        create_ising_animation(param_L, T_anim, param_H, # Pass H here
                               num_frames=animation_frames,
                               steps_per_frame=steps_p_frame,
                               filename=anim_filename,
                               init_state='auto')

    print("\nSimulation and analysis complete.") 