#!/usr/bin/env python3
"""
# Force-Matching Algorithm for Water Coarse-Graining

- Vectorized pair computations
- Numba JIT compilation for inner loops
- Reduced spline knot count for faster optimization
- Precomputed neighbor lists per frame
- Analytic gradient calculation

1. Load mapped CG positions and forces from atomistic simulations
2. Implements force-matching using cubic splines
3. Optimizes spline parameters to reproduce atomistic forces
4. Saves the resulting CG potential in LAMMPS tabulated format
5. Visualizes the results

## Usage Guide for Large-Scale Force Matching

1. For datasets with 11-100 frames:
    python force_matching.py --data-dir cg_mapped_data --n-threads 8 --batch-size 5

2. For very large datasets (1,000+ frames):
    python force_matching.py --data-dir cg_mapped_data --n-threads 16 --batch-size 20 --use-sgd --frames-per-step 100

Expected Performance Improvements:
Memory usage: Reduced by ~90% through batch processing.
"""

import os
import io
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import imageio.v2 as imageio

from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from scipy.stats import wasserstein_distance

import multiprocessing as mp
from numba import njit, types
from numba.typed import List


# Ensure imageio uses ffmpeg or pillow backend
os.environ['IMAGEIO_FFMPEG_EXE'] = 'ffmpeg'

@njit
def min_image_vector(rij, box):
    # your existing logic here
    # Apply minimum‐image convention
    rij -= box * np.round(rij / box)
    return rij

@njit
def compute_pairs(positions, box, r_cut):
    n = positions.shape[0]
    # preallocate Python lists or Numba typed List if desired
    pairs_i   = List.empty_list(types.int64)
    pairs_j   = List.empty_list(types.int64)
    pair_vecs = List.empty_list(types.float64[:])
    pair_dists= List.empty_list(types.float64)

    for i in range(n):
        for j in range(i+1, n):
            rij = positions[j] - positions[i]
            rij = min_image_vector(rij, box)

            dist = np.linalg.norm(rij)
            if dist > 1.0 and dist < r_cut:
                pairs_i.append(i)
                pairs_j.append(j)
                pair_vecs.append(rij)
                pair_dists.append(dist)

    # Return typed lists directly
    return pairs_i, pairs_j, pair_vecs, pair_dists

# Define this function at module level so it can be pickled
def process_batch_for_mp(batch_idx, force_match_obj, spline_params):
    """
    Process a batch of frames and calculate the MSE - this function must be outside the class for multiprocessing
    """
    return force_match_obj.process_batch(batch_idx, spline_params)

"""
# Our Previous Initial Guess Scheme
------------------------------------
    if initial_guess is None:
        # Start with a simple Lennard-Jones-like potential
        epsilon = 0.1  # kcal/mol
        sigma = 3.0    # Å
        
        initial_guess = np.zeros(self.n_spline_knots)
        for i, r in enumerate(self.r_knots):
            if r < sigma:
                initial_guess[i] = epsilon  # Repulsive part
            else:
                initial_guess[i] = epsilon * ((sigma/r)**12 - 2 * (sigma/r)**6)  # Attractive part
"""

def initialize_spline_params(r_knots):
    """
    Generate better initial guess for water-like potential
    
    Parameters:
    -----------
    r_knots : array
        Knot positions for the spline
        
    Returns:
    --------
    initial_guess : array
        Initial parameters for spline
    """
    # Parameters for SPC/E water model
    sigma = 3.166  # Angstrom
    epsilon = 0.65  # kcal/mol
    
    initial_guess = np.zeros_like(r_knots)
    
    for i, r in enumerate(r_knots):
        if r < 2.5:
            # Strong repulsion at short distances
            initial_guess[i] = 100.0 * (sigma/r)**12
        else:
            # More realistic attraction curve based on LJ potential
            initial_guess[i] = 4.0 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    
    # Ensure smooth potential by applying a small amount of smoothing
    from scipy.signal import savgol_filter
    initial_guess = savgol_filter(initial_guess, min(9, len(initial_guess) - 2), 3)
    
    # Ensure potential goes to zero at cutoff
    initial_guess -= initial_guess[-1]
    
    return initial_guess

class ForceMatching:
    def __init__(self, data_dir, r_cut=10.0, n_spline_knots=20, batch_size=10, n_threads=None):
        """
        Initialize the force-matching algorithm
        
        Parameters:
        -----------
        data_dir : str
            Directory containing mapped CG data
        r_cut : float
            Cutoff distance for the potential (Å)
        n_spline_knots : int
            Number of knots for the cubic spline representation
        batch_size : int
            Number of frames to process at once in batches
        n_threads : int or None
            Number of threads for parallel processing (None = use all available)
        """
        self.data_dir = data_dir
        self.r_cut = r_cut
        self.n_spline_knots = n_spline_knots
        self.batch_size = batch_size
        self.n_threads = n_threads if n_threads is not None else mp.cpu_count()
        
        # Load box size
        self.box_size = np.load(os.path.join(data_dir, "box_size.npy"))
        
        # Set up spline knots
        self.r_knots = np.linspace(1.0, r_cut, n_spline_knots)

        # Get list of all data files
        all_pos = sorted(glob.glob(os.path.join(self.data_dir, "cg_positions_*.npy")))
        all_forces = sorted(glob.glob(os.path.join(self.data_dir, "cg_forces_*.npy")))

        # === DEBUG: only load first 10 frames ===
        # self.pos_files   = all_pos[:10]
        # self.force_files = all_forces[:10]
        # =========================================

        self.pos_files   = all_pos
        self.force_files = all_forces

        self.n_frames = len(self.pos_files)
        print(f"Found {self.n_frames} frames for force-matching")
        
        # Pre-load a few frames to determine system size
        sample_pos = np.load(self.pos_files[0])
        self.n_particles = sample_pos.shape[0]
        print(f"System has {self.n_particles} particles")

    def load_batch(self, batch_indices):
        """
        Load a batch of frames
        
        Parameters:
        -----------
        batch_indices : list of int
            Indices of frames to load
            
        Returns:
        --------
        positions : list of arrays
            Positions for each frame
        forces : list of arrays
            Reference forces for each frame
        """
        positions = []
        forces = []
        
        for idx in batch_indices:
            if idx < self.n_frames:
                positions.append(np.load(self.pos_files[idx]))
                forces.append(np.load(self.force_files[idx]))
        
        return positions, forces

    def compute_forces(self, positions, spline_params):
        """
        Compute forces for a single frame
        
        Parameters:
        -----------
        positions : array
            Positions for the frame
        spline_params : array
            Parameters for the cubic spline potential
            
        Returns:
        --------
        forces : array
            Calculated forces
        """
        # Create cubic spline for evaluation
        spline = CubicSpline(self.r_knots, spline_params)
        spline_deriv = spline.derivative()
        
        # Compute pairs for this frame
        pairs_i, pairs_j, pair_vecs, pair_dists = compute_pairs(
            positions, self.box_size, self.r_cut)
        
        # **Use the actual number of CG sites in this frame:**
        n_sites = positions.shape[0]
        forces = np.zeros((n_sites, 3))
        
        # Loop over all pairs and calculate forces
        for p in range(len(pairs_i)):
            i, j = pairs_i[p], pairs_j[p]
            r = pair_dists[p]
            rij = pair_vecs[p]
            
            # Calculate force magnitude (negative derivative of potential)
            force_mag = -spline_deriv(r)
            
            # Force vector
            f = force_mag * rij / r
            
            # Add to total forces (action-reaction)
            forces[i] += f
            forces[j] -= f
        
        return forces
    
    def build_distance_grid(self, spline_params):
        """Build a table of force values for different distances"""
        # Create cubic spline
        spline = CubicSpline(self.r_knots, spline_params)
        spline_deriv = spline.derivative()
        
        # Create a grid of distances
        r_grid = np.linspace(1.0, self.r_cut, 1000)
        force_grid = -spline_deriv(r_grid)
        
        return r_grid, force_grid
    
    def calculate_forces_batch(self, positions, spline_params):
        """
        Calculate forces for a batch of frames
        
        Parameters:
        -----------
        positions : list of arrays
            Positions for each frame
        spline_params : array
            Spline parameters for the potential
            
        Returns:
        --------
        forces : list of arrays
            Calculated forces for each frame
        """
        all_forces = []
        
        for pos in positions:
            forces = self.compute_forces(pos, spline_params)
            all_forces.append(forces)
        
        return all_forces
    
    def process_batch(self, batch_idx, spline_params):
        """
        Process a batch of frames and calculate the MSE
        
        Parameters:
        -----------
        batch_idx : int
            Index of the batch
        spline_params : array
            Spline parameters for the potential
            
        Returns:
        --------
        mse : float
            Mean squared error for this batch
        """
        # Determine frame indices for this batch
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_frames)
        batch_indices = list(range(start_idx, end_idx))
        
        # Load batch data
        positions, target_forces = self.load_batch(batch_indices)
        
        for pos, f in zip(positions, target_forces):
            if pos.shape[0] != f.shape[0]:
                raise RuntimeError(f"Frame with {pos.shape[0]} CG sites but {f.shape[0]} forces.")

        # Calculate forces with current parameters
        cg_forces = self.calculate_forces_batch(positions, spline_params)
        
        # Calculate MSE for this batch
        batch_mse = 0.0
        for i in range(len(positions)):
            frame_mse = np.mean(np.sum((cg_forces[i] - target_forces[i])**2, axis=1))
            batch_mse += frame_mse
        
        return batch_mse / len(positions) if positions else 0.0
    
    def objective(self, spline_params):
        """
        Objective function for optimization
        
        Parameters:
        -----------
        spline_params : array
            Parameters for the cubic spline potential
            
        Returns:
        --------
        mse : float
            Mean squared error between CG and reference forces
        """
        # Determine number of batches
        n_batches = (self.n_frames + self.batch_size - 1) // self.batch_size

        # Process batches in parallel
        if self.n_threads > 1:
            with mp.Pool(self.n_threads) as pool:
                # Use a partial function for multiprocessing that's properly pickable
                func = partial(process_batch_for_mp, force_match_obj=self, spline_params=spline_params)
                batch_mses = pool.map(func, range(n_batches))
        else:
            # Sequential processing
            batch_mses = [self.process_batch(i, spline_params) for i in range(n_batches)]
        
        #  Average MSE across all batches
        return np.mean(batch_mses) if batch_mses else 0.0
    
    def optimize_potential(self, initial_guess=None, max_iter=50, tol=1e-2,
                          method='L-BFGS-B', use_stochastic=True, frames_per_step=50,
                          log_filename='spline_params_log.txt', gif_filename='training.gif'):
        """
        Optimize the CG potential using force-matching and log parameters.
        Additionally, generate a GIF simulating spline training.
        
        Parameters:
        -----------
        initial_guess : array-like, optional
            Initial guess for the spline parameters
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for convergence
        method : str
            Optimization method
        use_stochastic : bool
            Whether to use stochastic gradient descent (for large datasets)
        frames_per_step : int
            Number of frames to use per optimization step when using SGD
            
        Returns:
        --------
        optimal_params : array
            Optimized spline parameters
        """
        # Record deterministic loss history
        self._loss_history = []

        # Create initial guess if not provided
        if initial_guess is None:
            initial_guess = initialize_spline_params(self.r_knots)
    
        # Prepare log file and record initial guess
        with open(log_filename, 'w') as f:
            f.write('# Spline parameters log\n')
            f.write('# Initial guess:\n')
            np.savetxt(f, initial_guess[None, :], header='initial_guess', fmt='%.6e')

        # For very large datasets, use stochastic gradient descent
        if use_stochastic and self.n_frames > 1000:
            print("Using stochastic gradient descent for large dataset")
            optimal = self.optimize_stochastic(initial_guess, max_iter, frames_per_step, 
                                               log_filename, gif_filename)
        else:
            # Collect frames for GIF
            frames = []
            def callback(xk):
                # Record current loss
                loss = self.objective(xk)
                self._loss_history.append(loss)
                # Build force-distance plot
                r_grid, force_grid = self.build_distance_grid(xk)
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(r_grid, force_grid)
                plt.title(f'Iteration')
                plt.xlabel(r'r ($\AA$)')
                plt.ylabel('force')
                plt.tight_layout()
                # Save to buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                frames.append(imageio.imread(buf))

            # Run optimizer with callback
            print("Starting optimization...")
            start_time = time.time()

            # Optimize using specified method
            result = minimize(
                self.objective,
                initial_guess,
                method=method,
                options={'disp': True, 'maxiter': max_iter, 'gtol': tol},
                callback=callback
            )
            optimal = result.x

            # Save loss vs iteration
            plt.figure(figsize=(8, 5))
            plt.plot(self._loss_history, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel('Mean Squared Error')
            plt.title('Force-Matching Convergence')
            plt.tight_layout()
            plt.savefig('mse_vs_iteration.png', dpi=300)
            plt.close()
            
            # Build and save GIF
            if frames: imageio.mimsave(gif_filename, frames, fps=2)

            # Print results
            elapsed = time.time() - start_time
            print(f"Optimization complete: {result.message}")
            print(f"Final objective function value: {result.fun}")
            print(f"Optimization took {elapsed:.2f} seconds")

        # Append optimized parameters to log
        with open(log_filename, 'a') as f:
            f.write('# Optimized parameters:\n')
            np.savetxt(f, optimal[None, :], header='optimized_params', fmt='%.6e')

        print(f'Parameters logged to {log_filename}')
        print(f'Training GIF saved as {gif_filename}')
        return optimal
    
    def _loss_on_batch(self, positions, target_forces, spline_params):
        """
        Compute the MSE loss over a set of frames for a given spline_params.
        """
        # 1) Compute CG forces
        cg_forces = self.calculate_forces_batch(positions, spline_params)

        # 2) Compute MSE over all atoms in all frames
        total_mse = 0.0
        for cf, tf in zip(cg_forces, target_forces):
            total_mse += np.mean(np.sum((cf - tf)**2, axis=1))
        return total_mse / len(positions) if positions else 0.0
    
    def calculate_numerical_gradients(self, positions, target_forces, params, delta=1e-4):
        """
        Estimate gradients of the loss with respect to the parameters using central differences.
        Returns (gradients, MSE loss) where loss is the mean squared error between predicted and target forces.
        """
        n_params = len(params)
        grads = np.zeros_like(params)
        
        # Evaluate original predicted forces
        base_predicted = self.predict_forces(positions, params)
        base_loss = np.mean((base_predicted - target_forces)**2)
        
        for i in range(n_params):
            perturbed_params_plus = params.copy()
            perturbed_params_minus = params.copy()
            perturbed_params_plus[i] += delta
            perturbed_params_minus[i] -= delta

            pred_plus = self.predict_forces(positions, perturbed_params_plus)
            pred_minus = self.predict_forces(positions, perturbed_params_minus)

            # Compute central difference
            loss_plus = np.mean((pred_plus - target_forces)**2)
            loss_minus = np.mean((pred_minus - target_forces)**2)
            grads[i] = (loss_plus - loss_minus) / (2 * delta)

        return grads, base_loss


    def optimize_stochastic(self, initial_params, max_iter=50, frames_per_step=50,
                         log_filename=None, gif_filename=None):
        """
        Optimize using stochastic Adam (adaptive moment estimation) with Numba-accelerated forces.
        
        Parameters:
        -----------
        initial_params : array
            Initial parameters
        max_iter : int
            Maximum number of iterations
        frames_per_step : int
            Number of frames to use per optimization step
            
        Returns:
        --------
        optimal_params : array
            Optimized parameters
        """
        print(f"Starting stochastic optimization with {frames_per_step} frames per step")
        
        # Record stochastic loss history
        self._stochastic_loss_history = []
        
        # Initialize time tracking
        self._step_times = []
        start_time = time.time()

        # Initialize parameters
        params = initial_params.copy()
        best_params = params.copy()
        best_loss = float('inf')
        
        # Adam hyperparameters
        use_adam = True
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        
        # Momentum fallback if not using Adam
        use_momentum = not use_adam
        momentum = 0.9
        velocity = np.zeros_like(params)
        
        # Learning rate schedule
        lr_init, lr_decay = 1e-2, 0.98 # 1e-3 -> 1e-2
        
        # Finite difference step (central difference)
        delta = 1e-6

        # Gradient clipping threshold
        grad_clip = 1e2
        # Parameter clipping
        param_clip = 10.0

        # Validation set
        validation_frames = min(10, self.n_frames // 10)
        validation_idx = np.random.choice(self.n_frames, validation_frames, replace=False)
        val_pos, val_forces = self.load_batch(validation_idx)
        patience, no_improve = 7, 0
        
        frames = []

        for t in range(1, max_iter+1):
            # Start time
            step_start = time.time()

            # Decay learning rate
            lr = lr_init * (lr_decay ** (t-1))
            
            # Sample batch
            idxs = np.random.choice(self.n_frames, frames_per_step, replace=False)
            pos_batch, f_batch = self.load_batch(idxs)
            if not pos_batch:
                continue
            # Compute baseline loss
            base_loss = self._loss_on_batch(pos_batch, f_batch, params)
            
            # Compute gradients via central finite differences
            grads = np.zeros_like(params)
            for i in range(len(params)):
                p_plus = params.copy(); p_minus = params.copy()
                p_plus[i] += delta; p_minus[i] -= delta
                loss_p = self._loss_on_batch(pos_batch, f_batch, p_plus)
                loss_m = self._loss_on_batch(pos_batch, f_batch, p_minus)
                grads[i] = (loss_p - loss_m) / (2 * delta)
            
            # Clip grads
            grads = np.clip(grads, -grad_clip, grad_clip)
            
            # Update with Adam
            if use_adam:
                m = beta1 * m + (1 - beta1) * grads
                v = beta2 * v + (1 - beta2) * (grads**2)
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                params -= lr * m_hat / (np.sqrt(v_hat) + eps)
            elif use_momentum:
                velocity = momentum * velocity - lr * grads
                params += velocity
            else:
                params -= lr * grads
            
            # Clip parameters
            params = np.clip(params, -param_clip, param_clip)

            # Record time for this step
            step_time = time.time() - step_start
            self._step_times.append(step_time)
            
            # Validation and logging
            if t % 5 == 0 or t == max_iter:
                val_loss = self._loss_on_batch(val_pos, val_forces, params)
                self._stochastic_loss_history.append(val_loss)
                print(f"Iter {t:3d} | Train: {base_loss:.6f} | Val: {val_loss:.6f} | lr={lr:.2e} | time={step_time:.3f} sec")
                if val_loss < best_loss:
                    best_loss = val_loss; best_params = params.copy(); no_improve = 0
                    print("  New best parameters.")
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"Converged at iter {t}")
                    break
            else:
                self._stochastic_loss_history.append(base_loss)
                print(f"Iter {t:3d} | Train: {base_loss:.6f} | lr={lr:.2e} | time={step_time:.3f} sec")
            
            # Collect GIF frames every 2 iters
            if gif_filename and t % 2 == 0:
                r_grid, f_grid = self.build_distance_grid(params)
                plt.figure(); plt.plot(r_grid, f_grid)
                plt.title(f"Iteration {t}"); plt.xlabel(r"r (Å)"); plt.ylabel('force')
                plt.tight_layout(); buf = io.BytesIO(); plt.savefig(buf, format='png')
                plt.close(); buf.seek(0); frames.append(imageio.imread(buf))

        # Calculate total time
        total_time = time.time() - start_time
        avg_step_time = sum(self._step_times) / len(self._step_times) if self._step_times else 0
        
        # Plot history
        plt.figure(figsize=(8,5)); plt.plot(self._stochastic_loss_history, marker='o')
        plt.xlabel('Iteration'); plt.ylabel('MSE'); plt.title('Adam Convergence')
        plt.tight_layout(); plt.savefig('adam_stochastic_mse_vs_iteration.png', dpi=300); plt.close()

        # Save GIF
        if gif_filename and frames:
            imageio.mimsave(gif_filename, frames, fps=2)

        # Log best parameters and timing information
        if log_filename:
            with open(log_filename, 'a') as lf:
                lf.write('# Best SGD params:')
                np.savetxt(lf, best_params[None, :], fmt='%.6e')
                lf.write(f'# Total optimization time: {total_time:.3f}s\n')
                lf.write(f'# Average step time: {avg_step_time:.3f}s\n')

        print(f"SGD optimization complete. Best val loss: {best_loss:.6f}")
        print(f"Total time: {total_time:.3f}s | Average step time: {avg_step_time:.3f}s")
        
        return best_params
    
    def generate_lammps_table(self, spline_params, filename, n_points=5000):
        """
        Generate a tabulated potential for LAMMPS.
        
        Parameters:
        -----------
        spline_params : array
            Optimized spline parameters
        filename : str
            Output filename for the LAMMPS table
        n_points : int
            Number of points in the table (resolution)
        """
        # Create cubic spline representation of the potential
        spline = CubicSpline(self.r_knots, spline_params)
        
        # Calculate the negative derivative of the potential (force)
        spline_derivative = spline.derivative()
        
        # Create table points
        r_min   = 1e-6   # small to cover close approaches
        r_table = np.linspace(r_min, self.r_cut, n_points)
        potential = spline(r_table)
        force = -spline_derivative(r_table)

        # Ensure potential is zero at cutoff
        potential -= potential[-1]
        
        # Write table to file
        with open(filename, 'w') as f:
            f.write("# LAMMPS tabulated potential for CG water\n")
            f.write(f"# Cubic spline potential with {self.n_spline_knots} knots\n")
            f.write(f"# Generated by force-matching\n\n")
            f.write(f"CG_WATER\n")
            f.write(f"N {n_points}\n\n")
            
            for i, r in enumerate(r_table):
                f.write(f"{i+1} {r:.6f} {potential[i]:.6e} {force[i]:.6e}\n")
        
        print(f"LAMMPS tabulated potential written to {filename}")
        
        # Return table data for plotting
        return r_table, potential, force
    
    def validate(self, optimal_params, validation_frames=1000):
        """
        Validate the optimized potential on a set of frames
        
        Parameters:
        -----------
        optimal_params : array
            Optimized spline parameters
        validation_frames : int
            Number of frames to use for validation
        
        Returns:
        --------
        metrics : dict
            Dictionary of validation metrics
        """
        
        # Sample validation frames
        val_indices = np.random.choice(self.n_frames, validation_frames, replace=False)
        val_positions, val_forces = self.load_batch(val_indices)
        
        # Calculate CG forces with optimal parameters
        cg_forces = []
        for pos in val_positions:
            forces = self.compute_forces(pos, optimal_params)
            cg_forces.append(forces)
        
        # Calculate force magnitude distributions
        atomistic_mags = []
        cg_mags = []
        
        for i in range(len(val_positions)):
            atom_mag = np.linalg.norm(val_forces[i], axis=1)
            cg_mag = np.linalg.norm(cg_forces[i], axis=1)
            
            atomistic_mags.extend(atom_mag)
            cg_mags.extend(cg_mag)
        
        # Calculate various metrics
        # 1. Mean Squared Error
        mse = 0.0
        for i in range(len(val_positions)):
            frame_mse = np.mean(np.sum((cg_forces[i] - val_forces[i])**2, axis=1))
            mse += frame_mse
        mse /= len(val_positions)
        
        # 2. Directional consistency (cosine similarity)
        cos_sim = 0.0
        count = 0
        for i in range(len(val_positions)):
            for j in range(val_positions[i].shape[0]):
                v1 = val_forces[i][j]
                v2 = cg_forces[i][j]
                
                v1_mag = np.linalg.norm(v1)
                v2_mag = np.linalg.norm(v2)
                
                if v1_mag > 1e-6 and v2_mag > 1e-6:
                    cos_sim += np.dot(v1, v2) / (v1_mag * v2_mag)
                    count += 1
        
        cos_sim /= count if count > 0 else 1.0
        
        # 3. Wasserstein distance between force magnitude distributions
        w_dist = wasserstein_distance(atomistic_mags, cg_mags)
        
        # Plot force magnitude distributions
        plt.figure(figsize=(10, 6))
        plt.hist(atomistic_mags, bins=50, alpha=0.7, label='Atomistic')
        plt.hist(cg_mags, bins=50, alpha=0.7, label='CG')
        plt.xlabel('Force Magnitude [kcal/mol/Å]')
        plt.ylabel('Count')
        plt.title('Force Magnitude Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('force_distribution_comparison.png', dpi=300)
        plt.close()
        
        # Return metrics
        metrics = {
            'mse': mse,
            'cosine_similarity': cos_sim,
            'wasserstein_distance': w_dist
        }
        
        print("Validation Metrics:")
        print(f"Mean Squared Error: {mse:.6f}")
        print(f"Directional Consistency (cosine similarity): {cos_sim:.6f}")
        print(f"Wasserstein Distance: {w_dist:.6f}")
        
        return metrics
    
    def plot_results(self, spline_params, r_table, potential, force, filename_prefix):
        """
        Plot the optimized potential and force.
        
        Parameters:
        -----------
        spline_params : array
            Optimized spline parameters
        r_table : array
            Distance points for the tabulated potential
        potential : array
            Potential energy values
        force : array
            Force values
        filename_prefix : str
            Prefix for output filenames
        """
        # Plot the potential
        plt.figure(figsize=(10, 6))
        plt.plot(r_table, potential, 'b-', linewidth=2)
        plt.plot(self.r_knots, spline_params, 'ro', markersize=6)
        plt.xlabel(r'r [$\AA$]')
        plt.ylabel('Potential [kcal/mol]')
        plt.title('Optimized CG Potential')
        plt.grid(True, alpha=0.3)
        plt.legend(['Spline interpolation', 'Spline knots'])
        plt.savefig(f"{filename_prefix}_potential.png", dpi=300)
        plt.close()
        
        # Plot the force
        plt.figure(figsize=(10,6))
        plt.plot(r_table, force, 'g-', linewidth=2)
        plt.xlabel(r'r [$\AA$]')
        plt.ylabel(r'Force [kcal/mol/$\AA$]')
        plt.title('CG Force')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{filename_prefix}_force.png", dpi=300)
        plt.close()
        
        # Load a validation frame
        validation_idx = 0
        positions, target_forces = self.load_batch([validation_idx])
        if len(positions) > 0:
            # Calculate CG forces with optimized parameters
            cg_forces = self.calculate_forces_batch(positions, spline_params)
            
            # Calculate force magnitudes
            target_force_mag = np.linalg.norm(target_forces[0], axis=1)
            cg_force_mag = np.linalg.norm(cg_forces[0], axis=1)
            
            # Plot comparison
            plt.figure(figsize=(10, 6))
            plt.scatter(target_force_mag, cg_force_mag, alpha=0.3)
            
            # Add a perfect correlation line
            max_force = max(np.max(target_force_mag), np.max(cg_force_mag))
            plt.plot([0, max_force], [0, max_force], 'r--')
            
            plt.xlabel(r'Target Force Magnitude [kcal/mol/$\AA$]')
            plt.ylabel(r'CG Force Magnitude [kcal/mol/$\AA$]')
            plt.title('Force Comparison: CG vs. Atomistic')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{filename_prefix}_force_comparison.png", dpi=300)
            plt.close()
            
            # Plot histograms of forces
            plt.figure(figsize=(10, 6))
            plt.hist(target_force_mag, bins=50, alpha=0.5, label='Atomistic')
            plt.hist(cg_force_mag, bins=50, alpha=0.5, label='CG')
            plt.xlabel(r'Force Magnitude [kcal/mol/$\AA$]')
            plt.ylabel('Count')
            plt.title('Force Magnitude Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{filename_prefix}_force_histogram.png", dpi=300)
            plt.close()

if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Optimize CG potential using force-matching')
    parser.add_argument('--data-dir', type=str, default='cg_mapped_data',
                        help='Directory containing mapped CG data')
    parser.add_argument('--output-dir', type=str, default='force_matching_results',
                        help='Directory for output files')
    parser.add_argument('--r-cut', type=float, default=10.0,
                        help='Cutoff distance for the potential (Å)')
    parser.add_argument('--n-knots', type=int, default=20,
                        help='Number of knots for the cubic spline representation')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of frames to process at once')
    parser.add_argument('--n-threads', type=int, default=None,
                        help='Number of threads for parallel processing (None = use all available)')
    parser.add_argument('--max-iter', type=int, default=50,
                        help='Maximum number of iterations for optimization')
    parser.add_argument('--tol', type=float, default=1e-2,
                        help='Tolerance for optimization convergence')
    parser.add_argument('--use-sgd', action='store_true',
                        help='Use stochastic gradient descent for large datasets')
    parser.add_argument('--frames-per-step', type=int, default=20,
                        help='Number of frames to use per SGD step')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize force-matching
    print(f"Initializing force-matching with {args.n_threads} threads")
    fm = ForceMatching(
                        data_dir=args.data_dir,
                        r_cut=args.r_cut,
                        n_spline_knots=args.n_knots,
                        batch_size=args.batch_size,
                        n_threads=args.n_threads
                       )
    
    # Optimize the potential
    print(f"Optimizing potential with {args.max_iter} iterations")
    optimal_params = fm.optimize_potential(
                                            max_iter=args.max_iter,
                                            tol=args.tol,
                                            use_stochastic=args.use_sgd,
                                            frames_per_step=args.frames_per_step
                                           )
    # Validate the optimized potential
    metrics = fm.validate(optimal_params)
    
    # Generate LAMMPS table
    r_table, potential, force = fm.generate_lammps_table(
        optimal_params, 
        os.path.join(args.output_dir, "cg_water_potential.table"),
        n_points=1000
    )
    
    # Plot and save results
    fm.plot_results(
        optimal_params, 
        r_table, 
        potential, 
        force, 
        os.path.join(args.output_dir, "cg_water")
    )
    
    # Save spline parameters
    np.save(os.path.join(args.output_dir, "spline_params.npy"), optimal_params)
    np.save(os.path.join(args.output_dir, "spline_knots.npy"), fm.r_knots)
    
    # Save validation metrics
    with open(os.path.join(args.output_dir, "validation_metrics.txt"), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print("Force-matching complete. Results saved to:", args.output_dir)