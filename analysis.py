# The analysis will help you assess how well the CG model reproduces 
# the structural properties of the atomistic system.

#!/usr/bin/env python3
"""
Analysis and Validation of Coarse-Grained Water Model

1. Compare RDFs between atomistic and CG simulations
2. Analyze structural properties including Voronoi tessellation
3. Validate the CG model through structural analysis
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def parse_lammps_rdf(filename):
    """Parse RDF data from LAMMPS output
    
    Columns of cg_rdf.dat file:
    ---------------------------
    c_rdf[1]: The center of the radial distance bin (in distance units, such as Ångströms).

    c_rdf[2]: The radial distribution function g(r) value for that bin. 
              This function describes how particle density varies as a function of distance 
              from a reference particle.

    c_rdf[3]: The coordination number coord(r), representing the average number of particles 
              within a sphere of radius r around a reference particle.
    """
    r_vals = []
    g_vals = []
    coord_vals = []
    block_data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # Block header: two columns (timestep and nrows)
            if len(parts) == 2:
                # start of new block: reset block_data
                block_data = []
                continue
            # Data line: index, r, g(r), coordination
            if len(parts) >= 4:
                _, ri, gi, ci = parts[:4]
                block_data.append((float(ri), float(gi), float(ci)))
        # After loop, block_data holds last block
    if not block_data: raise ValueError(f"No data found in RDF file: {filename}")
    r_vals, g_vals, coord_vals = zip(*block_data)
    return np.array(r_vals), np.array(g_vals), np.array(coord_vals)

def load_aa_rdf(directory, frame=0):
    """Load atomistic RDF from saved numpy files"""
    rdf_file = os.path.join(directory, f"rdf_{frame}.npy")
    if os.path.exists(rdf_file):
        data = np.load(rdf_file)
        return data[:, 0], data[:, 1]
    else:
        print(f"RDF file {rdf_file} not found")
        return None, None

def plot_rdf_comparison(aa_r, aa_g_r, cg_r, cg_g_r, output_file):
    """Plot comparison of atomistic and CG RDFs"""
    plt.figure(figsize=(10, 6))
    
    if aa_r is not None and aa_g_r is not None:
        plt.plot(aa_r, aa_g_r, 'b-', linewidth=2, label='Atomistic')
    
    plt.plot(cg_r, cg_g_r, 'r--', linewidth=2, label='Coarse-Grained')
    
    plt.xlabel(r'r [$\AA$]')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_rdf_and_coordination(r, g_r, coord, output_file):
    """Plot RDF and coordination number"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel(r'r [$\AA$]')
    ax1.set_ylabel('g(r)', color=color)
    ax1.plot(r, g_r, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    if coord is not None:
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Coordination Number', color=color)
        ax2.plot(r, coord, color=color, linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('RDF and Coordination Number')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_cg_potential(output_dir):
    """Plot the optimized CG potential and force"""
    # Load spline parameters
    spline_params = np.load(os.path.join(output_dir, "spline_params.npy"))
    spline_knots = np.load(os.path.join(output_dir, "spline_knots.npy"))
    
    # Load table data
    table_file = os.path.join(output_dir, "cg_water_potential.table")
    if os.path.exists(table_file):
        # Skip header lines
        with open(table_file, 'r') as f:
            lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith("1 "):
                    data_start = i
                    break
        
        # Parse data
        data = np.loadtxt(table_file, skiprows=data_start)
        r = data[:, 1]
        potential = data[:, 2]
        force = data[:, 3]
        
        # Plot potential
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(r, potential, 'b-', linewidth=2)
        plt.plot(spline_knots, spline_params, 'ro', markersize=6)
        plt.xlabel(r'r [$\AA$]')
        plt.ylabel('Potential [kcal/mol]')
        plt.title('Optimized CG Potential')
        plt.grid(True, alpha=0.3)
        plt.legend(['Spline interpolation', 'Spline knots'])
        
        # Plot force
        plt.subplot(2, 1, 2)
        plt.plot(r, force, 'g-', linewidth=2)
        plt.xlabel(r'r [$\AA$]')
        plt.ylabel(r'Force [kcal/mol/$\AA$]')
        plt.title('CG Force')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cg_potential_and_force.png"), dpi=300)
        plt.close()

def parse_voronoi_data(filename):
    """Parse Voronoi data from LAMMPS output"""
    # Data columns: id, type, x, y, z, voro_vol, voro_neighbors
    data = np.loadtxt(filename, skiprows=9)
    return data

def plot_voronoi_volume_distribution(files_pattern, output_file):
    """Plot histogram of Voronoi volumes"""
    files = sorted(glob.glob(files_pattern))
    if not files:
        print(f"No Voronoi data files found matching pattern: {files_pattern}")
        return

    all_volumes = []
    for f in files:
        data = parse_voronoi_data(f)
        volumes = data[:, 5]  # voro_vol
        all_volumes.extend(volumes)
    all_volumes = np.array(all_volumes)

    plt.figure(figsize=(10, 6))
    plt.hist(all_volumes, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel(r'Voronoi Cell Volume [$\AA^3$]')
    plt.ylabel('Frequency')
    plt.title('Distribution of Voronoi Cell Volumes')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300)
    plt.close()

    return {
        'mean': np.mean(all_volumes),
        'median': np.median(all_volumes),
        'std': np.std(all_volumes),
        'min': np.min(all_volumes),
        'max': np.max(all_volumes)
    }


def plot_voronoi_neighbors_distribution(files_pattern, output_file):
    """Plot histogram of number of Voronoi neighbors"""
    files = sorted(glob.glob(files_pattern))
    if not files:
        print(f"No Voronoi data files found matching pattern: {files_pattern}")
        return

    all_neighbors = []
    for f in files:
        data = parse_voronoi_data(f)
        neighbors = data[:, 6].astype(int)  # voro_neighbors
        all_neighbors.extend(neighbors)
    all_neighbors = np.array(all_neighbors, dtype=int)

    plt.figure(figsize=(10, 6))
    bins = np.arange(all_neighbors.min()-0.5, all_neighbors.max()+1.5, 1)
    plt.hist(all_neighbors, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Voronoi Neighbors')
    plt.ylabel('Frequency')
    plt.title('Distribution of Voronoi Neighbor Counts')
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(all_neighbors.min(), all_neighbors.max()+1, 1))
    plt.savefig(output_file, dpi=300)
    plt.close()

    return {
        'mean': np.mean(all_neighbors),
        'median': np.median(all_neighbors),
        'std': np.std(all_neighbors),
        'min': np.min(all_neighbors),
        'max': np.max(all_neighbors)
    }

def plot_volume_vs_neighbors(voronoi_dir, output_file, n_frames=3, bins_neighbors=None, bins_volume=50):
    """
    Plot a 2D histogram (heatmap) of Voronoi cell volume vs. neighbor count.

    Parameters
    ----------
    voronoi_dir : str
        Directory containing per-frame Voronoi dumps named 'voronoi_*.data'.
    output_file : str
        Path to save the resulting PNG.
    n_frames : int, optional
        Number of most-recent frames to include (default=3).
    bins_neighbors : array-like or int, optional
        Bin specification for neighbor counts. If None, inferred from data.
    bins_volume : int, optional
        Number of bins along the volume axis (default=50).
    """
    # 1. Gather files
    pattern = os.path.join(voronoi_dir, "voronoi_*.data")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Voronoi data files found: {pattern}")

    # 2. Read last n_frames
    vols = []
    neigh = []
    for f in files[-n_frames:]:
        data = np.loadtxt(f, skiprows=9)
        vols.append(data[:, 5])
        neigh.append(data[:, 6].astype(int))
    vols = np.concatenate(vols)
    neigh = np.concatenate(neigh)

    # 3. Determine neighbor bins if not provided
    if bins_neighbors is None:
        n_min, n_max = neigh.min(), neigh.max()
        bins_neighbors = np.arange(n_min - 0.5, n_max + 1.5, 1)

    # 4. Build 2D histogram
    h, xedges, yedges = np.histogram2d(
        neigh,
        vols,
        bins=[bins_neighbors, bins_volume]
    )
    h = h.T  # transpose for correct orientation

    # 5. Plot
    plt.figure(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list('WhiteToBlue', ['white', 'steelblue'])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    img = plt.imshow(
        h,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap=cmap,
        interpolation='nearest'
    )
    plt.xlabel('Number of Voronoi Neighbors')
    plt.ylabel(r'Voronoi Cell Volume [$\mathrm{\AA}^3$]')
    plt.title('Volume vs. Neighbors')
    cbar = plt.colorbar(img)
    cbar.set_label('Frequency')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()


def analyze_time_evolution(voronoi_dir, output_file):
    """Analyze time evolution of Voronoi properties"""
    global_stats_file = os.path.join(voronoi_dir, "voronoi_global_stats.dat")
    if not os.path.exists(global_stats_file):
        print(f"Global stats file not found: {global_stats_file}")
        return

    data = np.loadtxt(global_stats_file, skiprows=1)
    # Columns: step, vol_avg, vol_min, vol_max, neigh_avg, neigh_min, neigh_max
    steps = data[:, 0]
    time_ns = steps * 2e-6

    vol_mean = data[:, 1]
    vol_min  = data[:, 2]
    vol_max  = data[:, 3]

    neigh_mean = data[:, 4]
    neigh_min  = data[:, 5]
    neigh_max  = data[:, 6]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    ax1.plot(time_ns, vol_mean, linewidth=2, label='Mean Volume')
    ax1.fill_between(time_ns, vol_min, vol_max, alpha=0.2)
    ax1.set_ylabel(r'Voronoi Cell Volume [$\AA^3$]')
    ax1.set_title('Time Evolution of Voronoi Volume')
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_ns, neigh_mean, linewidth=2, label='Mean Neighbors')
    ax2.fill_between(time_ns, neigh_min, neigh_max, alpha=0.2)
    ax2.set_xlabel('Time [ns]')
    ax2.set_ylabel('Number of Neighbors')
    ax2.set_title('Time Evolution of Voronoi Neighbors')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def main():
    # Create analysis directory
    analysis_dir = "analysis_lr_1e-2"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load CG RDF from LAMMPS output
    cg_rdf_file = "cg_rdf.dat"
    if os.path.exists(cg_rdf_file):
        cg_r, cg_g_r, cg_coord = parse_lammps_rdf(cg_rdf_file)
        
        # Load atomistic RDF
        aa_r, aa_g_r = load_aa_rdf("cg_mapped_data_new")
        
        # Plot RDF comparison
        plot_rdf_comparison(
            aa_r, aa_g_r, 
            cg_r, cg_g_r, 
            os.path.join(analysis_dir, "rdf_comparison.png")
        )
        
        # Plot RDF and coordination number
        if cg_coord is not None:
            plot_rdf_and_coordination(
                cg_r, cg_g_r, cg_coord,
                os.path.join(analysis_dir, "rdf_and_coordination.png")
            )
    else:
        print(f"CG RDF file {cg_rdf_file} not found")
    
    # Plot CG potential
    plot_cg_potential("force_matching_results")
    
    # Voronoi analysis directory
    voro_dir = "voronoi_cg_data"
    if os.path.exists(voro_dir):
        plot_voronoi_volume_distribution(
            os.path.join(voro_dir, "voronoi_*.data"),
            os.path.join(analysis_dir, "voro_volume_dist.png")
        )
        plot_voronoi_neighbors_distribution(
            os.path.join(voro_dir, "voronoi_*.data"),
            os.path.join(analysis_dir, "voro_neighbors_dist.png")
        )
        plot_volume_vs_neighbors(
            voronoi_dir="voronoi_cg_data",
            output_file=os.path.join(analysis_dir, "volume_vs_neighbors.png"),
            n_frames=100
        )
        analyze_time_evolution(
            voro_dir,
            os.path.join(analysis_dir, "voro_time_evolution.png")
        )
    else:
        print(f"Voronoi data directory {voro_dir} not found")
    
    # Additional analysis
    print("\nAnalysis complete. Results saved to:", analysis_dir)

if __name__ == "__main__":
    main()