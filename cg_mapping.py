#!/usr/bin/env python3
"""
CG Mapping Script for Water Molecules

1. Read LAMMPS trajectory and force files
2. Map atomistic coordinates to CG sites (COM of water molecules)
3. Map atomistic forces to CG sites
4. Save mapped CG coordinates and forces for force-matching
5. Calculate radial distribution functions (RDFs) for structural comparison
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

#KD-Trees for nearest neighbor searches (O(N_o log(N_h)) lookup time), 
#which is much faster than the brute force approach (O(N_h x N_o)).
from scipy.spatial import cKDTree # Improved Performance: use a neighbor search

def read_lammps_custom(filename):
    """Read LAMMPS custom dump file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find ITEM: ATOMS line to determine column indices
    for i, line in enumerate(lines):
        if "ITEM: ATOMS" in line:
            header = line.strip().split()[2:]  # Get column names
            data_start = i + 1
            break
    
    # Parse data
    data = np.array([line.strip().split() for line in lines[data_start:]], dtype=float)
    
    # Create a dictionary with the data
    result = {}
    for i, col in enumerate(header):
        result[col] = data[:, i]
    
    return result, header

def _tile_positions(pos, box_size):
    """
    Tile positions into the 3x3x3 neighborhood (including original cell).
    Returns:
      tiled_pos: (n*27,3) array of positions
      tiled_idx: (n*27,) array mapping each tiled row back to 0..n-1
    """
    shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                                for j in (-1, 0, 1)
                                for k in (-1, 0, 1)])
    n = pos.shape[0]
    # repeat each atom 27 times with all shifts
    tiled_pos = (pos[:, None, :] + shifts[None, :, :] * box_size[None, None, :]).reshape(-1, 3)
    tiled_idx = np.repeat(np.arange(n), 27)
    return tiled_pos, tiled_idx

def group_water_molecules(atom_ids, atom_types, positions, box_size=None):
    """
    Group atoms into water molecules based on geometry:
      - O-H distance within [min_oh_dist, max_oh_dist]
      - H-H distance >= min_hh_dist
      - H-O-H angle within [min_angle, max_angle] (optional)
    Handles periodic boundary conditions (PBC) via minimum image.

    Returns list of dicts with keys 'O', 'H1', 'H2'.
    """

    # Constants for water geometry
    min_oh_dist = 0.8  # Minimum O-H distance (Å)
    max_oh_dist = 1.2  # Maximum O-H distance (Å)
    min_hh_dist = 1.4  # Minimum H-H distance (Å)
    # (optional) angle cutoffs:  [min_angle, max_angle] in degrees
    # min_angle, max_angle = 90, 120

    # Identify O and H indices
    O_idx = [i for i, t in enumerate(atom_types) if t == 1]
    H_idx = [i for i, t in enumerate(atom_types) if t == 2]
    
    if not O_idx or len(H_idx) < 2:
        raise ValueError("Not enough O or H atoms for water grouping.")

    if len(O_idx) == 0 or len(H_idx) < 2:
        print("Error: Not enough oxygen or hydrogen atoms found.")
        return []
    
    # Tile H positions to correctly handle bonds across PBC
    H_pos = positions[H_idx]
    tiled_H_pos, tiled_map = _tile_positions(H_pos, box_size)
    
    # account for PBC by tiling images (or use specialized PBC-KDTree)
    # here we brute-force with minimum image in the distance callback
    tree = cKDTree(tiled_H_pos)
    
    # Group atoms into water molecules
    molecules = []
    used_H = set()
    
    for o in O_idx:
        o_pos = positions[o]
        # find H's within max_oh_dist
        dists, hs = tree.query(o_pos, k=len(tiled_H_pos), distance_upper_bound=max_oh_dist)
        # filter and map back to unique H indices
        candidates = []
        for h_idx, d in zip(hs, dists):
            if d == np.inf:
                break
            h_global_idx = H_idx[tiled_map[h_idx]]
            if atom_ids[h_global_idx] in used_H:
                continue
            if d < min_oh_dist:
                continue
            candidates.append((h_global_idx, d))
        # need at least 2 candidates
        if len(candidates) < 2:
            continue

        # try combinations to satisfy H-H distance
        found = False
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                h1, d1 = candidates[i]
                h2, d2 = candidates[j]
                # compute H-H separation with PBC
                delta = positions[h1] - positions[h2]
                # apply minimum image
                if box_size is not None:
                    delta -= box_size * np.round(delta / box_size)
                hh_dist = np.linalg.norm(delta)
                if hh_dist < min_hh_dist:
                    continue
                # assign molecule
                molecules.append({'O': int(atom_ids[o]),
                                  'H1': int(atom_ids[h1]),
                                  'H2': int(atom_ids[h2])})
                used_H.update({int(atom_ids[h1]), int(atom_ids[h2])})
                found = True
                break
            if found:
                break
    return molecules

def map_to_cg(positions, atom_ids, atom_types, box_size=None):
    """Map atomistic water positions to CG sites (1 site per water molecule)"""
    # Group atoms into water molecules
    molecules = group_water_molecules(atom_ids, atom_types, positions, box_size)
    n_molecules = len(molecules)
    
    # Create a mapping from atom ID to position index
    id_to_idx = {int(atom_ids[i]): i for i in range(len(atom_ids))}
    
    # Initialize CG positions array
    cg_positions = np.zeros((n_molecules, 3))
    
    # Water masses
    m_O = 15.9994
    m_H = 1.008
    total_mass = m_O + 2 * m_H
    
    # Map each water molecule to a CG site based on center of mass
    for i, mol in enumerate(molecules):
        o_idx = id_to_idx[mol['O']]
        h1_idx = id_to_idx[mol['H1']]
        h2_idx = id_to_idx[mol['H2']]
        
        # Extract positions
        o_pos = positions[o_idx].copy()
        h1_pos = positions[h1_idx].copy()
        h2_pos = positions[h2_idx].copy()
        
        # Handle periodic boundary conditions if box_size is provided
        if box_size is not None:
            for pos in [h1_pos, h2_pos]:
                for dim in range(3):
                    while pos[dim] - o_pos[dim] > box_size[dim]/2:
                        pos[dim] -= box_size[dim]
                    while pos[dim] - o_pos[dim] < -box_size[dim]/2:
                        pos[dim] += box_size[dim]
        
        # Calculate center of mass
        com = (m_O * o_pos + m_H * h1_pos + m_H * h2_pos) / total_mass
        cg_positions[i] = com
    
    return cg_positions, molecules

def map_forces_to_cg(forces, atom_ids, molecules):
    """Map atomistic forces to CG sites (sum of forces on all atoms in a molecule)"""
    n_molecules = len(molecules)
    
    # Create a mapping from atom ID to force index
    id_to_idx = {int(atom_ids[i]): i for i in range(len(atom_ids))}
    
    # Water masses for proper weighting
    m_O = 15.9994
    m_H = 1.008
    total_mass = m_O + 2 * m_H

    # Initialize CG forces array
    cg_forces = np.zeros((n_molecules, 3))

    # Map forces with mass weighting
    for i, mol in enumerate(molecules):
        o_idx = id_to_idx[mol['O']]
        h1_idx = id_to_idx[mol['H1']]
        h2_idx = id_to_idx[mol['H2']]
        
        # Apply mass weighting to preserve momentum
        cg_forces[i] = (m_O * forces[o_idx] + m_H * forces[h1_idx] + m_H * forces[h2_idx]) / total_mass
    
    return cg_forces

def calculate_rdf(positions, box_size, n_bins=100, r_max=10.0):
    """Calculate radial distribution function from CG positions"""
    n_molecules = len(positions)
    
    # Calculate all pairwise distances
    distances = np.zeros((n_molecules, n_molecules))
    
    # Apply minimum image convention for periodic boundaries
    for i in range(n_molecules):
        for j in range(i+1, n_molecules):
            diff = positions[i] - positions[j]
            
            # Apply minimum image convention
            for dim in range(3):
                while diff[dim] > box_size[dim]/2:
                    diff[dim] -= box_size[dim]
                while diff[dim] < -box_size[dim]/2:
                    diff[dim] += box_size[dim]
            
            # Calculate distance
            dist = np.linalg.norm(diff)
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Flatten the upper triangle of the distance matrix
    flat_distances = distances[np.triu_indices(n_molecules, k=1)]
    
    # Create histogram
    bins = np.linspace(0, r_max, n_bins+1)
    hist, bin_edges = np.histogram(flat_distances, bins=bins)
    
    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Calculate volume of each shell
    bin_volume = 4.0/3.0 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    
    # Calculate RDF
    rho = n_molecules / np.prod(box_size)  # Number density
    rdf = hist / (bin_volume * rho * n_molecules)
    
    return bin_centers, rdf

def process_trajectory_data(traj_dir, forces_dir, output_dir):
    """Process all trajectory and force files, map to CG and save results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all trajectory files
    traj_files = sorted(glob.glob(os.path.join(traj_dir, "water_*.lammpstrj")))
    force_files = sorted(glob.glob(os.path.join(forces_dir, "force_*.dump")))
    
    # Make sure we have matching trajectory and force files
    assert len(traj_files) == len(force_files), "Number of trajectory and force files don't match"
    
    print(f"Found {len(traj_files)} trajectory files to process")
    
    # Process each frame
    for frame_idx, (traj_file, force_file) in enumerate(zip(traj_files, force_files)):
        # Read files
        traj_data, traj_header = read_lammps_custom(traj_file)
        force_data, force_header = read_lammps_custom(force_file)
        
        # Get box size (from LAMMPS box bounds)
        box_size = np.array([
            np.max(traj_data['x']) - np.min(traj_data['x']),
            np.max(traj_data['y']) - np.min(traj_data['y']),
            np.max(traj_data['z']) - np.min(traj_data['z'])
        ])
        
        # Extract relevant data
        positions = np.column_stack([traj_data['x'], traj_data['y'], traj_data['z']])
        atom_ids = traj_data['id']
        atom_types = traj_data['type']
        forces = np.column_stack([force_data['fx'], force_data['fy'], force_data['fz']])
        
        # Ensure force and trajectory data are for the same atoms and in the same order
        force_ids = force_data['id']
        
        # Check if IDs match
        if not np.array_equal(atom_ids, force_ids):
            print(f"Warning: Atom IDs don't match in frame {frame_idx}. Reordering force data...")
            
            # Create a mapping from atom ID to force
            id_to_force = {int(force_ids[i]): forces[i] for i in range(len(force_ids))}
            
            # Reorder forces to match trajectory atom order
            forces = np.array([id_to_force[int(aid)] for aid in atom_ids])
        
        # Map to CG representation
        cg_positions, molecules = map_to_cg(positions, atom_ids, atom_types, box_size)
        cg_forces = map_forces_to_cg(forces, atom_ids, molecules)
        
        # Save mapped data
        np.save(os.path.join(output_dir, f"cg_positions_{frame_idx}.npy"), cg_positions)
        np.save(os.path.join(output_dir, f"cg_forces_{frame_idx}.npy"), cg_forces)
        
        # Calculate RDF for the first 10 frames (for visualization purposes)
        if frame_idx < 10:
            r, g_r = calculate_rdf(cg_positions, box_size)
            np.save(os.path.join(output_dir, f"rdf_{frame_idx}.npy"), np.column_stack([r, g_r]))
        
        # Report progress
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx+1}/{len(traj_files)}")
            if frame_idx == 0:
                print(f"Successfully identified {len(molecules)} water molecules")
                print(f"Box size: {box_size}")
            
    # Save box size
    np.save(os.path.join(output_dir, "box_size.npy"), box_size)
    
    print(f"Processed {len(traj_files)} frames. Mapped data saved to {output_dir}")

if __name__ == "__main__":
    # Directories
    traj_dir = "traj"
    forces_dir = "forces"
    output_dir = "cg_mapped_data_new"
    
    # Process all trajectory data
    process_trajectory_data(traj_dir, forces_dir, output_dir)
    
    # Plot RDF from first frame (for visualization)
    try:
        rdf_file = os.path.join(output_dir, "rdf_0.npy")
        if os.path.exists(rdf_file):
            r, g_r = np.load(rdf_file).T
            
            plt.figure(figsize=(10, 6))
            plt.plot(r, g_r)
            plt.xlabel(r'r [$\AA$]')
            plt.ylabel('g(r)')
            plt.title('Radial Distribution Function (CG Water)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, "rdf_plot.png"), dpi=300)
            plt.close()
            print(f"RDF plot saved to {os.path.join(output_dir, 'rdf_plot.png')}")
    except Exception as e:
        print(f"Error plotting RDF: {e}")
    
    print("CG mapping complete.")