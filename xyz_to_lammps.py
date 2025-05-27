"""
# To create the initial water box, you can use Packmol
# Below is a Packmol input file (save as water_box.inp)

# Packmol input file to create a box of 1000 water molecules
tolerance 2.0
filetype xyz
output water_initial.xyz

structure water.xyz
  number 1000
  inside box 0. 0. 0. 40. 40. 40.
end structure

# water.xyz file content:
3
water molecule
O     0.000   0.000   0.000
H     0.957   0.000   0.000
H     0.240   0.927   0.000

# After running Packmol, convert the XYZ to LAMMPS data file
# You can use a script like TopoTools in VMD or other conversion tools
"""
# Convert the XYZ to LAMMPS data format

# Run the Packmol command:
# packmol < water_box.inp

# Then run the conversion script:
# python xyz_to_lammps.py

def xyz_to_lammps_data(xyz_file, data_file):
    # Read XYZ file
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    # Parse number of atoms
    num_atoms = int(lines[0].strip())
    n_molecules = num_atoms // 3
    
    # Create box dimensions
    box_size = 40.0  # Assuming 40x40x40 Å box
    
    # Create LAMMPS data file
    with open(data_file, 'w') as f:
        f.write('LAMMPS data file for TIP3P water\n\n')
        f.write(f'{num_atoms} atoms\n')
        f.write(f'{n_molecules * 2} bonds\n')
        f.write(f'{n_molecules} angles\n\n')
        
        f.write('2 atom types\n')
        f.write('1 bond types\n')
        f.write('1 angle types\n\n')
        
        f.write(f'0.0 {box_size} xlo xhi\n')
        f.write(f'0.0 {box_size} ylo yhi\n')
        f.write(f'0.0 {box_size} zlo zhi\n\n')
        
        f.write('Masses\n\n')
        f.write('1 15.9994   # O\n')
        f.write('2 1.008     # H\n\n')
        
        f.write('Pair Coeffs\n\n')
        f.write('1 0.102 3.188   # O\n')
        f.write('2 0.000 0.000   # H\n\n')
        
        f.write('Bond Coeffs\n\n')
        f.write('1 450.0 0.9572  # O-H\n\n')
        
        f.write('Angle Coeffs\n\n')
        f.write('1 55.0 104.52   # H-O-H\n\n')
        
        # Atoms section
        f.write('Atoms\n\n')
        
        atom_id = 1
        mol_id = 1
        atom_data = []
        
        for i in range(2, 2 + num_atoms):
            if (i - 2) % 3 == 0:  # Oxygen
                atom_type = 1
                charge = -0.834
            else:  # Hydrogen
                atom_type = 2
                charge = 0.417
            
            parts = lines[i].strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            atom_data.append((atom_id, mol_id, atom_type, charge, x, y, z))
            
            atom_id += 1
            if atom_id % 3 == 1:
                mol_id += 1
        
        for atom in atom_data:
            atom_id, mol_id, atom_type, charge, x, y, z = atom
            f.write(f'{atom_id} {mol_id} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}\n')
        
        # Bonds section
        f.write('\nBonds\n\n')
        
        bond_id = 1
        for i in range(1, num_atoms + 1, 3):
            # O-H bonds
            f.write(f'{bond_id} 1 {i} {i+1}\n')
            bond_id += 1
            f.write(f'{bond_id} 1 {i} {i+2}\n')
            bond_id += 1
        
        # Angles section
        f.write('\nAngles\n\n')
        
        angle_id = 1
        for i in range(1, num_atoms + 1, 3):
            # H-O-H angle
            f.write(f'{angle_id} 1 {i+1} {i} {i+2}\n')
            angle_id += 1

if __name__ == "__main__": 
    xyz_to_lammps_data('water_initial.xyz', 'initial_water.data')
