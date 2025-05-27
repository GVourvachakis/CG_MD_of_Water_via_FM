#!/bin/bash
# Complete workflow for force-matching coarse-graining of water
# Run the entire pipeline from AA simulation to CG simulation

# Step 0: Create directories
mkdir -p logs

# Step 1: Create initial water configuration
echo "Creating initial water configuration..."
# Use packmol if available
packmol < water_box.inp
python xyz_to_lammps.py

# Step 2: Run atomistic simulation
echo "Running atomistic simulation with LAMMPS..."
OMP_NUM_THREADS=8 mpirun -np 8 --bind-to core --map-by core $CONDA_PREFIX/bin/lmp -in in.aa_simulation > logs/cg_simulation.log

# Step 3: Map atomistic to CG representation
echo "Mapping atomistic to CG representation..."
python cg_mapping.py > logs/cg_mapping.log

# Step 4: Run force-matching 
echo "Running force-matching algorithm..."
python force_matching.py --n-threads 8 --batch-size 20 --frames-per-step 100 --max-iter 30 > logs/force_matching.log

# Step 5: Run CG simulation
echo "Running CG simulation with LAMMPS..."
OMP_NUM_THREADS=8 mpirun -np 8 --bind-to core --map-by core $CONDA_PREFIX/bin/lmp -in in.cg_simulation > logs/cg_simulation.log

# Step 6: Analyze results
echo "Analyzing results..."
python analysis.py

echo "Pipeline complete. Check the logs directory for detailed output."