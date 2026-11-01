#!/bin/bash

#SBATCH --job-name=generate_mass_file           # Job name
#SBATCH --partition=core64
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=64          # Tasks per node 
#SBATCH --time=01:00:00               # Walltime
#SBATCH --output=mass.%j.out    # Standard output


#
# Parameters
#

simulation_name='seed1e3_se_boostmax' # seed1e2_se_boost, seed1e3_se_boostmax, seed1e4_se_boost
flags='-v'    # -v, --verbose: Verbose mode


# paths
sim_dir=/beegfs/general/skoudmani/aesopica_proj/data_release/$simulation_name

details_dir=$sim_dir/blackhole_details
mass_file=$sim_dir/bh_masses.hdf5

# job setup
SBATCH_JOB_NAME=merger_$sim 
SBATCH_OUTPUT=merger_$sim.%j.out 
SBATCH_ERROR=merger_$sim.%j.err



export MV2_USE_ALIGNED_ALLOC=1 # better efficiency

mpirun -np $SLURM_NTASKS python -u -m mpi4py ../src/generate_mass_file.py \
    --details_dir $details_dir \
    --mass_file   $mass_file
    $flags

