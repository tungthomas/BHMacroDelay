#!/bin/bash

#SBATCH --job-name=merger2           # Job name
#SBATCH --partition=core64
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=64           # Tasks per node 
#SBATCH --time=1-00:00:00               # Walltime
#SBATCH --output=merger.%j.out    # Standard output


#
# Parameters
#
min_mass=1e5
simulation_name='seed1e4_se_boost' # seed1e2_se_boost, seed1e3_se_boostmax, seed1e4_se_boost
flags='-v'    # -v, --verbose: Verbose mode


# paths
sim_dir=/beegfs/general/skoudmani/aesopica_proj/data_release/$simulation_name
output_dir=/home/pctung/beegfs_pctung/bh_merger_project/output/$simulation_name

merger_dir=$sim_dir/blackhole_mergers
tree_dir=$sim_dir/sublink_galaxies
expansion_list=$sim_dir/../ExpansionList_new
mass_file=$output_dir/bh_masses.hdf5
output_dir_merger=$output_dir/new_merger_events_extended_$min_mass




export MV2_USE_ALIGNED_ALLOC=1 # better efficiency

mpirun -np $SLURM_NTASKS python -u -m mpi4py ../src/generate_merger_files.py \
    --sim_dir        $sim_dir \
    --merger_dir     $merger_dir \
    --tree_dir       $tree_dir \
    --expansion_list $expansion_list \
    --mass_file      $mass_file \
    --output_dir_merger     $output_dir_merger \
    --min_mass       $min_mass \
    $flags