#!/bin/bash

#SBATCH --job-name=macrophysical_delays           # Job name
#SBATCH --partition=core256
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=256          # Tasks per node 
#SBATCH --time=1-00:00:00               # Walltime
#SBATCH --output=delay.%j.out    # Standard output


#
# Parameters
#

simulation_name='seed1e2_se_boost' # seed1e2_se_boost, seed1e3_se_boostmax, seed1e4_se_boost
flags='-v'    # -v, --verbose: Verbose mode


# paths
# sim_dir=/beegfs/general/skoudmani/aesopica_proj/data_release/$simulation_name
sim_dir=/home/pctung/beegfs_pctung/bh_merger_project/data_release_link/$simulation_name
output_dir=/home/pctung/beegfs_pctung/bh_merger_project/output/$simulation_name

expansion_list=$sim_dir/../ExpansionList_new
tree_dir=$sim_dir/sublink_galaxies
offset_dir=$sim_dir/sublink_galaxies/offsets_illustris

output_dir_merger=$output_dir/new_merger_events_extended
output_dir_delay=$output_dir/delay_data
output_dir_accretion=$output_dir/accretion_details



export MV2_USE_ALIGNED_ALLOC=1 # better efficiency

mpirun -np $SLURM_NTASKS python -u -m mpi4py ../src/macrophysical_delays.py \
    --sim_dir              $sim_dir \
    --expansion_list       $expansion_list \
    --tree_dir             $tree_dir \
    --offset_dir           $offset_dir \
    --output_dir_merger    $output_dir_merger \
    --output_dir_delay     $output_dir_delay \
    --output_dir_accretion $output_dir_accretion \
    $flags