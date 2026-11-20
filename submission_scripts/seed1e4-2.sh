#!/bin/bash

#SBATCH --job-name=seed1e4-2         # Job name
#SBATCH --partition=core96
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=96            # Tasks per node 
#SBATCH --time=7-00:00:00               # Walltime
#SBATCH --output=seed1e4-2.%j.out    # Standard output

set -e    # stop if any stage fails


#
# Parameters
#
min_mass=1e6
box_size=4e4  # unit: ckpc/h
subhalo_selection='aesopica'  # fable (Buttigieg et al. 2025), aesopica (Koudmani et al. 2026)
simulation_suite='aesopica'   # aesopica, fable: For identifying the final snapshot and BH-related parameters (boost factor and Eddington ratio)
simulation_name='seed1e4_se_boost' # seed1e2_se_boost, seed1e3_se_boostmax, seed1e4_se_boost
flags='-v'    # -v, --verbose: Verbose mode
single_output=true    # true: single-file outputs (snap_XXX.hdf5), run make_links first

# Stages to run (true/false)
run_offsets=false       # This only needs to be run once per simulation (regardless of min_mass)
run_mass_file=false     # This only needs to be run once per simulation (regardless of min_mass)
run_merger_files=true
run_delay=true
run_eddington=true


# paths
data_release=/beegfs/general/skoudmani/aesopica_proj/data_release
data_release_link=/home/pctung/beegfs_pctung/bh_merger_project/data_release_link  # Modify this for single file outputs!!

raw_dir=$data_release/$simulation_name
if [ "$single_output" = true ]; then
    sim_dir=$data_release_link/$simulation_name
else
    sim_dir=$raw_dir
fi
output_dir=/home/pctung/beegfs_pctung/bh_merger_project/output/$simulation_name



details_dir=$raw_dir/blackhole_details
merger_dir=$raw_dir/blackhole_mergers
expansion_list=$sim_dir/../ExpansionList_new
tree_dir=$sim_dir/sublink_galaxies
offset_dir=$sim_dir/sublink_galaxies/offsets_illustris

mass_file=$output_dir/bh_masses.hdf5
output_dir_merger=$output_dir/new_merger_events_extended_$min_mass
output_dir_delay=$output_dir/delay_data_${subhalo_selection}_$min_mass
output_dir_accretion=$output_dir/accretion_details_${subhalo_selection}_$min_mass
output_dir_eddington=$output_dir/eddington_data_${subhalo_selection}_$min_mass

mkdir -p $output_dir



export MV2_USE_ALIGNED_ALLOC=1 # better efficiency

# 0. Symbolic links for single-file outputs (serial)
if [ "$single_output" = true ]; then
    python -u ../src/make_links.py \
        --src_root $data_release \
        --dst_root $data_release_link
fi

# 1. Illustris-style offset files (serial, needs the multi-file format from step 0)
if [ "$run_offsets" = true ]; then
    python -u ../src/generate_offsets.py \
        --sim_dir    $sim_dir \
        --tree_dir   $tree_dir \
        --offset_dir $offset_dir \
        --simulation_suite $simulation_suite
fi

# 2. BH mass file
if [ "$run_mass_file" = true ]; then
    mpirun -np $SLURM_NTASKS python -u -m mpi4py ../src/generate_mass_file.py \
        --details_dir $details_dir \
        --mass_file   $mass_file \
        $flags
fi

# 3. Merger event files
if [ "$run_merger_files" = true ]; then
    mpirun -np $SLURM_NTASKS python -u -m mpi4py ../src/generate_merger_files.py \
        --sim_dir           $raw_dir \
        --merger_dir        $merger_dir \
        --tree_dir          $raw_dir/sublink_galaxies \
        --expansion_list    $expansion_list \
        --mass_file         $mass_file \
        --output_dir_merger $output_dir_merger \
        --min_mass          $min_mass \
        $flags
fi

# 4. Macrophysical delays and accretion
if [ "$run_delay" = true ]; then
    mpirun -np $SLURM_NTASKS python -u -m mpi4py ../src/macrophysical_delays.py \
        --sim_dir              $sim_dir \
        --expansion_list       $expansion_list \
        --tree_dir             $tree_dir \
        --offset_dir           $offset_dir \
        --output_dir_merger    $output_dir_merger \
        --output_dir_delay     $output_dir_delay \
        --output_dir_accretion $output_dir_accretion \
        --box_size             $box_size \
        --subhalo_selection    $subhalo_selection \
        --simulation_suite     $simulation_suite \
        $flags
fi

# 5. Eddington ratios
if [ "$run_eddington" = true ]; then
    mpirun -np $SLURM_NTASKS python -u -m mpi4py ../src/eddington_rates.py \
        --sim_dir              $sim_dir \
        --expansion_list       $expansion_list \
        --offset_dir           $offset_dir \
        --output_dir_delay     $output_dir_delay \
        --output_dir_eddington $output_dir_eddington
fi