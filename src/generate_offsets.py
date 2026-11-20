"""
This script generates the Illustris-style offset files for each snapshot, which are required by illustris_python
to load the particles of a given group/subhalo. The SubLink offsets of each snapshot are also copied into these files.
The output directory is used as offset_dir in macrophysical_delays.py and eddington_rates.py.

Note that sim_dir must be in the multi-file format (groups_XXX/fof_subhalo_tab_XXX.N.hdf5, snapdir_XXX/snap_XXX.N.hdf5).
For single-file outputs, run make_links.py first.

[Parameters] (Input through command line. Refer to initialize.py for more information.)
------
    (INPUT_NAME: PARAMETER_NAME)
    sim_dir: basepath
    tree_dir: sublink_path (tree_dir/offsets)
    offset_dir: output_path

----------------------------------------------------------------------------------------------------------------------------

Stephanie Buttigieg (sb2583@cam.ac.uk)

Modified by Pei-Cheng Tung (pt26aae@herts.ac.uk)
"""

import h5py
import numpy as np
import os
import re

from initialize import initialize, get_final_snapshot, parameters

def generate_offset_file(snap_number, basepath, output_path, sublink_path):
    """
    Generate the offset file for a given snapshot.

    Parameters:
    snap_number (int): The snapshot number for which to generate the offset file.
    basepath (str): The basepath to the FABLE simulation.
    output_path (str): The path to save the offset file.
    """
    # loop through all the group files
    group_path = os.path.join(basepath, f'groups_{snap_number:03d}')
    if simulation_suite == 'fable' and not os.path.exists(group_path):
        group_path = os.path.join(basepath, f'groups_{snap_number:03d}_fable')

    group_files = [f for f in os.listdir(group_path) if f.startswith('fof_subhalo_tab') and f.endswith('.hdf5')]
    def extract_number(filename):
        parts = filename.split('.')  # Split at dot
        return int(parts[1])  # Extract the number after the first dot
    # Sort files based on extracted number
    group_files_sorted = sorted(group_files, key=extract_number)
    n_group_files = len(group_files)
    group_offsets = np.zeros(n_group_files)
    subhalo_offsets = np.zeros(n_group_files)

    group_first_particle = [np.zeros(6, dtype=int)]
    subhalo_first_particle = [np.zeros(6, dtype=int)]

    all_group_len_type = []
    all_subhalo_len_type = []
    all_group_nsubs = []
    all_group_first_subs = []

    for i,group_file in enumerate(group_files_sorted):
        file_path = os.path.join(group_path, group_file)
        with h5py.File(file_path, 'r') as f:
            header = dict(f['Header'].attrs.items())
            n_groups = header['Ngroups_ThisFile']
            n_subgroups = header['Nsubgroups_ThisFile']
            total_n_groups = header['Ngroups_Total']
            total_n_subgroups = header['Nsubgroups_Total']
            if 'GroupLenType' in f['Group']:
                group_len_type = np.array(f['Group']['GroupLenType'])
                group_nsubs = np.array(f['Group']['GroupNsubs'])
                group_first_sub = np.array(f['Group']['GroupFirstSub'])
                all_group_len_type.append(group_len_type)
                all_group_nsubs.append(group_nsubs)
                all_group_first_subs.append(group_first_sub)
            else:
                group_len_type = None
            if 'SubhaloLenType' in f['Subhalo']:
                subhalo_len_type = np.array(f['Subhalo']['SubhaloLenType'])
                all_subhalo_len_type.append(subhalo_len_type)
            else:
                subhalo_len_type = None
            if (i+1) < n_group_files:
                group_offsets[i+1] = n_groups + group_offsets[i]
                subhalo_offsets[i+1] = n_subgroups + subhalo_offsets[i]

            for j in range(n_groups):
                group_first_particle.append(group_first_particle[-1] + group_len_type[j])

            # for j in range(n_subgroups): #TODO: this is not true because of the halo inner fuzz.
            #     subhalo_first_particle.append(subhalo_first_particle[-1] + subhalo_len_type[j])

    # remove the last element of the list as this won't correspond to the next (sub)group
    group_first_particle = np.array(group_first_particle[:-1])

    if len(all_group_len_type) > 0:
        all_group_len_type = np.concatenate(all_group_len_type, axis=0)
        all_group_nsubs = np.concatenate(all_group_nsubs, axis=0)
        all_group_first_subs = np.concatenate(all_group_first_subs, axis=0)
    if len(all_subhalo_len_type) > 0:
        all_subhalo_len_type = np.concatenate(all_subhalo_len_type, axis=0)

    subhalo_first_particle = []

    for i in range(len(all_group_len_type)):
        nsubs = all_group_nsubs[i]
        if nsubs == 0:
            continue
        first_particle = np.array(group_first_particle[i])
        local_subhalo_first_particle = [first_particle]
        for j in range(nsubs):
            subhalo_len = np.array(all_subhalo_len_type[all_group_first_subs[i]+j])
            local_subhalo_first_particle.append(local_subhalo_first_particle[-1] + subhalo_len)
        local_subhalo_first_particle = np.array(local_subhalo_first_particle[:-1])
        subhalo_first_particle.append(local_subhalo_first_particle)

    if len(subhalo_first_particle) > 0:
        subhalo_first_particle = np.concatenate(subhalo_first_particle, axis=0)

    snap_path = os.path.join(basepath, f'snapdir_{snap_number:03d}')
    if simulation_suite == 'fable' and not os.path.exists(snap_path):
        snap_path = os.path.join(basepath, f'snapdir_{snap_number:03d}_fable')

    snap_files = [f for f in os.listdir(snap_path) if f.startswith('snap') and f.endswith('.hdf5')]
    snap_files_sorted = sorted(snap_files, key=extract_number)
    n_snap_files = len(snap_files)

    snap_offsets = np.zeros((n_snap_files, 6))

    for i,snap_file in enumerate(snap_files_sorted):
        file_path = os.path.join(snap_path, snap_file)
        with h5py.File(file_path, 'r') as f:
            header = dict(f['Header'].attrs.items())
            n_particles = header['NumPart_ThisFile']
            if (i+1) < n_snap_files:
                snap_offsets[i+1] = n_particles + snap_offsets[i]

    # put together snap_offsets as an Nx6 array

    sublink_file = os.path.join(sublink_path, f'offsets_{snap_number:03d}.hdf5')
    sublink_exist = os.path.exists(sublink_file)     # to avoid missing offsets of the first tens of snapshots of the simulation
    if sublink_exist:
        with h5py.File(sublink_file, 'r') as f:
            sublink_offsets = np.array(f['FileOffsets'])
            last_progenitor_id = np.array(f['LastProgenitorID'])
            main_progenitor_id = np.array(f['MainLeafProgenitorID'])  # this is also required by macrophysical_delays.py
            row_num = np.array(f['RowNum'])
            subhalo_id = np.array(f['SubhaloID'])

    # create directory
    os.makedirs(output_path, exist_ok=True)

    offset_file = os.path.join(output_path, f'offsets_{snap_number:03d}.hdf5')
    with h5py.File(offset_file, 'w') as f:
        file_offsets = f.create_group('FileOffsets')

        file_offsets.create_dataset('Group', data = group_offsets, dtype='int')
        file_offsets.create_dataset('Subhalo', data = subhalo_offsets, dtype='int')
        file_offsets.create_dataset('SnapByType', data = snap_offsets, dtype='int')
        if sublink_exist:
            file_offsets.create_dataset('Sublink', data = sublink_offsets, dtype='int')

        if total_n_groups > 0:
            group = f.create_group('Group')
            group.create_dataset('SnapByType', data = group_first_particle, dtype='int')

        if total_n_subgroups > 0:
            subhalo = f.create_group('Subhalo')
            subhalo.create_dataset('SnapByType', data = subhalo_first_particle, dtype='int')
            if sublink_exist:
                subhalo.create_group('Sublink')
                subhalo['Sublink'].create_dataset('LastProgenitorID', data = last_progenitor_id, dtype='int')
                subhalo['Sublink'].create_dataset('MainLeafProgenitorID', data = main_progenitor_id, dtype='int')
                subhalo['Sublink'].create_dataset('RowNum', data = row_num, dtype='int')
                subhalo['Sublink'].create_dataset('SubhaloID', data = subhalo_id, dtype='int')

    print(f'Offset file for snapshot {snap_number} generated.')

def generate_files(basepath, output_path, sublink_path):
    """
    Generate the offset files for all snapshots.

    Parameters:
    basepath (str): The basepath to the FABLE simulation.
    output_path (str): The path to save the offset files.
    """
    for i in range(get_final_snapshot(basepath, simulation_suite = simulation_suite) + 1):
        generate_offset_file(i, basepath, output_path, sublink_path)

if __name__ == '__main__':
    # basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100'
    # output_path = '/cosma7/data/dp012/dc-butt3/FABLE_100_offsets'
    # sublink_path = '/cosma7/data/dp012/dc-butt3/MergerTrees/output/Galaxies/FABLE/Fid_test/offsets'

    # basepath = f"/home/pctung/beegfs_pctung/bh_merger_project/data_release_link/{simulation_name}/"
    # sublink_path = f"{basepath}/sublink_galaxies/offsets"
    # output_path = f"{sublink_path}/../offsets_illustris"


    # Parameter initialization
    initialize('generate_offsets')

    # Set global parameters
    basepath = parameters.sim_dir
    sublink_path = os.path.join(parameters.tree_dir, 'offsets')
    output_path = parameters.offset_dir
    simulation_suite = parameters.simulation_suite
    
    # Main
    generate_files(basepath, output_path, sublink_path)