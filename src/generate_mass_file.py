"""
BH masses can be incorrect in the BH merger catalogue.
This script sorts through the BH details text files and collects the masses of the BHs, by particle ID.
This is necessary for quick access when calculating the mass of the BH (by interpolation) at a specific scale factor.

The script uses MPI to parallelize the process of reading the files and collecting the data.

[Parameters] (Input throgh command line. Refer to initialize.py for more information.)
------
    (INPUT_NAME: PARAMETER_NAME)
    details_dir: basepath
    mass_file: h5_output

-----------------------------------------------------------------------------------------------

Stephanie Buttigieg (sb2583@cam.ac.uk)

Modified by Pei-Cheng Tung (pt26aae@herts.ac.uk)
"""

from mpi4py import MPI
import os
import numpy as np
import h5py

from mpi4py.util import pkl5 # an extension to handle large datasets
import argparse

from initialize import initialize, parameters


# # Set the base path and output file as necessary
# simulation_name = args.simulation_name  # seed1e2_se_boost, seed1e3_se_boostmax, seed1e4_se_boost
# basepath = f'/beegfs/general/skoudmani/aesopica_proj/data_release/{simulation_name}/blackhole_details'  # path to "blackhole_details" directory
# h5_output = f'/home/pctung/beegfs_pctung/bh_merger_project/output/{simulation_name}/bh_masses.hdf5'     # output black hole masses hdf5 file


# Read blackhole_details files and collect mass and scale factor into dictionaries
def process_file(file_path):
    '''
    Read a blackhole_details file and collect the masses and scale factors of black holes by their IDs.

    Input: path to a blackhole_details file
    Output: masses andscale factors dictionaries
    '''
    masses = {}
    scale_factors = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = line.split()

            # skip the peculiar data
            try:
                bh_id = int(data[0].split("=")[1])
            except (ValueError, IndexError):
                print(f"Error processing line: {line.strip()} in file: {file_path}", flush=True)
                print(f"Data: {data}", flush=True)
                continue
            
            try:
                values = list(map(float, data[1:]))
            except ValueError:
                print(f"Error converting values to float in line: {line.strip()} in file: {file_path}")
                print(f"Data: {data}")
                continue
            
            if bh_id in masses:
                masses[bh_id].append(values[1])
                scale_factors[bh_id].append(values[0])
            else:
                masses[bh_id] = [values[1]]
                scale_factors[bh_id] = [values[0]]
    return masses, scale_factors

def merge_dicts(dict_list):
    merged_masses = {}
    merged_scales = {}
    for masses, scales in dict_list:
        for bh_id in masses:
            if bh_id in merged_masses:
                merged_masses[bh_id].extend(masses[bh_id])
                merged_scales[bh_id].extend(scales[bh_id])
            else:
                merged_masses[bh_id] = masses[bh_id]
                merged_scales[bh_id] = scales[bh_id]
    return merged_masses, merged_scales

def main(verbose=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # # Set the base path and output file as necessary
    # basepath = '/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/blackhole_details'
    # h5_output = '/cosma8/data/dp012/dc-butt3/bh_masses.hdf5'

    # Distirbute files to MPI processors and read
    files = sorted(os.listdir(basepath))
    my_files = files[rank::size]

    local_data = []
    for file in my_files:   # remove "[:10]" from the original version
        path = os.path.join(basepath, file)
        if verbose:
            print(f'Rank {rank} processing file: {path}')
        local_data.append(process_file(path))

    # Gather data at rank 0
    comm = pkl5.Intracomm(MPI.COMM_WORLD)  # enable large dataset (>2GB)
    gathered = comm.gather(local_data, root=0)

    if verbose:
        print('Gathered all data, now merging.')

    # Create and save dataset
    if rank == 0:
        flat_data = [item for sublist in gathered for item in sublist]
        masses, scale_factors = merge_dicts(flat_data)

        if verbose:
            print('Merged data, now writing to HDF5.')

        # create file dir
        os.makedirs(os.path.dirname(h5_output), exist_ok=True)

        bh_len = len(masses) # length of the dataset
        with h5py.File(h5_output, 'w', libver='latest') as f:
            for i, bh_id in enumerate(masses):
                data = np.column_stack((scale_factors[bh_id], masses[bh_id]))
                f.create_dataset(str(bh_id), data=data)

                if verbose and i % 100000 == 0:
                    print(f'Finished writing {i}/{bh_len} BHs to HDF5 file.')

        if verbose:
            print("Finished writing to HDF5 file.")

if __name__ == "__main__":
    # Parameter initialization
    initialize('generate_mass_file', verbose=(MPI.COMM_WORLD.Get_rank() == 0))

    # Set global parameters
    basepath = parameters.details_dir
    h5_output = parameters.mass_file

    # flags
    verbose = parameters.verbose

    # main
    main(verbose=verbose)
