"""
Read command-line parameters for the BHMacroDelay scripts.
Define the physical constants here!

    initialize('generate_merger_files'), get_bh_params(sim_dir), get_final_snapshot(sim_dir)
    
----------------------------------------------------------------------------------------------------------------------------

Pei-Cheng Tung (pt26aae@herts.ac.uk)
"""

import argparse
import os
from types import SimpleNamespace
import datetime
from time import sleep
import re

#
# Define physical values
#

# Current params: Fiducial setup for AESOPICA and FABLE
LITTLE_H = 0.679
OMEGA_0 = 0.3065
# BOX_SIZE = 4e4  # 40 Mpc/h box
# SOFTENING_LENGTH = 14.3606
DM_MASS = 3.4e7 / LITTLE_H
STELLAR_MASS = 6.4e6/LITTLE_H



# Parameter container
parameters = SimpleNamespace()

# All parameters: name -> (type, default, help)
# default None: required parameter
PARAMS = {
    # paths
    'details_dir':    (str,   None, 'blackhole_details directory'),
    'merger_dir':     (str,   None, 'blackhole_mergers directory'),
    'sim_dir':        (str,   None, 'simulation directory (snapshots, group catalogs)'),
    'expansion_list': (str,   None, 'output scale factor list (ExpansionList)'),
    'tree_dir':       (str,   None, 'merger tree directory'),
    'offset_dir':     (str,   None, 'Illustris-style offset directory (output of generate_offsets.py)'),
    'src_root':       (str,   None, 'source directory tree to mirror (e.g. data_release)'),
    'dst_root':       (str,   None, 'destination directory for the symbolic links'),

    'mass_file':      (str,   None, 'black hole mass file from generate_mass_file.py (bh_masses.hdf5)'),
    'output_dir_merger':     (str,   None, 'directory for merger-event hdf5 files'),
    'output_dir_delay':      (str,   None, 'directory for macrophysical delay text files'),
    'output_dir_accretion':  (str,   None, 'directory for accretion detail files'),
    'output_dir_eddington':  (str,   None, 'directory for Eddington ratio text files'),

    # physical parameters
    'min_mass':       (float, 1e6,  'minimum mass of merger event for both BHs [M_sun]'),
    'box_size':       (float, None, 'simulation box size [ckpc/h]'),
    'subhalo_selection': (str, 'fable', 'host subhalo selection criteria: fable (Buttigieg et al. 2025) or aesopica (Koudmani et al. 2026)'),
    'simulation_suite':  (str, 'aesopica', 'simulation suite: aesopica or fable'),  # Easier for this project. This parameter should be modified in the future for more general usage

    # flags
    'verbose':        (bool,  False, 'print progress messages'),
    'dry_run':        (bool,  False, 'only print what would be done'),
}

# Parameters for scripts 
# 'in': existing paths, 'out': output path
SCRIPTS = {
    # Main routines
    'generate_mass_file': {
        'details_dir': 'in', 
        'mass_file': 'out',
        'verbose': None,
    },
    'generate_merger_files': {
        'sim_dir': 'in', 'merger_dir': 'in', 'expansion_list': 'in', 'mass_file': 'in', 'tree_dir': 'in', 
        'output_dir_merger': 'out', 
        'min_mass': None,
        'verbose': None,
    },
    'macrophysical_delays':{
        'sim_dir': 'in', 'expansion_list': 'in', 'tree_dir': 'in', 'output_dir_merger': 'in', 'offset_dir': 'in',
        'output_dir_delay': 'out', 'output_dir_accretion': 'out',
        'box_size': None, 'subhalo_selection': None, 'simulation_suite': None,
        'verbose': None,
    },
    'eddington_rates':{
        'sim_dir': 'in', 'expansion_list': 'in', 'offset_dir': 'in', 'output_dir_delay': 'in',
        'output_dir_eddington': 'out',
    },

    # Helper routines
    'make_links': {
        'src_root': 'in',
        'dst_root': 'out',
        'dry_run': None,
    },
    'generate_offsets': {
        'sim_dir': 'in', 'tree_dir': 'in',
        'offset_dir': 'out',
        'simulation_suite': None,
    },
}

# Short version for input
SHORT_FLAGS = {
    'verbose': '-v',
}

# Allowed values for input
CHOICES = {
    'subhalo_selection': ['fable', 'aesopica'],
    'simulation_suite': ['fable', 'aesopica'],
}

# Boost factor and Eddington ratio of Aesopica simulations
bh_params_map_aesopica = {
    (2, 'fiducial'):    (1e2, 1),
    (3, 'fiducial'):    (1e2, 1),
    (4, 'fiducial'):    (1e2, 1),
    (5, 'fiducial'):    (1e2, 1),

    (2, 'se_boost'):    (1e3, 10),
    (3, 'se_boost'):    (1e3, 10),
    (4, 'se_boost'):    (1e3, 10),
    (5, 'se_boost'):    (1e3, 10),

    (2, 'se_boostmax'): (1e5, 10),
    (3, 'se_boostmax'): (1e4, 10),
    (4, 'se_boostmax'): (1e3, 10),
    (5, 'se_boostmax'): (1e3, 10),
}


def initialize(script, argv=None, verbose=True):
    '''
    Initialize parameters for all the scripts
    '''
    # Check script
    if script not in SCRIPTS:
        raise ValueError(f'Unknown script {script!r}, expected one of {list(SCRIPTS)}')

    # Read parameters
    parser = argparse.ArgumentParser(prog=script, allow_abbrev=False)
    for name in SCRIPTS[script]:
        d_type, default, help_ = PARAMS[name]

        # Parse flags
        if d_type is bool:
            flags = [SHORT_FLAGS[name]] if name in SHORT_FLAGS else []
            parser.add_argument(*flags, f'--{name}', action='store_true', help=help_)
            continue
        
        # Parse paths and parameters
        choices = CHOICES.get(name)
        if default is None:
            parser.add_argument(f'--{name}', type=d_type, required=True, choices=choices, help=help_)
        else:
            parser.add_argument(f'--{name}', type=d_type, default=default, choices=choices, help=help_)

    args = parser.parse_args(argv)   # missing or unknown arguments -> error and exit

    # Verify inputs
    for name, kind in SCRIPTS[script].items():
        if kind == 'in' and not os.path.exists(getattr(args, name)):
            raise FileNotFoundError(f'--{name}: {getattr(args, name)} does not exist')

    # Update parameter container
    parameters.__dict__.update(vars(args))
    parameters.script = script

    # Print parameters
    if verbose:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"----BHMacroDelay----")
        print(f"Executing script: {parameters.script} ({time})")
        print('======== parameters ========')
        for name, value in vars(args).items():
            print(f'  {name:15s} = {value}')
        print('============================', flush=True)
        print()
        sleep(1)
    return parameters


def get_bh_params(sim_dir, simulation_suite = 'aesopica'):

    if simulation_suite == 'aesopica':
        bh_params_map = bh_params_map_aesopica
        feature_string = r"seed1e(\d)_(\S+)"
    elif simulation_suite == 'fable':
        return 100, 1
    else:
        raise NotImplementedError(f"Simulation suite {simulation_suite} is not implemented yet")

    # Extract parametes from simulation dircetory
    simulation_name = os.path.basename(os.path.normpath(sim_dir))
    params = re.search(feature_string, simulation_name)
    if params is None:
        raise ValueError(f'Cannot parse seed mass and accretion model from {simulation_name!r}')

    # Find BH parameters
    seed_mass_index = int(params.group(1))
    accretion_boost = params.group(2)

    key = (seed_mass_index, accretion_boost)
    if key not in bh_params_map:
        raise KeyError(f'No accretion parameters for {simulation_name!r} (key {key})')

    return bh_params_map[key]

def get_final_snapshot(sim_dir, simulation_suite = 'aesopica'):

    dir_list = os.listdir(sim_dir)

    snap_dirs = [d for d in dir_list if 'snapdir_' in d]  
    snap_list1 = [d for d in dir_list if 'snap_' in d]  
    snap_list2 = [d for d in dir_list if 'snapshot_' in d]  
    if len(snap_dirs) != 0:
        # For group output
        if simulation_suite == 'fable':
            # remove the file with "fable" in the name
            snap_dirs = [d for d in snap_dirs if 'fable' not in d]
        last_snap_index = max(int(d.split('_')[-1]) for d in snap_dirs)
    elif len(snap_list1) != 0:
        # For single output
        last_snap_index = max(int(d.split('_')[-1].replace(".hdf5", "")) for d in snap_list1)
    elif len(snap_list2) != 0:
        # For single output
        last_snap_index = max(int(d.split('_')[-1].replace(".hdf5", "")) for d in snap_list2)
    else:
        raise ValueError(f'This simulation format is not supported. Please implement here')

    return last_snap_index