"""
Mirror a directory tree with symbolic links, renaming single-file HDF5 outputs
to the multi-file naming convention:

    name.hdf5                    -> name.0.hdf5
    fof_subhalo_tab_XXX.hdf5     -> groups_XXX/fof_subhalo_tab_XXX.0.hdf5
    snap_XXX.hdf5                -> snapdir_XXX/snap_XXX.0.hdf5

Files that already carry a chunk number (name.N.hdf5) and files inside
'offsets' directories keep their names. Original data are never modified.

Usage:
    python make_links.py /beegfs/general/skoudmani/aesopica_proj/data_release /path/to/parent [--dry-run]

----------------------------------------------------------------------------------------------------------------------------

Pei-Cheng Tung (pt26aae@herts.ac.uk)
"""


import argparse
import os
import re

CHUNKED = re.compile(r'\.\d+\.hdf5$')
FOF = re.compile(r'^fof_subhalo_tab_(\d+)\.hdf5$')
SNAP = re.compile(r'^snap_(\d+)\.hdf5$')
KEEP_NAME_DIRS = {'offsets'}
IGNORE_DIRS = {'arepo'}


def link_name(rel_dir, fname):
    """Return the link path (relative to the mirror root) for a source file."""
    if not fname.endswith('.hdf5') or CHUNKED.search(fname) \
            or os.path.basename(rel_dir) in KEEP_NAME_DIRS:
        return os.path.join(rel_dir, fname)
    
    m = FOF.match(fname)
    if m:
        num = m.group(1)
        return os.path.join(rel_dir, f'groups_{num}', f'fof_subhalo_tab_{num}.0.hdf5')
    
    m = SNAP.match(fname)
    if m:
        num = m.group(1)
        return os.path.join(rel_dir, f'snapdir_{num}', f'snap_{num}.0.hdf5')
    return os.path.join(rel_dir, fname[:-len('.hdf5')] + '.0.hdf5')


def make_link(target, link, dry_run, stats):
    if os.path.lexists(link):
        if os.path.islink(link) and os.readlink(link) == target:
            stats['skipped'] += 1
        else:
            print(f'WARNING: {link} already exists, not overwritten (wanted -> {target})')
            stats['conflicts'] += 1
        return
    if dry_run:
        print(f'{link} -> {target}')
    else:
        os.makedirs(os.path.dirname(link), exist_ok=True)
        os.symlink(target, link)
    stats['created'] += 1


def mirror(src_root, dst_root, dry_run=False):
    src_root = os.path.abspath(src_root)
    stats = {'created': 0, 'skipped': 0, 'conflicts': 0}
    
    for dirpath, dirnames, filenames in os.walk(src_root, onerror=lambda e: print(f'WARNING: cannot read {e.filename}')):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORE_DIRS)
        rel_dir = os.path.relpath(dirpath, src_root)
        rel_dir = '' if rel_dir == '.' else rel_dir

        # print(dirnames, rel_dir)

        if not dry_run:
            os.makedirs(os.path.join(dst_root, rel_dir), exist_ok=True)

        # Symlinked directories are not walked into: link them as they are
        for d in [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            dirnames.remove(d)
            make_link(os.path.join(dirpath, d), os.path.join(dst_root, rel_dir, d), dry_run, stats)

        for fname in sorted(filenames):
            target = os.path.join(dirpath, fname)
            link = os.path.join(dst_root, link_name(rel_dir, fname))
            make_link(target, link, dry_run, stats)

    print(f"created {stats['created']}, already existed {stats['skipped']}, conflicts {stats['conflicts']}")


if __name__ == '__main__':
    # # Command line option
    # parser = argparse.ArgumentParser(description='Mirror a directory tree with symbolic links.')
    # parser.add_argument('src_root')
    # parser.add_argument('dst_parent', help='data_release_links is created inside this directory')
    # parser.add_argument('--dry-run', action='store_true', help='only print what would be done')
    # args = parser.parse_args()

    # src_root = os.path.abspath(args.src_root)
    # dst_root = os.path.join(os.path.abspath(args.dst_parent), 'data_release_links')
    # dry_run = args.dry_run
    # if os.path.commonpath([src_root, dst_root]) == src_root:       # make sure "dst_root" is not inside "src_root"
    #     parser.error('Destination must not be inside the source tree!')


    # Direct input
    src_root = "/beegfs/general/skoudmani/aesopica_proj/data_release/"
    dst_root = "/home/pctung/beegfs_pctung/bh_merger_project/data_release_link/"
    dry_run  = False
    if os.path.commonpath([os.path.abspath(src_root), os.path.abspath(dst_root)]) == os.path.abspath(src_root):    # make sure "dst_root" is not inside "src_root"
        raise ValueError('Destination must not be inside the source tree!')

    # Main
    mirror(src_root, dst_root, dry_run)