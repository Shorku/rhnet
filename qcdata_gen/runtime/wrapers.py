"""
Implementation of the interfaces between package-specific functions from
mol_model module and general workflow from run.py module
"""

import os
import random
import tarfile
import shutil


from datetime import datetime
from multiprocessing import Process

try:
    from mol_model.rdkit_driver import rdkit_gen
except ModuleNotFoundError:
    print('Warning: RDKit is missing, this may cause script execution failure')
from mol_model.conformers_driver import conformers_filter
from mol_model.orca_driver import orca_gen, orca_cube, orca_rescale_cubes
from mol_model.orca_driver import orca_tmp_clear
from utils.utils import sort_xyz_into_jobs, write_conf_from_list


def generate_qc_conf(input_dir, output_dir, level, params):
    """ Sort conformers geometries from .xyz files in input_dir according to
    the chemical compound they belong to, optimize on the requested level of
    theory, filter by energy and write optimized and filtered geometries to
    output_dir

    Args:
        input_dir (str): directory path, contains .xyz files to be optimized
        output_dir (str): directory path, directory to write optimized
                          geometries in
        level (str): level of theory to optimize geometries, can be 'semi' or
                     'dft'
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    job_names = sort_xyz_into_jobs(input_dir)
    thresh_keep = params.orca_thresh_keep
    pal = params.pal
    dft_scheme = params.dft_scheme
    for job, jobs_files in job_names.items():
        conformers = orca_gen(input_dir, jobs_files, thresh_keep, level, pal,
                              dft_scheme)
        write_conf_from_list(job, conformers, output_dir)


def generate_ff_conf(input_dir, output_dir, params):
    """ For every compound in .mol files in the input_dir directory generate
    a set of conformers using RDKit module and write every conformer into .xyz
    file in the output_dir directory

    Args:
        input_dir (str): directory path, contains .mol files with individual
                         compounds
        output_dir (str): directory path, directory to write a set of
                          geometries (.xyz files) of the conformers
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    for infile in os.listdir(input_dir):
        if infile.endswith('.mol') or infile.endswith('.sdf'):
            job_name = os.path.splitext(infile)[0]
            infile_path = os.path.join(input_dir, infile)
            thresh = params.rdkit_thresh
            thresh_keep = params.rdkit_thresh_keep
            nconf = params.rdkit_nconf

            conformers = rdkit_gen(infile_path, thresh, nconf, thresh_keep)
            write_conf_from_list(job_name, conformers, output_dir)


def generate_qc_cube(input_dir, output_dir, params):
    """ Calculate electron and spin density for every geometry (.xyz files) in
    the input_dir directory and generate density plots (.cube files) with
    params.extend_cube -  times larger spatial resolution, than needed for
    training and write the files with densities to the output_dir directory

    Args:
        input_dir (str): directory path, contains .xyz files
        output_dir (str): directory path, directory to write densities
                          (.cube files)
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    xyz_file_list = [file for file in os.listdir(input_dir)
                     if file.endswith('.xyz')]
    pal = params.pal
    cube_n = params.cube_n
    dft_scheme = params.dft_scheme
    nospin = params.nospin
    for xyz_file in xyz_file_list:
        if (params.component == 'solvent'
                or (params.use_name_convention
                    and xyz_file.split('_')[0] == 's')):
            ispolymer = False
        elif (params.component == 'polymer'
                or (params.use_name_convention
                    and xyz_file.split('_')[0] == 'p')):
            ispolymer = True
        else:
            raise NameError('Type of compound unrecognized. Prepend input '
                            'files with s_ or p_ and use --use_name_convention'
                            'or use --component with solvent of polymer arg')
        xyz_path = os.path.join(input_dir, xyz_file)
        orca_cube(xyz_path, pal, ispolymer, cube_n, dft_scheme, nospin)
    # Check whether output_dir contains already some of the required .cube's
    cube_file_list_done = [f for f in os.listdir(output_dir)
                           if f.endswith('.cube')]
    cube_file_list = [f for f in os.listdir()
                      if (f.endswith('.cube')
                          and f not in cube_file_list_done)]
    random.shuffle(cube_file_list)
    num_cub = len(cube_file_list) // pal
    # High resolution .cube's are generated using orca_plot utility, which has
    # only serial mode, hence it is important to use multiprocessing.
    if pal > 1:
        processes = [Process(target=orca_rescale_cubes,
                             args=(cube_file_list[i * num_cub:(i +
                                                               1) * num_cub],
                                   params)) for i in range(pal-1)]
        processes.append(Process(target=orca_rescale_cubes,
                                 args=(cube_file_list[(pal - 1) * num_cub:],
                                       params)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    else:
        orca_rescale_cubes(cube_file_list, params)

    for file in cube_file_list:
        shutil.move(file, os.path.join(output_dir, file))

    compress = True
    orca_tmp_clear('.', compress)


def remove_duplicates(data_dir):
    """ Sort conformers geometries from .xyz files in data_dir directory
    according to the chemical compound they belong to, write them into a single
    .xyz-file and invoke external conformers script to remove duplicates i.e.
    identical geometries. The non-filtered geometries are archived.

    Args:
        data_dir (str): directory path, contains .xyz files to be filtered

    Return:
        None

    """
    cwd = os.getcwd()
    os.chdir(data_dir)
    label = os.path.split(data_dir)[1]
    job_names = sort_xyz_into_jobs('.')
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    for job, jobs_files in job_names.items():
        all_xyz_file = f'{job}.xyz'
        all_xyz_tar = f'{job}_nonfiltered_{label}_{now}.tar.xz'
        with open(all_xyz_file, 'w') as xyz:
            for file in jobs_files:
                file_stream = open(file, 'r')
                xyz.write(file_stream.read())
                file_stream.close()
                os.remove(file)

        conformers = conformers_filter(all_xyz_file)
        write_conf_from_list(job, conformers, '.')

        with tarfile.open(all_xyz_tar, 'w:xz') as tar:
            tar.add(all_xyz_file)
        os.remove(all_xyz_file)

    os.chdir(cwd)
