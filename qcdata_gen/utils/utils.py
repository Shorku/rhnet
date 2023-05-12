"""Utility functions: directory listing, temporary files clean up etc"""


import os
import re
import tarfile

from datetime import datetime


def dump_temp_files(data_dir, ext, compress=True, remove=True):
    """ Compress (optionally) and remove (optionally) files with defined
    extension in the defined directory

    Args:
        data_dir (str): directory path
        ext (str): extension of the files to compress and remove
        compress (bool): whether to compress files
        remove (bool): whether to remove files after compression

    Return:
        None

    """
    label = os.path.split(data_dir)[1]
    files = []
    # navigate to the target directory, otherwise tarfile will produce an
    # archive with many nested directories
    cwd = os.getcwd()
    os.chdir(data_dir)
    for name in os.listdir():
        if name.endswith(ext):
            files.append(name)
    if compress:
        # use time and date to generate unique archive name
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        tar_name = f"{'' if label == '.' else label}_{ext[1:]}_{now}.tar.xz"
        tar = tarfile.open(tar_name, 'w:xz')
        for name in files:
            tar.add(name)
        tar.close()
    if remove:
        for name in files:
            os.remove(name)
    # get back to the original directory
    os.chdir(cwd)


def sort_xyz_into_jobs(input_dir):
    """ Sort files in the target directory according to their belonging to
    a particular compound. I.e. a set of files

    'p_2_1_1.xyz'
    'p_2_1_2.xyz'
    'p_2_5_1.xyz'
    's_45_1_1.xyz'
    's_45_1_2.xyz'

    will be converted to a dictionary

    {'p_2_1': ['p_2_1_1.xyz', 'p_2_1_2.xyz'],
     'p_2_5': ['p_2_5_1.xyz'],
     's_45_1':['s_45_1_1.xyz', 's_45_1_2.xyz']}

    Args:
        input_dir (str): directory path

    Return:
        job_names (Dict[str, List[str]]): dictionary with compounds notations
                                          as keys and lists of corresponding
                                          files as values

    """
    pattern = re.compile(r'(.+)_\d+\.xyz')
    job_names = dict()
    for file in os.listdir(input_dir):
        check_file = pattern.match(file)
        if check_file:
            job = check_file.group(1)
            if job in job_names:
                job_names[job].append(file)
            else:
                job_names[job] = [file]
    return job_names


# TODO replace read-write with shutil.copy
def write_conf_from_list(job_name, conformers, output_dir):
    """ Read files from conformers list and write them to the target directory
    under the job_name and new conformer indexing

    Args:
        job_name (str): compound notation
        conformers (List[str]): list of filenames with compound conformers
        output_dir (str): target directory path

    Return:
        None

    """
    for index, conf in enumerate(conformers):
        outfile = f'{job_name}_{index + 1}.xyz'
        outfile_path = os.path.join(output_dir, outfile)
        with open(outfile_path, 'w') as xyz:
            xyz.write(conf)


def bin_average(cube, times):
    """
    Downsize NxNxN array to N/times*N/times*N/times array averaging over
    times*times*times blocks.

    Args:
        cube (numpy.array): 3D array
        times (int): times the cube should be resized

    Return:
        cube (numpy.array): downsized 3D array

    """

    extended_shape = (cube.shape[0]//times, times,
                      cube.shape[1]//times, times,
                      cube.shape[2]//times, times,)
    cube = cube.reshape(extended_shape)
    for i in range(3):
        cube = cube.mean(-1*(i+1))
    return cube
