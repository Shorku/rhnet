"""
Functions that use conformers script
(http://limor1.nioch.nsc.ru/quant/program/conformers/) to remove coinciding
conformers from a given set of conformers (i.e. removing duplicates)
"""


import os
import re
import subprocess


def conformers_filter(xyz_file):
    """ Call conformers script to find duplicate geometries in a .xyz-file
    containing multiple geometries corresponding to the different conformations
    of a single compound

    Args:
        xyz_file (str): path to the .xyz-file

    Return:
        List[str]: list with the strings containing cartesian coordinates of
                   the unique conformers
    """
    conformers = []
    conformers_path = f'''{os.environ['CONFORMERS']}/conformers'''
    cli_opts = [conformers_path, '-no_mopac', '-logfile', xyz_file]
    subprocess.run(cli_opts,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    job_name = os.path.splitext(os.path.split(xyz_file)[1])[0]
    pattern = re.compile(job_name+r'\.(\d\d\d\d)\.xyz')
    for file in os.listdir():
        check_file = pattern.match(file)
        if check_file:
            with open(file, 'r') as xyz:
                conformers.append(xyz.read())
                os.remove(file)
    return conformers
