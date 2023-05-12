"""Pre- and runtime utility functions"""

import pandas as pd


from qcdata_gen.utils.utils import sort_xyz_into_jobs


def calc_mass():
    """ Read xyz-files in the current folder and write .csv files with
    molecular masses

    Args:
        None

    Return:
        None

    """
    mass_dict = {'Ar': 39.95, 'C': 12.01, 'Cl': 35.45, 'F': 19.0, 'H': 1.01,
                 'N': 14.01, 'O': 16.0, 'S': 32.07, 'Si': 28.09}
    solvent_mass = {'solvent': [], 'solv_mass': []}
    polymer_mass = {'polymer': [], 'cut': [], 'poly_mass': []}
    job_names = sort_xyz_into_jobs('.')
    for job in job_names:
        compound_type, compound_num, compound_cut = job.split('_')
        compound_num = int(compound_num)
        compound_cut = int(compound_cut)
        mass = 0.0
        with open(job_names[job][0], 'r') as xyz:
            temp = xyz.readlines()
            for j in temp[2:]:
                mass += mass_dict[j.split()[0]]
        if compound_type == 'p':
            polymer_mass['polymer'].append(compound_num)
            polymer_mass['cut'].append(compound_cut)
            polymer_mass['poly_mass'].append(mass)
        elif compound_type == 's':
            solvent_mass['solvent'].append(compound_num)
            solvent_mass['solv_mass'].append(mass)
    polymer_mass = pd.DataFrame(polymer_mass)
    solvent_mass = pd.DataFrame(solvent_mass)
    polymer_mass.to_csv('polymer_mass.csv', index=False)
    solvent_mass.to_csv('solvent_mass.csv', index=False)


def trailing_zeros(number):
    """ Calculate how many times a number can be divided by 2

    Args:
        number (int): Number to be divided

    Return:
        count (int): power of two the number can be divided by

    """
    bin_string = bin(number)
    count = 0
    i = -1
    while bin_string[i] == '0':
        count += 1
        i -= 1
    return count
