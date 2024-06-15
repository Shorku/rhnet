"""Encapsulates all rdkit related functions"""


import pathlib

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import AllChem
from rdkit.Chem import rdmolfiles


def rdkit_gen(file_name, thresh, nconf, thresh_keep, threads=0, seed=None):
    """ Read input geometry from .mol-file and generate conformations

    Args:
        file_name (str): path to .mol file with input geometry
        thresh (float): conformers energy difference criteria
        nconf (int): maximum number of conformers to be generated
        thresh_keep (float): Energy window to keep conformers within, kJ/mol
        threads (int): number of threads rdkit will use. Defaults to 0 meaning
                       all available
        seed (int): random seed to be used by rdkit

    Return:
        List[str]: a list of strings with cartesian coordinates of generated
                   conformers

    """
    print('\n', file_name, '\n')
    extension = pathlib.Path(file_name).suffix
    if extension == '.mol':
        mol = Chem.MolFromMolFile(file_name, removeHs=False)
    elif extension == '.sdf':
        mol = Chem.AddHs(next(rdmolfiles.SDMolSupplier(file_name)))
    else:
        print(f'{extension} is not an appropriate extension')
    param = Chem.rdDistGeom.ETKDGv2()
    param.pruneRmsThresh = thresh
    # param.clearConfs = False
    # param.enforceChirality = True
    if seed:
        param.randomSeed = seed
    param.numThreads = threads
    rdDistGeom.EmbedMultipleConfs(mol, nconf, param)
    conv = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=threads)
    e_mmff = [i[1] for i in conv]
    e_min = min(e_mmff)
    filtered_confs = []
    for i in range(Chem.rdchem.Mol.GetNumConformers(mol)):
        if e_mmff[i]-e_min < thresh_keep:
            filtered_confs.append(Chem.rdmolfiles.MolToXYZBlock(mol, i))

    return filtered_confs
