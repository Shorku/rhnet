"""ORCA input files templates"""


def orca_xtb(pal):
    """ Returns a template for an input file for geometry optimization using
    XTB

    Args:
        pal (int): the number of processes ORCA will use

    Return:
        (str): a string to be written in the input file
    """
    return f'''! XTB2 verytightopt
%pal nprocs {pal} end
*xyz 0 1
{{}}
*'''


def orca_dft(pal, dft):
    """ Returns a template for an input file for geometry optimization using
    DFT

    Args:
        pal (int): the number of processes ORCA will use
        dft (str): DFT methods choice. Valid options are 'normal' for r2SCAN-3c
                   and 'fast' for B97-3c

    Return:
        (str): a string to be written in the input file
    """
    # B97-3c
    if dft == 'normal':
        method = 'r2SCAN-3c'
    elif dft == 'fast':
        method = 'B97-3c'
    else:
        raise ValueError(f'Unknown DFT methods scheme {dft}')
    return f'''! RKS {method} tightscf tightopt
%pal nprocs {pal} end
*xyz 0 1
{{}}
*
'''


def orca_dft_cube(pal, mult, n, dft):
    """ Returns a template for an input file for a single-point calculation
     using DFT. ORCA job will make a single-point calculation and generate
     electron and spin densities in the .cube-files

    Args:
        pal (int):  the number of processes ORCA will use
        mult (int): molecular system multiplicity
        n (int):    resolution (n x n x n) of the electron and spin densities
                    in the .cube-files ORCA will write after the DFT
                    calculation is converged
        dft (str):  DFT methods choice. Valid options are 'normal' for
                    B3LYP/def2-TZVP and 'fast' for PBE/def2-SVP

    Return:
        (str): a string to be written in the input file
    """
    if dft == 'normal':
        method = 'B3LYP def2-TZVP'
    elif dft == 'fast':
        method = 'PBE def2-SVP'
    else:
        raise ValueError(f'Unknown DFT methods scheme {dft}')
    return f'''! UKS {method} tightscf
%pal nprocs {pal} end
%scf
maxiter 300
end
%plots Format Cube
dim1 {n}
dim2 {n}
dim3 {n}
ElDens("{{}}.eldens.cube");
SpinDens("{{}}.spindens.cube");
end
*xyz 0 {mult}
{{}}
*
'''
