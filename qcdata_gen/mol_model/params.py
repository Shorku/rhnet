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


def orca_dft(pal):
    """ Returns a template for an input file for geometry optimization using
    DFT

    Args:
        pal (int): the number of processes ORCA will use

    Return:
        (str): a string to be written in the input file
    """
    return f'''! RKS r2SCAN-3c tightscf tightopt
%pal nprocs {pal} end
*xyz 0 1
{{}}
*
'''


def orca_dft_cube(pal, mult, n):
    """ Returns a template for an input file for a single-point calculation
     using DFT. ORCA job will make a single-point calculation and generate
     electron and spin densities in the .cube-files

    Args:
        pal (int): the number of processes ORCA will use
        mult (int): molecular system multiplicity
        n (int): resolution (n x n x n) of the electron and spin densities
                 in the .cube-files ORCA will write after the DFT calculation
                 is converged

    Return:
        (str): a string to be written in the input file
    """
    return f'''! UKS B3LYP def2-TZVP notrah tightscf
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
