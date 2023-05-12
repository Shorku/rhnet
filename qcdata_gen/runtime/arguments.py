"""Command line argument parsing"""
import argparse
from munch import Munch

PARSER = argparse.ArgumentParser(description="ConfGen")

PARSER.add_argument('--exec_mode',
                    choices=['conf', 'cube', 'conf_and_cube'],
                    type=str,
                    default='conf_and_cube',
                    help="""Generate conformers and/or cubes""")

PARSER.add_argument('--mol_dir',
                    type=str,
                    default='/mol',
                    help="""Input .mol or .xyz structures directory""")

PARSER.add_argument('--out_dir',
                    type=str,
                    required=True,
                    help="""Directory with output data""")

PARSER.add_argument('--pal',
                    type=int,
                    default=12,
                    help="""Number of CPU cores to use in calculations""")

PARSER.add_argument('--conf_level',
                    choices=['ff', 'semi', 'dft', 'ff_dft'],
                    type=str,
                    default='dft',
                    help="""Highest level of conf geometry optimization,
                    successively optimize at ff, xtb and dft levels by default.
                    For example xtb option will do ff and xtb optimizations.
                    ff-dft will omit xtb step.""")

PARSER.add_argument('--use_name_convention',
                    dest='use_name_convention',
                    action='store_true',
                    help="""Suggest structure with xyz-filename starting with:
                    's_' to be a solvent molecule and with
                    'p_' to be a monomer - 8 last atoms are suggested to be
                     screening Me- groups and will be removed for cube eval""")

PARSER.add_argument('--component',
                    choices=['solvent', 'polymer'],
                    type=str,
                    default=None,
                    help="""Suggest all structures in xyz-files to be:
                    solvent - calculate/interpolate cube for the whole molecule
                    polymer - 8 last atoms are suggested to be screening
                    Me- groups and will be removed for cube eval""")

PARSER.add_argument('--rdkit_thresh',
                    type=float,
                    default=0.1,
                    help="""Conformers energy difference criteria""")

PARSER.add_argument('--rdkit_nconf',
                    type=int,
                    default=1000,
                    help="""Max number of conformers rdkit will generate""")

PARSER.add_argument('--rdkit_thresh_keep',
                    type=float,
                    default=20,
                    help="""Energy window to keep conformers within, kJ/mol""")

PARSER.add_argument('--orca_thresh_keep',
                    type=float,
                    default=5,
                    help="""Energy window to keep conformers within, kJ/mol""")

PARSER.add_argument('--cube_n',
                    type=int,
                    default=80,
                    help="""Dimension of interpolated .cube, points""")

PARSER.add_argument('--cube_spacing',
                    type=float,
                    default=0.4,
                    help="""Spacing in interpolated .cube, Angstroem""")

PARSER.add_argument('--cube_scaling',
                    type=float,
                    default=75.0,
                    help="""Scale electron density""")

PARSER.add_argument('--cube_aug',
                    type=int,
                    default=25,
                    help="""Number of augmented .cubes, max is 25""")

PARSER.add_argument('--extend_cube',
                    type=int,
                    default=6,
                    help="""Number of points to average cube density""")

PARSER.add_argument('--orca_path',
                    type=str,
                    help="""ORCA location, can also be taken from env""")

PARSER.add_argument('--conf_path',
                    type=str,
                    help="""CONFORMERS location, can also be taken from env""")


def parse_args(flags):
    return Munch({
        'exec_mode': flags.exec_mode,
        'mol_dir': flags.mol_dir,
        'out_dir': flags.out_dir,
        'pal': flags.pal,
        'conf_level': flags.conf_level,
        'use_name_convention': flags.use_name_convention,
        'component': flags.component,
        'rdkit_thresh': flags.rdkit_thresh,
        'rdkit_nconf': flags.rdkit_nconf,
        'rdkit_thresh_keep': flags.rdkit_thresh_keep,
        'orca_thresh_keep': flags.orca_thresh_keep,
        'cube_n': flags.cube_n,
        'cube_spacing': flags.cube_spacing,
        'cube_scaling': flags.cube_scaling,
        'cube_aug': flags.cube_aug,
        'extend_cube': flags.extend_cube,
        'orca_path': flags.orca_path,
        'conf_path': flags.conf_path
            })
