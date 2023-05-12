"""Set environment variables, prepare directories"""

import os


def set_env(params):
    """ Set environmental variables: define path to ORCA quantum chemical
    package and Conformers script (removes duplicate conformers)

    Args:
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    if params.orca_path:
        os.environ['ORCA'] = os.path.abspath(params.orca_path)
    elif 'ORCA' not in os.environ:
        raise EnvironmentError('ORCA executables path not specified. '
                               'Use --orca_path.')

    if params.conf_path:
        os.environ['CONFORMERS'] = os.path.abspath(params.conf_path)
    elif 'CONFORMERS' not in os.environ:
        raise EnvironmentError('conformers script path not specified. '
                               'Use --conf_path.')


def prepare_dir(params):
    """ Create if not exist folders within params.out_dir mother directory to
    store results of calculations:
    - params.out_dir/ff_geom to store generated conformations
    - params.out_dir/semi_geom to store conformations optimized on
                               semi-empirical level (.xyz files)
    - params.out_dir/dft_geom to store conformations optimized on DFT level
                              (.xyz files)
    - params.out_dir/cut_geom to store conformations without shielding groups
                              (.xyz files)
    - params.out_dir/npy_data to store interpolated and rescaled training-ready
                              electron and spin densities (.npy files)
    - params.out_dir/qc_temp to use while performing quantum chemical
                             calculations and store calculation logs

    Args:
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    if params.exec_mode == 'conf' or params.exec_mode == 'conf_and_cube':
        os.makedirs(os.path.join(params.out_dir, 'ff_geom'),
                    exist_ok=True)
        if params.conf_level == 'semi' or params.conf_level == 'dft':
            os.makedirs(os.path.join(params.out_dir, 'semi_geom'),
                        exist_ok=True)
        if params.conf_level == 'dft' or params.conf_level == 'ff_dft':
            os.makedirs(os.path.join(params.out_dir, 'dft_geom'),
                        exist_ok=True)
    if params.exec_mode == 'cube' or params.exec_mode == 'conf_and_cube':
        os.makedirs(os.path.join(params.out_dir, 'cut_geom'),
                    exist_ok=True)
        os.makedirs(os.path.join(params.out_dir, 'cubes'),
                    exist_ok=True)
        os.makedirs(os.path.join(params.out_dir, 'npy_data'),
                    exist_ok=True)
    if not (params.exec_mode == 'conf' and params.conf_level == 'ff'):
        os.makedirs(os.path.join(params.out_dir, 'qc_temp'),
                    exist_ok=True)
        os.chdir(os.path.join(params.out_dir, 'qc_temp'))
