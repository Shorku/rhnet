"""Data dimensions and scaling parameters """

from munch import Munch


def scale_params():
    """ Returns predefined dataset scaling parameters

    Args:

    Return:
        scale_options (munch.Munch): scaling parameters

    """
    scale_options = Munch({
                        'el_scale': 20.0,
                        'sp_scale': 20.0,
                        'mnw_shift': 1.0,
                        'mnw_scale': 4.0,
                        'p_shift': -4.0,
                        'p_scale': 6.0,
                        't_shift': 258.0,
                        't_scale': 300.0,
                        'd_shift': -3.0,
                        'd_scale': 2.0,
                        })

    return scale_options


def image_dim():
    """ Returns predefined electron or spin density images dimensions

    Return:
        (tuple): density image dimensions

    """
    return 80, 80, 80, 2


def macro_dim(params):
    """ Returns predefined experimental data dimensions

    Args:
        params (munch.Munch): Command line parameters

    Return:
        num_features (int): number of macroscopic features

    """
    num_features = 5
    if params.use_only_mw:
        num_features -= 1
    if params.use_only_amorph:
        num_features -= 1
    if params.use_tg:
        num_features += 1
    if params.use_dens:
        num_features += 1
    if params.use_bt:
        num_features += 1
    if params.use_ctp:
        num_features += 2

    return num_features
