"""
Manipulate common data formats:
- remove shielding groups from geometries in .xyz-files
- interpolate densities from .cube-files
- reduce densities resolutions keeping integral charge
- augment densities (rotate and shift)
"""


import os
import math
import shutil
import random
import numpy as np
from scipy.interpolate import RegularGridInterpolator as interpolator


from utils.utils import bin_average


def cut_geometries(input_dir, output_dir, params):
    """ Read molecular geometries from .xyz files in input_dir directory,
    remove the last 8 atoms (supposed to be two shielding methyl-groups) if the
    geometry is supposed to be a monomer/oligomer and save the geometry to
    .xyz-file in the output_dir directory. Do nothing and copy to .xyz-file to
    the output directory if the geometry belongs to a solvent

    Args:
        input_dir (str): directory path, contains .xyz files
        output_dir (str): directory path, directory to write densities
                          unshielded geometries
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    xyz_list = []
    for file in os.listdir(input_dir):
        if file.endswith('.xyz'):
            xyz_list.append(file)
    for file in xyz_list:
        in_xyz_file = os.path.join(input_dir, file)
        out_xyz_file = os.path.join(output_dir, file)
        if (params.component == 'solvent'
                or (params.use_name_convention and file.split('_')[0] == 's')):
            shutil.copyfile(in_xyz_file, out_xyz_file)
        elif (params.component == 'polymer'
                or (params.use_name_convention and file.split('_')[0] == 'p')):
            uncut_xyz = open(in_xyz_file, 'r')
            cut_xyz = open(out_xyz_file, 'w')
            cut_text = uncut_xyz.readlines()[:-8]
            cut_text[0] = f'{str(int(cut_text[0]) - 8)}\n'
            cut_xyz.write(''.join(cut_text))
            uncut_xyz.close()
            cut_xyz.close()
        else:
            raise NameError('Type of compound unrecognized. Prepend input '
                            'files with s_ or p_ and use --use_name_convention'
                            'or use --component with solvent of polymer arg')


def convert_qc_cubes(input_dir, output_dir, params):
    """ Read electron and spin densities from .cube-files in input_dir
    directory, call the convert_qc_cube function to move densities to the
    training-ready spatial grid. Then generate multiple shifted and rotated
    densities, stack electron and spin densities as channels and write them
    to .npy-files in numpy format into the output_dir directory

    Args:
        input_dir (str): directory path, contains .cube files
        output_dir (str): directory path, directory to write augmented
                          densities in .npy-format
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    unique_cube_rot_states = ((0, 0, 0), (0, 0, 1), (0, 0, 2),
                              (0, 0, 3), (0, 1, 0), (0, 1, 1),
                              (0, 1, 2), (0, 1, 3), (0, 2, 0),
                              (0, 2, 1), (0, 2, 2), (0, 2, 3),
                              (0, 3, 0), (0, 3, 1), (0, 3, 2),
                              (0, 3, 3), (1, 0, 0), (1, 0, 1),
                              (1, 0, 2), (1, 0, 3), (1, 2, 0),
                              (1, 2, 1), (1, 2, 2), (1, 2, 3))
    aug = params.cube_aug - 1

    for file in os.listdir(input_dir):
        if file.endswith('eldens.cube'):
            job_name = os.path.splitext(os.path.splitext(file)[0])[0]
            input_file = os.path.join(input_dir, file)
            eldens, eldens_disp = convert_qc_cube(input_file, params)
            if not params.nospin:
                input_file = \
                    os.path.join(input_dir, f'{job_name}.spindens.cube')
                spdens, spdens_disp = convert_qc_cube(input_file, params)
                dens = np.concatenate((eldens[:, :, :, np.newaxis],
                                       spdens[:, :, :, np.newaxis]), axis=3)
            else:
                dens = eldens
            if 'predict' in params.exec_mode:
                output_file = os.path.join(output_dir, f'{job_name}.npy')
                np.save(output_file, dens.astype(np.float32))
                continue
            output_file = os.path.join(output_dir, f'{job_name}_1.npy')
            np.save(output_file, dens.astype(np.float32))
            if not params.nospin:
                max_disp = [max(i) for i in zip(eldens_disp, spdens_disp)]
            else:
                max_disp = eldens_disp
            random_states = random.sample(unique_cube_rot_states, aug)
            for j in range(aug):
                rot_state = random_states[j]
                disp_state = [random.randint(-i, i) for i in max_disp]

                aug_eldens = np.roll(eldens, disp_state, axis=(0, 1, 2))
                aug_eldens = np.rot90(aug_eldens, rot_state[0], axes=(1, 2))
                aug_eldens = np.rot90(aug_eldens, rot_state[1], axes=(2, 0))
                aug_eldens = np.rot90(aug_eldens, rot_state[2], axes=(0, 1))

                if not params.nospin:
                    aug_spdens = \
                        np.roll(spdens, disp_state, axis=(0, 1, 2))
                    aug_spdens = \
                        np.rot90(aug_spdens, rot_state[0], axes=(1, 2))
                    aug_spdens = \
                        np.rot90(aug_spdens, rot_state[1], axes=(2, 0))
                    aug_spdens = \
                        np.rot90(aug_spdens, rot_state[2], axes=(0, 1))
                if params.nospin:
                    aug_dens = aug_eldens
                else:
                    aug_dens = \
                        np.concatenate((aug_eldens[:, :, :, np.newaxis],
                                        aug_spdens[:, :, :, np.newaxis]),
                                       axis=3)
                output_file = os.path.join(output_dir, f'{job_name}_{j+2}.npy')
                np.save(output_file, aug_dens.astype(np.float32))


def convert_qc_cube(input_file, params):
    """ Read a high resolution electron or spin density from a .cube-file in
    the input_dir directory, interpolate density on a params.extend_cube -
    times finer grid than needed for training (so that the density is
    integrated to the correct charge), reduce resolution using averaging to
    keep correct integrated charge. Finally, calculate the number of grid
    points the density can be shifted by within the training-ready grid not
    to be cropped by the border of the image.

    Args:
        input_file (str): path to a .cube-file
        params (munch.Munch): Command line parameters

    Return:
        interp_cube_vals (numpy.array): interpolated density
        max_disp (List[int,int,int]): max possible displacements in numbers of
                                      grid points

    """
    extend_cube = params.extend_cube
    n_target = params.cube_n
    n_target_ext = params.cube_n * extend_cube
    nxt, nyt, nzt = n_target_ext, n_target_ext, n_target_ext
    spacing_bohr = (params.cube_spacing * 1.889726
                    * (n_target - 1) / (n_target_ext - 1))
    xmax = (nxt - 1) / 2.0 * spacing_bohr
    ymax = (nyt - 1) / 2.0 * spacing_bohr
    zmax = (nzt - 1) / 2.0 * spacing_bohr

    gridx, gridy, gridz = np.meshgrid(np.linspace(-xmax, xmax, nxt),
                                      np.linspace(-ymax, ymax, nyt),
                                      np.linspace(-zmax, zmax, nzt),
                                      indexing='ij')
    points = np.hstack((gridx.reshape((-1, 1)), gridy.reshape((-1, 1)),
                        gridz.reshape((-1, 1))))
    with open(input_file, 'r') as inp_file:
        inp_file.readline()
        inp_file.readline()
        n_atoms, x0, y0, z0 = [float(j) for j in inp_file.readline().split()]
        nx, dx, x2, x3 = [float(j) for j in inp_file.readline().split()]
        ny, y1, dy, y3 = [float(j) for j in inp_file.readline().split()]
        nz, z1, z2, dz = [float(j) for j in inp_file.readline().split()]
        atom_coord = []
        for _ in range(int(n_atoms)):
            atom_coord.append(inp_file.readline())
        cube_vals = np.zeros((int(nx), int(ny), int(nz)))
        for x in range(int(nx)):
            for y in range(int(ny)):
                z_list_cube = []
                for z in range(int(math.ceil(int(nz) / 6.0))):
                    z_list_cube += [float(j) for j
                                    in inp_file.readline().split()]
                z_list_cube = np.array(z_list_cube)
                cube_vals[x, y, :] = z_list_cube

    gx = np.arange(0, int(nx)) * dx + x0
    gy = np.arange(0, int(ny)) * dy + y0
    gz = np.arange(0, int(nz)) * dz + z0
    cube_interpolator = interpolator((gx, gy, gz),
                                     cube_vals,
                                     bounds_error=False,
                                     fill_value=0.0)
    interp_cube_vals = cube_interpolator(points).reshape((nxt, nyt, nzt))
    interp_cube_vals = bin_average(interp_cube_vals, params.extend_cube)
    spacing_bohr = params.cube_spacing * 1.889726
    max_disp = [math.ceil((xmax + x0 if (xmax + x0) > 0 else 0)/spacing_bohr),
                math.ceil((ymax + y0 if (ymax + y0) > 0 else 0)/spacing_bohr),
                math.ceil((zmax + z0 if (zmax + z0) > 0 else 0)/spacing_bohr)]

    return interp_cube_vals, max_disp
