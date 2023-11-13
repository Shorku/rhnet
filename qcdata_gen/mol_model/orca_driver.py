import os
import subprocess


from mol_model.params import orca_xtb, orca_dft, orca_dft_cube
from utils.utils import dump_temp_files


def orca_tmp_clear(data_dir, compress):
    extensions_del = ['.inp', '.txt', '.engrad', '.opt', '.xyz',
                      '.xtbrestart', '.densities', '.tmp', '.cube']
    extensions_dump = ['.out', '.gbw']

    for ext in extensions_dump:
        dump_temp_files(data_dir, ext, compress=compress)

    for ext in extensions_del:
        dump_temp_files(data_dir, ext, compress=False, remove=True)


def orca_gen(input_dir, job, jobs_files, thresh_keep, level, pal):
    qc_input_string = orca_xtb(pal) if level == 'semi' else orca_dft(pal)
    orca_path = f'''{os.environ['ORCA']}/orca'''
    e_dft = []
    for file in jobs_files:
        file_path = os.path.join(input_dir, file)
        e_dft.append(0)
        with open(file_path, 'r') as xyz:
            xyz_lines = xyz.readlines()
            if len(xyz_lines[2:]) == 1:
                orca_xyz_name = f'{os.path.splitext(file)[0]}.{level}.xyz'
                with open(orca_xyz_name, 'w') as out_xyz:
                    out_xyz.write(''.join(xyz_lines))
                break
            xyz_block = ''.join(xyz_lines[2:])
            input_string = qc_input_string.format(xyz_block)
        orca_job_name = f'{os.path.splitext(file)[0]}.{level}'
        orca_inp = f'{orca_job_name}.inp'
        orca_out = f'{orca_job_name}.out'
        with open(orca_inp, 'w') as inp:
            inp.write(input_string)
        cli_opts = [orca_path, orca_inp]
        with open(orca_out, 'w') as out:
            subprocess.run(cli_opts, stdout=out)
        cli_opts = ['grep', 'ORCA TERMINATED NORMALLY', orca_out]  # TODO Fix!
        conv_flag = float(subprocess.run(cli_opts).returncode)
        conv_flag = 1.0 if conv_flag == 0.0 else 0.0
        with open(orca_out) as out:
            for j in out:
                if 'FINAL SINGLE POINT ENERGY' in j:
                    e_dft[-1] = float(j.split()[4]) * conv_flag
    e_dft_min = min(e_dft)
    conformers = []
    for file, e in zip(jobs_files, e_dft):
        orca_xyz_name = f'{os.path.splitext(file)[0]}.{level}.xyz'
        if (e - e_dft_min) * 2625.5 < thresh_keep:
            with open(orca_xyz_name, 'r') as xyz:
                conformers.append(xyz.read())
    compress = True if level == 'dft' else False
    orca_tmp_clear('.', compress)

    return conformers


def orca_cube(xyz_path, pal, ispolymer, params):
    n_target = params.cube_n
    qc_input_string = orca_dft_cube(pal, 3 if ispolymer else 1, n_target)
    orca_path = f'''{os.environ['ORCA']}/orca'''
    xyz_file = os.path.split(xyz_path)[1]
    orca_job_name = os.path.splitext(xyz_file)[0]
    orca_inp = f'{orca_job_name}.inp'
    orca_out = f'{orca_job_name}.out'

    with open(xyz_path, 'r') as xyz:
        xyz_block = ''.join(xyz.readlines()[2:])
        input_string = qc_input_string.format(os.path.splitext(xyz_file)[0],
                                              os.path.splitext(xyz_file)[0],
                                              xyz_block)
    with open(orca_inp, 'w') as inp:
        inp.write(input_string)
    cli_opts = [orca_path, orca_inp]
    with open(orca_out, 'w') as out:
        subprocess.run(cli_opts, stdout=out)

    eldens_cube_file = f'{orca_job_name}.eldens.cube'
    spdens_cube_file = f'{orca_job_name}.spindens.cube'

    return (os.path.abspath(eldens_cube_file),
            os.path.abspath(spdens_cube_file))


def orca_rescale_cubes(cube_files, params):
    extend_cube = params.extend_cube
    n_target = params.cube_n
    n_target_ext = params.cube_n * extend_cube
    spacing_bohr_ext = (params.cube_spacing * 1.889726
                        * (n_target - 1) / (n_target_ext - 1))
    orca_plot_path = f'''{os.environ['ORCA']}/orca_plot'''
    for cube_file in cube_files:
        cube_name = os.path.splitext(cube_file)[0]
        orca_job_name, plot_type = os.path.splitext(cube_name)
        cli_opts = [orca_plot_path, f'{orca_job_name}.gbw', '-i']

        with open(cube_file, 'r') as inp_file:
            for i in range(3):
                inp_file.readline()
            nx, dx, x2, x3 = [float(j) for j in
                              inp_file.readline().split()]
            ny, y1, dy, y3 = [float(j) for j in
                              inp_file.readline().split()]
            nz, z1, z2, dz = [float(j) for j in
                              inp_file.readline().split()]
        nx = round(dx * (nx - 1) / spacing_bohr_ext + 1)
        ny = round(dy * (ny - 1) / spacing_bohr_ext + 1)
        nz = round(dz * (nz - 1) / spacing_bohr_ext + 1)
        typ = 2 if plot_type == '.eldens' else 3
        interactive_input = f'1\n{typ}\ny\n4\n{nx} {ny} {nz}\n5\n7\n10\n11\n'
        subprocess.run(cli_opts,
                       input=interactive_input.encode(),
                       stdout=subprocess.DEVNULL)
