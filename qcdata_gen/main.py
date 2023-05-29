"""Entry point of the automated electron density images generation.

Example:
    ...

"""


from runtime.arguments import PARSER, parse_args
from runtime.run import generate_conf, generate_cube
from runtime.setup import set_env, prepare_dir


def main():
    """
    Starting point of the application generating electron density set
    """
    params = parse_args(PARSER.parse_args())
    set_env(params)
    prepare_dir(params)
    # Read .mol files and generate/optimize conformations
    if 'conf' in params.exec_mode:
        params.mol_dir = generate_conf(params)
    # Read .xyz files and generate corresponding electron densities
    if 'cube' in params.exec_mode:
        generate_cube(params)


if __name__ == '__main__':
    main()
