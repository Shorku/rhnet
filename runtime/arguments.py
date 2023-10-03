"""Command line argument parsing"""

import argparse
from munch import Munch

PARSER = argparse.ArgumentParser(description="RhNet")

PARSER.add_argument('--exec_mode',
                    choices=['train', 'evaluate', 'predict', 'error_analysis',
                             'train_and_error_analysis'],
                    type=str,
                    default='train',
                    help="""Execution mode of running the model""")

# General code execution related parameters

PARSER.add_argument('--model_dir',
                    type=str,
                    default='results',
                    help="""Output directory for saved model and
                    checkpoints""")

PARSER.add_argument('--data_dir',
                    type=str,
                    required=True,
                    help="""Directory with the dataset""")

PARSER.add_argument('--data_csv',
                    type=str,
                    default='experimental_dataset.csv',
                    help="""CSV file with experimental part of the dataset
                    relative to data_dir""")

PARSER.add_argument('--to_pred_csv',
                    type=str,
                    default='to_predict_ranges.csv',
                    help="""CSV file with intended for prediction parameters
                    ranges""")

PARSER.add_argument('--api',
                    choices=['builtin', 'custom'],
                    type=str,
                    default='builtin',
                    help="""Whether to use Keras builtin or custom training
                    loop""")

PARSER.add_argument('--store_density',
                    choices=['ram', 'file', 'cache'],
                    type=str,
                    default='cache',
                    help="""Where to store density images""")

PARSER.add_argument('--store_sparse', '--sparse', dest='store_sparse',
                    action='store_true', help="""Compress images""")

PARSER.add_argument('--parallel_preproc',
                    type=int,
                    default=0,
                    help="""Perform preprocessing in parallel""")

PARSER.add_argument('--timeout',
                    type=int,
                    default=10,
                    help="""Delay (sec) to sync data generators init""")

PARSER.add_argument('--log_dir',
                    type=str,
                    default='.',
                    help="""Output directory for training logs""")

PARSER.add_argument('--log_name',
                    type=str,
                    default=None,
                    help="""Suffix for different log files""")

PARSER.add_argument('--log_every',
                    type=int,
                    default=100,
                    help="""Log performance every n steps""")

PARSER.add_argument('--use_amp', '--amp', dest='use_amp',
                    action='store_true', help="""Train using TF-AMP""")

PARSER.add_argument('--use_xla', '--xla', dest='use_xla', action='store_true',
                    help="""Train using XLA""")

# Model definition parameters

PARSER.add_argument('--model',
                    type=str,
                    default='linear',
                    help="""Model index""")

PARSER.add_argument('--activation',
                    choices=['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu'],
                    type=str,
                    default='relu',
                    help="""Activation function""")

PARSER.add_argument('--pooling',
                    choices=['av', 'max'],
                    type=str,
                    default='av',
                    help="""Pooling function""")

PARSER.add_argument('--save_model', dest='save_model',
                    action='store_true', help="""Save model after training""")

PARSER.add_argument('--resume_training', dest='resume_training',
                    action='store_true',
                    help="""Resume training from a checkpoint""")

PARSER.add_argument('--load_model', dest='load_model',
                    action='store_true',
                    help="""Resume training from a saved_model""")

PARSER.add_argument('--use_only_mw', '--mw', dest='use_only_mw',
                    action='store_true', help="""Use only Mw and drop Mn""")

PARSER.add_argument('--use_only_amorph', '--amorph', dest='use_only_amorph',
                    action='store_true', help="""Drop semi-crystalline poly""")

PARSER.add_argument('--use_tg', '--tg', dest='use_tg',
                    action='store_true', help="""Use polymer Tg feature""")

PARSER.add_argument('--use_dens', '--dens', dest='use_dens',
                    action='store_true', help="""Use polymer density feature"""
                    )

PARSER.add_argument('--use_solvent_boil_temp', '--bt', dest='use_bt',
                    action='store_true', help="""Use solvent Tb feature""")

PARSER.add_argument('--use_solvent_cryt_point', '--ctp', dest='use_ctp',
                    action='store_true', help="""Use solvent critical point
                    feature""")

# Data related parameters

PARSER.add_argument('--augment',
                    type=int,
                    default=25,
                    help="""Number of shifted and rotated densities per
                    geometry. Note: the option affects only indexing,
                    augmentation itself is done elsewhere""")

PARSER.add_argument('--augment_onthefly', dest='augment_onthefly',
                    action='store_true',
                    help="""Shift and rotate images on the fly""")

PARSER.add_argument('--nonint_shift', dest='nonint_shift',
                    action='store_true',
                    help="""Use non-integer steps shifting images """)

PARSER.add_argument('--even_ratios_distrib', '--make_even', dest='make_even',
                    action='store_true', help="""Tune sampling weights
                    to equalize impacts of the examples with medium and high
                    solvent content""")

PARSER.add_argument('--analysis_n',
                    type=int,
                    default=None,
                    help="""Number of samples to be used in error analysis.
                    Use full table if not specified""")

# Evaluation split parameters

PARSER.add_argument('--eval_split',
                    type=int,
                    default=None,
                    help="""Turn on validation during training. Randomly sample
                    1/val_split examples into validation set""")

PARSER.add_argument('--eval_define',
                    type=str,
                    default=None,
                    help="""Turn on validation during training. Filepath to a
                    .csv file defining polymer-solvent pairs to be used in
                    evaluation (validation). Will override --eval_split
                    option.""")

PARSER.add_argument('--fold',
                    type=int,
                    default=None,
                    help="""Turn on cross-validation during training. Randomly
                    split the dataset into a defined number of parts for
                    cross-validation. None - no cross-val. Will override
                    --eval_split and --eval_define""")

PARSER.add_argument('--holdout_define',
                    type=str,
                    default=None,
                    help="""Exclude the data associated with the defined
                    polymer-solvent pairs from the currently used dataset.
                    Filepath to a .csv file defining polymer-solvent pairs
                    to be excluded.""")

PARSER.add_argument('--restrict_to_define',
                    type=str,
                    default=None,
                    help="""Include in the currently used dataset only the data
                    associated with the defined polymer-solvent pairs. Filepath
                    to a .csv file defining polymer-solvent pairs to be
                    included.""")

# Training/evaluation parameters

PARSER.add_argument('--batch_size',
                    type=int,
                    default=1,
                    help="""Size of each minibatch per GPU""")

PARSER.add_argument('--learning_rate',
                    type=float,
                    default=0.0001,
                    help="""Learning rate coefficient for AdamOptimizer""")

PARSER.add_argument('--learning_rate_decay',
                    type=float,
                    default=0,
                    help="""Learning rate exponential decay coefficient """)

PARSER.add_argument('--learning_rate_decay_steps',
                    type=float,
                    default=600000,
                    help="""Learning rate exponential decay rate""")

PARSER.add_argument('--max_steps',
                    type=int,
                    default=0,
                    help="""Maximum number of steps (batches) in training
                    If --max_steps 0 consume the whole set""")

PARSER.add_argument('--epochs',
                    type=int,
                    default=1,
                    help="""Number of epochs used in training""")

PARSER.add_argument('--evaluate_every',
                    type=int,
                    default=1,
                    help="""Evaluate performance (validate) every n epochs""")

PARSER.add_argument('--checkpoint_every',
                    type=int,
                    default=0,
                    help="""Save checkpoint every ... epochs""")

PARSER.add_argument('--initial_epoch',
                    type=int,
                    default=0,
                    help="""Initial epoch for training""")

PARSER.add_argument('--prune_model',
                    type=float,
                    default=None,
                    help="""Define final sparsity for model pruning""")

PARSER.add_argument('--prune_start',
                    type=int,
                    default=None,
                    help="""Define model pruning starting step""")

PARSER.add_argument('--prune_end',
                    type=int,
                    default=None,
                    help="""Define model pruning final step""")

# Regularization parameters

PARSER.add_argument('--dnn_dropout',
                    type=float,
                    default=None,
                    help="""Use dropout regularization for dense part""")

PARSER.add_argument('--cnn_dropout',
                    type=float,
                    default=None,
                    help="""Use dropout regularization for conv part""")

PARSER.add_argument('--dnn_l2',
                    type=float,
                    default=0,
                    help="""Use l2 regularization for dense part""")

PARSER.add_argument('--cnn_l2',
                    type=float,
                    default=0,
                    help="""Use l2 regularization for conv part""")

# Miscellaneous

PARSER.add_argument('--zero_density_test', '--zdt', dest='zdt',
                    action='store_true', help="""Perform a second run of error
                    analysis using zero densities""")


def parse_args(flags):
    return Munch({
        'exec_mode': flags.exec_mode,
        # General code execution related parameters
        'model_dir': flags.model_dir,
        'data_dir': flags.data_dir,
        'data_csv': flags.data_csv,
        'to_pred_csv': flags.to_pred_csv,
        'api': flags.api,
        'store_density': flags.store_density,
        'store_sparse': flags.store_sparse,
        'parallel_preproc': flags.parallel_preproc,
        'timeout': flags.timeout,
        'log_dir': flags.log_dir,
        'log_name': flags.log_name,
        'log_every': flags.log_every,
        'use_amp': flags.use_amp,
        'use_xla': flags.use_xla,
        # Model definition parameters
        'model': flags.model,
        'activation': flags.activation,
        'pooling': flags.pooling,
        'save_model': flags.save_model,
        'resume_training': flags.resume_training,
        'load_model': flags.load_model,
        'use_only_mw': flags.use_only_mw,
        'use_only_amorph': flags.use_only_amorph,
        'use_tg': flags.use_tg,
        'use_dens': flags.use_dens,
        'use_bt': flags.use_bt,
        'use_ctp': flags.use_ctp,
        # Data related parameters
        'augment': flags.augment,
        'augment_onthefly': flags.augment_onthefly,
        'nonint_shift': flags.nonint_shift,
        'make_even': flags.make_even,
        'analysis_n': flags.analysis_n,
        # Evaluation split parameters
        'eval_split': flags.eval_split,
        'eval_define': flags.eval_define,
        'fold': flags.fold,
        'holdout_define': flags.holdout_define,
        'restrict_to_define': flags.restrict_to_define,
        # Training/evaluation parameters
        'batch_size': flags.batch_size,
        'learning_rate': flags.learning_rate,
        'learning_rate_decay': flags.learning_rate_decay,
        'learning_rate_decay_steps': flags.learning_rate_decay_steps,
        'max_steps': flags.max_steps,
        'epochs': flags.epochs,
        'evaluate_every': flags.evaluate_every,
        'checkpoint_every': flags.checkpoint_every,
        'initial_epoch': flags.initial_epoch,
        'prune_model': flags.prune_model,
        'prune_start': flags.prune_start,
        'prune_end': flags.prune_end,
        # Regularization parameters
        'dnn_dropout': flags.dnn_dropout,
        'cnn_dropout': flags.cnn_dropout,
        'dnn_l2': flags.dnn_l2,
        'cnn_l2': flags.cnn_l2,
        # Miscellaneous
        'zdt': flags.zdt,
    })
