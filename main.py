"""Entry point of the RhNet training/evaluation/prediction script.

Example:
    $ python3 main.py --exec_mode train --data_dir /data ...

"""

import os

import tensorflow as tf

from model.rhnet import model_choice
from runtime.run import train, evaluate, error_analysis, predict
from runtime.setup import get_logger, set_flags, prepare_model_dir
from runtime.arguments import PARSER, parse_args
from data_loading.data_loader import DatasetFit, DatasetPred


def main():
    """
    Starting point of the RhNet application
    """
    params = parse_args(PARSER.parse_args())
    set_flags(params)
    model_dir = prepare_model_dir(params)
    params.model_dir = model_dir
    if params.exec_mode != 'predict':
        logger = get_logger(params)

    # TODO workaround for GPU-trained models used for inference on CPU
    if params.load_model:
        model = tf.keras.models.load_model(
            os.path.join(params.model_dir, 'saved_model'))
    else:
        model = model_choice(params)

    if 'predict' in params.exec_mode:
        dataset = DatasetPred(params)
    else:
        dataset = DatasetFit(params)

    if 'train' in params.exec_mode:
        # Training with n-fold validation
        if params.fold:
            for i in range(params.fold):
                print('Doing ', i, 'fold')
                train(params, model, dataset, logger, i)
                model = model_choice(params.model)
        # Training without validation or with a randomly sampled validation set
        else:
            train(params, model, dataset, logger)

    if 'evaluate' in params.exec_mode:
        evaluate(params, model, dataset, logger, 0, False)

    if 'predict' in params.exec_mode:
        predict(params, model, dataset)

    if 'error_analysis' in params.exec_mode:
        # similar to evaluate, but loops over the whole dataset and logs
        # predicted values alongside the true values,
        # similar to predict, but loops over the training dataset
        error_analysis(params, model, dataset)


if __name__ == '__main__':
    main()
