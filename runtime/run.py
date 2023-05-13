"""Implementation of the training, evaluation and predictions loops"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf


def train(params, model, dataset, logger, fold_no=None):
    """Wrapper for RhNet training loop (with optional evaluation): define
    optimizer and checkpoint, run the chosen (builtin of custom) training loop
    and save the trained model

    Args:
        params (munch.Munch): Command line parameters
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)
        logger (module): dllogger (for custom loop)
        fold_no (int): dataset split sequence number in n-fold cross-validation

    Return:
        None

    """
    # Define learning rate decay
    if params.learning_rate_decay:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            params.learning_rate, decay_steps=params.learning_rate_decay_steps,
            decay_rate=params.learning_rate_decay, staircase=True)
    else:
        lr_schedule = params.learning_rate
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    if params.use_amp:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer,
                                                                dynamic=True)
    # Define checkpoint and read saved data if needed
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    if (not params.load_model) and params.resume_training:
        checkpoint.restore(tf.train.latest_checkpoint(params.model_dir))
        model.load_weights(os.path.join(params.model_dir, "checkpoint"))
    # Choose and run training loop
    if params.api == 'builtin':
        train_builtin(params, model, dataset, optimizer, fold_no)
    elif params.api == 'custom':
        train_custom(params, model, dataset, optimizer, checkpoint,
                     logger, fold_no)
    if params.save_model:
        model.save(os.path.join(params.model_dir, 'saved_model'))


def evaluate(params, model, dataset, logger, epoch, fold_no):
    """Wrapper for RhNet evaluation loop: optionally read checkpoint and run
    the chosen (builtin of custom) evaluation loop

    Args:
        params (munch.Munch): Command line parameters
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)
        logger (module): dllogger (for custom loop)
        epoch (int): epoch sequence number from training loop to be
            logged (for custom loop)
        fold_no (int): dataset split sequence number in n-fold cross-validation

    Return:
        None

    """
    if (not params.fold and
            not params.eval_split and
            params.exec_mode != 'evaluate'):
        print("No split specified for evaluation. Use --fold or --eval_split.")

    if params.exec_mode == 'evaluate' \
            and params.resume_training\
            and (not params.load_model):
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(params.model_dir)).\
            expect_partial()

    if params.api == 'builtin':
        evaluate_builtin(model, dataset, fold_no)
    elif params.api == 'custom':
        evaluate_custom(model, dataset, logger, epoch, fold_no)


def error_analysis(params, model, dataset):
    """Wrapper for RhNet error analysis loop: optionally read checkpoint and
    run the chosen (builtin of custom) error analysis loop over the whole
    training dataset and save the whole dataset to
    full_table_{params.log_name}.csv and predicted values to
    detailed_pred_{params.log_name}.csv. On demand repeat
    evaluation using zero CNN output.

    Args:
        params (munch.Munch): Command line parameters
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)

    Return:
        None

    """
    if params.exec_mode == 'error_analysis' \
            and params.resume_training\
            and (not params.load_model):
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(params.model_dir)).\
            expect_partial()
    if params.api == 'builtin':
        error_analysis_builtin(params, model, dataset)
    elif params.api == 'custom':
        error_analysis_custom(params, model, dataset)


def train_builtin(params, model, dataset, optimizer, fold_no):
    """Functional API training loop with optional evaluation

    Args:
        params (munch.Munch): Command line parameters
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim),
                              label_shape)
        optimizer(tf.keras.optimizers.Optimizer): keras optimizer instance
        fold_no (int): dataset split sequence number in n-fold cross-validation

    Return:
        None

    """
    max_steps = params.max_steps if params.max_steps else None
    tb_log_dir = params.log_dir + "/" + params.log_name
    callbacks = [tf.keras.callbacks.TensorBoard(
        log_dir=tb_log_dir, histogram_freq=1)]

    if params.checkpoint_every > 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(params.model_dir, "checkpoint"),
            save_weights_only=True,
            save_freq=params.checkpoint_every))

    if params.evaluate_every > 0 and \
            (params.fold or params.eval_split or params.eval_define):
        validation_data = dataset.data_gen(is_evaluation=True, fold_no=fold_no)
    else:
        validation_data = None

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MeanSquaredError(name='mse_loss'),
                  metrics=[tf.keras.metrics.MeanSquaredError(name='mse_loss'),
                           tf.keras.metrics.MeanAbsoluteError(name='mаe_loss')
                           ],
                  run_eagerly=False)

    model.fit(x=dataset.data_gen(fold_no=fold_no),
              epochs=params.epochs,
              steps_per_epoch=max_steps,
              validation_data=validation_data,
              validation_freq=params.evaluate_every,
              callbacks=callbacks)


def train_custom(params, model, dataset, optimizer, checkpoint,
                 logger, fold_no=None):
    """Custom training loop with optional evaluation

    Args:
        params (munch.Munch): Command line parameters
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)
        optimizer(tf.keras.optimizers.Optimizer): keras optimizer instance
        checkpoint(tf.train.Checkpoint): checkpoint instance
        logger (module): dllogger
        fold_no (int): dataset split sequence number in n-fold cross-validation

    Return:
        None

    """
    max_steps = params.max_steps
    epochs = params.epochs

    mse_metr = tf.keras.metrics.MeanSquaredError(name='mse_loss')

    @tf.function
    def train_step(features, frac_true):
        with tf.GradientTape() as tape:
            frac_pred = model(features, True)
            mse_loss = tf. \
                keras. \
                losses. \
                MeanSquaredError(name='mse_loss')
            mse = mse_loss(frac_true, frac_pred)
            mse_metr(frac_true, frac_pred)

            if params.dnn_l2 or params.cnn_l2:
                l2 = 0.0
                if params.dnn_l2:
                    l2 += params.dnn_l2 \
                          * tf.add_n([tf.nn.l2_loss(v)
                                      for v in
                                      model.layers[1].trainable_weights])
                if params.cnn_l2:
                    l2 += params.cnn_l2 \
                          * tf.add_n([tf.nn.l2_loss(v)
                                      for v in
                                      model.layers[0].trainable_weights])
                if params.use_amp:
                    loss = tf.cast(mse, tf.float32) + l2
                    loss = tf.cast(loss, tf.float16)
                else:
                    loss = mse + l2
            else:
                loss = mse
            if params.use_amp:
                loss = optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        if params.use_amp:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    step = 0

    for epoch in range(epochs):
        for i, (mixtures, fractions) \
                in enumerate(dataset.data_gen(fold_no=fold_no)):
            train_step(mixtures, fractions)
            if i % params.log_every == 0:
                logger.log(step=(epoch + 1, i, max(step, max_steps)),
                           data={"train_mse_loss":
                                 float(mse_metr.result())})
                mse_metr.reset_states()

            if epoch == 0:
                step += 1

            if max_steps != 0 and i >= max_steps:
                break
        if (params.evaluate_every > 0
                and (params.fold or params.eval_split or params.eval_define)
                and (epoch + 1) % params.evaluate_every == 0):
            evaluate(params, model, dataset, logger, epoch, fold_no)
        if (params.checkpoint_every > 0
                and (epoch + 1) % params.checkpoint_every == 0):
            checkpoint.save(file_prefix=os.path.join(params.model_dir,
                                                     "checkpoint"))

    logger.flush()


def evaluate_builtin(model, dataset, fold_no):
    """Functional API evaluation loop

    Args:
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): training or evaluation dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)
        fold_no (int): dataset split sequence number in n-fold cross-validation

    Return:
        None

    """
    model.evaluate(x=dataset.data_gen(is_evaluation=True, fold_no=fold_no))


def evaluate_custom(model, dataset, logger, epoch, fold_no):
    """Custom evaluation loop

    Args:
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): training or evaluation dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)
        logger (module): dllogger
        epoch (int): epoch sequence number from training loop to be logged
        fold_no (int): dataset split sequence number in n-fold cross-validation

    Return:
        None

    """
    mse_eval = tf.keras.metrics.MeanSquaredError(name='mse_eval')

    @tf.function
    def evaluation_step(features, frac_true):
        frac_pred = model(features, False)
        mse_eval(frac_true, frac_pred)

    for i, (mixtures, fractions) in enumerate(dataset.
                                              data_gen(is_evaluation=True,
                                                       fold_no=fold_no)):
        evaluation_step(mixtures, fractions)

    logger.log(step=(epoch + 1, epoch + 1),
               data={"eval_mse_loss": float(mse_eval.result())})
    logger.flush()


def error_analysis_builtin(params, model, dataset):
    """Functional API error analysis loop. Loop over the whole training dataset
    and save the whole dataset to full_table_{params.log_name}.csv and
    predicted values to detailed_pred_{params.log_name}.csv. On demand repeat
    evaluation using zero CNN output and save the whole dataset to
    full_table_zeros_{params.log_name}.csv and predicted values to
    detailed_pred_zeros_{params.log_name}.csv.

    Args:
        params (munch.Munch): Command line parameters
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)

    Return:
        None

    """
    y_hat = model.predict(x=dataset.data_gen(is_analysis=True))
    log_path = os.path.join(params.log_dir,
                            f'detailed_pred_{params.log_name}.csv')
    pd.DataFrame(y_hat, columns=['y_hat']).to_csv(log_path, index=False)
    # Repeat evaluation with zero CNN output
    if params.zdt:
        y_hat = model.predict(
            x=dataset.data_gen(is_analysis=True, with_zeros=True))
        log_path = os.path.join(params.log_dir,
                                f'detailed_pred_zeros_{params.log_name}.csv')
        pd.DataFrame(y_hat, columns=['y_hat']).to_csv(log_path, index=False)


def error_analysis_custom(params, model, dataset):
    """Custom error analysis loop. Loop over the whole training dataset
    and save the whole dataset to full_table_{params.log_name}.csv and
    predicted values to detailed_pred_{params.log_name}.csv. On demand repeat
    evaluation using zero CNN output and save the whole dataset to
    full_table_zeros_{params.log_name}.csv and predicted values to
    detailed_pred_zeros_{params.log_name}.csv.

    Args:
        params (munch.Munch): Command line parameters
        model (tf.keras.Model): RhNet instance
        dataset (tensorflow.data.Dataset): dataset with
            output_signature=((cube_dim, cube_dim, exp_param_dim), label_shape)

    Return:
        None

    """
    @tf.function
    def step(features):
        return model(features, training=False)

    y = np.concatenate([np.concatenate([step(mixtures).numpy(),
                                        fractions.numpy()], axis=1)
                        for (mixtures, fractions)
                        in dataset.data_gen(is_analysis=True)],
                       axis=0)

    log_path = os.path.join(params.log_dir,
                            f'detailed_pred_{params.log_name}.csv')
    pd.DataFrame(y, columns=['y_hat', 'y']).to_csv(log_path, index=False)
    # Repeat evaluation with zero CNN output
    if params.zdt:
        y = np.concatenate([np.concatenate([step(mixtures).numpy(),
                                            fractions.numpy()], axis=1)
                            for (mixtures, fractions)
                            in dataset.data_gen(is_analysis=True,
                                                with_zeros=True)],
                           axis=0)

        log_path = os.path.join(params.log_dir,
                                f'detailed_pred_zeros_{params.log_name}.csv')
        pd.DataFrame(y, columns=['y_hat', 'y']).to_csv(log_path, index=False)
