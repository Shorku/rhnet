"""Set environment variables, prepare directories, initialize logger"""

import os
import subprocess

import tensorflow as tf
import dllogger as logger

from datetime import datetime
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend


def set_flags(params):
    """ Set environmental variables, turn on AMP and XLA if specified

    Args:
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    # Disable JIT caching
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    # Set TensorFlow logging to minimal level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Set GPU to use threads dedicated to this device
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    # Try to optimize utilization of memory bandwidth
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    # Enable the use of the non-fused Winograd convolution algorithm
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    # Enable auto-tuning process to select the fastest convolution algorithms
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
    # Enable Tensor Core math for float32 matrix multiplication operations
    os.environ["TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32"] = '1'
    # Tensor Core math for float32 convolutions
    os.environ["TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32"] = '1'

    if params.use_xla:
        tf.config.optimizer.set_jit(True)

    if params.use_amp:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


def prepare_model_dir(params):
    """ Create or overwrite a folder to store checkpoints and the training res

    Args:
        params (munch.Munch): Command line parameters

    Return:
        None

    """
    model_dir = params.model_dir
    if ('train' in params.exec_mode) and (not (params.resume_training
                                               or params.load_model)):
        subprocess.run(['rm', '-rf', model_dir])
    os.makedirs(model_dir, exist_ok=True)

    return model_dir


def get_logger(params):
    """ Create log folder if needed, initialize dllogger for a general logging
    and logging for a detailed evaluation logging in the 'evaluate' exec_mode

    Args:
        params (munch.Munch): Command line parameters

    Return:
        logger (module)

    """

    log_dir = params.log_dir
    if log_dir != '.':
        os.makedirs(log_dir, exist_ok=True)
    if params.log_name:
        log_name = f'{params.log_name}.log'
    else:
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        log_name = f'{now}.log'
    log_path = os.path.join(log_dir, log_name)
    backends = [StdOutBackend(Verbosity.VERBOSE)]
    if params.log_dir:
        backends += [JSONStreamBackend(Verbosity.VERBOSE, log_path)]
    logger.init(backends=backends)

    return logger
