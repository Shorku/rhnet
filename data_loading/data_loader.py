""" Dataset class encapsulates the data loading"""

import gc
import os
import ast
import time
import sparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from scipy.ndimage import rotate, shift
from model.params import scale_params, image_dim, macro_dim


class Dataset:
    """Base class with common functionality needed to load, separate and
    prepare the data for training and prediction"""

    def __init__(self, params):
        # Initialize general parameters
        if not os.path.exists(params.data_dir):
            raise FileNotFoundError('Cannot find data dir: {}'.
                                    format(params.data_dir))
        self._data_dir = params.data_dir
        self._batch_size = params.batch_size
        self.log_dir = params.log_dir
        self.log_name = params.log_name
        self.aug_onthefly = params.augment_onthefly
        self.precision = tf.float16 if params.use_amp else tf.float32
        self.load_precision = np.float16 if params.use_amp else np.float32
        self.image_dim = image_dim()
        self.cube_dim = tf.TensorSpec(shape=self.image_dim,
                                      dtype=self.precision)
        self.macr_dim = tf.TensorSpec(shape=(macro_dim(params),),
                                      dtype=self.precision)
        self.labl_dim = tf.TensorSpec(shape=(1,),
                                      dtype=self.precision)
        self.use_only_mw = params.use_only_mw
        self.use_only_amorph = params.use_only_amorph
        self.use_tg = params.use_tg
        self.use_dens = params.use_dens
        self.use_bt = params.use_bt
        self.use_ctp = params.use_ctp
        # Get predefined scaling parameters
        self.data_scale = scale_params()
        # Load oligomers/solvents masses from polymer_mass.csv/solvent_mass.csv
        self.poly_mass, self.solv_mass = self._load_mass()
        # Make a list of selected macro features
        self.features = ['pressure', 'temperature']
        self.solv_macro_features = []
        self.poly_macro_features = []
        if not self.use_only_amorph:
            self.features = ['cryst'] + self.features
        if self.use_only_mw:
            self.features = ['mw'] + self.features
            self.poly_macro_features = self.poly_macro_features + ['mw']
        else:
            self.features = ['mn', 'mw'] + self.features
            self.poly_macro_features = self.poly_macro_features + ['mn', 'mw']
        if self.use_tg:
            self.features = self.features + ['tg']
            self.poly_macro_features = self.poly_macro_features + ['tg']
        if self.use_dens:
            self.features = self.features + ['dens']
            self.poly_macro_features = self.poly_macro_features + ['dens']
        if self.use_bt:
            self.features = self.features + ['bt']
            self.solv_macro_features = self.solv_macro_features + ['bt']
        if self.use_ctp:
            self.features = self.features + ['ct', 'cp']
            self.solv_macro_features = self.solv_macro_features + ['ct', 'cp']
        # Load solvents macroscopic features from solvent_macro_features.csv
        if self.solv_macro_features:
            self.macro_feature = self._load_macro(self.data_scale)
        # Define a place to store electron density loaded during run
        self.poly_cube_set = {}
        self.solv_cube_set = {}

    def _load_cube(self, name):
        """Load one electron density file, expects filename.npy as name"""
        cube = np.load(os.path.join(self._data_dir, 'cubes', name))
        cube[:, :, :, 0] = cube[:, :, :, 0] / self.data_scale.el_scale
        cube[:, :, :, 1] = np.clip(cube[:, :, :, 1], 0.0,
                                   1.0) / self.data_scale.sp_scale
        return cube

    def _load_cube_sparse(self, name):
        """Load one electron density file and return sparse tensor,
        expects filename.npy as name"""
        return sparse.COO.from_numpy(
            self._load_cube(name).astype(self.load_precision))

    def _load_cube_dense(self, name):
        """Load one electron density file and return dense tensor,
         expects filename.npy as name"""
        return self._load_cube(name).astype(self.load_precision)

    def _load_mass(self):
        """Load masses of compounds to Pandas dataframe"""
        poly_mass = pd.read_csv(os.path.join(self._data_dir,
                                             'polymer_mass.csv'),
                                dtype={'polymer': np.uint8,
                                       'cut': np.uint8,
                                       'poly_mass': np.float32})

        solv_mass = pd.read_csv(os.path.join(self._data_dir,
                                             'solvent_mass.csv'),
                                dtype={'solvent': np.uint8,
                                       'solv_mass': np.float32})

        return poly_mass, solv_mass

    def _load_macro(self, scale):
        """Load solvent macroscopic parameters to Pandas dataframe"""
        data_types = {i: np.float32 for i in self.solv_macro_features}
        data_types['solvent'] = np.uint8
        macro_feature = pd.read_csv(os.path.join(self._data_dir,
                                                 'solvent_macro_features.csv'),
                                    dtype=data_types,
                                    usecols=['solvent'] +
                                    self.solv_macro_features)
        if self.use_ctp:
            macro_feature['cp'] = macro_feature['cp']. \
                apply(np.log10).sub(scale.p_shift).div(scale.p_scale)
            macro_feature['ct'] = macro_feature['ct']. \
                sub(scale.t_shift).div(scale.t_scale)
        if self.use_bt:
            macro_feature['bt'] = macro_feature['bt']. \
                sub(scale.t_shift).div(scale.t_scale)
        macro_feature[self.solv_macro_features] = macro_feature[
            self.solv_macro_features].apply(self.load_precision)
        return macro_feature

    def _scale_exp_set(self, exp_set, use_columns):
        # TODO fix solvent macro features scaling
        feat_to_scale = list(set(use_columns) & {'dens', 'mn', 'mw'})
        exp_set.loc[:, feat_to_scale] = exp_set.loc[:, feat_to_scale]. \
            div(exp_set['poly_mass'], axis=0)
        feat_to_scale = list(set(use_columns) & {'mn', 'mw', 'dens',
                                                 'pressure', 'wa'})
        exp_set[feat_to_scale] = exp_set[feat_to_scale].apply(np.log10)
        feat_to_scale = ['mw'] if self.use_only_mw else ['mn', 'mw']
        exp_set[feat_to_scale] = exp_set[feat_to_scale]. \
            sub(self.data_scale.mnw_shift).div(self.data_scale.mnw_scale)
        exp_set['pressure'] = exp_set['pressure']. \
            sub(self.data_scale.p_shift).div(self.data_scale.p_scale)
        feat_to_scale = ['temperature',
                         'tg'] if self.use_tg else ['temperature']
        exp_set[feat_to_scale] = exp_set[feat_to_scale]. \
            sub(self.data_scale.t_shift).div(self.data_scale.t_scale)
        if self.use_dens:
            exp_set['dens'] = exp_set['dens']. \
                sub(self.data_scale.d_shift).div(self.data_scale.d_scale)
        feat_to_scale = list(set(use_columns) & {'mn', 'mw', 'tg', 'dens',
                                                 'pressure', 'temperature',
                                                 'wa'})
        exp_set[feat_to_scale] = exp_set[feat_to_scale]. \
            apply(self.load_precision)

    def _rotate_cube(self, cube):
        cube_rot = [random.random() * 360 for i in range(3)]
        cube_copy = np.copy(cube)
        for i in range(2):
            for angle, axes in zip(cube_rot, [(1, 0), (2, 0), (2, 1)]):
                cube_copy[:, :, :, i] = \
                    rotate(cube_copy[:, :, :, i], angle, axes=axes,
                           reshape=False, order=3, cval=0.0, prefilter=False)
        return cube_copy

    def _shift_cube(self, cube):
        margins = [[0, 0], [0, 0], [0, 0]]
        dim = cube.shape[0]
        profiles = [[0 for i in range(dim)],
                    [0 for i in range(dim)],
                    [0 for i in range(dim)]]
        for i in range(dim):
            profiles[0][i] = cube[i, :, :, 0].sum()
            profiles[1][i] = cube[:, i, :, 0].sum()
            profiles[2][i] = cube[:, :, i, 0].sum()
        for axis in range(3):
            for i in range(dim):
                if profiles[axis][i] != 0:
                    margins[axis][0] = -i
                    break
            for i in range(dim):
                if profiles[axis][dim - i - 1] != 0:
                    margins[axis][1] = i
                    break
        cube_copy = np.copy(cube)
        if self.nonint_shift:
            cube_shift = [random.uniform(*i) for i in margins]
            for i in range(2):
                cube_copy[:, :, :, i] = \
                    shift(cube_copy[:, :, :, i], cube_shift, cval=0,
                          prefilter=False, order=3)
        else:
            cube_shift = [random.randint(*i) for i in margins]
            for i in range(2):
                cube_copy[:, :, :, i] = \
                    np.roll(cube_copy[:, :, :, i], cube_shift, axis=(0, 1, 2))
        return cube_copy


class DatasetFit(Dataset):
    """Load, separate and prepare the data for training and analysis"""

    def __init__(self, params):
        # Initialize general parameters
        super().__init__(params)
        self.analysis_n = params.analysis_n
        self.store_density = params.store_density
        self.make_even = params.make_even
        self.data_csv = params.data_csv
        if params.augment_onthefly:
            self.aug = 1
            self.store_sparse = False
        else:
            self.aug = params.augment
            self.store_sparse = params.store_sparse
        if params.parallel_preproc:
            self.parallel_preproc = params.parallel_preproc
        else:
            self.parallel_preproc = self._batch_size
        self.timeout = params.timeout
        self.nonint_shift = params.nonint_shift
        self.fold = params.fold
        self.eval_split = params.eval_split
        self.eval_define = params.eval_define
        self.holdout_define = params.holdout_define
        self.restrict_to_define = params.restrict_to_define
        # Load number of available conformations from polymers.txt/solvents.txt
        poly_conf_num, solv_conf_num = self._load_num_conf()
        # Prepare dataset tables
        self.index_table, \
            self.exp_set = self._form_index_tables(poly_conf_num,
                                                   solv_conf_num)
        # Load electron density
        if self.store_density == 'ram':
            self._load_cubes()
        elif self.store_density == 'file':
            self.store_sparse = False
        # Get sampling number for a single experiment
        self.exp_mapper = self._get_exp_sampling(self.exp_set)
        self.exp_set = self.exp_set.set_index(['expno', 'cut'])
        # Split dataset (index tables) into validation and training sets
        if self.fold:
            self.index_table_folds = self._index_crossval_split(self.exp_set)
            if 'error_analysis' not in params.exec_mode:
                del self.index_table
                gc.collect()
        elif self.eval_split or self.eval_define:
            self.index_table_train, \
                self.index_table_eval = self._index_eval_split(self.exp_set)
            # actual training table will be sampled each epoch while
            # evaluation table is sampled here once per training run
            self.index_table_eval = self. \
                _index_table_sampling(self.index_table_eval)
            self.index_table_train = self.index_table_train. \
                reset_index(drop=True)
            self.index_table_eval = self.index_table_eval. \
                reset_index(drop=True)
            self.eval_sample_size = len(self.index_table_eval)
            # remove parent index table after split if it isn't needed anymore
            if 'error_analysis' not in params.exec_mode:
                del self.index_table
                gc.collect()
        # Drop some columns from index tables to save some memory
        try:
            self.index_table
        except AttributeError:
            pass
        else:
            self.index_table = self.index_table. \
                drop(['polymer', 'solvent'], axis=1)
        try:
            self.index_table_train
        except AttributeError:
            pass
        else:
            self.index_table_train = self.index_table_train. \
                drop(['polymer', 'solvent'], axis=1)
        try:
            self.index_table_eval
        except AttributeError:
            pass
        else:
            self.index_table_eval = self.index_table_eval. \
                drop(['polymer', 'solvent'], axis=1)
        self.sample_table = None
        self.train_sample_size = self._index_table_chooser()
        if 'error_analysis' in params.exec_mode:
            self.analysis_sample_size = \
                self.analysis_n if self.analysis_n else len(self.index_table)
        else:
            self.analysis_sample_size = 0

    def _load_experimental(self):
        """Load experimental data set to Pandas dataframe"""
        data_types = {i: np.float32 for i in self.poly_macro_features}
        data_types.update({'expno': np.uint16,
                           'polymer': np.uint8,
                           'solvent': np.uint8,
                           'pressure': np.float32,
                           'temperature': np.float32,
                           'cryst': self.load_precision,
                           'wa': np.float32})
        use_columns = ['expno', 'polymer', 'solvent', 'pressure',
                       'temperature', 'cryst', 'wa'] + self.poly_macro_features
        exp_set = pd.read_csv(os.path.join(self._data_dir, self.data_csv),
                              dtype=data_types, usecols=use_columns)
        if self.restrict_to_define:
            restrict_to_pairs = pd.read_csv(self.restrict_to_define,
                                            dtype={'polymer': np.uint8,
                                                   'solvent': np.uint8})
            exp_set = exp_set.merge(restrict_to_pairs,
                                    on=['polymer', 'solvent'])
        if self.holdout_define:
            holdout_pairs = pd.read_csv(self.holdout_define,
                                        dtype={'polymer': np.uint8,
                                               'solvent': np.uint8})
            exp_set = exp_set. \
                merge(holdout_pairs,
                      on=['polymer', 'solvent'],
                      how="left",
                      indicator=True). \
                query('_merge=="left_only"'). \
                drop(['_merge'], axis=1)

        exp_set = exp_set. \
            merge(self.poly_mass[['polymer', 'cut', 'poly_mass']],
                  on='polymer'). \
            merge(self.solv_mass[['solvent', 'solv_mass']],
                  on='solvent')

        exp_set['wa'] = (exp_set['wa'] *
                         exp_set['poly_mass'] /
                         exp_set['solv_mass'] /
                         (1 - exp_set['wa']))

        self._scale_exp_set(exp_set, use_columns)
        if self.use_only_amorph:
            exp_set = exp_set.drop(exp_set[exp_set.cryst > 0.001].index)
            exp_set = exp_set.drop(['cryst'], axis=1)
        exp_set = exp_set.drop(['poly_mass', 'solv_mass'], axis=1)

        return exp_set

    def _load_num_conf(self):
        """Load number of available conformers of monomers and solvents"""
        with open(os.path.join(self._data_dir, 'polymers.txt')) as f:
            poly_dict = ast.literal_eval(f.read())
        with open(os.path.join(self._data_dir, 'solvents.txt')) as f:
            solv_dict = ast.literal_eval(f.read())

        return poly_dict, solv_dict

    def _conf_comb(self, poly_dict, solv_dict):
        """Get combinations of a polymer/oligomer-cut/conf/shift-rot"""
        poly_conf_list, solv_conf_list = [], []

        for poly, conf in poly_dict.items():
            for cut, iconf in enumerate(conf):
                df = pd.DataFrame({'polymer': poly,
                                   'cut': cut + 1,
                                   'pconformer': [i for i
                                                  in range(1, iconf + 1)]}
                                  ).astype(np.uint8)
                poly_conf_list.append(df)
        poly_df = pd.concat(poly_conf_list, ignore_index=True)

        for solv, conf in solv_dict.items():
            df = pd.DataFrame({'solvent': solv,
                               'sconformer': [i for i
                                              in range(1, conf + 1)]}
                              ).astype(np.uint8)
            solv_conf_list.append(df)
        solv_df = pd.concat(solv_conf_list, ignore_index=True)

        aug_df = pd.DataFrame({'aug': [i for i
                                       in range(1, self.aug + 1)]}
                              ).astype(np.uint8)
        return (poly_df.merge(aug_df, how='cross').
                rename(columns={'aug': 'paug'}),

                solv_df.merge(aug_df, how='cross').
                rename(columns={'aug': 'saug'}))

    def _load_cube_conditional(self, name):
        """Load one electron density file and return dense or sparse tensor,
         expects filename.npy as name"""
        if self.aug_onthefly:
            cube = self._load_cube(name)
        else:
            if self.store_sparse:
                cube = self._load_cube_sparse(name)
            else:
                cube = self._load_cube_dense(name)
        return cube

    # TODO make it load only required images
    def _load_cubes(self):
        """Load electron density into RAM"""
        compound_set = {'p': self.index_table['polymer'].unique(),
                        's': self.index_table['solvent'].unique()}
        for i in os.listdir(os.path.join(self._data_dir, 'cubes')):
            if ('.npy' in i
                    and int(i.split('_')[1]) in compound_set[i.split('_')[0]]):
                cube = self._load_cube_conditional(i)
                if i.split('_')[0] == 'p':
                    self.poly_cube_set[i.split('.')[0]] = cube
                elif i.split('_')[0] == 's':
                    self.solv_cube_set[i.split('.')[0]] = cube

    def _cube_from_df(self, df_sample, df_exp):
        """Form dictionary key to retrieve array with electron density"""
        poly = 'p_{}'.format('_'.join([str(int(df_exp.at['polymer'])),
                                       str(df_sample.at['cut']),
                                       str(df_sample.at['pconformer']),
                                       str(df_sample.at['paug'])
                                       ]))
        solv = 's_{}'.format('_'.join([str(int(df_exp.at['solvent'])),
                                       '1',
                                       str(df_sample.at['sconformer']),
                                       str(df_sample.at['saug'])
                                       ]))
        if self.store_density == 'file':
            return (self._load_cube_conditional(f'{poly}.npy'),
                    self._load_cube_conditional(f'{solv}.npy'))
        if poly not in self.poly_cube_set:
            self.poly_cube_set[poly] = \
                self._load_cube_conditional(f'{poly}.npy')
        if solv not in self.solv_cube_set:
            self.solv_cube_set[solv] = \
                self._load_cube_conditional(f'{solv}.npy')
        if self.store_sparse:
            return (self.poly_cube_set[poly].todense(),
                    self.solv_cube_set[solv].todense())
        else:
            return (self.poly_cube_set[poly],
                    self.solv_cube_set[solv])

    def _form_index_tables(self,
                           poly_conf_num,
                           solv_conf_num):
        """Combinations of experiment/polymers oligomer/conformer/augment"""
        # Form possible combinations of compounds/conformers/cuts/augmentations
        poly_set, solv_set = self._conf_comb(poly_conf_num, solv_conf_num)
        # Load and scale experimental data from experimental_dataset.csv
        exp_set = self._load_experimental()
        return (
            exp_set[['expno', 'polymer', 'solvent']].
            drop_duplicates(subset=['expno']).
            merge(poly_set, on='polymer').
            merge(solv_set, on='solvent').
            sample(frac=1, ignore_index=True),

            exp_set.
            merge(self.macro_feature, on='solvent') if self.solv_macro_features
            else exp_set
        )

    def _get_exp_sampling(self, exp_val_set):
        """Balanced amount of conformations/augmentations per experiment"""
        num_exp_per_pair = exp_val_set. \
            drop_duplicates(subset=['expno']). \
            groupby(['polymer', 'solvent'])['expno']. \
            count().reset_index(). \
            rename(columns={'expno': 'experiments'})

        min_num_exp_per_pair = num_exp_per_pair['experiments'].min()
        min_sample = self.index_table.groupby('expno')['expno'].count().min()

        target_samp_per_exp = num_exp_per_pair. \
            copy(). \
            rename(columns={'experiments': 'samples'})

        target_samp_per_exp['samples'] = min_sample \
            * min_num_exp_per_pair \
            // num_exp_per_pair['experiments']

        target_samp_per_exp['samples'] = \
            target_samp_per_exp['samples'].replace(0, 1)

        if target_samp_per_exp['samples'].eq(0).any():
            raise RuntimeError('Found zero usages of experimental points!')

        mapper_index = target_samp_per_exp. \
            merge(exp_val_set,
                  how='inner',
                  on=['polymer', 'solvent'])[['expno', 'samples']]

        if self.make_even:
            exp_set_cut = exp_val_set[exp_val_set['cut'] == 1].copy()
            mean = exp_set_cut['wa'].mean()
            std = exp_set_cut['wa'].std()
            exp_set_cut['freq'] = \
                1 / ((((exp_set_cut['wa'] - mean) / std).apply(np.square)
                      * (-0.5)).apply(np.exp))
            exp_set_cut.loc[exp_set_cut['wa'] < mean, ['freq']] = 1
            exp_set_cut.loc[:, ['freq']] = \
                exp_set_cut.loc[:, ['freq']].astype(np.int32)
            mapper_index = mapper_index.merge(exp_set_cut, on='expno')
            mapper_index.loc[:, 'samples'] = mapper_index['samples'] * \
                mapper_index['freq']
            mapper_index = mapper_index[['expno', 'samples']]

        log_path = os.path.join(self.log_dir,
                                f'per_experiment_sampling_{self.log_name}.csv')
        mapper_index.to_csv(log_path, index=False)

        return mapper_index.set_index('expno')['samples'].to_dict()

    def _index_eval_split(self, exp_val_set):
        """Split total dataset into train, validation and test sets"""
        pairs = exp_val_set[['polymer', 'solvent']]. \
            drop_duplicates(). \
            sample(frac=1)
        if self.eval_define:
            pairs_eval = pd.read_csv(self.eval_define)
            # remove absent in the experimental set pairs from the test pairs
            pairs_eval = pairs.merge(pairs_eval, how='inner',
                                     on=['polymer', 'solvent'])
        elif self.eval_split:
            pairs_eval = pairs.sample(frac=1 / self.eval_split)
        else:
            raise RuntimeError('For some reason we are trying to split' +
                               'the dataset with no way to split defined')
        # remove the evaluation pairs from the experimental set pairs to get
        # training pairs only
        pairs_train = pairs. \
            merge(pairs_eval, how='left', on=['polymer', 'solvent'],
                  indicator='True')
        pairs_train = pairs_train[pairs_train['True'] == 'left_only'].drop(
            'True', axis=1)

        log_path = os.path.join(self.log_dir,
                                f'train_pairs_{self.log_name}.csv')
        pairs_train.to_csv(log_path, index=False)
        log_path = os.path.join(self.log_dir,
                                f'evaluation_pairs_{self.log_name}.csv')
        pairs_eval.to_csv(log_path, index=False)

        return (
            self.index_table.
            merge(pairs_train, how='inner', on=['polymer', 'solvent']).
            reset_index(drop=True),
            self.index_table.
            merge(pairs_eval, how='inner', on=['polymer', 'solvent']).
            reset_index(drop=True)
        )

    def _index_crossval_split(self, exp_val_set):
        """Split total dataset into n chunks for n-fold cross-validation"""
        pairs = exp_val_set[['polymer', 'solvent']].drop_duplicates()
        shuffled_pairs = pairs.sample(frac=1)
        pairs_folds = np.array_split(shuffled_pairs, self.fold)
        return [self.index_table.
                merge(pairs_fold, how='inner', on=['polymer', 'solvent'])
                for pairs_fold in pairs_folds]

    def _index_table_sampling(self, index_table):
        """Sample rows: equal number for each polymer-solvent pair"""
        return index_table.groupby('expno'). \
            apply(lambda x: x.sample(n=self.exp_mapper.get(x.name),
                                     replace=True)). \
            reset_index(drop=True). \
            sample(frac=1)

    def _index_table_chooser(self,
                             is_evaluation=False,
                             is_analysis=False,
                             fold_no=False,
                             gen_id=0):
        """Choose appropriate index table and sample for training/evaluation"""
        if gen_id:
            return 0
        if is_analysis:
            if self.analysis_n:
                self.sample_table = self.index_table.head(self.analysis_n)
            else:
                self.sample_table = self.index_table
            return len(self.sample_table)
        if is_evaluation:
            if self.fold:
                self.sample_table = self. \
                    _index_table_sampling(self.index_table_folds[fold_no])
            elif self.eval_split or self.eval_define:
                # self.sample_table = self.index_table_eval
                pass
            else:
                raise RuntimeError('Generator error: fold/split not defined')
        elif self.fold:
            self.sample_table = self. \
                _index_table_sampling(
                    pd.concat([df for i, df
                               in enumerate(self.index_table_folds)
                               if i != fold_no]))
        elif self.eval_split or self.eval_define:
            self.sample_table = \
                self._index_table_sampling(self.index_table_train)
        else:
            self.sample_table = self._index_table_sampling(self.index_table)
        return len(self.sample_table)

    def set_generator(self,
                      is_evaluation=False,
                      is_analysis=False,
                      with_zeros=False,
                      fold_no=False,
                      gen_id=0):
        """Combine experiment with related solvent/polymer electron density"""
        self._index_table_chooser(is_evaluation, is_analysis, fold_no, gen_id)
        if is_analysis and gen_id == 0:
            if with_zeros:
                log_name = f'full_table_zeros_{self.log_name}.csv'
            else:
                log_name = f'full_table_{self.log_name}.csv'
            log_path = os.path.join(self.log_dir, log_name)
            self.sample_table.to_csv(log_path, index=False)
        if is_analysis:
            chunk_size = self.analysis_sample_size  # // self.parallel_preproc
        elif is_evaluation:
            chunk_size = self.eval_sample_size // self.parallel_preproc
        else:
            chunk_size = self.train_sample_size // self.parallel_preproc
        if gen_id:
            time.sleep(self.timeout)
        for s in range(chunk_size * gen_id, chunk_size * (gen_id + 1)):
            if is_evaluation and not fold_no:
                table_slice = self.index_table_eval.iloc[s]
            else:
                table_slice = self.sample_table.iloc[s]
            exp_slice = self.exp_set.loc[tuple(table_slice[['expno', 'cut']])]
            # TODO debug DataFrame instead of Series occurencies
            if isinstance(exp_slice, pd.DataFrame):
                exp_slice = exp_slice.iloc[0]
            if is_analysis and with_zeros:
                yield (np.zeros(self.image_dim, dtype=self.load_precision),
                       np.zeros(self.image_dim, dtype=self.load_precision),
                       np.array(exp_slice.loc[self.features])), \
                    np.array(exp_slice.at['wa']).reshape((1,))
            else:
                poly_cube, solv_cube = \
                    self._cube_from_df(table_slice, exp_slice)
                if self.aug_onthefly:
                    poly_cube = self._rotate_cube(poly_cube)
                    solv_cube = self._rotate_cube(solv_cube)
                    poly_cube = self._shift_cube(poly_cube)
                    solv_cube = self._shift_cube(solv_cube)
                    poly_cube = poly_cube.astype(self.load_precision)
                    solv_cube = solv_cube.astype(self.load_precision)
                yield (poly_cube, solv_cube,
                       np.array(exp_slice.loc[self.features])), \
                    np.array(exp_slice.at['wa']).reshape((1,))

    def data_gen(self, is_evaluation=False, is_analysis=False,
                 with_zeros=False, fold_no=False):
        """Input function for training/evaluation"""
        if is_analysis:
            dataset = tf.data.Dataset.from_generator(
                lambda: self.set_generator(is_evaluation, is_analysis,
                                           with_zeros, fold_no, 0),
                output_signature=(
                    (self.cube_dim, self.cube_dim, self.macr_dim),
                    self.labl_dim, ))
            dataset = dataset.batch(self._batch_size, drop_remainder=True)
            dataset = dataset.prefetch(self._batch_size)
            return dataset
        if (not is_evaluation) or (self.fold or
                                   self.eval_split or
                                   self.eval_define):
            gen_ids = [i for i in range(self.parallel_preproc)]
            dataset = tf.data.Dataset.from_tensor_slices(gen_ids)
            dataset = dataset.interleave(
                lambda gen_id: tf.data.Dataset.from_generator(
                    self.set_generator,
                    output_signature=((self.cube_dim,
                                       self.cube_dim,
                                       self.macr_dim), self.labl_dim,),
                    args=(is_evaluation, is_analysis,
                          with_zeros, fold_no, gen_id,)),
                cycle_length=self.parallel_preproc,
                block_length=1,
                num_parallel_calls=self.parallel_preproc)

            dataset = dataset.batch(self._batch_size, drop_remainder=True)
            dataset = dataset.prefetch(self._batch_size)
        else:
            dataset = None
        return dataset


class DatasetPred(Dataset):
    """Load and prepare the data for inference"""

    def __init__(self, params):
        # Initialize general parameters
        super().__init__(params)
        self.to_pred_csv = params.to_pred_csv
        self.logcm = params.logcm
        self.index_table = self._index_table()

    def _index_table(self):
        data_types = {i: np.float32 for i in self.poly_macro_features}
        data_types.update({'polymer': np.uint8,
                           'solvent': np.uint8,
                           'pmin': np.float32,
                           'pmax': np.float32,
                           'npstep': np.uint8,
                           'tmin': np.float32,
                           'tmax': np.float32,
                           'ntstep': np.uint8,
                           'cryst': self.load_precision})
        use_columns = ['polymer', 'solvent', 'pmin', 'pmax', 'npstep',
                       'tmin', 'tmax', 'ntstep',
                       'cryst'] + self.poly_macro_features
        if self.logcm and 'dens' not in use_columns:
            use_columns = use_columns + ['dens']
        ranges = pd.read_csv(os.path.join(self._data_dir, self.to_pred_csv),
                             dtype=data_types, usecols=use_columns)
        ranges_len = len(ranges)
        range_num, pressure, temperature = [], [], []
        for s in range(0, ranges_len):
            range_slice = ranges.iloc[s]
            examples_num = int(range_slice.at['npstep'] *
                               range_slice.at['ntstep'])
            range_num += [s for i in range(examples_num)]
            ref_temperature = list(np.linspace(range_slice.at['tmin'],
                                               range_slice.at['tmax'],
                                               int(range_slice.at['ntstep'])))
            ref_pressure = list(np.linspace(range_slice.at['pmin'],
                                            range_slice.at['pmax'],
                                            int(range_slice.at['npstep'])))
            for t in ref_temperature:
                pressure += ref_pressure
                temperature += [t for i in range(len(ref_pressure))]
        index_table = pd.DataFrame({'range': range_num,
                                    'temperature': temperature,
                                    'pressure': pressure})
        ranges = ranges.reset_index().rename(columns={'index': 'range'})
        ranges = ranges. \
            drop(['pmin', 'pmax', 'npstep', 'tmin', 'tmax', 'ntstep'],
                 axis=1)
        if self.use_only_amorph:
            ranges = ranges.drop(['cryst'], axis=1)
        index_table = index_table.merge(ranges, on='range')
        index_table = index_table. \
            merge(self.poly_mass[['polymer', 'poly_mass']].
                  drop_duplicates(subset=['polymer', 'poly_mass']),
                  on='polymer'). \
            merge(self.solv_mass[['solvent', 'solv_mass']],
                  on='solvent')
        self._scale_exp_set(index_table, use_columns + ['pressure',
                                                        'temperature'])
        return index_table

    def _cube_from_df(self, df_sample):
        poly = int(df_sample.at['polymer'])
        solv = int(df_sample.at['solvent'])
        return (
            self.poly_cube_set.setdefault(poly,
                                          self._shift_cube(
                                              self._rotate_cube(
                                                  self._load_cube(
                                                      f'p_{poly}.npy'))).
                                          astype(self.load_precision)
                                          if self.aug_onthefly
                                          else self._load_cube_dense(
                                              f'p_{poly}.npy')),
            self.solv_cube_set.setdefault(solv,
                                          self._shift_cube(
                                              self._rotate_cube(
                                                  self._load_cube(
                                                      f's_{solv}.npy'))).
                                          astype(self.load_precision)
                                          if self.aug_onthefly
                                          else self._load_cube_dense(
                                              f's_{solv}.npy'))
                )

    def set_generator(self):
        """Combine defined macroscopic parameters with related solvent/polymer
        electron density"""

        sample_size = len(self.index_table)
        for s in range(0, sample_size):
            table_slice = self.index_table.iloc[s]
            yield ((*self._cube_from_df(table_slice),
                   np.array(table_slice.loc[self.features])),)

    def data_gen(self):
        """Input function for inference"""
        dataset = tf.data.Dataset.from_generator(self.set_generator,
                                                 output_signature=(
                                                   (self.cube_dim,
                                                    self.cube_dim,
                                                    self.macr_dim),))
        dataset = dataset.batch(self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._batch_size)
        return dataset
