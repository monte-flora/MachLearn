import numpy as np
import keras
from build_convolution_neural_network import ConvNet
model_builder = ConvNet( )
import xarray as xr 
from glob import glob
import itertools
import json
import copy

import os
from os.path import join
from wofs.util import config
import random
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from MachLearn import MachLearn
import keras_metrics
from wofs.evaluation.verification_metrics import Metrics, brier_skill_score

#plt.rc('xtick',labelsize=18)
#plt.rc('ytick',labelsize=18)

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

STORM_IDS_KEY = 'storm_ids'
STORM_STEPS_KEY = 'storm_steps'
PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
TARGET_NAME_KEY = 'target_name'
TARGET_MATRIX_KEY = 'target_matrix'

TRAINING_FILES_KEY = 'training_file_names'
NORMALIZATION_DICT_KEY = 'normalization_dict'
BINARIZATION_THRESHOLD_KEY = 'binarization_threshold'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
VALIDATION_FILES_KEY = 'validation_file_names'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
CNN_FILE_KEY = 'cnn_file_name'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'

MIN_XENTROPY_DECREASE_FOR_EARLY_STOP = 0.005
MIN_MSE_DECREASE_FOR_EARLY_STOP = 0.005
NUM_EPOCHS_FOR_EARLY_STOPPING = 5

LIST_OF_METRIC_FUNCTIONS = [
    keras_metrics.accuracy, keras_metrics.binary_accuracy,
    keras_metrics.binary_csi, keras_metrics.binary_frequency_bias,
    keras_metrics.binary_pod, keras_metrics.binary_pofd,
    keras_metrics.binary_peirce_score, keras_metrics.binary_success_ratio,
    keras_metrics.binary_focn
]

METRIC_FUNCTION_DICT = {
    'accuracy': keras_metrics.accuracy,
    'binary_accuracy': keras_metrics.binary_accuracy,
    'binary_csi': keras_metrics.binary_csi,
    'binary_frequency_bias': keras_metrics.binary_frequency_bias,
    'binary_pod': keras_metrics.binary_pod,
    'binary_pofd': keras_metrics.binary_pofd,
    'binary_peirce_score': keras_metrics.binary_peirce_score,
    'binary_success_ratio': keras_metrics.binary_success_ratio,
    'binary_focn': keras_metrics.binary_focn
}

DEFAULT_NUM_BWO_ITERATIONS = 200
DEFAULT_BWO_LEARNING_RATE = 0.01


class DeepLearn:
    '''
    The DeepLearn class builds convolution neural network models using keras and implements them in a cross-fold validation. 

    Parameters:
    ------------  
        n_cv_folds, int, number of cross-validation folds 
        percent_dates_for_training, float, percentage of the total dates for training (set between 0-1 )
        percent_dates_for_validation, float, percentage of the total dates for validation (set between 0-1 )
        percent_dates_for_testing, float, percentage of the total dates for testing (set between 0-1 )
        percent_positive_examples_in_training, float, percentage of the examples that have a positive label (between 0-1; default = 0.5 for balanced examples) 
        percent_positive_examples_in_validation, float, percentage of the examples that have a positive label
                                                        for the validation dataset( between 0-1, default = None) 
    Attributes:
    ------------ 
        num_dates, int, total number of dates
        n_training_dates, int, number of training dates
        n_validation_dates, int, number of validation dates 
    '''
    def __init__(self, fcst_time_idx=6, n_cv_folds=8):
        self.fcst_time_idx = fcst_time_idx 
        self.num_dates = len(config.ml_dates)
        if self.num_dates % n_cv_folds != 0:
            self.fold_interval = int((self.num_dates / n_cv_folds)+1)
        else:
            self.fold_interval = int((self.num_dates / n_cv_folds))
        print ("Fold interval: ", self.fold_interval)
        self.n_training_dates = int(0.8*self.num_dates)
        self.n_validation_dates = int(round(0.1*self.num_dates))
        self.n_testing_dates = int(round(0.1*self.num_dates))
        self.num1 = self.n_training_dates+self.n_validation_dates
        self.num2 = self.n_training_dates+self.n_validation_dates+self.n_testing_dates

    def find_dates(self, r ):
        '''
        '''
        folds_training = (np.arange(self.n_training_dates) + r) % self.num_dates
        folds_validation = (np.arange(self.n_training_dates, self.num1) + r) % self.num_dates
        folds_testing = (np.arange(self.num1, self.num2) + r) % self.num_dates
        
        this_folds_training_dates = config.ml_dates[folds_training]
        this_folds_validation_dates = config.ml_dates[folds_validation]
        this_folds_testing_dates = config.ml_dates[folds_testing]

        return (this_folds_training_dates, this_folds_validation_dates, this_folds_testing_dates)

    def get_netcdf_file_names(self, fold, generic_file_name_str ):
        '''
        Get the NetCDF file names for the deep_learning_generator
        '''
        print (self.fcst_time_idx)
        #'/DL_WOFS_*_{}_generic_patches.nc'.format(self.fcst_time_idx)
        training_dates, validation_dates, testing_dates = self.find_dates( r=fold )             
      
        train_file_names = [ glob( join(config.ML_INPUT_PATH, str(date)+ generic_file_name_str))
                                                                 for date in training_dates]
        valid_file_names = [ glob( join(config.ML_INPUT_PATH, str(date)+ generic_file_name_str))
                                                                 for date in validation_dates] 
        
        test_file_names = [ glob( join(config.ML_INPUT_PATH, str(date)+ generic_file_name_str))
                                                                 for date in testing_dates] 

        return ( sorted( list(itertools.chain.from_iterable(train_file_names))), 
                 sorted( list(itertools.chain.from_iterable(valid_file_names))),
                 sorted( list(itertools.chain.from_iterable(test_file_names)))
                )

    def read_netcdf_file(self, netcdf_file_name, target_var = 'matched_to_LSRs_15km'):
        '''
        Reads deep learning images from NETCDF file.
        '''
        dataset_object = xr.open_dataset(netcdf_file_name)
        label_vars = ['matched_to_LSRs_15km', 'matched_to_LSRs_30km', 'matched_to_azshr_30km', 'matched_to_Tornado_LSRs_15km', 'matched_to_Tornado_LSRs_30km',
              'matched_to_Severe_Wind_LSRs_15km', 'matched_to_Severe_Wind_LSRs_30km',
              'matched_to_Hail_LSRs_15km', 'matched_to_Hail_LSRs_30km', 'ensemble member', 'label'] 
              
        dataset_object_sub = dataset_object.drop(label_vars)

        predictor_names = list(dataset_object_sub.data_vars)        

        predictor_matrix = np.stack( [dataset_object[v].values for v in predictor_names], axis=-1)
        target_matrix = dataset_object[target_var].values

        dataset_object.close()
        return {
                'predictor_names': predictor_names,
                'predictor_matrix': predictor_matrix,
                'target_matrix': target_matrix
                } 
   
    def read_many_netcdf_files(self, netcdf_file_names):
        """Reads storm-centered images from many NetCDF files.

        :param netcdf_file_names: 1-D list of paths to input files.
        :return: image_dict: See doc for `read_image_file`.
        """

        image_dict = None
        keys_to_concat = [
            PREDICTOR_MATRIX_KEY, TARGET_MATRIX_KEY
        ]

        for this_file_name in netcdf_file_names:
            print('Reading data from: "{0:s}"...'.format(this_file_name))
            this_image_dict = self.read_netcdf_file(this_file_name)

            if image_dict is None:
                image_dict = copy.deepcopy(this_image_dict)
                continue

            for this_key in keys_to_concat:
                image_dict[this_key] = np.concatenate(
                    (image_dict[this_key], this_image_dict[this_key]), axis=0)

        return image_dict

    def _update_normalization_params(self, intermediate_normalization_dict, new_values):
        """Updates normalization params for one predictor.

        :param intermediate_normalization_dict: Dictionary with the following keys.
               intermediate_normalization_dict['num_values']: Number of values on which
               current estimates are based.
               intermediate_normalization_dict['mean_value']: Current estimate for mean.
               intermediate_normalization_dict['mean_of_squares']: Current mean of squared
               values.

        :param new_values: numpy array of new values (will be used to update
                                                     `intermediate_normalization_dict`).
        :return: intermediate_normalization_dict: Same as input but with updated
            values.
        """

        if 'mean_value' not in intermediate_normalization_dict:
            intermediate_normalization_dict = {
                'num_values': 0,
                'mean_value': 0.,
                'mean_of_squares': 0.
            }

        these_means = np.array([
            intermediate_normalization_dict['mean_value'], np.mean(new_values) ])
        
        these_weights = np.array([
            intermediate_normalization_dict['num_values'], new_values.size  ])

        intermediate_normalization_dict['mean_value'] = np.average(
            these_means, weights=these_weights)

        these_means = np.array([
            intermediate_normalization_dict['mean_of_squares'],
            np.mean(new_values ** 2)  ])

        intermediate_normalization_dict['mean_of_squares'] = np.average(
            these_means, weights=these_weights)

        intermediate_normalization_dict['num_values'] += new_values.size
        
        return intermediate_normalization_dict 

    def _get_standard_deviation(self, intermediate_normalization_dict):
        """Computes stdev from intermediate normalization params.

        :param intermediate_normalization_dict: See doc for
            `_update_normalization_params`.
        :return: standard_deviation: Standard deviation.
        """

        num_values = float(intermediate_normalization_dict['num_values'])
        multiplier = num_values / (num_values - 1)

        return np.sqrt(multiplier * (
            intermediate_normalization_dict['mean_of_squares'] -
            intermediate_normalization_dict['mean_value'] ** 2
            ))    
    
    def get_image_normalization_params(self, netcdf_file_names, norm_dict_file_name=None):
        """Computes normalization params (mean and stdev) for each predictor.

        :param netcdf_file_names: 1-D list of paths to input files.
        :return: normalization_dict: See input doc for `normalize_images`.
        """

        print ('Number of NETCDF files: {}'.format(len(netcdf_file_names)))

        predictor_names = None
        norm_dict_by_predictor = None

        for i, this_file_name in enumerate(netcdf_file_names):
            print('Reading data from: {}...{} out of {} files'.format(this_file_name, i, len(netcdf_file_names)))
            this_image_dict = self.read_netcdf_file(this_file_name)

            if predictor_names is None:
                predictor_names = this_image_dict['predictor_names']
                norm_dict_by_predictor = [{}] * len(predictor_names)

            for m in range(len(predictor_names)):
                norm_dict_by_predictor[m] = self._update_normalization_params(
                    intermediate_normalization_dict=norm_dict_by_predictor[m],
                    new_values=this_image_dict['predictor_matrix'][..., m])
            del this_image_dict     
            
        
        print('\n')
        normalization_dict = {}

        for m in range(len(predictor_names)):
            this_mean = norm_dict_by_predictor[m]['mean_value']
            this_stdev = self._get_standard_deviation(norm_dict_by_predictor[m])
            normalization_dict[predictor_names[m]] = np.array(
                [this_mean, this_stdev]).tolist()

            print(
                ('Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
                ).format(predictor_names[m], this_mean, this_stdev)
            )

        #with open( norm_dict_file_name, 'w') as this_file:
        #    json.dump(normalization_dict, this_file)
    
        return normalization_dict

    def normalize_images(
        self, predictor_matrix, predictor_names, normalization_dict):
        """Normalizes images to z-scores.

        E = number of examples (storm objects) in file
        M = number of rows in each storm-centered grid
        N = number of columns in each storm-centered grid
        C = number of channels (predictor variables)

        :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
        :param predictor_names: length-C list of predictor names.
        :param normalization_dict: Dictionary.  Each key is the name of a predictor
            value, and the corresponding value is a length-2 numpy array with
            [mean, standard deviation].  If `normalization_dict is None`, mean and
            standard deviation will be computed for each predictor.
        :return: predictor_matrix: Normalized version of input.
        :return: normalization_dict: See doc for input variable.  If input was None,
            this will be a newly created dictionary.  Otherwise, this will be the
            same dictionary passed as input.
        """
        if normalization_dict is None:
            normalization_dict = self.normalization_dict 

        num_predictors = len(predictor_names)

        for m in range(num_predictors):
            this_mean = normalization_dict[predictor_names[m]][0]
            this_stdev = normalization_dict[predictor_names[m]][1]

            predictor_matrix[..., m] = (
                (predictor_matrix[..., m] - this_mean) / float(this_stdev)
            )

        return predictor_matrix

    def balance_examples(self, predictor_matrix, target_matrix):
        '''
        Balance examples
        '''
        if np.mean(target_matrix) > 0.5:
            return predictor_matrix, target_matrix
        
        positive_idxs = np.where(( target_matrix ==1.0 ))[0]
        negative_idxs = np.where(( target_matrix ==0.0 ))[0]
        random_negative_idxs = random.sample( list(negative_idxs), len(positive_idxs))
        idxs = np.concatenate(( positive_idxs, random_negative_idxs ))
        
        return (predictor_matrix[idxs.astype(int)], target_matrix[idxs.astype(int)])


    def deep_learning_generator(self, netcdf_file_names, num_examples_per_batch, normalization_dict ):
        """Generates training examples for deep-learning model on the fly.

        E = number of examples (storm objects)
        M = number of rows in each storm-centered grid
        N = number of columns in each storm-centered grid
        C = number of channels (predictor variables)

        :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
        :param num_examples_per_batch: Number of examples per training batch.
        :param normalization_dict: See doc for `normalize_images`.  You cannot leave
            this as None.
        :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
        :return: target_values: length-E numpy array of target values (integers in
            0...1).
        """

        random.shuffle(netcdf_file_names)
        num_files = len(netcdf_file_names)
        file_index = 0

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None
        predictor_names = None

        while True:
            while num_examples_in_memory < num_examples_per_batch:
                print('Reading data from: "{0:s}"...'.format(
                    netcdf_file_names[file_index]))

                this_image_dict = self.read_netcdf_file(netcdf_file_names[file_index])
                predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]

                file_index += 1
                if file_index >= num_files:
                    file_index = 0

                if full_target_matrix is None or full_target_matrix.size == 0:
                    full_predictor_matrix = (
                        this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
                    )
                    full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.

                else:
                    full_predictor_matrix = np.concatenate(
                        (full_predictor_matrix,
                        this_image_dict[PREDICTOR_MATRIX_KEY]),
                        axis=0)

                    full_target_matrix = np.concatenate(
                        (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
                        axis=0)

                full_predictor_matrix, full_target_matrix = self.balance_examples(full_predictor_matrix, full_target_matrix)    
                num_examples_in_memory = full_target_matrix.shape[0]

            predictor_matrix  = self.normalize_images(
                predictor_matrix=full_predictor_matrix,
                predictor_names=predictor_names,
                normalization_dict=normalization_dict)

            predictor_matrix = predictor_matrix.astype('float32')
            target_values = full_target_matrix

            print('Fraction of examples in positive class: {0:.4f}'.format(
                np.mean(target_values)))

            num_examples_in_memory = 0
            full_predictor_matrix = None
            full_target_matrix = None

            yield (predictor_matrix, target_values)    
        
    def train_cnn(
        self,
        cnn_model_object, training_file_names, 
        num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, output_model_file_name,
        normalization_dict, 
        validation_file_names=None, num_validation_batches_per_epoch=None):
        """Trains CNN (convolutional neural net).

        :param cnn_model_object: Untrained instance of `keras.models.Model` (may be
            created by `setup_cnn`).
        :param training_file_names: 1-D list of paths to training files (must be
            readable by `read_image_file`).
        :param normalization_dict: See doc for `deep_learning_generator`.
        :param num_examples_per_batch: Same.
        :param num_epochs: Number of epochs.
        :param num_training_batches_per_epoch: Number of training batches furnished
            to model in each epoch.
        :param output_model_file_name: Path to output file.  The model will be saved
            as an HDF5 file (extension should be ".h5", but this is not enforced).
        :param validation_file_names: 1-D list of paths to training files (must be
            readable by `read_image_file`).  If `validation_file_names is None`,
            will omit on-the-fly validation.
        :param num_validation_batches_per_epoch:
            [used only if `validation_file_names is not None`]
            Number of validation batches furnished to model in each epoch.

        :return: cnn_metadata_dict: Dictionary with the following keys.
        cnn_metadata_dict['training_file_names']: See input doc.
        cnn_metadata_dict['normalization_dict']: Same.
        cnn_metadata_dict['num_examples_per_batch']: Same.
        cnn_metadata_dict['num_training_batches_per_epoch']: Same.
        cnn_metadata_dict['validation_file_names']: Same.
        cnn_metadata_dict['num_validation_batches_per_epoch']: Same.
        """
        self.normalization_dict = normalization_dict

        if validation_file_names is None:
            checkpoint_object = keras.callbacks.ModelCheckpoint(
                filepath=output_model_file_name, monitor='loss', verbose=1,
                save_best_only=False, save_weights_only=False, mode='min',
                period=1)
        else:
            checkpoint_object = keras.callbacks.ModelCheckpoint(
                filepath=output_model_file_name, monitor='val_loss', verbose=1,
                save_best_only=True, save_weights_only=False, mode='min',
                period=1)

        list_of_callback_objects = [checkpoint_object]

        cnn_metadata_dict = {
            TRAINING_FILES_KEY: training_file_names,
            NORMALIZATION_DICT_KEY: self.normalization_dict,
            NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
            NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
            VALIDATION_FILES_KEY: validation_file_names,
            NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
            }

        training_generator = self.deep_learning_generator(
            netcdf_file_names=training_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict)

        if validation_file_names is None:
            cnn_model_object.fit_generator(
                generator=training_generator,
                steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
                verbose=1, callbacks=list_of_callback_objects, workers=0, 
                use_multiprocessing=True)

            return cnn_metadata_dict

        early_stopping_object = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=MIN_XENTROPY_DECREASE_FOR_EARLY_STOP,
            patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

        list_of_callback_objects.append(early_stopping_object)

        validation_generator = self.deep_learning_generator(
            netcdf_file_names=validation_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict)

        cnn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0,
            validation_data=validation_generator,
            validation_steps=num_validation_batches_per_epoch,
            use_multiprocessing=True)

        return cnn_metadata_dict


    def _apply_cnn(self, cnn_model_object, predictor_matrix, verbose=True,
               output_layer_name=None):
        """Applies trained CNN (convolutional neural net) to new data.

        E = number of examples (storm objects) in file
        M = number of rows in each storm-centered grid
        N = number of columns in each storm-centered grid
        C = number of channels (predictor variables)

        :param cnn_model_object: Trained instance of `keras.models.Model`.
        :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
        :param verbose: Boolean flag.  If True, progress messages will be printed.
        :param output_layer_name: Name of output layer.  If
            `output_layer_name is None`, this method will use the actual output
            layer, so will return predictions.  If `output_layer_name is not None`,
            will return "features" (outputs from the given layer).

        If `output_layer_name is None`...

        :return: forecast_probabilities: length-E numpy array with forecast
            probabilities of positive class (label = 1).

        If `output_layer_name is not None`...

        :return: feature_matrix: numpy array of features (outputs from the given
            layer).  There is no guarantee on the shape of this array, except that
            the first axis has length E.
        """

        num_examples = predictor_matrix.shape[0]
        num_examples_per_batch = 1000

        if output_layer_name is None:
            model_object_to_use = cnn_model_object
        else:
            model_object_to_use = keras.models.Model(
                inputs=cnn_model_object.input,
                outputs=cnn_model_object.get_layer(name=output_layer_name).output)

        output_array = None

        for i in range(0, num_examples, num_examples_per_batch):
            this_first_index = i
            this_last_index = min(
                [i + num_examples_per_batch - 1, num_examples - 1]
            )

            if verbose:
                print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                    this_first_index, this_last_index, num_examples))

            these_indices = np.linspace(
                this_first_index, this_last_index,
                num=this_last_index - this_first_index + 1, dtype=int)

            this_output_array = model_object_to_use.predict(
                predictor_matrix[these_indices, ...],
                batch_size=num_examples_per_batch)

            if output_layer_name is None:
                this_output_array = this_output_array[:, -1]

            if output_array is None:
                output_array = this_output_array + 0.
            else:
                output_array = np.concatenate(
                    (output_array, this_output_array), axis=0)

        return output_array 

    def evaluate_cnn(
        self, cnn_model_object, image_dict, cnn_metadata_dict, calibrated_model):
        """Evaluates trained CNN (convolutional neural net).

        :param cnn_model_object: Trained instance of `keras.models.Model`.
        :param image_dict: Dictionary created by `read_image_file` or
            `read_many_image_files`.  Should contain validation or testing data (not
            training data), but this is not enforced.
        :param cnn_metadata_dict: Dictionary created by `train_cnn`.  This will
            ensure that data in `image_dict` are processed the exact same way as the
            training data for `cnn_model_object`.
        :param output_dir_name: Path to output directory.  Figures will be saved
            here.
         """

        predictor_matrix = self.normalize_images(
            predictor_matrix=image_dict[PREDICTOR_MATRIX_KEY] + 0.,
            predictor_names=image_dict[PREDICTOR_NAMES_KEY],
            normalization_dict=cnn_metadata_dict[NORMALIZATION_DICT_KEY])
        
        predictor_matrix = predictor_matrix.astype('float32')
        target_values = image_dict[TARGET_MATRIX_KEY]
        forecast_probabilities = self._apply_cnn(cnn_model_object=cnn_model_object,
                                        predictor_matrix=predictor_matrix)

        if calibrated_model is not None:
            calibrated_probabilities = calibrated_model.predict( forecast_probabilities )
            auc_calibrated = roc_auc_score(target_values, calibrated_probabilities)    
            bss_calibrated = brier_skill_score(target_values, calibrated_probabilities)
            print ('Calibrated BSS: {0:.3f} and Calibrated AUC: {1:.3f}'.format( bss_calibrated, auc_calibrated ))
            
        auc = roc_auc_score(target_values, forecast_probabilities)    
        bss = brier_skill_score(target_values, forecast_probabilities)
        print ('BSS: {0:.3f} and AUC: {1:.3f}'.format( bss, auc ))

        return forecast_probabilities


def read_keras_model(hdf5_file_name):
    """Reads Keras model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    return keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT)


def find_model_metafile(model_file_name, raise_error_if_missing=False):
    """Finds metafile for machine-learning model.

    :param model_file_name: Path to file with trained model.
    :param raise_error_if_missing: Boolean flag.  If True and metafile is not
        found, this method will error out.
    :return: model_metafile_name: Path to file with metadata.  If file is not
        found and `raise_error_if_missing = False`, this will be the expected
        path.
    :raises: ValueError: if metafile is not found and
        `raise_error_if_missing = True`.
    """

    model_directory_name, pathless_model_file_name = os.path.split(
        model_file_name)
    model_metafile_name = '{0:s}/{1:s}_metadata.json'.format(
        model_directory_name, os.path.splitext(pathless_model_file_name)[0]
    )

    if not os.path.isfile(model_metafile_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            model_metafile_name)
        raise ValueError(error_string)

    return model_metafile_name


def _metadata_numpy_to_list(model_metadata_dict):
    """Converts numpy arrays in model metadata to lists.

    This is needed so that the metadata can be written to a JSON file (JSON does
    not handle numpy arrays).

    This method does not overwrite the original dictionary.

    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :return: new_metadata_dict: Same but with lists instead of numpy arrays.
    """

    new_metadata_dict = copy.deepcopy(model_metadata_dict)

    if NORMALIZATION_DICT_KEY in new_metadata_dict.keys():
        this_norm_dict = new_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            if isinstance(this_norm_dict[this_key], np.ndarray):
                this_norm_dict[this_key] = this_norm_dict[this_key].tolist()

    return new_metadata_dict


def _metadata_list_to_numpy(model_metadata_dict):
    """Converts lists in model metadata to numpy arrays.

    This method is the inverse of `_metadata_numpy_to_list`.

    This method overwrites the original dictionary.

    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :return: model_metadata_dict: Same but numpy arrays instead of lists.
    """

    if NORMALIZATION_DICT_KEY in model_metadata_dict.keys():
        this_norm_dict = model_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            this_norm_dict[this_key] = np.array(this_norm_dict[this_key])

    return model_metadata_dict


def write_model_metadata(model_metadata_dict, json_file_name):
    """Writes metadata for machine-learning model to JSON file.

    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :param json_file_name: Path to output file.
    """

    new_metadata_dict = _metadata_numpy_to_list(model_metadata_dict)
    with open(json_file_name, 'w') as this_file:
        json.dump(new_metadata_dict, this_file)


def read_model_metadata(json_file_name):
    """Reads metadata for machine-learning model from JSON file.

    :param json_file_name: Path to output file.
    :return: model_metadata_dict: Dictionary with keys listed in doc for
        `train_cnn` or `train_ucn`.
    """

    with open(json_file_name) as this_file:
        model_metadata_dict = json.load(this_file)
        return _metadata_list_to_numpy(model_metadata_dict)        

