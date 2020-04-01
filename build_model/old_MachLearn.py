# Import Modules
import os 
from os.path import join, exists
import numpy as np
import xarray as xr
from os.path import join, exists
from glob import glob
import itertools 
import random
from joblib import dump, load
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import xgboost as xgb
import keras
from keras.models import load_model
import pandas as pd
from datetime import datetime

# Personal Modules 
from build_sklearn_model import classifier_model, calibration_model
from wofs.util import config
from PermutationImportance.permutation_importance import sklearn_permutation_importance
from machine_learning.plotting.plot_feature_importance import plot_variable_importance
from wofs.util import feature_names
from wofs.util.MultiProcessing import multiprocessing_per_date
from wofs.main.forecasts.StoringObjectProperties import save_object_properties
from build_convolution_neural_network import ConvNet
from wofs.util.feature_names import _feature_names_for_traditional_ml
from wofs.plotting.Plot import Plotting
from wofs.plotting.Plot import verification_plots
from wofs.evaluation.verification_metrics import ContingencyTable, Metrics, brier_skill_score, _get_binary_xentropy
from wofs.processing.ObjectIdentification import QualityControl
from wofs.processing.ObjectIdentification import label as label_id
import feature_selection

qc = QualityControl()
my_plt = Plotting(  )
model_builder = ConvNet( )

plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)

class MachLearn:
    '''
    The MachLearn class builds sklearn models and implements them in a leave-one-day-out cross-fold validation. 

    Parameters:
    ------------  
        percent_positive_examples_in_training, float, percentage of the examples that have a positive label (between 0-1; default = 0.5 for balanced examples) 
        percent_positive_examples_in_validation, float, percentage of the examples that have a positive label
                                                        for the validation dataset( between 0-1, default = None) 
    Attributes:
    ------------ 
        num_dates, int, total number of dates
        n_training_dates, int, number of training dates
    '''
    def __init__(self, n_cv_folds=5, percent_positive_examples_in_training = None, percent_positive_examples_in_validation = None):
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
        self.percent_positive_examples_in_training = percent_positive_examples_in_training
        self.percent_positive_examples_in_validation = percent_positive_examples_in_validation
        mode = {'LogisticRegression': 'ML', 'RandomForest': 'ML', 'XGBoost': 'ML', 'CNN': 'DL'}
        self.mode = mode     
        tags = {'LogisticRegression': '.joblib', 'RandomForest': '.joblib', 'XGBoost': '.joblib', 'CNN': '.h5'}
        self.tags = tags


    def fit( self, model_name, params, training_data, other_params=None, validation_data=None, testing_data=None):
        '''
        Fits the classifier model to the training data with parameters 'params'
        Args:
        ------------ 
            model_name, str, name of the sklearn model 
            params, dict, dictionary of model parameters for the sklearn model 
            training_data, 2-tuple of training examples and training labels
            validation_data, 2-tuple of validation examples and validation labels (used for Xgboost models)
        Returns:
        ------------ 
        self: object
        '''
        self.model_name = model_name 
        if model_name == 'XGBoost':
            clf = classifier_model( model_name=model_name, params=params)
            clf.fit( X=training_data[0], y=training_data[1], eval_set = [training_data,validation_data], early_stopping_rounds=200)

        else:
            clf = classifier_model( model_name=model_name, params=params)
            clf.fit( training_data[0], training_data[1])

        self.training_data = training_data
        self.validation_data = validation_data
        self.testing_data = testing_data
        self.clf = clf        
        
        return self 

    def drop_columns(self, inp_data, to_drop):
        '''
        '''
        # Drops the correlated columns
        drop_cols = set(to_drop)
        inp_data = inp_data.drop(columns=drop_cols)
        # Return same type as inp
        return inp_data

    def filter_df_by_corr(self, inp_data, target_var, cc_val=0.8):
        '''
        Returns an array or dataframe (based on type(inp_data) adjusted to drop \
            columns with high correlation to one another. Takes second arg corr_val
            that defines the cutoff

        ----------
        inp_data : np.array, pd.DataFrame
            Values to consider
        corr_val : float
            Value [0, 1] on which to base the correlation cutoff
        '''
        # Creates Correlation Matrix
        if isinstance(inp_data, np.ndarray):
            inp_data = pd.DataFrame(data=inp_data)
            array_flag = True
        else:
            array_flag = False
        corr_matrix = inp_data.corr()

        # Iterates through Correlation Matrix Table to find correlated columns
        drop_cols = []
        n_cols = len(corr_matrix.columns)

        print ('Calculating correlations between features...')
        for i in range(n_cols):
            for k in range(i+1, n_cols):
                val = corr_matrix.iloc[k, i]
                col = corr_matrix.columns[i]
                row = corr_matrix.index[k]
                col_to_target = corr_matrix.loc[col,target_var]
                row_to_target = corr_matrix.loc[row,target_var]
                if abs(val) >= cc_val: 
                    # Prints the correlated feature set and the corr val
                    if col_to_target > row_to_target and row not in drop_cols:
                        print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {row}')
                        drop_cols.append(row)
                    if row_to_target > col_to_target and col not in drop_cols:
                        print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {col}')
                        drop_cols.append(col)

        # Drops the correlated columns
        print ('Dropping {} highly correlated features...'.format(len(drop_cols)))
        print ( drop_cols )
        inp_data = self.drop_columns(inp_data, drop_cols) 
        
        return inp_data, drop_cols 

    def load_dataframe(self, fold, target_var_name, fcst_time_idx, data_extraction_method, variables_to_remove=[] ):
        '''
        Load pandas dataframe. 
        '''
        print ('Loading training, validation, and testing data for fold {}...'.format(fold))
        valid_df = pd.read_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'validation_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_{data_extraction_method}.pkl' ))
        train_df = pd.read_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'training_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_{data_extraction_method}.pkl' ))
        test_df = pd.read_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'testing_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_{data_extraction_method}.pkl' )) 

        variables_to_remove += ['matched_to_LSRs_15km', 'matched_to_LSRs_30km', 'matched_to_Tornado_LSRs_15km', 'matched_to_Tornado_LSRs_30km', 
                                'matched_to_Severe_Wind_LSRs_15km', 'matched_to_Severe_Wind_LSRs_30km', 
                                'matched_to_Hail_LSRs_15km', 'matched_to_Hail_LSRs_30km', 'Run Time', 'Run Date', 'label', 'ensemble_member']

        if self.remove_highly_correlated_features:
            variables_to_remove.pop( variables_to_remove.index(target_var_name))

        train_examples = train_df.drop( columns = variables_to_remove)
        valid_examples = valid_df.drop( columns = variables_to_remove)
        test_examples = test_df.drop( columns = variables_to_remove)

        if self.use_select_features:
            print ('Using these selected features: {}'.format(feature_selection.selected_features))
            train_examples = train_examples[feature_selection.selected_features]
            valid_examples = valid_examples[feature_selection.selected_features]
            test_examples = test_examples[feature_selection.selected_features]

        self.test_ens_mem = test_df['ensemble_member'].astype(float)

        train_examples = train_examples.astype(float)
        valid_examples = valid_examples.astype(float)
        test_examples = test_examples.astype(float)

        #Remove correlated Features
        if self.remove_highly_correlated_features:
            train_examples, dropped_cols = self.filter_df_by_corr(train_examples, target_var=target_var_name, cc_val=0.8)
            valid_examples = self.drop_columns(valid_examples, to_drop=dropped_cols)
            test_examples = self.drop_columns(test_examples, to_drop=dropped_cols)
        
        train_targets = train_df[target_var_name].astype(float)
        valid_targets = valid_df[target_var_name].astype(float)
        test_targets = test_df[target_var_name].astype(float)

        # Remove the target variable
        if self.remove_highly_correlated_features:
            train_examples = train_examples.drop( columns = target_var_name)
            valid_examples = valid_examples.drop( columns = target_var_name)
            test_examples = test_examples.drop( columns = target_var_name)

        self.feature_names = list(train_examples.columns)
        print ('Num of Features: {}'.format(len(self.feature_names)))

        return ( train_examples.values, train_targets.values), (valid_examples.values, valid_targets.values), (test_examples.values, test_targets.values) 

    def fitCV(self, model_name, params, var, verify_var, fcst_time_idx, data_extraction_method, removed_variables, 
            norm=False, feature_importance=False, calibrate=False, plot_roc_curve=False, plot_training_curve=False,
            use_select_features=False, remove_highly_correlated_features=False):
        '''
        Performs cross-validation to find the best parameters for the model given. 
        '''
        self.use_select_features = use_select_features
        self.remove_highly_correlated_features = remove_highly_correlated_features 
        self.model_name = model_name
        self.verify_var = verify_var 
        self.fcst_time_idx = fcst_time_idx
        scaling_values = None
        for fold, r in enumerate(range(0,self.num_dates,self.fold_interval)):
            print('\nfold: ', fold)
            self.fold = fold
            fname_base = f'{model_name}_uncalibrated_verify_var={verify_var}_norm={norm}_fold={fold}_fcst_time_idx={fcst_time_idx}'
                
            training_data, validation_data, testing_data = self.load_dataframe( fold, verify_var, fcst_time_idx, data_extraction_method, removed_variables) 

            if norm:
                print ('Performing normalization...')  
                training_examples, validation_examples, testing_examples = self.normalize_sklearn(training_data[0], validation_data[0], testing_data[0], fname_base=fname_base)
                validation_data = (validation_examples, validation_data[1]) 
                training_data = (training_examples, training_data[1])
                testing_data = (testing_examples, testing_data[1])

            print ('Balancing the training data....')
            if self.percent_positive_examples_in_training is not None and model_name != 'CNN':
                training_examples, training_labels = self.balance_examples( training_data[0], training_data[1], percent_positive = self.percent_positive_examples_in_training )
                training_data = (training_examples, training_labels)
                print(("Number of training examples: ", training_examples.shape))
            
            if self.percent_positive_examples_in_validation is not None:
                validation_examples, validation_labels = self.balance_examples(
                            validation_data[0], validation_data[1], percent_positive = self.percent_positive_examples_in_validation )
            
            
            print('Performing imputation...')
            training_examples, validation_examples, testing_examples = self.imputer(training_data, validation_data, testing_data, fname_base=fname_base) 
            validation_data = (validation_examples, validation_data[1])
            training_data = (training_examples, training_data[1])
            testing_data = (testing_examples, testing_data[1]) 
        
            print ("Fitting the model...") 
            self.fit(model_name=model_name, params=params, training_data=training_data, validation_data=validation_data, testing_data=testing_data) 

            if calibrate: 
                self.calibrate_fit( validation_data=self.validation_data, fname_base=fname_base) 
            if plot_roc_curve:
                print ('\t Evaluating training curves...')
                roc_curve_fname = 'roc_curve_' + fname_base + '.png'
                self.plot_roc_curves( model = self.clf, 
                                      fig_filename = roc_curve_fname, scaling_values=scaling_values )
           
                perform_curve_fname = 'perform_curve_' + fname_base + '.png'
                self.plot_performance_curve(model = self.clf, 
                                        fig_filename = perform_curve_fname, scaling_values=scaling_values )
                
                reliability_fname = 'reliability_' + fname_base + '.png'
                self.plot_reliability(model = self.clf,
                                        fig_filename = reliability_fname, scaling_values=scaling_values )

            print ("Performance on the Validation Dataset:...")
            self._evaluate(model=self.clf, data=self.validation_data, calibrate=calibrate)

            #print ("Performance on the Testing Dataset:...")
            #self._evaluate(model=self.clf, data=self.testing_data, calibrate=calibrate)
            if feature_importance:
                self._feature_importance( )
            

            print( "\n Finished and saving the model..." )
            dump(self.clf, join( config.ML_MODEL_SAVE_PATH, fname_base+self.tags[model_name] ) )
            
            self.clf=None

    def imputer(self, training_data, validation_data, testing_data, fname_base=None, save=True):
        '''
        Imputation transformer for missing values.
        '''
        imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = 0.)
        #imp = IterativeImputer(random_state=0)
        imp.fit(training_data[0])
        training_examples = imp.transform(training_data[0])
        validation_examples = imp.transform(validation_data[0])
        testing_examples = imp.transform(testing_data[0])

        if save:
            dump(imp, join( config.ML_MODEL_SAVE_PATH, 'IMP_'+fname_base+self.tags[self.model_name] ) )

        return training_examples, validation_examples, testing_examples

    def normalize_sklearn(self, training_examples, validation_examples, testing_examples, fname_base=None, save=True):
        '''
        Normalize a dataset.
        '''
        scaler = StandardScaler( )
        # Fit on training set only.
        scaler.fit(training_examples)
        # Apply transform to both the training set and the validation set.
        training_examples = scaler.transform(training_examples)
        validation_examples = scaler.transform(validation_examples)
        testing_examples = scaler.transform(testing_examples)

        if save:
            dump(scaler, join( config.ML_MODEL_SAVE_PATH, 'NORM_'+fname_base+self.tags[self.model_name] ) )

        return training_examples, validation_examples, testing_examples

    def pca_transform( self, training_examples, validation_examples, fname_base ):
        '''
        Peforms Principal Component Analysis on the examples
        '''
        # Make an instance of the Model
        pca = PCA(n_components=5)
        pca.fit(training_examples)
        training_examples = pca.transform(training_examples)
        validation_examples = pca.transform(validation_examples)
        testing_examples = pca.transform(testing_examples)

        dump(pca, join( config.ML_MODEL_SAVE_PATH, 'PCA_'+fname_base+self.tags[self.model_name] ) )

        return training_examples, validation_examples 


    def generate_model_save_names(self, fname_base, **params): 
        '''
        '''
        for key, value in list(params.items( )):
            fname_base += '_{}={}'.format(key, value)

        return fname_base 

    def predict(self, model, examples):
        '''
        Returns the probabilistic predictions from a given machine learning model.
        Args:
            model, object, pre-fitted machine learning object 
            data, 2-tuple of examples and labels to evaluate 
        Returns:
            1-D array, Probabilistic predictions made by the machine learning model 
        '''
        if self.model_name == 'XGBoost':
            predictions = model.predict_proba( examples, ntree_limit=model.best_ntree_limit )[:,1] 

        else:
            predictions = model.predict_proba( X = examples )[:,1]

        return predictions

    def predict_probaCV(self, verify_var, model_name, fcst_time_idx, var, variables_to_remove=[], norm=False, calibrate=False, **params): 
        '''
        Predicts...
        '''
        self.model_name = model_name
        scaling_values = None

        for fold, r in enumerate(range(0,self.num_dates,self.fold_interval)):
            print('\nfold: ', fold)
            fname_base = f'{model_name}_uncalibrated_verify_var={verify_var}_norm={norm}_fold={fold}_fcst_time_idx={fcst_time_idx}'
            folds_testing = (np.arange(self.num1, self.num2) + r) % self.num_dates
            test_df = pd.read_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'testing_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_hagelslag.pkl' ))
            test_examples = test_df.drop( columns = ['matched_to_LSRs_15km', 'matched_to_LSRs_30km', 'matched_to_Tornado_LSRs_15km', 'matched_to_Tornado_LSRs_30km',
                                'matched_to_Severe_Wind_LSRs_15km', 'matched_to_Severe_Wind_LSRs_30km',
                                'matched_to_Hail_LSRs_15km', 'matched_to_Hail_LSRs_30km', 'Run Time', 'Run Date', 'label', 'ensemble_member']+variables_to_remove)
            additional_vars = test_df[['Run Date', 'Run Time', 'label', 'ensemble_member']]
            self.feature_names = list(test_examples.columns)
            if norm: 
                norm_model = load( join( config.ML_MODEL_SAVE_PATH, 'NORM_'+fname_base+self.tags[self.model_name] ))
                test_examples = norm_model.transform( test_examples )
            save_model_fname = join( config.ML_MODEL_SAVE_PATH, fname_base+self.tags[model_name] )
            print (save_model_fname) 
            if model_name == 'CNN':
                pd_filename = join( config.ML_INPUT_PATH, '%s_scaling_values_fold=%s' % ( params['dim_option'], fold  ) )
                scaling_values = save_object_properties( pd_filename )
                model = load_model( save_model_fname )
            else: 
                model = load( save_model_fname )
            
            predictions = self.predict(model, test_examples) 
            if calibrate:
                iso_model = load( join( config.ML_MODEL_SAVE_PATH, fname_base.replace('uncalibrated','calibrated') ))
                predictions = iso_model.transform( predictions ) 

            datetimes = config.datetime_dict_verify( config.verify_forecast_dates[folds_testing], config.verification_times )
            kwargs = { 'predictions':predictions, 'other_vars':additional_vars, 'obj_var': var, 
                       'fcst_time_idx':fcst_time_idx, 'model_name':model_name, 'fold':fold, 'verify_var':verify_var, 'calibrate':calibrate }           
            multiprocessing_per_date( datetimes=datetimes, n_date_per_chunk=8, func=unravel_fcst_probs, kwargs=kwargs)        
            
            
            
            #unravel_fcst_probs(date='20170516', time='0000', kwargs=kwargs)
                  
    def calibrate_fit(self, validation_data, fname_base): 
        '''
        Calibrate a pre-fitted classifier on the validation data set using isotonic regression
        '''     
        validation_predictions = self.predict(self.clf, validation_data[0])
        calibrated_clf = IsotonicRegression( out_of_bounds='clip' )    
        calibrated_clf.fit(validation_predictions.astype(float), validation_data[1].astype(float))

        self.calibrated_clf = calibrated_clf

        fname = fname_base.replace('uncalibrated', 'calibrated')
        dump( calibrated_clf, join( config.ML_MODEL_SAVE_PATH, fname) ) 

        return self

    def balance_examples(self, examples, labels, percent_positive):
        '''
        Returns examples and labels resampled to percent of positive labels given by the percent_positive
        '''
        positive_ratio = float(len(labels[labels==1.0]))/len(labels)
        if positive_ratio > 0.5:
            print ('Ratio of postive to negative examples ({}) already greater than 0.5!'.format(positive_ratio))
            return examples, labels 
        
        if percent_positive == 0.5: 
            positive_idxs = np.where(( labels ==1.0 ))[0]
            negative_idxs = np.where(( labels ==0.0 ))[0]
            random_negative_idxs = random.sample( list(negative_idxs), len(positive_idxs)) 
            idxs = np.concatenate(( positive_idxs, random_negative_idxs ))
            return (examples[idxs.astype(int)], labels[idxs.astype(int)])
        else:
            default_percent_positive = float(np.count_nonzero( labels )) / len(labels)
            reduced_idxs = np.arange(0, len(labels))
            reduced_idxs = np.ndarray.tolist( reduced_idxs )
            error = 1./len(labels)
            if percent_positive < default_percent_positive:
                # randomly remove positive labels
                positive_label_idxs = np.where(( labels == 1) )[0]
                np.random.shuffle( positive_label_idxs )
                for idx in positive_label_idxs:
                    reduced_idxs.remove( idx )
                    new_percent_positive = float(np.count_nonzero( labels[reduced_idxs] )) / len(labels[reduced_idxs])
                    if (new_percent_positive > percent_positive-error and new_percent_positive < percent_positive +error):
                        return (examples[reduced_idxs], labels[reduced_idxs])
            else:
                # randomly remove negative labels       
                negative_label_idxs = np.where(( labels == 0) )[0]
                np.random.shuffle( negative_label_idxs )
                for idx in negative_label_idxs:
                    reduced_idxs.remove( idx )
                    new_percent_positive = float(np.count_nonzero( labels[reduced_idxs] )) / len(labels[reduced_idxs])
                    if (new_percent_positive > percent_positive-error and new_percent_positive < percent_positive +error):
                        return (examples[reduced_idxs], labels[reduced_idxs]) 

    def save_data(self, model_name, var, fcst_time_idx, data_extraction_method ): 
        '''
        Save model data.
        '''
        print ( f'\n Forecast Time Index: {fcst_time_idx} ' )
        for fold, r in enumerate(range(0,self.num_dates,self.fold_interval)):
            print(f'\t fold: {fold}')
            folds_training = (np.arange(self.n_training_dates) + r) % self.num_dates
            folds_validation = (np.arange(self.n_training_dates, self.num1) + r) % self.num_dates
            folds_testing = (np.arange(self.num1, self.num2) + r) % self.num_dates

            training_df  = self.load_data( var=var, 
                                           data_file_paths=[ glob( join(config.ML_INPUT_PATH, str(date)+ f'/{self.mode[model_name]}_WOFS_{var}*_{fcst_time_idx}_{data_extraction_method}.nc')) 
                                               for date in config.ml_dates[folds_training] ]
                                           )
            validation_df  = self.load_data( var=var,
                                           data_file_paths=[ glob( join(config.ML_INPUT_PATH, str(date)+f'/{self.mode[model_name]}_WOFS_{var}*_{fcst_time_idx}_{data_extraction_method}.nc')) 
                                               for date in config.ml_dates[folds_validation] ]
                                           )
            test_df  = self.load_data( var=var,
                                       data_file_paths=[ glob( join(config.ML_INPUT_PATH, str(date)+f'/{self.mode[model_name]}_WOFS_{var}*_{fcst_time_idx}_{data_extraction_method}.nc')) 
                                               for date in config.ml_dates[folds_testing]]        
                                        )

            validation_df.to_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'validation_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_{data_extraction_method}.pkl' ))
            training_df.to_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'training_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_{data_extraction_method}.pkl' ))
            test_df.to_pickle( os.path.join(config.ML_DATA_STORAGE_PATH, f'testing_dataset_fold={fold}_fcst_time_idx={fcst_time_idx}_{data_extraction_method}.pkl' ))
           
            del validation_df, training_df, test_df

    def load_data( self, var, data_file_paths=None, vars_to_load=None ):
        '''
        Load the machine learning data for training and validation. 
        '''
        idx = 4 if 'UPDRAFT' in var else 2
        storm_files = sorted( list(itertools.chain.from_iterable(data_file_paths)))
        if vars_to_load is None:
            total_vars, _, _,_,_ = _feature_names_for_traditional_ml( ) 
        else:
            total_vars = vars_to_load

        data = [ ]
        run_times = [ ] 
        run_dates = [ ]
        for storm_file in storm_files: 
            print ('Loading {}...'.format(storm_file)) 
            ds = xr.open_dataset(storm_file)
            data.append( np.stack( [ds[v].values for v in total_vars], axis=-1) )         
            run_times.append([storm_file.split("/")[-1].split("_")[idx][9:]] * data[-1].shape[0])
            run_dates.append([storm_file.split("/")[-1].split("_")[idx][:8]] * data[-1].shape[0])
            ds.close()

        all_data = np.concatenate( data ) 
        all_run_times = np.concatenate( run_times )
        all_run_dates = np.concatenate( run_dates ) 
        
        data_concat = np.concatenate((all_data, all_run_times[:,np.newaxis], all_run_dates[:,np.newaxis]), axis=1)
        total_vars += ['Run Time', 'Run Date'] 

        return pd.DataFrame(data=data_concat, columns = total_vars)


    def permutation_importance(self):
        '''
        Single Pass Permutation Importance.
        '''
        scaling_values = None
        examples = self.validation_data[0]
        targets = self.validation_data[1]
        validation_predictions = self.predict(self.clf, examples, scaling_values=scaling_values)
        original_auc = roc_auc_score( y_true = targets, y_score = validation_predictions ) 
        importance = {}
        for i, var in enumerate(self.feature_names):
            print (i) 
            copy_examples = np.copy(examples) 
            copy_examples[:,i] = np.random.permutation(examples[:,i])
            new_predictions = self.predict(self.clf, copy_examples, scaling_values=scaling_values)
            auc = roc_auc_score( y_true = targets, y_score = new_predictions )
            deviation = original_auc - auc
            importance[var] = auc 
       
        ranked_importance = sorted( importance.items(), reverse=True )

        print (original_auc)
        print( ranked_importance )

    def _feature_importance(self, evaluation_fn = roc_auc_score ):
        '''
        Diagnose the important features using the permutation importance (calls Eli's code).

        The method 'fit' has to be called first     
        '''
        #self.permutation_importance() 
        model_list = [ 'RandomForest', 'GradientBoost', 'LogisticRegression', 'XGBoost']
        if self.model_name in model_list:
            _, _, _, _, (VARIABLE_NAMES_DICT, VARIABLE_COLORS_DICT) = _feature_names_for_traditional_ml( obj_props_train=True, feature_importance=True )
            print('\n Start Time:', datetime.now().time())
            result = sklearn_permutation_importance( model = self.clf, 
                                                     scoring_data = self.validation_data, 
                                                     evaluation_fn = roc_auc_score, 
                                                     variable_names = self.feature_names, 
                                                     scoring_strategy = 'argmin_of_mean', 
                                                     subsample=1, 
                                                     nimportant_vars = 10, 
                                                     njobs = 0.5, 
                                                     nbootstrap = 50) 
       
            print('End Time: ', datetime.now().time())
            print ('\n plotting the feature importances...')
            plot_variable_importance(
                    result, f'singlepass_permutation_{self.model_name}_{self.verify_var}_{self.fcst_time_idx}_{self.fold}.png', 
                    readable_feature_names=VARIABLE_NAMES_DICT, 
                    feature_colors=VARIABLE_COLORS_DICT, 
                    multipass=False, fold = self.fold) 
    
            plot_variable_importance(
                    result, f'multiplepass_permutation_{self.model_name}_{self.verify_var}_{self.fcst_time_idx}_{self.fold}.png',                                               
                    readable_feature_names=VARIABLE_NAMES_DICT, 
                    feature_colors=VARIABLE_COLORS_DICT, 
                    multipass=True, fold = self.fold)         
    

    def is_cross_validation_good(self):
        '''
        Testing if the cross-validation splits are good.
        '''

        test_list = [ ]

        for fold, r in enumerate(range(0,self.num_dates,self.fold_interval)):
            folds_training = (np.arange(self.n_training_dates) + r) % self.num_dates
            folds_validation = (np.arange(self.n_training_dates, self.num1) + r) % self.num_dates
            folds_testing = (np.arange(self.num1, self.num2) + r) % self.num_dates

            train_dates = config.ml_dates[folds_training]
            valid_dates = config.ml_dates[folds_validation]
            test_dates = config.ml_dates[folds_testing]

            test_list.extend(test_dates)

            #print ('Fold {}---> \n\t Testing Dates: {}'.format(fold, test_dates))
                
            train_count = len(train_dates)
            valid_count = len(valid_dates)
            test_count = len(test_dates)

            print ('Checking if this fold is good...')
            print (any(item in train_dates for item in valid_dates),any(item in train_dates for item in test_dates) )
            print (f'\n Num. of training dates : {train_count},\
                   \n Num. of validation dates: {valid_count},\
                   \n Num. of testing dates   : {test_count}')
        
        unique_list = [] 
        for date in test_list:
            if date not in unique_list:
                unique_list.append(date)

        print ('Total number of dates in the testing set: {}'.format(len(unique_list)))


    def _cross_validation_for_best_params(self, model_name, verify_var, fcst_time_idx, data_extraction_method, 
                                            removed_variables, use_select_features, param_grid):
        '''
        Performs cross-validation to find the best parameters for the model given. 
        '''
        self.use_select_features = use_select_features
        self.model_name = model_name
        self.verify_var = verify_var
        self.fcst_time_idx = fcst_time_idx 
        scores_per_fold = [ ]
        #for fold, r in enumerate(range(0,self.num_dates,self.fold_interval)):
        for fold in range(2): 
            print('\nfold: ', fold)

            training_data, validation_data, testing_data = self.load_dataframe( fold, verify_var, fcst_time_idx, data_extraction_method, removed_variables)
            training_examples, training_labels = self.balance_examples( training_data[0], training_data[1], percent_positive = self.percent_positive_examples_in_training )
            training_data = (training_examples, training_labels)
            print(("Number of training examples: ", training_examples.shape))

            scores = self._determine_score_per_params( model_name=model_name, param_grid=param_grid, training_data=training_data,
                                validation_data=validation_data )
            scores_per_fold.append( scores )

        avg_scores = np.mean( scores_per_fold, axis = 0)
        best_params = self._best_params( param_grid, avg_scores )

        return best_params, avg_scores

            
    def _determine_score_per_params(self, model_name, param_grid, training_data, validation_data):
        '''
        Find the scores for a training/validation fold.
        '''
        scores = [ ]
        keys, values = list(zip(*list(param_grid.items())))
        for v in itertools.product(*values):
            params = dict(list(zip(keys, v)))
            print ('Evaluating {} with the following params: {}'.format(model_name, params))
            self.fit( model_name=model_name, params=params, training_data=training_data)
            auc = self._evaluate( model = self.clf, data = validation_data)
            scores.append( auc )

        return scores

    def _best_params(self, param_grid, avg_scores):
        '''
        Find the best parameters. 
        '''
        keys, values = list(zip(*list(param_grid.items())))
        possible_params = np.array( [ dict(list(zip(keys, v))) for v in itertools.product(*values) ] )
        idx = np.argmax( avg_scores )

        return possible_params[idx] 

    def _evaluate(self, model, data, calibrate=False):
        '''
        Evaluate the model using area under the curve. 
        '''
        prediction = model.predict_proba( data[0] )[:,1]
        auc = roc_auc_score( y_true = data[1], y_score = prediction )
        bss = brier_skill_score(data[1], prediction)
        bxe = _get_binary_xentropy(data[1], prediction)

        if calibrate:
            calibrated_predictions = self.calibrated_clf.predict(prediction)
            auc_c = roc_auc_score( y_true = data[1], y_score = calibrated_predictions )
            bss_c = brier_skill_score(data[1], calibrated_predictions)
            bxe_c = _get_binary_xentropy(data[1], calibrated_predictions) 
            print ( 'CAL:....AUC: {0:.3f}   BSS: {1:.3f}   BXE: {2:.3f}'.format(auc_c, bss_c, bxe_c))

        #print ( 'UNCAL:....AUC: {0:.3f}   BSS: {1:.3f}   BXE: {2:.3f}'.format(auc, bss, bxe))
        return auc 


    def calc_climo(self, var, fcst_time_idx, verify_var, data_extraction_method='hagelslag' ):
        '''
        Calculate the climatology. 
        '''
        df = self.load_data( var=var, vars_to_load = [verify_var],
                        data_file_paths=[ glob( join(config.ML_INPUT_PATH, str(date)+ f'/ML_WOFS_{var}*_{fcst_time_idx}_{data_extraction_method}.nc'))
                                               for date in config.ml_dates]) 
        
        labels = df[verify_var].values.astype(float)
        ratio = round( np.count_nonzero( labels ) / float(len(labels)), 3)
       
        print("Full climatology: ", ratio)
        print("Total Number of examples: ", np.shape( labels )) 
    
    def plot_reliability(self, model, scaling_values, fig_filename):
        '''
        Produce reliability diagram.
        '''
        uncalibrated_testing_predictions = self.predict(self.clf, self.testing_data[0])
        calibrated_testing_predictions = self.calibrated_clf.predict(uncalibrated_testing_predictions)

        mean_fcst_prob1, event_frequency1 = Metrics.reliability_curve( uncalibrated_testing_predictions, self.testing_data[1], bins=np.arange(0, 1.1, 0.1) )       
        mean_fcst_prob2, event_frequency2 = Metrics.reliability_curve( calibrated_testing_predictions, self.testing_data[1], bins=np.arange(0, 1.1, 0.1) )       
        
        fig, axes = my_plt._create_fig( fig_num = 0, figsize = (8, 9))
        subpanel_labels  = [ ('', '')]
        linestyles  = [ '-', '-', '-', '-', '-' ]
        line_colors = [ 'k', 'r' ]
        line_labels = [ 'Uncalibrated', 'Calibrated' ]

        verification_plots.plot_attribute_diagram( ax = axes,
                mean_prob = [mean_fcst_prob1, mean_fcst_prob2],
                event_frequency = [event_frequency1, event_frequency2],
                fcst_probs = [ ],
                line_colors = line_colors,
                line_labels = line_labels,
                linestyles = linestyles,
                subpanel_labels = subpanel_labels[0],
                counter= 2,
                event_freq_err = None,
                inset_loc='upper left',
                error = False )

        my_plt._add_major_frame( fig, xlabel_str='Mean Forecast Probability', ylabel_str='Observed Frequency', title = '' )
        my_plt._save_fig( fig=fig, fname = fig_filename)

    def plot_performance_curve(self, model, scaling_values, fig_filename):
        '''
        Produce Performance Diagram
        '''
        uncalibrated_testing_predictions = self.predict(self.clf, self.testing_data[0])
        calibrated_testing_predictions = self.calibrated_clf.predict(uncalibrated_testing_predictions) #self.predict(self.calibrated_clf, self.testing_data[0], scaling_values=scaling_values)
       
        pod1,sr1 = Metrics.performance_curve( uncalibrated_testing_predictions, self.testing_data[1], bins=np.arange(0, 1.0, 0.01) )  
        pod2,sr2 = Metrics.performance_curve( calibrated_testing_predictions, self.testing_data[1], bins=np.arange(0, 1.0, 0.01) )

        fig, axes = my_plt._create_fig( fig_num = 0, figsize = (8, 9))
        subpanel_labels  = [ ('', '')]
        linestyles  = [ '-', '-', '-', '-', '-' ]
        line_colors = [ 'k', 'r' ]
        line_labels = [ 'Uncalibrated', 'Calibrated' ]

        csiLines = verification_plots.plot_performance_diagram( ax = axes,
            pod = [pod1, pod2],
            sr = [sr1, sr2],
            line_colors = line_colors,
            line_labels = line_labels,
            linestyles = linestyles,
            subpanel_labels = subpanel_labels[0],
            counter = 2,
            error = False)

        my_plt._add_major_frame( fig, xlabel_str='Success Ratio (1-FAR)', ylabel_str='Probability of Detection', title = '' )
        my_plt._add_major_colobar( fig, contours=csiLines, label = 'Critical Success Index' )
        my_plt._save_fig( fig=fig, fname = fig_filename) 

    def plot_roc_curves(self, model, scaling_values, fig_filename):
        '''
        Produce ROC curves on the training and validation datasets.
        '''
        uncalibrated_testing_predictions = self.predict(self.clf, self.testing_data[0])
        calibrated_testing_predictions = self.calibrated_clf.predict(uncalibrated_testing_predictions)

        pod1, _, pofd1 = Metrics.performance_curve( uncalibrated_testing_predictions, self.testing_data[1], bins=np.arange(0, 1.0, 0.05), roc_curve=True )
        pod2, _, pofd2 = Metrics.performance_curve( calibrated_testing_predictions, self.testing_data[1], bins=np.arange(0, 1.0, 0.05), roc_curve=True )        

        auc1 = roc_auc_score( y_true = self.testing_data[1], y_score = uncalibrated_testing_predictions )
        auc2 = roc_auc_score( y_true = self.testing_data[1], y_score = calibrated_testing_predictions )
        
        fig, axes = my_plt._create_fig( fig_num = 0, figsize = (8, 9))
        subpanel_labels  = [ ('', '')]
        linestyles  = [ '-', '-', '-', '-', '-' ]
        line_colors = [ 'k', 'r' ]
        line_labels = [ 'Uncalibrated', 'Calibrated' ]

        verification_plots.plot_roc_curve( ax = axes,
            pod = [pod1, pod2],
            pofd = [pofd1, pofd2],
            line_colors = line_colors,
            line_labels = ['AUC:{}'.format(round(auc1,2)),'AUC:{}'.format(round(auc2,2))], 
            subpanel_labels = subpanel_labels[0],
            counter = 2)
        
        my_plt._add_major_frame( fig, xlabel_str='Probability of False Detection (POFD)', ylabel_str='Probability of Detection', title = '' )
        my_plt._save_fig( fig=fig, fname = fig_filename) 

    def plot_training_curves(self, fold, fig_filename):
        '''
        Plots the binary cross-entropy as a function of the training epoches to diagnose if the 
        CNN is learning and whether it is potentially overfitting. 
        '''
        plt.figure( fold , figsize = (8,8) )
        plt.plot(self.model_training_hist.epoch, self.model_training_hist.history["val_loss"], label="validation", linewidth = 3.0, color = 'r')
        plt.plot(self.model_training_hist.epoch, self.model_training_hist.history["loss"], label="train", linewidth = 3.0, color ='b')
        plt.ylim(0, 1.3)
        plt.legend(fontsize = 20 )
        plt.grid( alpha = 0.5 )
        plt.ylabel("Loss", fontsize = 20 )
        plt.xlabel("Epoch", fontsize = 20 )
        plt.savefig( fig_filename, bbox_inches = 'tight', format = 'png')
        plt.close( )

def fill(original_probs, labels):
    copy_probs = np.zeros(labels.shape)
    for label in np.unique(labels)[1:]:
        ens_probs = original_probs[:,labels==label]
        copy_probs[labels==label] = np.average(np.amax(ens_probs, axis=1))

    return copy_probs

def unravel_fcst_probs( date, time, kwargs):
    '''
    Unravels the 1D prediction array into the 2D forecast probabilities.
    '''
    df = kwargs['other_vars'].astype({'ensemble_member':float, 'label':float, 'Run Date': str, 'Run Time': str})
    prediction = kwargs['predictions']

    # Load 2D storm object labels 
    in_path = join( config.OBJECT_SAVE_PATH, date )
    object_file = join(in_path, 'WOFS_%s_OBJECTS_%s-%s_%02d.nc' % (kwargs['obj_var'], date, time, kwargs['fcst_time_idx']))
    ds = xr.open_dataset( object_file)
    ens_object_labels_2D = ds['Objects'].values
    ds.close() 

    fcst_probs_2D  = np.zeros(( ens_object_labels_2D.shape ))
    for mem_idx in range(config.N_ENS_MEM):
        idx = df[(df['Run Date'] == str(date)) & (df['Run Time'] == str(time)) & (df['ensemble_member'] == float(mem_idx)) ].index.values.astype(int)
        object_labels_1D = df['label'].values[idx]
        predictions_temp = prediction[idx]
        for i, label in enumerate(object_labels_1D):
            fcst_probs_2D[mem_idx, ens_object_labels_2D[mem_idx, :, :] == label] = predictions_temp[i]

    qc_params = {'min_area':12., 'merge_thresh': 2.}
    watershed_params_fcst = {'min_thresh': 1,
                         'max_thresh': 75,
                         'data_increment': 10,
                         'delta': 100,
                         'size_threshold_pixels': 500,
                         'dist_btw_objects': 15 } 


    avg_ens_probs = np.average(fcst_probs_2D, axis=0)
    processed_data = np.round(100.*avg_ens_probs,5)
    labels_fcst, props_fcst = label_id( processed_data, method='watershed', params=watershed_params_fcst)
    qc_object_labels, qc_object_props = qc.quality_control(object_labels=labels_fcst, object_properties=props_fcst, input_data = avg_ens_probs, qc_params=qc_params )
    
    new_ens_probs = fill(fcst_probs_2D, qc_object_labels)         

    out_path = join( config.ML_FCST_PATH, str(date) )
    fname = join( out_path, '%s_fcst_calibrate=%s_verify_var=%s_%s-%s_%s_fold=%s.nc' % ( kwargs['model_name'], kwargs['calibrate'], kwargs['verify_var'], date, time, kwargs['fcst_time_idx'], kwargs['fold'] ))
    data = { }
    data['Objects'] = ( ['Ensemble Member', 'Y', 'X'], ens_object_labels_2D )
    data['3D Probabilities'] = ( ['Ensemble Member', 'Y', 'X'], fcst_probs_2D )
    data['2D Probabilities'] = ( ['Y', 'X'], new_ens_probs )
    data['Probability Objects'] = ( ['Y', 'X'], qc_object_labels )

    ds = xr.Dataset( data )
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    ds.to_netcdf( path = fname )
    ds.close( )


