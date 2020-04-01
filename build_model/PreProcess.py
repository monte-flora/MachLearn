# Import Modules
import os
import numpy as np
import xarray as xr
from os.path import join, exists
from glob import glob
import itertools
import random
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import pandas as pd
from datetime import datetime
from collections import Counter
import pickle

# Personal Modules
from imblearn.under_sampling import RandomUnderSampler
from wofs.util import config

ratio_dict = {
    "matched_to_tornado_0km": float(1.123 / (100.0 - 1.123)),
    "matched_to_severe_hail_0km": float(3.943 / (100.0 - 3.943)),
    "matched_to_severe_wind_0km": float(2.503 / (100.0 - 2.503)),
}


class PreProcess:
    """
    PreProcess is a class for pre-processing a traditional ML dataset
    Also includes loading and saving data and checking the 
    cross-validation.
    """

    EXAMPLES = "examples"
    TARGETS = "targets"
    OTHER_TARGETS = "other_targets"
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATE = "evaluate"
    TESTING = "testing"
    INFO = "info"

    EXTRA_VARIABLES = [
        "Run Time",
        "Run Date",
        "FCST_TIME_IDX",
        "label",
    ]

    TARGETS_VARIABLES = [
        "matched_to_LSRs_15km",
        "matched_to_LSRs_0km",
        "matched_to_tornado_15km",
        "matched_to_tornado_0km",
        "matched_to_severe_wind_15km",
        "matched_to_severe_wind_0km",
        "matched_to_severe_hail_15km",
        "matched_to_severe_hail_0km",
    ]

    def _is_cross_validation_good(
        self, training_dates, validation_dates, testing_dates
    ):
        """
        Ensure the training, validation, and testing folds do not share a single date!
        All values should be False!
        """
        values = [
            any(item in training_dates for item in validation_dates),
            any(item in testing_dates for item in validation_dates),
            any(item in training_dates for item in testing_dates),
            any(item in validation_dates for item in testing_dates),
            any(item in testing_dates for item in training_dates),
            any(item in validation_dates for item in training_dates),
        ]

        return sum(values)

    def _calc_cross_validation_fold_parameters(
        self,
        n_cv_folds=15,
        list_of_dates=config.ml_dates,
        percent_training=0.8,
        percent_testing=6,
        verbose=True,
        percent_validation=0.1,
    ):
        """
        Internal function for my purposes.

        Args:
        -----------
            n_cv_folds, int, number of cross-validation folds
            dates, list, list of all dates in the dataset
            percent_training, float, percent of dates used in the training set (between 0-1)
            percent_testing, float, percent of dates used in the testing set (between 0-1)
            percent_validation, float, percent of dates used in the validation set (between 0-1)
        """
        num_of_dates = len(list_of_dates)
        if num_of_dates % n_cv_folds != 0:
            fold_interval = int((num_of_dates / n_cv_folds) + 1)
        else:
            fold_interval = int((num_of_dates / n_cv_folds))
        num_of_training_dates = int(percent_training * num_of_dates)
        if 0 <= percent_testing <= 1:
            num_of_testing_dates = int(percent_testing * num_of_dates)
            num_of_validation_dates = int(percent_validation * num_of_dates)
        else:
            num_of_testing_dates = percent_testing
            num_of_validation_dates = (
                num_of_dates - num_of_training_dates - num_of_testing_dates
            )

        count_training_and_validation = num_of_training_dates + num_of_validation_dates
        count_training_validation_testing = (
            count_training_and_validation + num_of_testing_dates
        )

        dates_dict = {}
        map_date_to_fold = {}

        testing_dates = []
        for fold, r in enumerate(range(0, num_of_dates, fold_interval)):
            if verbose:
                print("fold: ", fold)
            this_training_folds_dates = list_of_dates[
                (np.arange(num_of_training_dates) + r) % num_of_dates
            ]
            this_validation_folds_dates = list_of_dates[
                (np.arange(num_of_training_dates, count_training_and_validation) + r)
                % num_of_dates
            ]
            this_testing_folds_dates = list_of_dates[
                (
                    np.arange(
                        count_training_and_validation, count_training_validation_testing
                    )
                    + r
                )
                % num_of_dates
            ]

            if verbose:
                print(
                    "Number of Training dates  : {}".format(
                        len(this_training_folds_dates)
                    )
                )
                print(
                    "Number of Validation dates: {}".format(
                        len(this_validation_folds_dates)
                    )
                )
                print(
                    "Number of Testing dates   : {}".format(
                        len(this_testing_folds_dates)
                    )
                )

            testing_dates.extend(this_testing_folds_dates)
            if verbose:
                print("Checking if this fold is good...")
            value = self._is_cross_validation_good(
                training_dates=this_training_folds_dates,
                validation_dates=this_validation_folds_dates,
                testing_dates=this_testing_folds_dates,
            )
            if value > 0:
                raise ValueError("Cross-validation is not pure!")

            dates_dict[fold] = {
                "training": this_training_folds_dates,
                "validation": this_validation_folds_dates,
                "testing": this_testing_folds_dates,
            }

            for date in this_testing_folds_dates:
                if date not in list(map_date_to_fold.keys()):
                    map_date_to_fold[date] = fold

        if verbose:
            unique_dates = []
            for date in testing_dates:
                if date not in unique_dates:
                    unique_dates.append(date)

            print(
                "Number of Unique dates in the testing set: {} vs. \
                \n Number of all testing set :{}".format(
                    len(unique_dates), len(testing_dates)
                )
            )

            print(
                f"Number of dates in map_date_to_fold: {len(list(map_date_to_fold.keys()))}"
            )
            print(map_date_to_fold)

        return dates_dict, map_date_to_fold
    
    def load_cv_data(
        self,
        fold_to_load=None,
        modes=[],
    ):
        """
        Load the training, validation, and testing datasets for 
        all cross-validation folds
        """
        if len(modes) == 0:
            modes = [self.TRAINING, self.VALIDATION, self.TESTING]

        target_var_name = self.fname_params["target_var"]
        fcst_time_idx = self.fname_params['fcst_time_idx']
        
        load_filename_template = "{}_f:{}_t:{}_raw_probability_objects.pkl"
        blank_load_filename_dict = self._to_save_filenames(
            modes, load_filename_template, target_var_name
        )

        total = sorted(list(self.dates_dict.keys()))
        if fold_to_load is not None:
            total = [fold_to_load]
        
        data_dict = {}
        for fold in total:
            print("\n Current fold: ", fold)
            print("Start Time: ", datetime.now().time())
            self.fname_params["fold"] = fold
            load_filename_dict = {
                mode: join(
                    config.ML_DATA_STORAGE_PATH,
                    mode,
                    blank_load_filename_dict[mode].format(mode, fold, fcst_time_idx),
                )
                for mode in modes
            }

            data = self.load_dataframe(
                target_var_name=target_var_name, load_filename_dict=load_filename_dict, additional_vars_to_drop=self.additional_vars_to_drop
            )

            filename_dict = self._generate_filenames()

            if self.model_name == 'LogisticRegression':
                data = self.normalize(data, filename_dict["norm_clf"])

            if self.load_model:
                print("Loading {}...".format(filename_dict["main_clf"]))
                model = load(filename_dict["main_clf"])
                cali_model = load(filename_dict["cal_clf"])
            else:
                model = None
                cali_model = None
        
            data_dict["fold_{}".format(fold)] = {
                "model": model,
                "cal_model": cali_model,
                "filename_dict": filename_dict,
                "data": data,
            }

        return data_dict

    def _generate_filenames(self):
        """
        Generates filename for a machine learning model
        """
        print('self.opt: {}'.format(self.opt))

        path = join(
            join(
                config.ML_MODEL_SAVE_PATH,
                f'FCST_TIME_IDX={self.fname_params["fcst_time_idx"]}',
            ),
            f'{self.model_name}',
        )

        # Need to save the main classifier and the imputation model
        fname_of_main_clf = f'model:{self.model_name}_{self.opt}:{self.fname_params["target_var"]}_fold:{self.fname_params["fold"]}.joblib'
        fname_of_isotonic_model = f'model:{self.model_name}_isotonic_{self.opt}:{self.fname_params["target_var"]}_fold:{self.fname_params["fold"]}.joblib'
        fname_of_norm_model = f'model:{self.model_name}_norm_{self.opt}:{self.fname_params["target_var"]}_fold:{self.fname_params["fold"]}.joblib'

        fname_of_results = join(
            config.ML_RESULTS_PATH,
            f'verifyData_{self.model_name}_target:{self.fname_params["target_var"]}_fcst_time_idx={self.fname_params["fcst_time_idx"]}_{self.opt}.nc',
        )

        fname_of_main_clf = join(path, fname_of_main_clf)
        fname_of_isotonic_model = join(path, fname_of_isotonic_model)
        fname_of_norm_model = join(path, fname_of_norm_model)

        return {
            "main_clf": fname_of_main_clf,
            "cal_clf": fname_of_isotonic_model,
            "results": fname_of_results,
            "norm_clf" : fname_of_norm_model
        }

    def _to_save_filenames(self, modes, load_filename_template, target_var_name):
        """
        Internal use only. 
        Returns the filenames for training, validation, and testing files
        
        Parameters: 
            modes : list 
                lists of including some combination of 'training', 'validation', and 'testing' 
                these are what will be loaded per CV fold
            load_filename_template_dict : str
                A string template for filename to save to (no path included)
                The format used has three empty spaces ({}) for mode (see above), fold (as int), 
                and the forecast time index (as int)
        
        """
        blank_save_filename_dict = {}
        for mode in modes:
            if mode == self.TRAINING or mode == self.VALIDATION:
                save_filename_template = load_filename_template.replace(
                    "raw", "resampled_to_{}".format(target_var_name),
                )
                blank_save_filename_dict[mode] = save_filename_template
            else:
                blank_save_filename_dict[mode] = load_filename_template

        return blank_save_filename_dict

    def preprocess(self, fname_params, modes, load_filename_template):
        """
        Preprocess the ML data by performing imputation, resampling the examples to 1:1, and then 
        removing highly correlated features
        
        Args:
        ----------
        dict_of_dataframes: nested dictionary where the keys can be 'training', 'validation, or 'testing
                            and items are dictionary of EXAMPLES and TARGETS
        """
        target_var_name = fname_params["target_var"]
        fcst_time_idx = fname_params["fcst_time_idx"]

        blank_save_filename_dict = self._to_save_filenames(
            modes, load_filename_template, target_var_name
        )

        dates_dict, _ = self._calc_cross_validation_fold_parameters(verbose=False)
        for fold in sorted(list(dates_dict.keys())):
            print("\n Current fold: ", fold)
            fname_params["fold"] = fold

            # Build the filenames for saving the processed dataframes
            save_filename_dict = {
                mode: join(
                    config.ML_DATA_STORAGE_PATH,
                    mode,
                    blank_save_filename_dict[mode].format(mode, fold, fcst_time_idx),
                )
                for mode in modes
            }

            load_filename_dict = {
                mode: join(
                    config.ML_DATA_STORAGE_PATH,
                    mode,
                    load_filename_template.format(mode, fold, fcst_time_idx),
                )
                for mode in modes
            }

            data = self.load_dataframe(
                target_var_name=target_var_name, load_filename_dict=load_filename_dict
            )

            # Perform imputation
            print("Performing imputation...")
            data = self._imputer(data=data, simple=True, save=False)

            # Combine other data back into examples before resampling
            for mode in modes:
                data[mode][self.EXAMPLES] = pd.concat(
                    [
                        data[mode][self.EXAMPLES],
                        data[mode][self.OTHER_TARGETS],
                        data[mode][self.INFO],
                    ],
                    axis=1,
                    sort=False
                )

            # Resample the training dataset to 1:1 and validation dataset
            # to the climatology of the given target variable
            print("Resampling the training and validation datasets...")
            data = self.resample(
                data=data,
                modes=[self.TRAINING, self.VALIDATION],
                ratio=ratio_dict[target_var_name],
            )

            for mode in modes:
                df = data[mode][self.EXAMPLES] 
                # Add the info of the concatenate dict back in.
                save_name = save_filename_dict[mode]
                print("Saving {}...".format(save_name))
                df.to_pickle( save_name )
                del df

    def resample(self, data, ratio=None):
        """
        Resamples a dataset to 1:1 using the imblearn python package
        """
        for mode in list(data.keys()):
            if mode == self.TRAINING:
                model = RandomUnderSampler(random_state=42)
            elif mode == self.VALIDATION:
                model = RandomUnderSampler(
                    random_state=42, sampling_strategy=ratio, replacement=True
                )

            examples_resampled, targets_resampled = model.fit_resample(
                data[mode][self.EXAMPLES], data[mode][self.TARGETS]
            )

            data[mode][self.EXAMPLES] = examples_resampled
            data[mode][self.TARGETS] = targets_resampled
            print(
                "Targets after resampling.. {}".format(
                    Counter(data[mode][self.TARGETS])
                )
            )
            print(
                np.shape(data[mode][self.EXAMPLES]), np.shape(data[mode][self.TARGETS])
            )

        return data

    def correlation_filtering(self, df, cc_val):
        """
        filter a dataframe by correlation
        2. Determine the pair of predictors with the highest correlation 
        greater than a given threshold

        3. Determine the average correlation between these predictors and
            the remaining features

        4. Remove the predictors with the highest average correlation. 
        """
        corr_matrix = df.corr().abs()
        feature_names = df.columns.to_list()

        correlated_pairs = {}
        columns_to_drop = []
        # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
        sol = (
            corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            .stack()
            .sort_values(ascending=False)
        )

        all_feature_pairs_above_thresh = sol[sol > cc_val]

        all_feature_pairs = sol.index.values
        all_feature_pairs_above_thresh_idx = all_feature_pairs_above_thresh.index.values

        for pair in all_feature_pairs_above_thresh_idx:
            predictor_A = pair[0]
            predictor_B = pair[1]
            if predictor_A in columns_to_drop or predictor_B in columns_to_drop:
                continue
            predictors_A_set = [
                (predictor_A, feature)
                for feature in feature_names
                if (feature != predictor_B and feature not in columns_to_drop)
            ]
            predictors_B_set = [
                (predictor_B, feature)
                for feature in feature_names
                if (feature != predictor_A and feature not in columns_to_drop)
            ]

            avg_corr_with_A = np.nanmean(
                sol[sol.index.isin(set(predictors_A_set))].values
            )
            avg_corr_with_B = np.nanmean(
                sol[sol.index.isin(set(predictors_B_set))].values
            )
            if avg_corr_with_A > avg_corr_with_B:
                columns_to_drop.append(predictor_A)
                correlated_pairs[predictor_A] = (
                    predictor_B,
                    all_feature_pairs_above_thresh[pair],
                )
            else:
                columns_to_drop.append(predictor_B)
                correlated_pairs[predictor_B] = (
                    predictor_A,
                    all_feature_pairs_above_thresh[pair],
                )

        return columns_to_drop, correlated_pairs

    def filter_df_by_correlation(self, inp_data, target_var, cc_val=0.8):
        """
        Returns an array or dataframe (based on type(inp_data) adjusted to drop \
            columns with high correlation to one another. Takes second arg corr_val
            that defines the cutoff

        ----------
        inp_data : np.array, pd.DataFrame
            Values to consider
        corr_val : float
            Value [0, 1] on which to base the correlation cutoff
        """
        # Creates Correlation Matrix
        corr_matrix = inp_data.corr()

        # Iterates through Correlation Matrix Table to find correlated columns
        columns_to_drop = []
        n_cols = len(corr_matrix.columns)

        correlated_features = []
        print("Calculating correlations between features...")
        for i in range(n_cols):
            for k in range(i + 1, n_cols):
                val = corr_matrix.iloc[k, i]
                col = corr_matrix.columns[i]
                row = corr_matrix.index[k]
                col_to_target = corr_matrix.loc[col, target_var]
                row_to_target = corr_matrix.loc[row, target_var]
                if abs(val) >= cc_val:
                    # Prints the correlated feature set and the corr val
                    if (
                        abs(col_to_target) > abs(row_to_target)
                        and (row not in columns_to_drop)
                        and (row not in self.IGNORED_VARIABLES)
                    ):
                        # print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {row}')
                        columns_to_drop.append(row)
                        correlated_features.append((row, col))
                    if (
                        abs(row_to_target) > abs(col_to_target)
                        and (col not in columns_to_drop)
                        and (col not in self.IGNORED_VARIABLES)
                    ):
                        # print( f'{col} ({col_to_target:.3f}) | {row} ({row_to_target:.3f}) | {val:.2f}....Dropped {col}')
                        columns_to_drop.append(col)
                        correlated_features.append((row, col))

        # Drops the correlated columns
        print("Dropping {} highly correlated features...".format(len(columns_to_drop)))
        print(len(columns_to_drop) == len(correlated_features))
        df = self.drop_columns(inp_data=inp_data, to_drop=columns_to_drop)

        return df, columns_to_drop, correlated_features

    def drop_columns(self, inp_data, to_drop):
        """
        """
        # Drops the correlated columns
        columns_to_drop = list(set(to_drop))
        inp_data = inp_data.drop(columns=columns_to_drop)
        # Return same type as inp
        return inp_data

    def _imputer(self, data, simple=True, save=False):
        """
        Imputation transformer for missing values.
        """
        if simple:
            imp = SimpleImputer(missing_values=np.nan, strategy="mean")
        else:
            imp = IterativeImputer(random_state=0)
        imp.fit(data[self.TRAINING][self.EXAMPLES])

        for mode in list(data.keys()):
            data[mode][self.EXAMPLES] = pd.DataFrame(
                data=imp.transform(data[mode][self.EXAMPLES]),
                columns=data[mode]["feature_names"],
            )

        # save the model
        if save:
            print(f"Saving {self.fname_of_imputer}...")
            dump(imp, self.fname_of_imputer)

        return data

    def pca_transform(self, data):
        """
        Peforms Principal Component Analysis on the examples
        """
        # Make an instance of the Model
        pca = PCA(n_components=5)
        pca.fit(training_examples)
        for mode in list(data.keys()):
            data[mode][self.EXAMPLES = pd.DataFrame(
                data=pca.transform(data[mode][self.EXAMPLES]),
                columns=data[mode]["feature_names"],
            )
        return data, pca 

    def normalize(self, data, fname_of_norm_model):
        """
        Normalize a dataset.
        """
        if os.path.exists(fname_of_norm_model):
            print(f"Loading {fname_of_norm_model}...")
            scaler = load(fname_of_norm_model)
        else:
            scaler = StandardScaler()

        # Fit on training set only.
        scaler.fit(data[self.TRAINING][self.EXAMPLES])
        # Apply transform to both the training set and the validation set.
        for mode in list(data.keys()):
            data[mode][self.EXAMPLES] = pd.DataFrame(
                data=scaler.transform(data[mode][self.EXAMPLES]),
                columns=data[mode]["feature_names"],
            )

        #print(f"Saving {fname_of_norm_model}...")
        #if fname_of_norm_model is not None:
        #    dump(scaler, fname_of_norm_model)

        return data

    def getfiles(self, key, dates):
        """get filenames with globbing"""
        if key == "first_hour":
            fname_strs = (
                "PROBABILITY_OBJECTS*[0][0-9].nc",
                "PROBABILITY_OBJECTS*[1][0-2].nc",
            )
        elif key == "second_hour":
            fname_strs = (
                "PROBABILITY_OBJECTS*[1][3-9].nc",
                "PROBABILITY_OBJECTS*[2][0-4].nc",
            )

        data_file_paths = []
        for fname_str in fname_strs:
            data_file_paths.extend(
                [
                    glob(join(config.ML_INPUT_PATH, str(date), fname_str))
                    for date in dates
                ]
            )

        return data_file_paths

    def save_data(
        self,
        key,
        save_filename_template="{}_f:{}_t:{}_raw.pkl",
        var="UPDRAFT_SWATH",
        modes=[TRAINING, VALIDATION, TESTING],
    ):
        """
        Save model data.
        """
        dates_dict, _ = self._calc_cross_validation_fold_parameters()
        for fold in sorted(list(dates_dict.keys())):
            print(f"\t fold: {fold}")
            for mode in modes:
                print("Loading data for this mode: {}".format(mode))
                data_file_paths = self.getfiles(key, dates = dates_dict[fold][mode])
                df = self.load_data(var=var, data_file_paths=data_file_paths)
                print(df.values.shape)
                df.to_pickle(
                    join(
                        config.ML_DATA_STORAGE_PATH,
                        mode,
                        save_filename_template.format(mode, fold, key),
                    )
                )
                del df

    def _save_operational_dataset(self, key):
        """
        Save a random 80% of dates for operational model 
        """
        with open('/home/monte.flora/machine_learning/util/operational_training_dates.pkl', 'rb') as pkl_file:
            training_dates = pickle.load(pkl_file)

        with open('/home/monte.flora/machine_learning/util/operational_validation_dates.pkl', 'rb') as pkl_file:
            validation_dates = pickle.load(pkl_file)

        for mode, dates in zip( [self.TRAINING, self.VALIDATION], [training_dates, validation_dates]):
            print(f'Number of dates: {len(dates)}')
            data_file_paths = self.getfiles(key, dates=dates)
            df = self.load_data(var='PROB', data_file_paths=data_file_paths)
            df.to_pickle(join(config.ML_DATA_STORAGE_PATH, f"full_operational_{mode}_{key}_dataset.pkl"))


    def _save_wholedataset(self):
        """
        Save the whole dataset into a single file
        """
        fname_strs = (
            "PROBABILITY_OBJECTS*[0][0-9].nc",
            "PROBABILITY_OBJECTS*[1][0-2].nc",
        )
        data_file_paths = []
        for fname_str in fname_strs:
            data_file_paths.extend(
                [
                    glob(join(config.ML_INPUT_PATH, str(date), fname_str))
                    for date in config.ml_dates
                ]
            )

        df = self.load_data(var="PROB", data_file_paths=data_file_paths)

        df.to_pickle(join(config.ML_DATA_STORAGE_PATH, "whole_dataset.pkl"))

    def load_data(self, var, data_file_paths=None, vars_to_load=None):
        """
        Load the machine learning data for training and validation. 
        """
        idx = 4 if "UPDRAFT" in var else 2
        storm_files = sorted(list(itertools.chain.from_iterable(data_file_paths)))
        if vars_to_load is None:
            ds = xr.open_dataset(storm_files[0])
            total_vars = list(ds.data_vars)
        else:
            total_vars = vars_to_load

        print("Loading a total of {} variables...".format(len(total_vars)))
        data = []
        run_times = []
        run_dates = []
        run_time_idx = []

        total_num_of_files = len(storm_files)
        print('Loading files...')
        for i, storm_file in enumerate(storm_files):
            #print(f"Loading file {i} out of {total_num_of_files}...")
            ds = xr.open_dataset(storm_file)
            try:
                data.append(np.stack([ds[v].values for v in total_vars], axis=-1))
                run_times.append(
                    [storm_file.split("/")[-1].split("_")[idx][9:]] * data[-1].shape[0]
                )
                run_dates.append(
                    [storm_file.split("/")[-1].split("_")[idx][:8]] * data[-1].shape[0]
                )
                run_time_idx.append(
                    [int(storm_file.split("_")[-1][:2])] * data[-1].shape[0]
                )
                ds.close()
            except KeyError:
                print(
                    f"{storm_file} did not contain a given variable; likely is empty!"
                )
                continue

        all_data = np.concatenate(data)
        all_run_times = np.concatenate(run_times)
        all_run_dates = np.concatenate(run_dates)
        all_run_time_idx = np.concatenate(run_time_idx)

        data_concat = np.concatenate(
            (
                all_data,
                all_run_times[:, np.newaxis],
                all_run_dates[:, np.newaxis],
                all_run_time_idx[:, np.newaxis],
            ),
            axis=1,
        )
        total_vars += ["Run Time", "Run Date", "FCST_TIME_IDX"]

        return pd.DataFrame(data=data_concat, columns=total_vars)

    def split_testing_dataset(self, fcst_time_idx, info, examples, targets):
        """ Load testing dataset at multiple times """
        if fcst_time_idx == "first_hour":
            fcst_time_idx_set = [0, 6, 12]
        elif fcst_time_idx == "second_hour":
            fcst_time_idx_set = [18, 24]

        testing_data = {}
        for f in fcst_time_idx_set:
            idx = info.loc[info["FCST_TIME_IDX"] == float(f)].index.values.astype(int)
            testing_data[f] = (examples.iloc[idx, :], targets.iloc[idx])

        return testing_data

    def load_dataframe(self, target_var_name, load_filename_dict, additional_vars_to_drop=[], info=None, other_targets=None):
        """             
        Load pandas dataframe from the training, validation, and testing dataset for a particular fold.
        Loaded data includes the examples, target, additional targets, feature names of the examples,
        and additional data such as date and time.
        
        Args:
        ---------------
            target_var_name: str 
                name of the target value
            save_filename_dict: dict 
                dict of filenames to load for a particular CV fold
                where the keys are 'training', 'validation', and/or 'testing'

        Returns:
        ----------------
            data : dict 
                keys are 'training, validation, and testing'
                items include examples, targets, other targets
                and additional info such as date and time.          
        """
        vars_to_drop = [
            "matched_to_severe_wx_warn_polys_15km",
            "matched_to_severe_wx_warn_polys_30km",
            "matched_to_tornado_warn_ploys_15km",
            "matched_to_tornado_warn_ploys_30km",
        ]

        data = {}

        # Initialize an empty dict for storing the data
        data = {}
        for mode in list(load_filename_dict.keys()):
            print(f"Loading the {mode} dataframe from {load_filename_dict[mode]}...")
            df = pd.read_pickle(load_filename_dict[mode])
            if info is not None:
                info = df[["Run Time", "Run Date", "FCST_TIME_IDX", "label"]]
                info = info[self.EXTRA_VARIABLES].astype(
                    {
                        "label": float,
                        "Run Date": str,
                        "Run Time": str,
                        "FCST_TIME_IDX": float,
                    })

            if other_targets is not None:
                other_targets = df[self.TARGETS_VARIABLES].astype(float)
            
            # Always remove the target variables
            vars_to_drop += self.TARGETS_VARIABLES
            # Always remove the extra variables (e.g., date, time, etc.)
            vars_to_drop += self.EXTRA_VARIABLES
            # Drop variables
            vars_to_drop += additional_vars_to_drop

            examples = df.drop(columns=vars_to_drop, errors="ignore").astype(float)

            data[mode] = {
                self.EXAMPLES: examples,
                self.TARGETS: df[target_var_name].astype(float),
                self.OTHER_TARGETS: other_targets,
                self.INFO: info,
                "feature_names": list(examples.columns),
            }

        return data
