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

