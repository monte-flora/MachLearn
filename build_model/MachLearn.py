# Import Modules
import os
from os.path import join
import numpy as np
import xarray as xr
from glob import glob
import itertools
from joblib import dump, load
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import pandas as pd
from datetime import datetime
import pickle

# Personal Modules
from build_sklearn_model import classifier_model, calibration_model
from wofs.util import config
from wofs.evaluation.verification_metrics import (
    ContingencyTable,
    Metrics,
    brier_skill_score,
    _get_binary_xentropy,
)
from PreProcess import PreProcess


class MachLearn(PreProcess):
    """

    """
    def __init__(self, fname_params={}, load_model=False, drop_correlated_features=False, fold_to_load=None, preload=True):
        dates_dict, date_to_fold = self._calc_cross_validation_fold_parameters(verbose=False)
        self.dates_dict = dates_dict
        self.date_to_fold_dict = date_to_fold
        self.fname_params = fname_params
        self.model_name = self.fname_params.get('model_name', None)
        self.load_model = load_model

        print(f'drop_correlated_features: {drop_correlated_features}')

        if drop_correlated_features:
            print('***DROPPING CORRELATED FEATURES!***')
            filename = '/home/monte.flora/machine_learning/main/correlated_features_to_drop.pkl'
            with open(filename,'rb') as f:
                correlated_features = pickle.load(f)
            self.additional_vars_to_drop = correlated_features
            self.opt = 'correlated_features_removed'
        else:
            self.additional_vars_to_drop = [ ]
            self.opt = 'all_features'

        print(f'Preload: {preload}')
        if preload:
            print(f'Droppping {len(self.additional_vars_to_drop)} features ...')
            self.data_dict = self.load_cv_data(fold_to_load)

    def fit(self, params, data, other_params=None):
        """
        Fits the classifier model to the training data with parameters 'params'
        Args:
        ------------ 
            model_name, str, name of the sklearn model 
            params, dict, dictionary of model parameters for the sklearn model 
            training_data, 2-tuple of training examples and training labels
        Returns:
        ------------ 
            self: object
        """
        if self.model_name == "XGBoost":
            clf = classifier_model(model_name=self.model_name, params=params)
            X_train = data[self.TRAINING][self.EXAMPLES]
            y_train = data[self.TRAINING][self.TARGETS]
            X_valid = data[self.VALIDATION][self.EXAMPLES]
            y_valid = data[self.VALIDATION][self.TARGETS]
            if isinstance(X_train, (pd.DataFrame)):
                X_train = X_train.values
            if isinstance(y_train, (pd.DataFrame)):
                y_train = y_train.values
            if isinstance(X_valid, (pd.DataFrame)):
                X_valid = X_valid.values
            if isinstance(y_valid, (pd.DataFrame)):
                y_valid = y_valid.values

            clf.fit(
                X=X_train,
                y=y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                early_stopping_rounds=100,
                verbose=False
            )
        else:
            clf = classifier_model(model_name=self.model_name, params=params)
            clf.fit(data[self.TRAINING][self.EXAMPLES], data[self.TRAINING][self.TARGETS])

        self.clf = clf
        return self

    def fitCV(self, model_params):
        """
        Fit a model across a given set of cross-validation data
        Args:
        ------------
            fname_params: dict
                dictionary for naming the model filename
            feature_importance: boolean 
                if True, perform feature importance (see _feature_importance)
            calibrate: boolean
                if True, will use validation dataset to train an isotonic regression model
        """
        predictions = []
        targets = []
        
        for fold in list(self.data_dict.keys()):
            data = self.data_dict[fold]["data"]
           
            print('\n Fitting the base classifier...')
            self.fit(params=model_params, data=data)
            
            print("\n Fitting the isotonic regression model...")
            self.calibrate_fit(data=data, save_name=self.data_dict[fold]["filename_dict"]["cal_clf"])

            print("\n Get calibrated predictions...")
            calibrated_predictions = self.predict(examples=data[self.TESTING][self.EXAMPLES], calibrate=True)

            predictions.extend(calibrated_predictions)
            targets.extend(data[self.TESTING][self.TARGETS])

            print(
                f"\n Finished and saving the model to {self.data_dict[fold]['filename_dict']['main_clf']}..."
            )
            dump(self.clf, self.data_dict[fold]["filename_dict"]["main_clf"])
            self.clf = None

        predictions = np.array(predictions)
        targets = np.array(targets)
        self.assess_model_performance(
            predictions=predictions,
            targets=targets,
            save_name=self.data_dict[fold]["filename_dict"]["results"],
        )

    def evaluateCV(self):
        """ evaluate different splits of the testing dataset """
        if self.fname_params["fcst_time_idx"] == "first_hour":
            fcst_time_idx_set = [0, 6, 12]
        elif self.fname_params["fcst_time_idx"] == "second_hour":
            fcst_time_idx_set = [18, 24]

        prediction_sets = {t: [] for t in fcst_time_idx_set}
        target_sets = {t: [] for t in fcst_time_idx_set}
        
        for fold in list(self.data_dict.keys()):
            self.calibrated_clf = self.data_dict[fold]["cal_model"]
            self.clf = self.data_dict[fold]["model"]
            data = self.data_dict[fold]["data"]
            examples = data[self.TESTING][self.EXAMPLES]
            targets = data[self.TESTING][self.TARGETS]
            fcst_time_idx = self.fname_params["fcst_time_idx"]
            info = data[self.TESTING]["info"]

            testing_dataset = self.split_testing_dataset(
                fcst_time_idx=fcst_time_idx,
                info=info,
                examples=examples,
                targets=targets,
            )
            for t in list(testing_dataset.keys()):
                examples_temp, targets_temp = testing_dataset[t]
                calibrated_predictions = self.predict(examples=examples_temp, calibrate=True)

                prediction_sets[t].extend(calibrated_predictions)
                target_sets[t].extend(list(targets_temp.values))

        save_names = {}
        for t in fcst_time_idx_set:
            fname_of_results = join(
                config.ML_RESULTS_PATH,
                f'verifyData_{self.model_name}_target:{self.fname_params["target_var"]}_fcst_time_idx={t}_{self.opt}.nc',
            )
            save_names[t] = fname_of_results

        for t in fcst_time_idx_set:
            predictions_temp = np.array(prediction_sets[t])
            targets_temp = np.array(target_sets[t])
            save_name = save_names[t]
            self.assess_model_performance(
                predictions=predictions_temp, targets=targets_temp, save_name=save_name,
            )

    def assess_model_performance(self, predictions, targets, save_name):
        """
        Assess model performance.
        """
        num_bootstraps = 1000
        pod_n = []
        sr_n = []
        pofd_n = []
        mean_fcst_probs_n = []
        event_frequency_n = []
        auc_n = []
        auprc_n = []
        bss_n = []
        predictions_n = []
        targets_n = []

        print("Performing bootstrapping of results...")
        for i in range(num_bootstraps):
            # print (f'bootstrapp index {i}')
            random_idx = np.random.choice(
                np.arange(len(predictions)), size=len(predictions)
            )

            predictions_i = predictions[random_idx]
            targets_i = targets[random_idx]

            pod, sr, pofd = Metrics.performance_curve(
                predictions_i, targets_i, bins=np.arange(0, 1.0, 0.01), roc_curve=True
            )
            mean_fcst_probs, event_frequency = Metrics.reliability_curve(
                predictions_i, targets_i, bins=np.arange(0, 1.0, 0.1)
            )
            auc = roc_auc_score(targets_i, predictions_i)
            auprc = average_precision_score(targets_i, predictions_i)
            bss = brier_skill_score(targets_i, predictions_i)

            pod_n.append(pod)
            sr_n.append(sr)
            pofd_n.append(pofd)
            mean_fcst_probs_n.append(mean_fcst_probs)
            event_frequency_n.append(event_frequency)
            auc_n.append(auc)
            auprc_n.append(auprc)
            bss_n.append(bss)
            predictions_n.append(predictions_i)
            targets_n.append(targets_i)

        data = {}
        data["pod"] = (["N", "thresholds"], pod_n)
        data["sr"] = (["N", "thresholds"], sr_n)
        data["pofd"] = (["N", "thresholds"], pofd_n)
        data["mean fcst prob"] = (["N", "bins"], mean_fcst_probs_n)
        data["event frequency"] = (["N", "bins"], event_frequency_n)
        data["predictions"] = (["N", "examples"], predictions_n)
        data["targets"] = (["N", "examples"], targets_n)
        data["auc"] = (["N"], auc_n)
        data["auprc"] = (["N"], auprc_n)
        data["bss"] = (["N"], bss_n)

        print("Saving {}...".format(save_name))
        ds = xr.Dataset(data)
        ds.to_netcdf(save_name)
        ds.close()

    def calibrate_fit(self, data, save_name):
        """
        Calibrate a pre-fitted classifier on the validation data set using isotonic regression
        """
        validation_predictions = self.predict(examples=data[self.VALIDATION][self.EXAMPLES])
        calibrated_clf = IsotonicRegression(out_of_bounds="clip")
        calibrated_clf.fit(
            validation_predictions.astype(float),
            data[self.VALIDATION][self.TARGETS].astype(float),
        )

        self.calibrated_clf = calibrated_clf
        print(f"Saving {save_name}...")
        dump(calibrated_clf, save_name)

    def predict(self, examples, calibrate=False):
        """
       Returns the probabilistic predictions from a given machine learning model.
        Args:
            model, object, pre-fitted machine learning object 
            data, 2-tuple of examples and labels to evaluate 
        Returns:
            1-D array, Probabilistic predictions made by the machine learning model 
        """
        if not hasattr(self, 'clf'):
            raise AttributeError ('Must call .fit() first!')

        if self.model_name == "XGBoost":
            examples = examples.values
            predictions = self.clf.predict_proba(
                examples, ntree_limit=self.clf.best_ntree_limit
            )[:, 1]

        else:
            predictions = self.clf.predict_proba(X=examples)[:, 1]
    
        if calibrate:
            if not hasattr(self, 'calibrated_clf'):
                raise AttributeError ('Must call .calibrate_fit() first!')
            calibrated_predictions = self.calibrated_clf.predict(predictions)
            return calibrated_predictions

        return predictions

    def calc_auprc(self, targets, predictions):
        """
        Return 
        """
        return average_precision_score(targets, predictions)

    def hyperparameter_search(self, param_grid):
        """
        Performs cross-validation to find the best parameters for the model given. 
        """
        debug=False
        modes = [self.TRAINING, self.VALIDATION]
        scores_per_fold = [ ]
        for fold in list(self.data_dict.keys()):
            data = self.data_dict[fold]["data"]
            scores = self._determine_score_per_params( param_grid=param_grid, data=data)
            scores_per_fold.append(scores)

        avg_scores = np.mean(scores_per_fold, axis=0)
        best_params = self._best_params(param_grid, avg_scores)

        return best_params, avg_scores

    def _determine_score_per_params(self, param_grid, data):
        """
        Find the scores for a training/validation fold.
        """
        scores = []
        keys, values = list(zip(*list(param_grid.items())))
        for v in itertools.product(*values):
            params = dict(list(zip(keys, v)))
            print(
                "Evaluating {} with the following params: {}".format(self.model_name, params)
            )
            self.fit(params=params, data=data)
            validation_predictions = self.predict(examples=data[self.VALIDATION][self.EXAMPLES]) 
            auprc = self.calc_auprc(targets=data[self.VALIDATION][self.TARGETS], predictions=validation_predictions)
            scores.append(auprc)

        return scores

    def _best_params(self, param_grid, avg_scores):
        """
        Find the best parameters. 
        """
        keys, values = list(zip(*list(param_grid.items())))
        possible_params = np.array(
            [dict(list(zip(keys, v))) for v in itertools.product(*values)]
        )
        idx = np.argmax(avg_scores)

        return possible_params[idx]

    def calc_climo(self, df, verify_var):
        """
        Calculate the climatology. 
        """
        ratio = np.mean(df[verify_var].values.astype(float))

        print("Climatology of {}: {:.3f}%".format(verify_var, 100.0 * ratio))

    def correlated_features_to_remove(self, df):
        """
        Determines correlated features to drop
        """
        vars_to_drop = [
            "matched_to_severe_wx_warn_polys_15km",
            "matched_to_severe_wx_warn_polys_30km",
            "matched_to_tornado_warn_ploys_15km",
            "matched_to_tornado_warn_ploys_30km",
        ]

        vars_to_drop += self.TARGETS_VARIABLES
        vars_to_drop += self.EXTRA_VARIABLES
      
        self.IGNORED_VARIABLES = vars_to_drop
        print(f'Dropping {len(vars_to_drop)} features...')
        examples = df.drop( columns=vars_to_drop, errors = 'ignore' ).astype(float)
        
        columns_to_drop, correlated_pairs = self.correlation_filtering(df=examples, cc_val=0.8)

        fname = 'correlated_features_to_drop.pkl'
        with open(fname, 'wb') as fp:
            pickle.dump(columns_to_drop, fp)

        fname = 'correlated_feature_pairs.pkl'
        with open(fname, 'wb') as fp:
            pickle.dump(correlated_pairs, fp)


def load_probability_objects(date, time, fcst_time_idx):
    """
    load the 2d probability object netcdf files
    """
    object_file = join(
        config.OBJECT_SAVE_PATH,
        date,
        f"updraft_ensemble_objects_{date}-{time}_t:{fcst_time_idx}.nc",
    )
    print(f"Loading {object_file}...")
    ds = xr.open_dataset(object_file)
    objects = ds["Probability Objects"].values

    ds.close()
    del ds
    return objects


def get_examples_at_datetime(info, date, time, f):
    """ get indices of ..."""
    idx = info.loc[
        (info["FCST_TIME_IDX"] == float(f))
        & (info["Run Date"] == str(date))
        & (info["Run Time"] == str(time))
    ].index.values.astype(int)
    return idx

def _save_netcdf(data, fname):
    """ save a xarray dataset as netcdf """
    ds = xr.Dataset(data)
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in ds.data_vars}
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    print("Saving..{}".format(fname))
    ds.to_netcdf(path=fname, encoding=encoding)
    ds.close()
 
def to_predictions_2d(predictions, labeled_regions, labels_column, idx):
    """
    Unravels a 1D prediction array into the 2D forecast probabilities.
    """
    probabilities_2d = np.zeros(labeled_regions.shape)
    object_labels_1D = labels_column[idx]
    predictions_temp = predictions[idx]
    for i, label in enumerate(object_labels_1D):
        probabilities_2d[labeled_regions == label] = predictions_temp[i]

    return probabilities_2d

