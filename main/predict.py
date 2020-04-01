import sys
sys.path.append("/home/monte.flora/machine_learning/build_model")
import MachLearn as ml_script
from MachLearn import MachLearn, to_predictions_2d, get_examples_at_datetime, load_probability_objects, _save_netcdf
from os.path import join
from wofs.util import config
from joblib import dump, load
import pickle

TRAINING = "training"
TESTING = 'testing'

def worker(date, time, time_idx, model, target, return_data=False):
    """
    """
    drop_correlated_features = True
    vars_to_drop = [
                    'matched_to_severe_wx_warn_polys_15km', 'matched_to_severe_wx_warn_polys_30km',
                    'matched_to_tornado_warn_ploys_15km', 'matched_to_tornado_warn_ploys_30km']
    if drop_correlated_features:
        print('***DROPPING CORRELATED FEATURES!***')
        filename = '/home/monte.flora/machine_learning/main/correlated_features_to_drop.pkl'
        with open(filename,'rb') as f:
            correlated_features = pickle.load(f)
        vars_to_drop += correlated_features
        opt = 'correlated_features_removed'
    else:
        opt = 'all_features'
   
    ml = MachLearn(preload=False)
    fold = ml.date_to_fold_dict[date]
    fname_params = {
        "model_name": model,
        "target_var": target,
        "fcst_time_idx": "first_hour",
        "fold": ml.date_to_fold_dict[date]
    }
   
    # Load the features from a single date-time-fcst_time_idx
    fname = join(
        config.ML_INPUT_PATH,
        str(date),
        f"PROBABILITY_OBJECTS_{date}-{time}_{time_idx:02d}.nc",
    )
    ml = MachLearn(fname_params=fname_params, load_model=True, drop_correlated_features=True, preload=True, fold_to_load=fold)
    data = ml.data_dict[f'fold_{fold}']['data']
    filename_dict = ml.data_dict[f'fold_{fold}']['filename_dict']
    if model == 'LogisticRegression':
        norm_model = load(filename_dict['norm_clf'])

    # Load data as a pd.DataFrame
    dataframe = ml.load_data(var="PROB", data_file_paths=[[fname]])
    vars_to_drop = ml.additional_vars_to_drop
    vars_to_drop += ml.TARGETS_VARIABLES
    vars_to_drop += ml.EXTRA_VARIABLES

    examples = dataframe.drop(columns=vars_to_drop, errors='ignore').astype(float)
    data[ml.TESTING][ml.EXAMPLES] = examples
    data = ml._imputer(data)

    examples = data[ml.TESTING][ml.EXAMPLES]
    if model == 'LogisticRegression':
        examples = norm_model.transform(examples)

    targets = dataframe[target].astype(float).values
   
    info = dataframe[["Run Time", "Run Date", "FCST_TIME_IDX", "label"]].astype({"Run Time": str, "Run Date": str, 'FCST_TIME_IDX': float, "label": float})

    # Load the model for a fold
    print("Loading {}...".format(filename_dict["main_clf"]))
    model = load(filename_dict["main_clf"])
    cali_model = load(filename_dict["cal_clf"])

    ml.clf = model
    ml.calibrated_clf = cali_model
    calibrated_predictions = ml.predict( examples=examples, calibrate=True )

    # Load 2D storm object labels
    labeled_regions = load_probability_objects(date, time, time_idx)

    idx = get_examples_at_datetime(info, date, time, time_idx)
    labels_column = info["label"].values
    
    forecast_probabilities = to_predictions_2d(calibrated_predictions, labeled_regions, labels_column, idx)
    if return_data:
        data = { 
                'probs_2d': forecast_probabilities,
                'examples': examples,
                'targets': targets,
                'info': info,
                'model': model,
                'labeled_regions': labeled_regions
                }
        return data 
        
    data = {"ForecastProbabilities": (["Y", "X"], forecast_probabilities)}

    fname = join(
        config.ML_FCST_PATH,
        str(date),
        f"{fname_params['model_name']}_fcst_probs_target:{fname_params['target_var']}_{date}-{time}_t:{time_idx}.nc",
    )
    _save_netcdf(data, fname)

