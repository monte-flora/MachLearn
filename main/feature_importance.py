from ModelClarifier.class_ModelClarify import ModelClarify

""" usage: stdbuf -oL python feature_importance.py 2 > & log_importance & """

model_names = ['RandomForest',"LogisticRegression"]
target_vars = [
    "matched_to_tornado_0km",
    "matched_to_severe_wind_0km",
    "matched_to_severe_hail_0km",
]
opt_set = [True, False]
fname_params = {"fcst_time_idx": "first_hour"}

for drop_correlated_features in opt_set:
    for model_name in model_names:
        fname_params["model_name"] = model_name
        for target_var in target_vars:
            fname_params["target_var"] = target_var
            model = MachLearn(
                fname_params,
                load_model=True,
                drop_correlated_features=drop_correlated_features,
                fold_to_load=fold_to_load,
            )
            model.permutation_importance(evaluation_fn="auprc")
