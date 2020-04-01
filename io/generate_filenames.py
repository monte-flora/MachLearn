



def generate_filenames(self, paths, fname_params):
    """
    Generates various filename for a machine learning model.
    Includes the filename for the base classifier, calibration model,
    normalization, and imputer. 
    """
    model_name = fname_params['model_name']
    target = fname_params['target'] 
    fold = fname_params['fold']
    drop_correlated_feature_opt = fname_params['drop_correlated_feature_opt']
    fcst_time_idx = fname_params['fcst_time_idx']

    path = join(
            join(
                config.ML_MODEL_SAVE_PATH,
                f'FCST_TIME_IDX={self.fname_params["fcst_time_idx"]}',
            ),
            f'{self.model_name}',
        )

        # Need to save the main classifier and the imputation model
        fname_of_main_clf = f'model:{model_name}_{opt}:{target_var}_fold:{fold}.joblib'
        fname_of_isotonic_model = f'model:{model_name}_isotonic_{opt}:{target}_fold:{fold}.joblib'
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

