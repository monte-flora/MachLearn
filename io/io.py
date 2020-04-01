


class IO:

    def generate_filenames(self):
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
                f'FCST_TIME_IDX={fcst_time_idx}',
            ),
            f'{model_name}',
        )

        # Need to save the main classifier and the imputation model
        fname_of_main_clf = f'model:{model_name}_{opt}:{target_var}_fold:{fold}.joblib'
        fname_of_isotonic_model = f'model:{model_name}_isotonic_{opt}:{target_var}_fold:{fold}.joblib'
        fname_of_norm_model = f'model:{model_name}_norm_{opt}:{target_var}_fold:{fold}.joblib'

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


    



