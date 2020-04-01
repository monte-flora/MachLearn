

def CrossValidation:
    """
    A class for handling cross-validation where separation
    is based on dates. 
    """
    
    def load_data_for_all_cv_folds(
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



    def calc_cross_validation_fold_parameters(
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
        
        Returns:
        --------------
            dates_dict : dict
                a nested dictionary where the primary keys are the folds
                and the subdictionaries are the lists of dates in the training,
                testing, and validation sets per fold. 
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
        




