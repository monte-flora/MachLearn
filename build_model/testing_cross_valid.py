import sys, os
sys.path.append('/home/monte.flora/wofs/util')
import config
import numpy as np 

n_cv_folds = 8

num_dates = len(config.verify_forecast_dates)
fold_interval = int((num_dates / n_cv_folds))

print ("Total number of days: ", num_dates)
print (fold_interval)

n_training_dates = int(round(0.8*num_dates))
n_valid_dates = int(round(0.1*num_dates))+2
n_testing_dates = int(round(0.1*num_dates))+2

print ("Training: ", n_training_dates)
print ("Valid: ", n_valid_dates)
print ("Test: ", n_testing_dates)

num1 = n_training_dates + n_valid_dates
num2 = num1 + n_testing_dates

for fold, r in enumerate(range(0,num_dates,fold_interval)):
    folds_training = (np.arange(n_training_dates) + r) % num_dates
    folds_validation = (np.arange(n_training_dates, num1) + r) % num_dates 
    folds_testing = (np.arange(num1, num2) + r) % num_dates
    if fold == 6:
        print config.verify_forecast_dates[folds_training]

    print "Training Idxs: {}-{}   Valid Idxs: {}-{}   Testing Idxs: {}-{}".format(folds_training[0], folds_training[-1], folds_validation[0], folds_validation[-1], folds_testing[0], folds_testing[-1])

    #print config.verify_forecast_dates[folds_training]
    #print config.verify_forecast_dates[folds_validation]


