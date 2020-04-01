import sys
sys.path.append('/home/monte.flora/machine_learning/build_model')
from MachLearn import MachLearn

model = MachLearn(preload=False)
model._calc_cross_validation_fold_parameters(n_cv_folds=15, 
                                             percent_testing=6,
                                             verbose=True)


