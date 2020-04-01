from plot_feature_importance import plot_variable_importance
import pickle
from wofs.util import config
from os.path import join 
import numpy as np 
from wofs.util.feature_names import to_readable_names

model_name = 'LogisticRegression'
target_var = 'matched_to_tornado_0km'
fcst_time_idx = 'first_hour' 
drop_correlated_features = True
if drop_correlated_features:
    opt = 'correlated_features_removed'
else:
    opt = 'all_features'

def importance_ranking_cv(n_folds=15): 
    """
    Re-rank importance across cv folds
    """
        
    for fold in range(n_folds):
        save_fname =  f'singlepass_permutation_{model_name}_{target_var}_t:{fcst_time_idx}_f:fold_{fold}_{opt}.png'
        fname_of_feature_importance = \
                     join( config.ML_RESULTS_PATH, f'PermutationImportance_{model_name}_target:{target_var}_t:{fcst_time_idx}_f:fold_{fold}_{opt}.pkl')

        with open(fname_of_feature_importance, 'rb') as fp:
            result = pickle.load(fp)

        rankings = result.retrieve_multipass()
        var_names = list(rankings.keys())

        if fold == 0:
            original_score = result.original_score
            data = {var: rankings[var] for var in var_names}
        else:
            original_score += result.original_score
            for var in var_names:
                try:
                    data[var] = [data[var][0]+rankings[var][0], data[var][1]+rankings[var][1]]
                except KeyError:
                    data[var] = [rankings[var][0], rankings[var][1]]

    # Average ranking and average the mertics bootstrap array 
    for feature in list(data.keys()):
        data[feature] = (data[feature][0] / 15., data[feature][1]/15.)

    print ('plotting the feature importances...')    
    ranking = [ ]
    for i, feature in enumerate(list(data.keys())):
        ranking.append( [i, feature, np.mean(data[feature][1])] )

    sorted_ranking = sorted(ranking, reverse=True, key = lambda x:x[-1])
    for i, element in enumerate(sorted_ranking):
        sorted_ranking[i][0] = i

    for element in sorted_ranking:
        feature = element[1]
        rank = element[0]
        data[feature] = (rank, data[feature][1] )

    original_score = original_score / 15.

    with open('permutation_importance_cv.pkl', 'wb') as pkl_file:
        pickle.dump(data, pkl_file)

    new_data = {}
    for feature in list(data.keys()):
        readable_name = to_readable_names([feature])
        new_data[readable_name] = data[feature]


    plot_variable_importance(
                    new_data,
                    original_score = original_score,
                    filename = save_fname.replace('single', 'multiple'),
                         )

importance_ranking_cv(n_folds=15)




