from plot_feature_importance import plot_variable_importance
import pickle
from wofs.util import config
from os.path import join 
import numpy as np 
from wofs.util.feature_names import to_readable_names 

drop_correlated_features = True

if drop_correlated_features:
    opt = 'correlated_features_removed'
else:
    opt = 'all_features'

title_dict = {'matched_to_tornado_0km':'Tornadoes',
              'matched_to_severe_hail_0km':'Severe Hail',
              'matched_to_severe_wind_0km':'Severe Wind'
              }

corr_title_dict = {'correlated_features_removed': 'Correlated Features Removed',
                   'all_features' : 'All Features'
                  }


for model_name in ['RandomForest', 'LogisticRegression']:
    for target in ['matched_to_tornado_0km', 'matched_to_severe_hail_0km', 'matched_to_severe_wind_0km']:
        fname_of_feature_importance = f'/work/mflora/ML_DATA/MODEL_SAVES/OPERATIONAL/permutation_importance_{model_name}_{target}.pkl'

        print(fname_of_feature_importance) 

        with open(fname_of_feature_importance, 'rb') as fp:
            result = pickle.load(fp)

        rankings = result.retrieve_multipass()
        original_score = result.original_score

        new_rankings = { }
        feature_colors={}
        for feature in list(rankings.keys()):
            readable_name, color = to_readable_names([feature])
            feature_colors[readable_name] = color
            new_rankings[readable_name] = rankings[feature]

        plot_variable_importance(
                title = f'Multipass Permutation Importance \n {model_name} -- {title_dict[target]} -- {corr_title_dict[opt]}',
                metric = 'Training AUC',
                rankings=new_rankings,
                original_score = original_score,
                feature_colors=feature_colors, 
                filename = f'multipass_perm_import_{model_name}_{target}.png'
                )







