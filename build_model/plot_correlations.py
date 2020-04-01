from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

from sklearn.metrics import roc_auc_score, adjusted_mutual_info_score, average_precision_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from PreProcess import PreProcess
from datetime import datetime
from machine_learning.plotting.plot_feature_importance import plot_variable_importance
from PermutationImportance.permutation_importance import sklearn_permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from wofs.evaluation.verification_metrics import brier_skill_score
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from sklearn.decomposition import PCA
from wofs.util import feature_names

preprocess = PreProcess( )

fname_params = {
           'model_name' : 'RandomForest',
           'target_var': 'matched_to_LSRs_15km',
           'use_time_stats': True,
           'resampling_method': 'random',
           'fold': 1,
           'fcst_time_idx': 0
           }

variables_to_remove = [
                    'matched_to_severe_wx_warn_polys_15km', 'matched_to_severe_wx_warn_polys_30km',
                    'matched_to_tornado_warn_ploys_15km', 'matched_to_tornado_warn_ploys_30km']

save_data_str_valid = '{}_f:{}_t:{}_use_time_stats:{}_raw.pkl'
save_data_str_train = \
                save_data_str_valid.replace('raw', '{}_resampled_to_{}_highly_corr_vars_removed'.format(fname_params['resampling_method'], 
                                                                                                        fname_params['target_var']))

save_data_str = {
                    'training': save_data_str_train,
                    'validation': save_data_str_valid,
                 }


data = preprocess.load_dataframe( params = fname_params,
                                        modes = ['training', 'validation'],
                                        variables_to_remove = variables_to_remove,
                                        save_data_str = save_data_str,
                                        remove_variables=True
                                        )

data = preprocess._imputer(  data=data,
                       simple=True,
                       save=False
                      )

X = data['training']['examples']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=data['training']['feature_names'], ax=ax1,
                                      leaf_rotation=90)
dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])
fig.tight_layout()

plt.savefig( 'test.png' )
cluster_ids = hierarchy.fcluster(corr_linkage, 3, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
print(len(selected_features))

env_vars = feature_names.env_vars_smryfiles + feature_names.env_vars_wofsdata

#data = preprocess.resample( data=data )
X_train_sel = data['training']['examples'][:, selected_features]
X_test_sel = data['validation']['examples'][:, selected_features]
features = np.array(data['training']['feature_names'])[selected_features]
print (features)

clf_sel = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs = 25, min_samples_leaf=5)
clf_sel.fit(X_train_sel, data['training']['targets'])

prediction = clf_sel.predict_proba( X_test_sel )[:,1]

auc = roc_auc_score( data['validation']['targets'], prediction)
aps = average_precision_score( data['validation']['targets'], prediction)

print ('AUC: {0:.3f}.......APS:{1:.3f}'.format(auc, aps))

scoring_data = (X_test_sel, np.array(data['validation']['targets']))

'''
print('\n Calculating Permutation Importance...Start Time:', datetime.now().time())
result = sklearn_permutation_importance( model = clf_sel,
                                         scoring_data = scoring_data,
                                         evaluation_fn = roc_auc_score,
                                         variable_names = features,
                                         scoring_strategy = 'argmin_of_mean',
                                         subsample=1,
                                         nimportant_vars = 10,
                                         njobs = 0.5,
                                         nbootstrap = 100)

print('End Time: ', datetime.now().time())

feature_names = {f:f for f in features}
colors = {f: 'lightgreen' for f in features}

plot_variable_importance(
                    result, f'singlepass_permutation.png', 
                    readable_feature_names=feature_names, 
                    feature_colors=colors, 
                    multipass=False, fold = 0) 

plot_variable_importance(
                    result, f'multipass_permutation.png',
                    readable_feature_names=feature_names,
                    feature_colors=colors,
                    multipass=True, fold = 0)

'''


