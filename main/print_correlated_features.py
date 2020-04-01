import pickle

fname = 'correlated_features_to_drop.pkl'
with open(fname,'rb') as f:
    correlated_features = pickle.load(f)

print(len(correlated_features))
print(correlated_features)


"""
filename = 'correlated_feature_pairs.pkl'

with open(filename,'rb') as f:
    correlated_features = pickle.load(f)


for dropped_feature in list(correlated_features.keys()):
    if dropped_feature.startswith('uh'):
        kept_feature = correlated_features[dropped_feature][0]
        cc_val = correlated_features[dropped_feature][1]
        print(f'\n Dropped : {dropped_feature} | Kept : {kept_feature} | rho = {cc_val:.3f}')
"""

