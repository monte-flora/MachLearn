from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

def classifier_model( model_name, params=None ):
    '''
    Returns classifier machine learning model object.

    usage: clf.fit( ) 
    '''
    if model_name == 'RandomForest':
        if params is None:
            return RandomForestClassifier( n_jobs = 40 )
        else:
            return RandomForestClassifier( random_state=42, n_jobs = 40, n_estimators = params['n_estimators'], min_samples_leaf = params['min_samples_leaf'], 
                    criterion = params['criterion'], max_features = params['max_features'], max_depth = params['max_depth'], bootstrap = params['bootstrap']) 
    if model_name == 'GradientBoost':
        if params is None:
            return GradientBoostingClassifier()
        else:
            return GradientBoostingClassifier( loss = 'deviance', n_estimators = params['n_estimators'], learning_rate = params['alpha'], min_samples_leaf = 20  )

    if model_name == 'LogisticRegression':
        return LogisticRegression(n_jobs=40, solver='saga', penalty='elasticnet', C = params['C'], max_iter=200, l1_ratio=params['l1_ratio'] )

    if model_name == 'XGBoost':
        return XGBClassifier(**params)

def calibration_model( classifier ):
    '''
    Returns the calibration machine learning model.

    usage: clf.fit( ) 
    '''
    return CalibratedClassifierCV(classifier, cv='prefit', method='isotonic')

