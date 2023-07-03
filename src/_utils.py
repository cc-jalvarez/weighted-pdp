import statistics
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import StratifiedKFold
import optuna
import optuna.integration.lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier

import numpy as np
#from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import _get_weights
from sklearn.utils.validation import _num_samples
from sklearn.utils.extmath import weighted_mode


class KNeighborsClassifierW(KNeighborsClassifier):

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        """added sample_weight """
        self.sample_weight = sample_weight
        return super().fit(X, y)

    def predict(self, X):
        neigh_dist, neigh_ind = self.kneighbors(X)
        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = _num_samples(X)
        weights = _get_weights(neigh_dist, self.weights)
        """ added """
        if self.sample_weight is not None:
            nv = neigh_ind.reshape( (neigh_ind.shape[0]*neigh_ind.shape[1],) )
            weights = self.sample_weight.iloc[nv].to_numpy()
            weights = weights.reshape(neigh_ind.shape)
            #print(weights.shape)
            #print(weights[1,:])
        """ end """
        
        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

    def predict_proba(self, X):
        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_queries = _num_samples(X)

        weights = _get_weights(neigh_dist, self.weights)
        if weights is None:
            weights = np.ones_like(neigh_ind)
        """ added """
        if self.sample_weight is not None:
            nv = neigh_ind.reshape( (neigh_ind.shape[0]*neigh_ind.shape[1],) )
            weights = self.sample_weight.iloc[nv].to_numpy()
            weights = weights.reshape(neigh_ind.shape)
            #print(weights.shape)
            #print(weights[1,:])
        """ end """

        all_rows = np.arange(n_queries)
        probabilities = []
        for k, classes_k in enumerate(classes_):
            pred_labels = _y[:, k][neigh_ind]
            proba_k = np.zeros((n_queries, classes_k.size))

            # a simple ':' index doesn't work right
            for i, idx in enumerate(pred_labels.T):  # loop is O(n_neighbors)
                proba_k[all_rows, idx] += weights[:, i]

            # normalize 'votes' into real [0,1] probabilities
            normalizer = proba_k.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not self.outputs_2d_:
            probabilities = probabilities[0]

        return probabilities


def ci_auc(res, npos=None):
    # mean
    m = statistics.mean(res)
    # standard deviation
    s = statistics.stdev(res)
    #l, u = m-s, m+s
    # logit interval
    '''
    el = log(m/(1-m))
    sl = sqrt(m*(1-m)/npos)*(1/m+1/(1-m))
    cl = .95
    l = el + stats.norm.ppf((1-cl)/2)*sl
    u = el + stats.norm.ppf((1+cl)/2)*sl
    l = exp(l)/(1+exp(l))
    u = exp(u)/(1+exp(u)) 
    '''
    # normal interval
    l, u = stats.t.interval(0.95, len(res) - 1, loc=m, scale=stats.sem(res))
    return m, s, l, u

def ci_auc_l(res):
    return ci_auc(res)[2]
def ci_auc_u(res):
    return ci_auc(res)[3]

#Build the object function in such a way that can receive explicit parameters
#https://optuna.readthedocs.io/en/latest/faq.html#how-to-define-objective-functions-that-have-own-arguments

optuna.logging.set_verbosity(0)
   
class Objective(object):
    
    def __init__(self, clf_name, categorical_columns, trial_cv, X, y, W):
        
        """
        clf_name (string)
        X indipendent features of df
        y dipendent features of df
        X and y are derived from the same train folds when we use stratified 10-fold cross validation
        """
          
        self.clf_name = clf_name
        self.categorical_columns = categorical_columns
        self.trial_cv = trial_cv
        self.X = X
        self.y = y
        self.W = W        
        
    def __call__(self, trial):

        """
        the function structures the hyperparameter optimization for the classifier
        evaluates the model with cross validation, we want to maximize AUPR curve
        return result = is the mean of the highest AUPR-curve scores for each X
        """
        #apply the one hot just to certain columns
        one_hot_transformer = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False),
                                         self.categorical_columns)], 
                                        remainder='passthrough')
        
        #pipeline encapsulates all transformations
        if self.clf_name == "RF":
            estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                      ('clf', RandomForestClassifier(
                                        n_estimators = trial.suggest_int('n_estimators', 50, 200),
                                        max_depth = trial.suggest_int('max_depth', 3, 150),
                                        n_jobs = -1))])
        elif self.clf_name == "TABNET":
            estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                      ('clf', TabNetClassifier(
                          #n_d = trial.suggest_int('n_d', 8, 64, log=True),
                          #n_a = trial.suggest_int('n_a', 8, 64, log=True),
                          #n_steps  = trial.suggest_int('n_steps', 3, 10),
                          #gamma  = trial.suggest_float('gamma', 1.0, 2.0),
                          #n_independent  = trial.suggest_int('n_independent', 1, 5),
                          n_shared  = trial.suggest_int('n_shared', 1, 5)
                          #mask_type  = trial.suggest_categorical('mask_type', ['sparsemax', 'entmax'])
                          ))])
        elif self.clf_name == "DT":
            estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                  ('clf', DecisionTreeClassifier(max_depth = trial.suggest_int('max_depth', 3, 200)))])
        elif self.clf_name == "KNN":
            estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                      ('SC', MinMaxScaler()), # Good practice to use scaler 
                      ('clf', KNeighborsClassifierW(n_neighbors = trial.suggest_int('n_neighbors', 3, 10)))])
        
        elif self.clf_name == "LR":
            estimator = Pipeline(steps = [
                  ('one hot', one_hot_transformer),
                  ('SC', MinMaxScaler()), # Good practice to use scaler in logistic regression
                  ('clf', LogisticRegression(C = trial.suggest_float("C", 1e-10, 1e10, log=True)))])
  
        elif self.clf_name == "XGB":
            estimator = Pipeline(steps = [
                  ('one hot', one_hot_transformer),
                  ('clf', LGBMClassifier(
                                        n_estimators = trial.suggest_int('n_estimators', 32, 256),
                                        num_leaves = trial.suggest_int('num_leaves', 5, 100), 
                                        min_child_weight = trial.suggest_int('min_child_weight', 1, 20),
                                        scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 100),
                                        n_jobs = -1))])
        elif self.clf_name == "LGBM2":
            estimator = Pipeline(steps = [
                  ('one hot', one_hot_transformer),
                  ('clf', LGBMClassifier(boosting_type='gbdt', 
                                            objective='binary',
                                            metric='binary_logloss',
                                            n_estimators= trial.suggest_int('n_estimators', 32, 256), 
                                            num_leaves = trial.suggest_int('num_leaves', 5, 100), 
                                            min_child_samples = trial.suggest_int('min_child_samples', 2, 200),
                                            lambda_l1 = trial.suggest_float('lambda_l1', 0, 1),
                                            lambda_l2 = trial.suggest_float('lambda_l2', 0, 1),
                                            feature_fraction = trial.suggest_float('feature_fraction', 0.5, 1), 
                                            bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1), 
                                            bagging_freq = trial.suggest_int('bagging_freq', 0, 10), 
                                            verbose=-1, 
                                            n_jobs = -1))])
        """
        Measure the performance of the chosen classifier + hyperparameters
        by doing cross-validation and focusing on AUCPR.
        When we run the test for finding first the best classifier, here
        X and y are part of the same train folds that we took from stratified 10-fold cross validation
        
        """
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
        #scores = cross_val_score(estimator = estimator, X = self.X, y = self.y,
        #                cv = trial_cv, #cv is global variable declared in the exoerment settings cell
        #                scoring = score_metric) #scoring is global variable declared in the experment settings cell   
        #result = scores.mean()
        
        skf = StratifiedKFold(n_splits = self.trial_cv, shuffle = True, random_state=42)
        scores = []
        for train, test in skf.split(self.X, self.y):
            X_train = self.X.iloc[train]
            y_train = self.y.iloc[train]
            sw_train = self.W.iloc[train] if self.W is not None else None
            X_test = self.X.iloc[test]
            y_test = self.y.iloc[test]
            sw_test = self.W.iloc[test] if self.W is not None else None
            if sw_train is not None:
                estimator.fit(X_train, y_train, clf__sample_weight=sw_train)
            else:
                estimator.fit(X_train, y_train)
            y_scores = estimator.predict_proba(X_test)[:,1] 
            avg_pr = average_precision_score(y_test, y_scores, sample_weight=sw_test)
            # ll = log_loss(y, y_pred, sample_weight=None)
            scores.append(avg_pr)
        result = statistics.mean(scores)
        return result
		
#Optuna study normal now can receive parameters

def optuna_study_normal(clf_name, categorical_columns, trial_cv, n_trials, X, y, W):
    
    """
    Here Optuna starts the optimization process.
    Receive as input
        clf_name (string)
        X indipendent features of df
        y dipendent features of df
        X and y are derived from the same train folds when we use stratified 10-fold cross validation
    
    Returns a dictionary with the df version, the hyps that maximize AUPR-curve, classifier name
    
    """
    
    if clf_name == "TABNET0":
        return {'best_hyparams': {}, 'clf_name': clf_name }
    # Maximize because we are using AUC-PR (higher = better)
    study = optuna.create_study(direction='maximize') # 
    
    # Pass our objective function as parameter to the study
    #with cross validation, per each fold there is a fixed combination of hyperparam
    
    study.optimize(Objective(clf_name, categorical_columns, trial_cv, X, y, W), n_trials = n_trials) #n_trials is a global variable declared in the experiment settings
    
    # Create experiment row for the final table
    experiment_result = {
                'best_hyparams': study.best_trial.params,
                'clf_name': clf_name
    }
    
    return experiment_result  
	
#Also the Optuna's integration for LightGBM can receive parameters

def optuna_study_lgbm(categorical_columns, trial_cv, X, y, W):
    
    """
    Explicitly take as input global variable:
    Receive as input
        clf_name (string)
        X indipendent features of df
        y dipendent features of df
        X and y are derived from the same train folds when we use stratified 10-fold cross validation
    
    Optuna will search hyperparameters by optimizing
    for the binary_logloss instead of the AUC-PR,
    refer to: https://github.com/optuna/optuna/blob/master/examples/lightgbm_tuner_cv.py
    for lgbm parameters refer to: https://neptune.ai/blog/lightgbm-parameters-guide
    
    Returns a dictionary with the df version, the hyps that maximize AUPR-curve, classifier name
    """
    
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss", #"average_precision", #"binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    
    # Create dataset in optuna format
    # Indicate which are the categorical columns with low cardinality (encoded as integers)
    lgb_dataset = lgb.Dataset(data = X,
                              label = y,
                              weight = W,
                              categorical_feature = categorical_columns, 
                              free_raw_data=False)
    
    """
    Hyperparameter tuner for LightGBM with cross-validation.
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.LightGBMTunerCV.html
    
    tuner is returning the best param for minimizing the binary logloss
    https://readthedocs.org/projects/optuna/downloads/pdf/stable/
    https://github.com/optuna/optuna/blob/master/examples/lightgbm_tuner_cv.py
    """
    
    #LightGBMTuner inherited his param settings from lightgbmcv (API from LightGBM no skt compatible)
    #https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html
    #step wise approach
    tuner = lgb.LightGBMTunerCV(
        lgb_params, lgb_dataset, verbose_eval=1000, early_stopping_rounds=10, nfold = trial_cv, stratified = 1,
        categorical_feature = categorical_columns,
        show_progress_bar = False)
    
    tuner.run()
    
    experiment_result = {
                'best_hyparams': tuner.best_params,
                'clf_name': 'LGBM',
             }
    return experiment_result

#Once we have found the best hyperparameters we need to build a model
def build_model(clf_name, categorical_columns, hyparameters):

        """
        Receive
        classifier name (string)
        hyperparameters is a dictionary of hyperparams found with Optuna.
        we can access the dictionary with **
        https://stackoverflow.com/questions/4989850/dictionary-input-of-function-in-python
        
        Returns Classifier(**hyparameters)
        the estimator with its best hyperparameters. Is used  when the classifier is not LightGBM
        This function is used when we need to test
        the classifiers on the test set of the 10-stratified cross validation

        """
        one_hot_transformer = ColumnTransformer([('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False),
                                         categorical_columns)], 
                                         remainder='passthrough')
        if clf_name == "TABNET" or clf_name == "TABNET0":
                estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                      ('clf', TabNetClassifier(**hyparameters))])
        
        if clf_name == "RF":
                estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                      ('clf', RandomForestClassifier(**hyparameters))])
        
        elif clf_name == "DT":
            estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                      ('clf', DecisionTreeClassifier(**hyparameters))])
            
        elif clf_name == "KNN":
            estimator = Pipeline(steps = [
                      ('one hot', one_hot_transformer),
                      ('SC', MinMaxScaler()), # Good practice to use scaler 
                      ('clf', KNeighborsClassifierW(**hyparameters))])
        
        elif clf_name == "LR":
            estimator = Pipeline(steps = [
                  ('one hot', one_hot_transformer),
                  ('SC', MinMaxScaler()), # Good practice to use scaler in logistic regression
                  ('clf', LogisticRegression(**hyparameters))])
  
        elif clf_name == "XGB":
            estimator = Pipeline(steps = [
                  ('one hot', one_hot_transformer),
                  ('clf', XGBClassifier(**hyparameters))])
        
        elif clf_name == "LGBM2":
            estimator = Pipeline(steps = [
                  ('one hot', one_hot_transformer),
                  ('clf', LGBMClassifier(**hyparameters))])
        
        elif clf_name == 'LGBM':
            estimator =  Pipeline(steps = [ ('clf', LGBMClassifier(**hyparameters))])
            
        return estimator