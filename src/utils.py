import statistics
from scipy import stats
import pandas as pd
import os
import folktables as ft
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# from pytorch_tabnet.tab_model import TabNetClassifier
import random
import numpy as np

# from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._base import _get_weights
from sklearn.utils.validation import _num_samples
from sklearn.utils.extmath import weighted_mode


def set_seed(seed=None):
    """
    Set random seed for reproducibility.
    :param seed: the random seed. If None, no action is taken. Default: None
    :param seed_torch: if True, sets the random seed for pytorch. Default: True.
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed {seed} has been set.")


states = sorted(
    [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ]
)

# categorical attributes
cat_atts = [
    "SCHL",
    "MAR",
    "SEX",
    "DIS",
    "ESP",
    "CIT",
    "MIG",
    "MIL",
    "ANC",
    "NATIVITY",
    "DEAR",
    "DEYE",
    "DREM",
    "ESR",
    "ST",
    "FER",
    "RAC1P",
]


# subset of attributes: subset1 is the one used in the paper
def get_attributes(subset="all"):
    if subset == "subset1":
        atts = ["SCHL", "MAR", "AGEP", "SEX", "RAC1P"]
    elif subset == "subset2":
        atts = ["AGEP", "SEX", "RAC1P"]
    elif subset == "cat":
        atts = cat_atts
    else:
        atts = ft.ACSPublicCoverage.features
    return atts


# data split into training and test
def split_data(X, y, test_size=0.25, random_state=42):
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


# load folktables state data
def load_folktables_data(state, survey_year="2017", horizon="1-Year", survey="person"):
    # add check for data, so it doesn't need to download
    root_dir = ""
    state_codes = pd.read_csv(os.path.join(root_dir, "data", "state_codes.csv"))
    # To avoid downloading each time, check per state if downloaded, if not, download
    # Either way, append the state data to acs_data data frame, updating the region field
    # get state code
    code = state_codes.loc[state_codes["USPS"] == state]["numeric"].values[0]
    data_path = os.path.join(
        root_dir, "data", survey_year, horizon, f"psam_p{code}.csv"
    )
    # This file path works with person, not household survey
    if os.path.exists(data_path):
        print(data_path)
        # load from csv, update to region == i, append to acs_data
        state_data = pd.read_csv(data_path)
    else:
        # download that state (and save in .csv format)
        data_source = ft.ACSDataSource(
            survey_year=survey_year,
            horizon=horizon,
            survey=survey,
            root_dir=os.path.join(root_dir, "data"),
        )
        state_data = data_source.get_data(states=[state], download=True)
    return state_data


# load and split data about states
def load_ACSPublicCoverage(subset, states=states, year="2017"):
    # Dictionaries mapping states to train-test data
    X_train_s, X_test_s, y_train_s, y_test_s = dict(), dict(), dict(), dict()
    task_method = ft.ACSPublicCoverage
    for s in states:
        print(s, end=" ")
        source_data = load_folktables_data(s, year, "1-Year", "person")
        features_s, labels_s, group_s = task_method.df_to_numpy(source_data)
        X_s = pd.DataFrame(features_s, columns=task_method.features)
        X_s["y"] = labels_s
        y_s = X_s["y"]
        X_s = X_s[subset]
        X_train_s[s], X_test_s[s], y_train_s[s], y_test_s[s] = split_data(X_s, y_s)
    # Target is same as source, because in the same year
    X_train_t, X_test_t, y_train_t, y_test_t = X_train_s, X_test_s, y_train_s, y_test_s
    return (
        X_train_s,
        X_test_s,
        y_train_s,
        y_test_s,
        X_train_t,
        X_test_t,
        y_train_t,
        y_test_t,
    )


def load_ACSIncome(subset, states=states, year="2017"):
    # Dictionaries mapping states to train-test data
    X_train_s, X_test_s, y_train_s, y_test_s = dict(), dict(), dict(), dict()
    task_method = ft.ACSIncome
    for s in states:
        print(s, end=" ")
        source_data = load_folktables_data(s, year, "1-Year", "person")
        features_s, labels_s, group_s = task_method.df_to_numpy(source_data)
        X_s = pd.DataFrame(features_s, columns=task_method.features)
        X_s["y"] = labels_s
        y_s = X_s["y"]
        X_s = X_s[subset]
        X_train_s[s], X_test_s[s], y_train_s[s], y_test_s[s] = split_data(X_s, y_s)
    # Target is same as source, because in the same year
    X_train_t, X_test_t, y_train_t, y_test_t = X_train_s, X_test_s, y_train_s, y_test_s
    return (
        X_train_s,
        X_test_s,
        y_train_s,
        y_test_s,
        X_train_t,
        X_test_t,
        y_train_t,
        y_test_t,
    )


# extract metrics from confusion matrix
def cm_metrics(cm):
    TN, FP, FN, TP = cm.ravel()
    N = TP + FP + FN + TN  # Total population
    ACC = (TP + TN) / N  # Accuracy
    TPR = TP / (TP + FN)  # True positive rate
    FPR = FP / (FP + TN)  # False positive rate
    FNR = FN / (TP + FN)  # False negative rate
    PPP = (TP + FP) / N  # % predicted as positive
    return [ACC, TPR, FPR, FNR, PPP]


# calculate accuracy and fairness metrics
def get_metric(r, m):
    # accuracy - the higher the better
    if m == "acc":
        return cm_metrics(r["cm"])[0]
    # equal accuracy - the smaller the better
    if m == "eqacc":
        return abs(
            cm_metrics(r["cm_protected"])[0] - cm_metrics(r["cm_unprotected"])[0]
        )
    # equality of opportunity - the smaller the better
    if m == "eop":
        return abs(
            cm_metrics(r["cm_protected"])[1] - cm_metrics(r["cm_unprotected"])[1]
        )
    # demographic parity - the smaller the better
    if m == "dp":
        return abs(
            cm_metrics(r["cm_protected"])[4] - cm_metrics(r["cm_unprotected"])[4]
        )
    raise "unknown metric"


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
        """added sample_weight"""
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
            nv = neigh_ind.reshape((neigh_ind.shape[0] * neigh_ind.shape[1],))
            weights = self.sample_weight.iloc[nv].to_numpy()
            weights = weights.reshape(neigh_ind.shape)
            # print(weights.shape)
            # print(weights[1,:])
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
            nv = neigh_ind.reshape((neigh_ind.shape[0] * neigh_ind.shape[1],))
            weights = self.sample_weight.iloc[nv].to_numpy()
            weights = weights.reshape(neigh_ind.shape)
            # print(weights.shape)
            # print(weights[1,:])
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
    # l, u = m-s, m+s
    # logit interval
    """
    el = log(m/(1-m))
    sl = sqrt(m*(1-m)/npos)*(1/m+1/(1-m))
    cl = .95
    l = el + stats.norm.ppf((1-cl)/2)*sl
    u = el + stats.norm.ppf((1+cl)/2)*sl
    l = exp(l)/(1+exp(l))
    u = exp(u)/(1+exp(u)) 
    """
    # normal interval
    l, u = stats.t.interval(0.95, len(res) - 1, loc=m, scale=stats.sem(res))
    return m, s, l, u


def ci_auc_l(res):
    return ci_auc(res)[2]


def ci_auc_u(res):
    return ci_auc(res)[3]
