import time
import pickle
import numpy as np
from scipy.stats import wasserstein_distance
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

# local imports
import src.utils as utils



def epmf(y, values):
    d = y.value_counts(sort=False, normalize=True).to_dict()
    dist = np.array([d[c] if c in d else 0 for c in values])
    return dist



# calculate distances on an attribute between source and domain
def distances(X_train_s, X_train_t, X_test_t, y_train_s, y_train_t, source, target, att):
    # categorical?
    is_cat = att in utils.cat_atts
    # source training att
    xs = X_train_s[source][att]
    # target training att
    xt = X_train_t[target][att]
    # source y
    ys = y_train_s[source]
    # target y
    yt = y_train_t[target]
    # distinct values of att in source and target
    values = sorted(list(set(xs.unique()) | set(xt.unique())))
    # distinct classes in source and target
    classes = sorted(list(set(ys.unique()) | set(yt.unique())))
    # W(\hat{P}_S(X), \hat{P}_T(X))
    w_st = wasserstein_distance(epmf(xs, values), epmf(xt, values))
    # W(\hat{P}_T(Y|X), \hat{P}(Y|X)))
    w_y_cond = 0
    # accumulating \hat{P}(Y|X)
    y_est = 0
    for value in values:
        # \hat{P}_T(X==x)
        p_t = np.mean(xt==value)
        # \hat{P}_S(Y|X=x)*\hat{P}_T(X=x)
        y_est += epmf(ys[xs==value], classes)*p_t
        # add to w_y_cond
        ysv = ys[(xs==value) if is_cat else (xs<=value)]
        ytv = yt[(xt==value) if is_cat else (xt<=value)]
        # d(\hat{P}_S(Y|X=x)), \hat{P}_T(Y|X=x)))*\hat{P}_T(X=x)
        w_y_cond += wasserstein_distance(epmf(ysv, classes), epmf(ytv, classes))*p_t
    # W(\hat{P}_T(Y), \hat{P}(Y)))
    w_y = wasserstein_distance(y_est, epmf(yt, classes))
    return {
            'w_st':w_st,
            'w_y':w_y,
            'w_y_cond':w_y_cond,
            'len_x':len(values),
            'len_s_train':len(xs),
            'len_t_train':len(xt),
            'len_t_test':len(X_test_t[target][att])
           }

def main():
    dists = dict()
    attributes = utils.get_attributes('subset1')
    # load data
    X_train_s, X_test_s, y_train_s, y_test_s, X_train_t, X_test_t, y_train_t, y_test_t = utils.load_ACSIncome(
        attributes)
    for sr in utils.states:
        for tg in utils.states:
            dists[(sr, tg)] = dict()
            for att in attributes:
                dists[(sr, tg)][att] = distances(X_train_s, X_train_t, X_test_t, y_train_s, y_train_t, sr, tg, att)
                # print(sr, tg, att, dists[(sr, tg)][att])
    if not os.path.exists("results"):
        os.mkdir("results")
    pickle.dump(dists, open("results/distances_income.pkl", "wb"))

if __name__ == "__main__":
    main()