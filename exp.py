import sklearn.base
from xgboost import XGBClassifier
from folktables import ACSDataSource, ACSIncome
from src.utils import set_seed
from src import _partial_dependence_weighted as pdw
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pandas as pd
import random
import os
import argparse

#settings for the plots
plt.style.use('seaborn-whitegrid')
plt.rc('font', size=16)
plt.rc('legend', fontsize=16)
plt.rc('lines', linewidth=2)
plt.rc('axes', linewidth=2)
plt.rc('axes', edgecolor='k')
plt.rc('xtick.major', width=2)
plt.rc('xtick.major', size=10)
plt.rc('ytick.major', width=2)
plt.rc('ytick.major', size=10)
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)




def train_model(df_in_tr, atts, model_type='xgb', weight_column=None, seed=42):
    """
    Train a model on the training dataset
    df_in_tr: pd.DataFrame - training dataset
    atts: list - list of attributes to use for training
    model_type: str - type of model to use
    weight_column: pd.Series - weight column to use for training
    seed: int - seed for reproducibility
    """
    if model_type == 'xgb':
        clf = XGBClassifier(n_jobs=4, random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"],sample_weight=weight_column)
    elif model_type == 'lr':
        clf = LogisticRegression(random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"])
    elif model_type == 'rf':
        clf = RandomForestClassifier(n_jobs=4, random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"])
    elif model_type == 'lgbm':
        clf = LGBMClassifier(n_jobs=4, random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"], sample_weight=weight_column)
    elif model_type == 'mlp':
        clf = MLPClassifier(random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"])
    return clf

def get_weights(df_s, df_t, var_sel):
    """
    Get the weights for the source dataset
    these are the weights that are used to reweight the source dataset to look like the target dataset
    TODO: add a check for the number of bins
    TODO: double check it is ok how they are computed right now

    df_s: pd.DataFrame - source dataset
    df_t: pd.DataFrame - target dataset
    var_sel: str - variable to use for the reweighting
    """
    bins_ = np.histogram_bin_edges(df_t[var_sel], bins=10)
    occ_1 = np.digitize(df_s[var_sel], bins=bins_)
    occ_2 = np.digitize(df_t[var_sel], bins=bins_)
    # check if the bins are the same
    if occ_1.max() < occ_2.max():
        limit = occ_2.max() - occ_1.max()
        bins_ = bins_[:-limit]
        occ_2 = np.digitize(df_t[var_sel], bins=bins_)
    if occ_1.min() < occ_2.min():
        limit = occ_2.min() - occ_1.min()
        bins_ = bins_[limit:]
        occ_1 = np.digitize(df_s[var_sel], bins=bins_)
    # get the relative weights for the source dataset
    counts_1 = np.unique(occ_1, return_counts=True)[1]
    counts_2 = np.unique(occ_2, return_counts=True)[1]
    total_1 = np.sum(counts_1)
    total_2 = np.sum(counts_2)
    norm_weight_1 = counts_1 / total_1
    norm_weight_2 = counts_2 / total_2
    # get the weights
    weights = norm_weight_2/ (norm_weight_1)
    dict_map = {q: w for q, w in zip(np.unique(occ_1), weights)}
    df_s["weight"] = occ_1
    df_s["occ"] = occ_1
    df_s["weight"] = df_s["weight"].map(dict_map)
    return df_s, bins_



def get_data(filename, seed: int = 42, synth_weight: float = 0.10, state1 = "CA", state2 = "PR"):
    """
    Get the data for the experiment
    filename: str - name of the dataset to use
    seed: int - seed for reproducibility
    synth_weight: float - synthetic weight for the synthetic dataset
    state1: str - state 1 for the folktables dataset
    state2: str - state 2 for the folktables dataset
    """
    set_seed(seed)
    if filename == "folktables":
        data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
        ca_data = data_source.get_data(states=[state1], download=True)
        ca_data_ood = data_source.get_data(states=[state2], download=True)
        # features, label, group = ACSEmployment.df_to_numpy(acs_data)
        ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data)
        ca_features = ca_features.drop(columns="RAC1P")
        ca_features["group"] = ca_group
        df_in = ca_features.copy()
        df_in["TARGET"] = ca_labels.astype(int)
        atts = ca_features.columns
        ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data_ood)
        ca_features = ca_features.drop(columns="RAC1P")
        ca_features["group"] = ca_group
        df_ood = ca_features.copy()
        df_ood["TARGET"] = ca_labels.astype(int)
        df_in_tr, df_in_te = train_test_split(df_in, test_size=0.2, random_state=42, stratify=df_in["TARGET"])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(df_in_tr["AGEP"], alpha=1, label="{} Sample".format(state1))
        ax.hist(df_ood["AGEP"], alpha=.5, label="{} Sample".format(state2))
        ax.legend()
        ax.set_title("Age distribution in CA and PR")
        plt.savefig("plots/flktables_{}{}_agedistr.png".format(state1, state2), dpi=300, bbox_inches='tight')
    elif filename == "synth":
        X, y = make_classification(n_samples=10000, n_features=10, n_informative=2, n_redundant=0, random_state=seed)
        #add a sample selection method
        atts = ["X{}".format(i) for i in range(10)]
        df = pd.DataFrame(X, columns=atts)
        df["TARGET"] = y
        df["biased_choice"] = np.where(df["X0"] > 0, synth_weight, 1- synth_weight)
        df["biased_choiceRnd"] = df["biased_choice"].transform(lambda x: random.choices([0, 1], [x, 1-x])[0])
        df_in, df_ood = train_test_split(df, test_size=0.3, random_state=42, stratify=df["TARGET"])
        df_ood.drop(["biased_choice", "biased_choiceRnd"], inplace=True, axis=1)
        df_in_tr = df_in[df_in["biased_choiceRnd"] == 0].copy()
        df_in_val = df_in[df_in["biased_choiceRnd"] == 1].copy()
        df_in_tr, df_in_te = train_test_split(df_in_tr, test_size=.3, random_state=42, stratify=df_in_tr["TARGET"])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(df_in_tr["X0"], alpha=1, label="Sample Biased")
        ax.hist(df_ood["X0"], alpha=.5, label="True Distribution")
        ax.legend()
        ax.set_title("X0 distribution in training and true data - sw={}".format(synth_weight))
        plt.savefig("plots/synth_data_{}.png".format(synth_weight), dpi=300, bbox_inches='tight')
    return df_in_tr, df_in_te, df_ood, atts

def test_and_plot(dataset_name, classifier, seed: int = 42, sw: float = 0.10,
                  state1: str = "CA", state2: str = "PR", run = "unweighted"):
    """
    Test the model and plot the results
    dataset_name: str - name of the dataset to use
    classifier: str - name of the classifier to use
    seed: int - seed for reproducibility
    sw: float - synthetic weight for the synthetic dataset
    state1: str - state 1 for the folktables dataset
    state2: str - state 2 for the folktables dataset
    run: str - type of experiment, can be "unweighted", "sandbox" or "deployed"
    """
    df_in_tr, df_in_te, df_ood, atts = get_data(dataset_name, seed=seed, synth_weight=sw, state1=state1, state2=state2)
    if dataset_name == "synth":
        columns_of_interest = ["X0", "X1"]
        llabels = ["{} - Biased data".format((classifier).upper()),
                   "{} - True data".format((classifier).upper())]
        var_sel = "X0"
    elif dataset_name == "folktables":
        columns_of_interest = ['MAR', 'SCHL']
        llabels = ["{} - {}".format((classifier).upper(), state1),
                   "{} - {}".format((classifier).upper(), state2)]
        var_sel = "AGEP"
    else:
        raise ValueError("Dataset name not recognized")
    if run == "sandbox":
        df_in_tr, bins_ = get_weights(df_in_tr, df_ood, var_sel)
        clf = train_model(df_in_tr, atts, model_type=classifier,
                          seed=seed, weight_column=df_in_tr["weight"])
        tmp = df_in_tr[["occ", "weight"]].copy().groupby(["occ"], as_index=False).mean().reset_index()
        df_in_te["occ"] = np.digitize(df_in_te[var_sel], bins=bins_)
        df_in_te = df_in_te.merge(tmp, on="occ", how="left")
        df_ood["occ"] = np.digitize(df_ood[var_sel], bins=bins_)
        df_ood = df_ood.merge(tmp, on="occ", how="left")
        print(df_in_tr["weight"].isna().sum())
        print(df_in_te["weight"].isna().sum())
        print(df_ood["weight"].isna().sum())
        clf_oracle = train_model(df_ood, atts, model_type=classifier,
                                 seed=seed)
    elif run == "deployed":
        df_in_tr, bins_ = get_weights(df_in_tr, df_ood, var_sel)
        clf = train_model(df_in_tr, atts, model_type=classifier,
                          seed=seed, weight_column=None)
        tmp = df_in_tr[["occ", "weight"]].copy().groupby(["occ"], as_index=False).mean().reset_index()
        df_in_te["occ"] = np.digitize(df_in_te[var_sel], bins=bins_)
        df_in_te = df_in_te.merge(tmp, on="occ", how="left")
        df_ood["occ"] = np.digitize(df_ood[var_sel], bins=bins_)
        df_ood = df_ood.merge(tmp, on="occ", how="left")
        print(df_in_tr["weight"].isna().sum())
        print(df_in_te["weight"].isna().sum())
        print(df_ood["weight"].isna().sum())
        clf_oracle = train_model(df_ood, atts, model_type=classifier,
                                 seed=seed)
    elif run == "unweighted":
        clf = train_model(df_in_tr, atts, model_type=classifier,
                          seed=seed)
        clf_oracle = train_model(df_ood, atts, model_type=classifier,
                          seed=seed)
    scores_in = clf.predict_proba(df_in_te[atts])[:, 1]
    scores_ood = clf.predict_proba(df_ood[atts])[:, 1]
    preds_in = clf.predict(df_in_te[atts])
    preds_ood = clf.predict(df_ood[atts])
    res = pd.DataFrame()
    acc_in = skm.accuracy_score(df_in_te["TARGET"], preds_in)
    acc_ood = skm.accuracy_score(df_ood["TARGET"], preds_ood)
    roc_auc_in = skm.roc_auc_score(df_in_te["TARGET"], scores_in)
    roc_auc_ood = skm.roc_auc_score(df_ood["TARGET"], scores_ood)
    res["accuracy_in"] = [acc_in]
    res["accuracy_ood"] = acc_ood
    res["roc_auc_in"] = roc_auc_in
    res["roc_auc_ood"] = roc_auc_ood
    res["model"] = clf.__class__.__name__
    res["seed"] = seed
    res["run"] = run
    res["dataset"] = dataset_name
    res["sw"] = sw
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/{}".format(dataset_name)):
        os.mkdir("results/{}".format(dataset_name))
    if not os.path.exists("plots"):
        os.mkdir("plots")
    res.to_csv("results/{}/results_{}_{}.csv".format(dataset_name,clf.__class__.__name__, run), index=False)
    for col in columns_of_interest:
        pf_insample_unw = "plots/{}_{}_{}_{}_in_{}.png".format(clf.__class__.__name__,
                                                               dataset_name, sw, col, run)
        pf_ood_unw = "plots/{}_{}_{}_{}_ood_{}.png".format(clf.__class__.__name__,
                                                           dataset_name, sw, col, run)
        if run == "unweighted":
            plot_and_save_weighted_pdp(clf, df_in_te[atts], [col], weight_column=None, path_file=pf_insample_unw)
            plot_and_save_weighted_pdp(clf, df_ood[atts], [col], weight_column=None, path_file=pf_ood_unw)
            multiple_plot_and_save_weighted_pdp([clf, clf_oracle], [df_ood[atts], df_ood[atts]], [col], weight_column=None,
                                                list_labels=llabels,
                                                path_file=pf_insample_unw.replace("_in_", "_both_"))
        elif run == "sandbox":
            if dataset_name == "synth":
                llabels = ["unweighted - Biased data", "weighted - Biased data", "Oracle - whole data"]
            else:
                llabels = ["unweighted - {}".format(state2), "weighted - {}".format(state2), "{} - Oracle".format(state2)]
            # try:
            plot_and_save_weighted_pdp(clf, df_in_te[atts], [col], weight_column=df_in_te["weight"], path_file=pf_insample_unw)
            # except: import pdb; pdb.set_trace()
            plot_and_save_weighted_pdp(clf, df_ood[atts], [col], weight_column=None, path_file=pf_ood_unw)
            multiple_plot_and_save_weighted_pdp([clf, clf, clf_oracle], [df_ood[atts], df_ood[atts], df_ood[atts]],
                                                [col], weight_column=[None, df_ood["weight"], None],
                                                list_labels=llabels,
                                                path_file=pf_insample_unw.replace("_in_", "_both_"))
        elif run == "deployed":
            if dataset_name == "synth":
                llabels = ["unweighted - Biased data", "weighted - Biased data", "Oracle - whole data"]
            else:
                llabels = ["unweighted - {}".format(state2), "weighted - {}".format(state2), "{} - Oracle".format(state2)]
            plot_and_save_weighted_pdp(clf, df_in_te[atts], [col], weight_column=df_in_te["weight"], path_file=pf_insample_unw)
            plot_and_save_weighted_pdp(clf_oracle, df_ood[atts], [col], weight_column=None, path_file=pf_ood_unw)
            multiple_plot_and_save_weighted_pdp([clf, clf, clf_oracle], [df_ood[atts], df_ood[atts], df_ood[atts]],
                                                [col], weight_column=[None, df_ood["weight"], None],
                                                list_labels=llabels,
                                                path_file=pf_insample_unw.replace("_in_", "_both_"))


def plot_and_save_weighted_pdp(clf, X, columns_of_interest: list, weight_column: pd.Series = None,
                               path_file: str = "default.png"):
    """
    Plot and save the weighted PDP
    clf: sklearn.base.BaseEstimator - classifier to use
    X: pd.DataFrame - dataset to use
    columns_of_interest: list - list of columns to use
    weight_column: pd.Series - weight column to use
    path_file: str - path to save the plot
    """
    pdp_res = {}

    delta_pdp = pd.DataFrame(columns=['Theme', 'CLF'])  # here, delta = 0 - 10

    for theme_var in columns_of_interest:
        print(theme_var)

        if weight_column is None:
            print('running standard PDP')
        else:
            print('running weighted PDP')

        # to get the full range of X vals:
        grid_res = len(X[theme_var].unique()) + 1

        # calculate PD for both classifiers
        pd_clf = pdw.partial_dependence(clf, X, [theme_var],
                                   grid_resolution=grid_res,
                                   method='brute', kind='average', response_method='predict_proba',
                                   sample_weight=weight_column)
        # store results
        pdp_res[theme_var] = {}
        pdp_res[theme_var]['CLF'] = pd_clf

        # store deltas
        temp_delta_pdp = pd.DataFrame()
        temp_delta_pdp['Theme'] = [theme_var.split('_')[0]]
        temp_delta_pdp['CLF'] = round(pd_clf['average'][0][0] - pd_clf['average'][0][-1], 3)
        # temp_delta_pdp['LG'] = round(pd_lg['average'][0][0] - pd_lg['average'][0][-1], 3)
        delta_pdp = pd.concat([delta_pdp, temp_delta_pdp], axis=0)
        del temp_delta_pdp
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # create and save plot
        ax.plot(pd_clf['values'][0], pd_clf['average'][0], label=clf.__class__.__name__)
        # plt.plot(pd_lr['values'][0], pd_lr['average'][0], label='LR')

        ax.legend()
        # to be changed?
        ax.set_xlabel(theme_var.split('_')[0] + ' theme $t$')
        ax.set_ylabel('$\hat{b}_T (t)$')
        # plt.title('PDP for {} theme'.format(theme_var.split('_')[0]))
        # ax.set_xlim(X[theme_var].min() -0.01, X[theme_var].max() +0.01)
        # plt.ylim([0, np.max(pd_lg['average'] + pd_lr['average'])[0].round(1) + 0.25])
        # ax.set_ylim([pd_clf['average'][0].min().round(1) - 0.05, pd_clf['average'][0].max().round(1) + 0.05])
        # save
        plt.savefig(fname=path_file,
                    bbox_inches='tight',
                    dpi=300)
        # close with current plot
        plt.clf()

def multiple_plot_and_save_weighted_pdp(clf: list, X: list, columns_of_interest: list,
                                        weight_column: list = None,
                                        list_labels = ["CA", "PR"],
                                        path_file: str = "default.png"):
    """
    Plot and save the weighted PDP
    clf: list - list of classifiers to use
    X: list - list of datasets to use
    columns_of_interest: list - list of columns to use
    weight_column: list - list of weight columns to use
    list_labels: list - list of labels to use
    path_file: str - path to save the plot
    """
    pdp_res = {}
    delta_pdp = pd.DataFrame(columns=['Theme', 'CLF'])  # here, delta = 0 - 10
    for theme_var in columns_of_interest:
        print(theme_var)

        if weight_column is None:
            print('running standard PDP')
            weight_column = [None, None]
        else:
            print('running weighted PDP')

        # to get the full range of X vals:

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        for j, db in enumerate(X):
            grid_res = len(db[theme_var].unique()) + 1
            # calculate PD for both classifiers
            pd_clf = pdw.partial_dependence(clf[j], db, [theme_var],
                                            grid_resolution=grid_res,
                                            method='brute', kind='average', response_method='predict_proba',
                                            sample_weight=weight_column[j])
            # store results
            pdp_res[theme_var] = {}
            pdp_res[theme_var]['CLF'] = pd_clf

            # store deltas
            temp_delta_pdp = pd.DataFrame()
            temp_delta_pdp['Theme'] = [theme_var.split('_')[0]]
            temp_delta_pdp['CLF'] = round(pd_clf['average'][0][0] - pd_clf['average'][0][-1], 3)
            # temp_delta_pdp['LG'] = round(pd_lg['average'][0][0] - pd_lg['average'][0][-1], 3)
            delta_pdp = pd.concat([delta_pdp, temp_delta_pdp], axis=0)
            del temp_delta_pdp
            # create and save plot
            ax.plot(pd_clf['values'][0], pd_clf['average'][0], label = list_labels[j])
            # plt.plot(pd_lr['values'][0], pd_lr['average'][0], label='LR')
        ax.legend()
        # to be changed?
        ax.set_xlabel(theme_var.split('_')[0] + ' theme $t$')
        ax.set_ylabel('$\hat{b}_T (t)$')
        # plt.title('PDP for {} theme'.format(theme_var.split('_')[0]))
        ax.set_xlim(db[theme_var].min() - 0.01, db[theme_var].max() + 0.01)
        # plt.ylim([0, np.max(pd_lg['average'] + pd_lr['average'])[0].round(1) + 0.25])
        ax.set_ylim([pd_clf['average'][0].min().round(1) - 0.05, pd_clf['average'][0].max().round(1) + 0.05])
        # save
        plt.savefig(fname=path_file,
                    bbox_inches='tight',
                    dpi=300)
        # close with current plot
        plt.clf()

def main(dataset_name="synth", classifier="xgb", seed=42, sw=0.10, state1="CA", state2="PR"):
    """
    Main function to run the experiments
    """
    test_and_plot(dataset_name, classifier, seed=seed, sw=sw, state1=state1, state2=state2, run="unweighted")
    test_and_plot(dataset_name, classifier, seed=seed, sw=sw, state1=state1, state2=state2, run="sandbox")
    test_and_plot(dataset_name, classifier, seed=seed, sw=sw, state1=state1, state2=state2, run="deployed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="folktables")
    parser.add_argument("--model", type=str, default="xgb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sw", type=float, default=0.10)
    parser.add_argument("--state1", type=str, default="CA")
    parser.add_argument("--state2", type=str, default="PR")
    args = parser.parse_args()
    dataset = args.data
    model = args.model
    seed = args.seed
    sw = args.sw
    state1 = args.state1
    state2 = args.state2
    main(dataset_name=dataset, classifier=model, seed=seed, sw=sw, state1=state1, state2=state2)