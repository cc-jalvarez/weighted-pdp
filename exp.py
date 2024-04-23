from tqdm import tqdm
from xgboost import XGBClassifier
from folktables import ACSDataSource, ACSPublicCoverage, ACSIncome
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

# from joblib import dump, load

# settings for the plots
plt.rc("font", size=16)
plt.rc("legend", fontsize=16)
plt.rc("lines", linewidth=2)
plt.rc("axes", linewidth=2)
plt.rc("axes", edgecolor="k")
plt.rc("xtick.major", width=2)
plt.rc("xtick.major", size=10)
plt.rc("ytick.major", width=2)
plt.rc("ytick.major", size=10)
plt.rc("pdf", fonttype=42)
plt.rc("ps", fonttype=42)


def get_data(
    filename, seed: int = 42, synth_weight: float = 0.10, state1="CA", state2="PR"
):
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
        data_source = ACSDataSource(
            survey_year="2018", horizon="1-Year", survey="person"
        )
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
        df_in_tr, df_in_te = train_test_split(
            df_in, test_size=0.2, random_state=seed, stratify=df_in["TARGET"]
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(df_in_tr["AGEP"], alpha=1, label="{} Sample".format(state1))
        ax.hist(df_ood["AGEP"], alpha=0.5, label="{} Sample".format(state2))
        ax.legend()
        ax.set_title("Age distribution in {} and {}".format(state1, state2))
        plt.savefig(
            "plots/flktables_{}{}_agedistr.png".format(state1, state2),
            dpi=300,
            bbox_inches="tight",
        )
    elif filename == "synth":
        X, y = make_classification(
            n_samples=10000,
            n_features=10,
            n_informative=2,
            n_redundant=0,
            random_state=seed,
        )
        # add a sample selection method
        atts = ["X{}".format(i) for i in range(10)]
        df = pd.DataFrame(X, columns=atts)
        df["TARGET"] = y
        df["biased_choice"] = np.where(df["X0"] > 0, synth_weight, 1 - synth_weight)
        df["biased_choiceRnd"] = df["biased_choice"].transform(
            lambda x: random.choices([0, 1], [x, 1 - x])[0]
        )
        df_in, df_ood = train_test_split(
            df, test_size=0.3, random_state=42, stratify=df["TARGET"]
        )
        df_ood.drop(["biased_choice", "biased_choiceRnd"], inplace=True, axis=1)
        df_in_tr = df_in[df_in["biased_choiceRnd"] == 0].copy()
        df_in_val = df_in[df_in["biased_choiceRnd"] == 1].copy()
        df_in_tr, df_in_te = train_test_split(
            df_in_tr, test_size=0.3, random_state=42, stratify=df_in_tr["TARGET"]
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.hist(df_in_tr["X0"], alpha=1, label="Sample Biased")
        ax.hist(df_ood["X0"], alpha=0.5, label="True Distribution")
        ax.legend()
        ax.set_title(
            "X0 distribution in training and true data - sw={}".format(synth_weight)
        )
        plt.savefig(
            "plots/synth_data_{}.png".format(synth_weight), dpi=300, bbox_inches="tight"
        )

    return df_in_tr, df_in_te, df_ood, atts


def train_model(df_in_tr, atts, model_type="xgb", weight_column=None, seed=42):
    """
    Train a model on the training dataset
    df_in_tr: pd.DataFrame - training dataset
    atts: list - list of attributes to use for training
    model_type: str - type of model to use
    weight_column: pd.Series - weight column to use for training
    seed: int - seed for reproducibility
    """
    if model_type == "xgb":
        clf = XGBClassifier(n_jobs=4, random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"], sample_weight=weight_column)
    elif model_type == "lr":
        clf = LogisticRegression(random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"])
    elif model_type == "rf":
        clf = RandomForestClassifier(n_jobs=4, random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"])
    elif model_type == "lgbm":
        clf = LGBMClassifier(n_jobs=4, random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"], sample_weight=weight_column)
    elif model_type == "mlp":
        clf = MLPClassifier(random_state=seed)
        clf.fit(df_in_tr[atts], df_in_tr["TARGET"])
    return clf


def get_weights(df_s, df_t, var_sel, bins: int or np.array = 10, dataset_name="synth"):
    """
    Get the weights for the source dataset
    these are the weights that are used to reweight the source dataset to look like the target dataset
    TODO: add a check for the number of bins
    TODO: double check it is ok how they are computed right now

    df_s: pd.DataFrame - source dataset
    df_t: pd.DataFrame - target dataset
    var_sel: str - variable to use for the reweighting
    """
    if type(bins) == int:
        bins_ = np.histogram_bin_edges(df_t[var_sel], bins=bins)
    else:
        bins_ = bins
    # get the bins for the source dataset
    # if dataset_name == "synthV2":
    if "synth" in dataset_name:
        binned_source = np.digitize(df_s[var_sel], bins=np.linspace(-4, 4, bins))
        binned_target = np.digitize(df_t[var_sel], bins=np.linspace(-4, 4, bins))
        dict_source = {i: 0 for i, q in enumerate(np.linspace(-4, 4, bins))}
        dict_target = {i: 0 for i, q in enumerate(np.linspace(-4, 4, bins))}
    else:
        binned_source = np.digitize(df_s[var_sel], bins=bins_)
        binned_target = np.digitize(df_t[var_sel], bins=bins_)
        dict_source = {i: 0 for i, q in enumerate(bins_)}
        dict_target = {i: 0 for i, q in enumerate(bins_)}
    for i, q in enumerate(dict_source.keys()):
        dict_source[q] = np.sum(np.where(binned_source == i, 1, 0))
        dict_target[q] = np.sum(np.where(binned_target == i, 1, 0))
    dict_weights = {
        q: (dict_target[q] / dict_source[q]) * (df_s.shape[0]) / (df_t.shape[0])
        if dict_source[q] > 0
        else 1
        for q in dict_source.keys()
    }
    df_s["weight"] = binned_source
    df_s["occ"] = binned_source
    df_s["weight"] = df_s["weight"].map(dict_weights)
    total_weight = (df_s.shape[0]) / (df_t.shape[0])
    df_s["weight"] = df_s["weight"].fillna(total_weight)
    print(np.unique(df_s["weight"], return_counts=True))
    return df_s, bins_


def test_and_plot(
    dataset_name,
    classifier,
    seed: int = 42,
    sw: float = 0.10,
    state1: str = "CA",
    state2: str = "PR",
    run="deployed",
    img_fold="",
    bins: int or np.array = 10,
):
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
    set_seed(seed)
    df_in_tr, df_in_te, df_ood, atts = get_data(
        dataset_name, seed=seed, synth_weight=sw, state1=state1, state2=state2
    )
    if "synth" in dataset_name:
        columns_of_interest = ["X0", "X1"]
        llabels = [
            "{} - Biased data".format((classifier).upper()),
            "{} - True data".format((classifier).upper()),
        ]
        var_sel = "X0"
    elif dataset_name == "folktables":
        data_source = ACSDataSource(
            survey_year="2017", horizon="1-Year", survey="person"
        )
        ca_data = data_source.get_data(states=[state2], download=True)
        # features, label, group = ACSEmployment.df_to_numpy(acs_data)
        ca_features, ca_labels, ca_group = ACSIncome.df_to_pandas(ca_data)
        ca_features = ca_features.drop(columns="RAC1P")
        ca_features["group"] = ca_group
        df_ood_ly = ca_features.copy()
        df_ood_ly["TARGET"] = ca_labels.astype(int)
        columns_of_interest = ["MAR", "SCHL", "AGEP"]
        llabels = [
            "{} - {}".format((classifier).upper(), state1),
            "{} - {}".format((classifier).upper(), state2),
        ]
        var_sel = "AGEP"
    else:
        raise ValueError("Dataset name not recognized")
    if run == "sandbox":
        if dataset_name == "folktables":
            df_in_tr, bins_ = get_weights(
                df_in_tr, df_ood_ly, var_sel, bins=bins, dataset_name=dataset_name
            )
            print(df_in_tr["weight"].isna().sum())
            print(bins_)

        else:
            df_in_tr, bins_ = get_weights(df_in_tr, df_ood, var_sel, bins=bins)
        clf = train_model(
            df_in_tr,
            atts,
            model_type=classifier,
            seed=seed,
            weight_column=df_in_tr["weight"],
        )
        tmp = (
            df_in_tr[["occ", "weight"]]
            .copy()
            .groupby(["occ"], as_index=False)
            .mean()
            .reset_index()
        )
        df_in_te["occ"] = np.digitize(df_in_te[var_sel], bins=bins_)
        df_in_te = df_in_te.merge(tmp, on="occ", how="left")
        print(df_in_tr["weight"].isna().sum())
        print(df_in_te["weight"].isna().sum())
        if df_in_tr["weight"].isna().sum() == 0:
            min_weight = df_in_tr["weight"].max()
            df_in_te["weight"] = np.where(
                df_in_te["weight"].isna(), min_weight, df_in_te["weight"]
            )
            print(df_in_te["weight"].isna().sum())
        # print(df_ood["weight"].isna().sum())
        clf_oracle = train_model(df_in_tr, atts, model_type=classifier, seed=seed)
    elif run == "deployed":
        if dataset_name == "folktables":
            df_in_tr, bins_ = get_weights(
                df_in_tr, df_ood_ly, var_sel, bins=bins, dataset_name=dataset_name
            )
            print(df_in_tr["weight"].isna().sum())
            print(bins_)

        else:
            df_in_tr, bins_ = get_weights(df_in_tr, df_ood, var_sel, bins=bins)
        clf = train_model(
            df_in_tr, atts, model_type=classifier, seed=seed, weight_column=None
        )
        tmp = (
            df_in_tr[["occ", "weight"]]
            .copy()
            .groupby(["occ"], as_index=False)
            .mean()
            .reset_index()
        )
        df_in_te["occ"] = np.digitize(df_in_te[var_sel], bins=bins_)
        df_in_te = df_in_te.merge(tmp, on="occ", how="left")
        print(df_in_tr["weight"].isna().sum())
        print(df_in_te["weight"].isna().sum())
        if df_in_tr["weight"].isna().sum() == 0:
            min_weight = df_in_tr["weight"].max()
            df_in_te["weight"] = np.where(
                df_in_te["weight"].isna(), min_weight, df_in_te["weight"]
            )
            print(df_in_te["weight"].isna().sum())
        # print(df_ood["weight"].isna().sum())
        clf_oracle = train_model(df_in_tr, atts, model_type=classifier, seed=seed)
    elif run == "unweighted":
        clf = train_model(df_in_tr, atts, model_type=classifier, seed=seed)
        clf_oracle = train_model(df_ood, atts, model_type=classifier, seed=seed)
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
    if not os.path.exists("plots/{}".format(dataset_name)):
        os.mkdir("plots/{}".format(dataset_name))
    res.to_csv(
        "results/{}/results_{}_{}.csv".format(
            dataset_name, clf.__class__.__name__, run
        ),
        index=False,
    )
    for col in columns_of_interest:
        if dataset_name == "folktables":
            pf_insample_unw = "{}plots/{}/{}_{}_{}_{}_in_{}_nbins{}.png".format(
                img_fold,
                dataset_name,
                clf.__class__.__name__,
                dataset_name,
                state1 + "-" + state2,
                col,
                run,
                bins,
            )
            pf_ood_unw = "{}plots/{}/{}_{}_{}_{}_ood_{}_nbins{}.png".format(
                img_fold,
                dataset_name,
                clf.__class__.__name__,
                dataset_name,
                state1 + "-" + state2,
                col,
                run,
                bins,
            )
            delta_pdp_path_file = "results/{}/{}_{}_{}_{}_in_{}_nbins{}.png".format(
                dataset_name,
                clf.__class__.__name__,
                dataset_name,
                state1 + "-" + state2,
                col,
                run,
                bins,
            )
            all_pd_clf_path_file = "results/{}/allpdclf_{}_{}_{}_{}_bins{}.csv".format(
                dataset_name,
                clf.__class__.__name__,
                run,
                col,
                state1 + "-" + state2,
                bins,
            )

        else:
            pf_insample_unw = "{}plots/{}/{}_{}_{}_{}_in_{}_nbins{}.png".format(
                img_fold,
                dataset_name,
                clf.__class__.__name__,
                dataset_name,
                sw,
                col,
                run,
                bins,
            )
            pf_ood_unw = "{}plots/{}/{}_{}_{}_{}_ood_{}_nbins{}.png".format(
                img_fold,
                dataset_name,
                clf.__class__.__name__,
                dataset_name,
                sw,
                col,
                run,
                bins,
            )
            all_pd_clf_path_file = (
                "results/{}/all_pd_clf_{}_{}_{}_{}_nbins{}.csv".format(
                    dataset_name, clf.__class__.__name__, run, col, sw, bins
                )
            )

        if run == "unweighted":
            multiple_plot_and_save_weighted_pdp(
                [clf, clf_oracle],
                [df_in_te[atts], df_ood[atts]],
                [col],
                weight_column=None,
                list_labels=llabels,
                path_file=pf_insample_unw.replace("_in_", "_both_"),
                title=run,
            )
        elif run == "sandbox":
            if "synth" in dataset_name:
                llabels = [
                    "unweighted - source data",
                    "weighted - source data",
                    "Oracle - target data",
                ]
            else:
                llabels = [
                    "unweighted - {}".format(state2),
                    "weighted - {}".format(state2),
                    "{} - Oracle".format(state2),
                ]
            delta_pdp, all_pd_clf = multiple_plot_and_save_weighted_pdp(
                [clf, clf, clf],
                [df_in_te[atts], df_in_te[atts], df_ood[atts]],
                [col],
                weight_column=[None, df_in_te["weight"], None],
                list_labels=llabels,
                path_file=pf_insample_unw.replace("_in_", "_both_"),
                title=run,
            )
        elif run == "deployed":
            llabels = [
                "unweighted - source data",
                "weighted - source data",
                "Oracle - target data",
            ]
            delta_pdp, all_pd_clf = multiple_plot_and_save_weighted_pdp(
                [clf, clf, clf],
                [df_in_te[atts], df_in_te[atts], df_ood[atts]],
                [col],
                weight_column=[None, df_in_te["weight"], None],
                list_labels=llabels,
                path_file=pf_insample_unw.replace("_in_", "_both_"),
                title="{} - {}".format(run, clf.__class__.__name__),
            )
        all_pd_clf[col].to_csv(all_pd_clf_path_file, index=False)


def plot_and_save_weighted_pdp(
    clf,
    X,
    columns_of_interest: list,
    weight_column: pd.Series = None,
    path_file: str = "default.png",
    title: str = "default",
):
    """
    Plot and save the weighted PDP
    clf: sklearn.base.BaseEstimator - classifier to use
    X: pd.DataFrame - dataset to use
    columns_of_interest: list - list of columns to use
    weight_column: pd.Series - weight column to use
    path_file: str - path to save the plot
    """
    pdp_res = {}

    delta_pdp = pd.DataFrame(columns=["Theme", "CLF"])  # here, delta = 0 - 10

    for theme_var in columns_of_interest:
        print(theme_var)

        if weight_column is None:
            print("running standard PDP")
        else:
            print("running weighted PDP")

        # to get the full range of X vals:
        grid_res = len(X[theme_var].unique()) + 1

        # calculate PD for both classifiers
        pd_clf = pdw.partial_dependence(
            clf,
            X,
            [theme_var],
            grid_resolution=grid_res,
            method="brute",
            kind="individual",
            response_method="predict_proba",
            sample_weight=weight_column,
        )
        # store results
        pdp_res[theme_var] = {}
        pdp_res[theme_var]["CLF"] = pd_clf

        # store deltas
        temp_delta_pdp = pd.DataFrame()
        temp_delta_pdp["Theme"] = [theme_var.split("_")[0]]
        temp_delta_pdp["CLF"] = round(
            pd_clf["average"][0][0] - pd_clf["average"][0][-1], 3
        )
        # temp_delta_pdp['LG'] = round(pd_lg['average'][0][0] - pd_lg['average'][0][-1], 3)
        delta_pdp = pd.concat([delta_pdp, temp_delta_pdp], axis=0)
        del temp_delta_pdp
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # create and save plot
        ax.plot(pd_clf["values"][0], pd_clf["average"][0], label=clf.__class__.__name__)
        # plt.plot(pd_lr['values'][0], pd_lr['average'][0], label='LR')

        ax.legend()
        # to be changed?
        ax.set_xlabel(theme_var.split("_")[0] + " theme $t$")
        ax.set_ylabel("$\hat{b}_T (t)$")
        plt.title(title)
        # ax.set_xlim(X[theme_var].min() -0.01, X[theme_var].max() +0.01)
        # plt.ylim([0, np.max(pd_lg['average'] + pd_lr['average'])[0].round(1) + 0.25])
        # ax.set_ylim([pd_clf['average'][0].min().round(1) - 0.05, pd_clf['average'][0].max().round(1) + 0.05])
        # save
        plt.savefig(fname=path_file, bbox_inches="tight", dpi=300)
        # close with current plot
        plt.clf()


def multiple_plot_and_save_weighted_pdp(
    clf: list,
    X: list,
    columns_of_interest: list,
    weight_column: list = None,
    list_labels=["CA", "VT"],
    path_file: str = "default.png",
    title: str = "default",
):
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
    delta_pdp = pd.DataFrame(columns=["Theme", "CLF"])  # here, delta = 0 - 10
    dictionary_vars = {}
    for theme_var in columns_of_interest:
        print(theme_var)

        if weight_column is None:
            print("running standard PDP")
            weight_column = [None, None]
        else:
            print("running weighted PDP")

        # to get the full range of X vals:

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        all_pd_clf = pd.DataFrame()
        for j, db in enumerate(X):
            grid_res = len(db[theme_var].unique()) + 1
            # calculate PD for both classifiers
            pd_clf = pdw.partial_dependence(
                clf[j],
                db,
                [theme_var],
                grid_resolution=grid_res,
                method="brute",
                kind="average",
                response_method="predict_proba",
                sample_weight=weight_column[j],
            )
            # store results
            pdp_res[theme_var] = {}
            pdp_res[theme_var]["CLF"] = pd_clf

            # store deltas
            temp_delta_pdp = pd.DataFrame()
            temp_delta_pdp["Theme"] = [theme_var.split("_")[0]]
            temp_delta_pdp["CLF"] = round(
                pd_clf["average"][0][0] - pd_clf["average"][0][-1], 3
            )
            temp_delta_pdp["class"] = "classifier_{}".format(j)
            tmp = pd.DataFrame()
            tmp["X"] = pd_clf["values"][0]
            tmp["PDP"] = pd_clf["average"][0]
            tmp["class"] = "{}".format(list_labels[j])
            all_pd_clf = pd.concat([all_pd_clf, tmp], axis=0)
            # temp_delta_pdp['LG'] = round(pd_lg['average'][0][0] - pd_lg['average'][0][-1], 3)
            delta_pdp = pd.concat([delta_pdp, temp_delta_pdp], axis=0)
            del temp_delta_pdp
            # create and save plot
            ax.plot(pd_clf["values"][0], pd_clf["average"][0], label=list_labels[j])
            tmp_ = pd.DataFrame()
            tmp_["Theme"] = [theme_var.split("_")[0]]
            tmp_["min"] = pd_clf["average"][0].min()
            tmp_["max"] = pd_clf["average"][0].max()
            # plt.plot(pd_lr['values'][0], pd_lr['average'][0], label='LR')
        quants = [np.quantile(all_pd_clf["X"], q) for q in np.linspace(0, 1, 100)]
        all_pd_clf["quantile"] = np.digitize(all_pd_clf["X"], bins=quants)
        all_pd_clf = all_pd_clf.groupby(["quantile", "class"], as_index=False).mean()
        dictionary_vars[theme_var] = all_pd_clf
        ax.legend()
        # to be changed?
        ax.set_xlabel(theme_var)
        ax.set_ylabel("$\hat{f}(x)$")
        plt.title(title)
        ax.set_xlim(db[theme_var].min() - 0.01, db[theme_var].max() + 0.01)
        # plt.ylim([0, np.max(pd_lg['average'] + pd_lr['average'])[0].round(1) + 0.25])
        ax.set_ylim([all_pd_clf["PDP"].min() - 0.05, all_pd_clf["PDP"].max() + 0.05])
        # save
        plt.savefig(fname=path_file, bbox_inches="tight", dpi=300)
        # close with current plot
        plt.close(fig)
    return delta_pdp, dictionary_vars


def main(
    dataset_name="synth",
    classifier="xgb",
    seed=42,
    sw=0.10,
    state1="CA",
    state2="OH",
    bins=10,
):
    """
    Main function to run the experiments
    """
    # C:\\Users\\andre\\Dropbox\\Applicazioni\\Overleaf\\ECAI24 _ The weighted partial dependence plot
    # test_and_plot(dataset_name, classifier, seed=seed, sw=sw, state1=state1, state2=state2, run="unweighted")
    if dataset_name == "all_folktables":
        furthest_pairs = [
            ("KY", "VT"),
            ("VT", "KY"),
            ("TX", "VT"),
            ("VT", "TX"),
            ("CA", "VT"),
            ("VT", "CA"),
        ]
        closest_pairs = [
            ("IN", "NC"),
            ("NC", "IN"),
            ("AZ", "VA"),
            ("VA", "AZ"),
            ("IN", "VA"),
            ("VA", "IN"),
        ]
        for pair in tqdm(furthest_pairs + closest_pairs):
            test_and_plot(
                "folktables",
                classifier,
                seed=seed,
                sw=sw,
                state1=pair[0],
                state2=pair[1],
                run="sandbox",
                bins=10,
            )
            test_and_plot(
                "folktables",
                classifier,
                seed=seed,
                sw=sw,
                state1=pair[0],
                state2=pair[1],
                run="deployed",
                bins=10,
            )
    elif dataset_name == "all_synth":
        for sw in tqdm([0.05, 0.10, 0.25, 0.50]):
            for bins in tqdm([5, 10, 20]):
                test_and_plot(
                    "synth",
                    classifier,
                    seed=seed,
                    sw=sw,
                    state1=state1,
                    state2=state2,
                    run="sandbox",
                    bins=bins,
                )
                test_and_plot(
                    "synth",
                    classifier,
                    seed=seed,
                    sw=sw,
                    state1=state1,
                    state2=state2,
                    run="deployed",
                    bins=bins,
                )
    else:
        test_and_plot(
            dataset_name,
            classifier,
            seed=seed,
            sw=sw,
            state1=state1,
            state2=state2,
            run="sandbox",
            bins=bins,
        )
        test_and_plot(
            dataset_name,
            classifier,
            seed=seed,
            sw=sw,
            state1=state1,
            state2=state2,
            run="deployed",
            bins=bins,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="folktables")
    parser.add_argument("--model", type=str, default="lgbm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sw", type=float, default=0.10)
    parser.add_argument("--state1", type=str, default="CA")
    parser.add_argument("--state2", type=str, default="VT")
    parser.add_argument("--bins", type=int, default=20)
    args = parser.parse_args()
    dataset = args.data
    model = args.model
    seed = args.seed
    sw = args.sw
    state1 = args.state1
    state2 = args.state2
    bins = args.bins
    main(
        dataset_name=dataset,
        classifier=model,
        seed=seed,
        sw=sw,
        state1=state1,
        state2=state2,
        bins=bins,
    )
