import json
from itertools import chain
from collections import defaultdict
import numpy as np
import os
import glob
import pandas as pd
from hmmlearn import _hmmc
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from graphviz import Digraph


def get_markov_chain(matrix):
    n = matrix.shape[0]
    dot = Digraph(comment="Markov Chain")
    dot.attr(rankdir="LR", size="8,5")
    dot.attr("node", shape="circle")

    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                dot.edge(str(i), str(j), label=str(matrix[i][j]))

    return dot


def make_hmm_data(data_dir, cols, sort=True, first_n=1000):
    print(data_dir)
    print(glob.glob(data_dir + "*"))
    if sort:
        try:
            dfs = [
                pd.read_csv(file)
                .sort_values("step")
                .reset_index(drop=True)[cols]  # restrict to cols of interest
                .head(first_n)
                for file in glob.glob(data_dir + "*")
            ]
        except KeyError:
            dfs = [
                pd.read_csv(file)
                .sort_values("epoch")
                # .sort_index()
                .reset_index(drop=True)[cols]  # restrict to cols of interest
                .head(first_n)
                for file in glob.glob(data_dir + "*")
            ]
    else:
        dfs = [
            pd.read_csv(file)[cols].head(first_n)  # restrict to cols of interest
            for file in glob.glob(data_dir + "*")
        ]

    dfs = [df for df in dfs if not df.isnull().values.any()]  # remove invalid dfs

    train_dfs, test_dfs = train_test_split(dfs, test_size=0.2, random_state=0)

    train_data = np.vstack(
        [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in train_dfs]
    )
    test_data = np.vstack(
        [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in test_dfs]
    )

    return train_dfs, test_dfs, train_data, test_data


def unpack_vals(
    subsample,
    l1,
    l2,
    trace,
    spectral,
    code_sparsity,
    computational_sparsity,
    mean_lambda,
    var_lambda,
):
    l1.append(subsample["l1"])
    l2.append(subsample["l2"])
    trace.append(subsample["trace"])
    spectral.append(subsample["spectral"])
    code_sparsity.append(subsample["code_sparsity"])
    computational_sparsity.append(subsample["computational_sparsity"])
    mean_lambda.append(subsample["mean_singular_value"])
    var_lambda.append(subsample["var_singular_value"])


# TODO: delete this function, fold into cnn method
def get_stats_for_run(file_pths, is_transformer, has_loss=False):
    buf = defaultdict(list)

    for file_pth in file_pths:
        # logging formats have been inconsistent

        # name format: stats_{seed}_{step}.json
        # step = int(os.path.basename(file_pth).split("_")[-1].split(".")[0])

        # name format: stats_{step}epoch_{lr}lr_{optimizer}_{seed}seed
        # step = int(os.path.basename(file_pth).split("_")[1].split("epoch")[0])

        # name format: stats_{step}_losses.json
        # step = int(os.path.basename(file_pth).split("_")[1])

        # name format: epoch{step}.json
        step = int(os.path.basename(file_pth).split(".")[0].split("epoch")[1])

        # name format: stats_step{step}
        # step = int(os.path.basename(file_pth).split("step")[1])

        # name format: step{step}.json
        # step = int(os.path.basename(file_pth).split(".")[0].split("step")[1])

        l1_buf = []
        l2_buf = []
        trace_buf = []
        spectral_buf = []
        code_sparsity_buf = []
        computational_sparsity_buf = []
        mean_lambda_buf = []
        variance_lambda_buf = []

        with open(file_pth, "r") as f:
            data = json.loads(f.read())

        if is_transformer:
            samples = chain(
                data["k"],
                data["v"],
                data["q"],
                data["in_proj"],
                data["ffn_in"],
                data["ffn_out"],
            )
        else:
            samples = data["w"]
        for sample in samples:
            for subsample in sample.values():
                if isinstance(subsample, dict):
                    unpack_vals(
                        subsample,
                        l1_buf,
                        l2_buf,
                        trace_buf,
                        spectral_buf,
                        code_sparsity_buf,
                        computational_sparsity_buf,
                        mean_lambda_buf,
                        variance_lambda_buf,
                    )
                elif isinstance(subsample, list):
                    for subsubsample in subsample:
                        unpack_vals(
                            subsubsample,
                            l1_buf,
                            l2_buf,
                            trace_buf,
                            spectral_buf,
                            code_sparsity_buf,
                            computational_sparsity_buf,
                            mean_lambda_buf,
                            variance_lambda_buf,
                        )

        buf["l1"].append(np.mean(l1_buf))
        buf["l2"].append(np.mean(l2_buf))
        buf["trace"].append(np.mean(trace_buf))
        buf["spectral"].append(np.mean(spectral_buf))
        buf["code_sparsity"].append(np.mean(code_sparsity_buf))
        buf["computational_sparsity"].append(np.mean(computational_sparsity_buf))
        buf["mean_lambda"].append(np.mean(mean_lambda_buf))
        buf["variance_lambda"].append(np.nanmean(variance_lambda_buf))

        # global stats
        mean_w = data["w_all"]["mean"]
        median_w = data["w_all"]["median"]
        var_w = data["w_all"]["var"]
        mean_b = data["b_all"]["mean"]
        median_b = data["b_all"]["median"]
        var_b = data["b_all"]["var"]

        buf["mean_w"].append(mean_w)
        buf["median_w"].append(median_w)
        buf["var_w"].append(var_w)
        buf["mean_b"].append(mean_b)
        buf["median_b"].append(median_b)
        buf["var_b"].append(var_b)

        buf["step"].append(step)

        buf["grad_sym"].append(data["grad_sym"])
        buf["train_dist_irr"].append(data["train_dist_irr"])
        buf["test_dist_irr"].append(data["test_dist_irr"])

        if has_loss:
            buf["train_loss"].append(data["train_loss"])
            buf["eval_loss"].append(data["eval_loss"])
            buf["train_accuracy"].append(data["train_accuracy"])
            buf["eval_accuracy"].append(data["eval_accuracy"])

    return buf


def get_stats_for_cnn(file_pths, has_loss=False):
    # holds the csv data
    buf = defaultdict(list)

    for file_pth in file_pths:
        # name format: step{step}.json
        step = int(os.path.basename(file_pth).split(".")[0].split("step")[1])

        buffer_dict = defaultdict(list)

        with open(file_pth, "r") as f:
            data = json.loads(f.read())
            samples = data["w"]

        for sample in samples:
            for key, val in sample.items():
                if isinstance(val, list):
                    buffer_dict[key].extend(val)
                else:
                    buffer_dict[key].append(val)

        for key, val in buffer_dict.items():
            buf[key].append(np.mean(val))

        del buf["singular_values"]

        # global stats
        mean_w = data["w_all"]["mean"]
        median_w = data["w_all"]["median"]
        var_w = data["w_all"]["var"]
        mean_b = data["b_all"]["mean"]
        median_b = data["b_all"]["median"]
        var_b = data["b_all"]["var"]

        buf["mean_w"].append(mean_w)
        buf["median_w"].append(median_w)
        buf["var_w"].append(var_w)
        buf["mean_b"].append(mean_b)
        buf["median_b"].append(median_b)
        buf["var_b"].append(var_b)

        buf["step"].append(step)

        if has_loss:
            buf["train_loss"].append(data["train_loss"])
            buf["eval_loss"].append(data["eval_loss"])
            buf["train_accuracy"].append(data["train_accuracy"]["accuracy"])
            buf["eval_accuracy"].append(data["eval_accuracy"]["accuracy"])

    return buf


def break_list_by_lengths(lst, lengths):
    """
    Break the list in chunks of specified length

    Args:
        lst (list): input list
        lengths (int): lengh of one chunk

    Returns:
        result: list of chunks of specified length
    """
    result = []
    start_index = 0

    for length in lengths:
        sublist = lst[start_index:start_index + length]
        result.append(sublist)
        start_index += length

    return result


def softmax_with_overflow(logits):

    """
    log-sum-exp. convert logits into probabilities using softmax

    """
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def find_i_followed_by_j(lst, i, j):
    """
    Find transition in the estimated hidden state

    Returns:
        indexes: indexes of predictions where i is followed by j
    """
    indexes = [index for index in range(len(lst) - 1) if lst[index] == i and
               lst[index + 1] == j]
    return indexes


def load_model(model_path):

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


def save_model(model_path, model):

    with open(model_path + ".pkl", "wb") as f:
        pickle.dump(model, f)


def get_features_for_transition(model, data, best_predictions, lengths,
                                phase_1, phase_2):

    '''
    For each time a transition (phase_1 -> phase_2) happens,
    compute the derivatives for each feature.

    This computation is slightly inefficient, in that it computes the
    entire forward lattice of derivatives.

    In practice, this inefficiency doesn't seem to be an issue in
    terms of runtime.
    '''

    features = []
    for (i, datum) in enumerate(break_list_by_lengths(data, lengths)):

        preds = best_predictions[i]
        indexes = find_i_followed_by_j(preds, phase_1, phase_2)
        if indexes != []:
            derivatives = np.array(get_derivatives(datum, model))
            for idx in indexes:
                features.append(derivatives[idx, phase_2])
    return features


def get_difference_bt_means(model, phase_1, phase_2):
    """
    Returns the difference between mean values of two different phases

    Args:
        model : Trained HMM Model 
        phase_1 (int): Source Phase of a transition
        phase_2 (int): Destination phase of a transition

    """
    return model.means_[phase_2] - model.means_[phase_1]


def get_derivatives(X, model):
    '''
    Compute the derivative d/dz_t p(s_t = k | z_{1:t}) for the entire
    forward lattice.
    '''
    derivatives = []

    log_frameprob = model._compute_log_likelihood(X)
    log_probij, fwdlattice = _hmmc.forward_log(
                model.startprob_, model.transmat_, log_frameprob)
    # n_components = fwdlattice.shape[1]  # can be computed another way
    n_components = model.transmat_.shape[0]
    covars = [np.linalg.inv(model.covars_[i]) for i in range(n_components)]

    for i in range(len(X)):
        derivatives_i = []
        probs = softmax_with_overflow(fwdlattice[i]) 
        Z = np.sum([probs[j] * covars[j] @ (model.means_[j] - X[i]) for j in range(n_components)])

        for component in range(n_components):
            derivatives_i.append(
                covars[component] @ (model.means_[component] - X[i]) - Z 
            )

        derivatives.append(derivatives_i)

    return derivatives


def munge_data(hmm_model, model_pth, data_dir, cols, n_components,
               first_n=1000):
    dfs = hmm_model._make_hmm_data(data_dir, cols, sort=True, sort_col="epoch",
                                   first_n=first_n)
    data = np.vstack(
        [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in dfs]
    )
    lengths = [len(df) for df in dfs]

    with open(model_pth, 'rb') as f:
        models = pickle.load(f)

    model = models['best_models'][n_components-1]
    print(model.score(data, lengths=lengths))
    best_predictions = break_list_by_lengths(
        model.predict(data, lengths=lengths), lengths)

    return model, data, best_predictions, lengths


def characterize_transition_between_phases(model, data, best_predictions, cols,
                                           lengths, i, j):
    '''
    Compute the average derivative for each feature, sort features by
    highest absolute value
    '''
    features = get_features_for_transition(model, data, best_predictions,
                                           lengths, i, j)

    if len(features) != 0:
        features = np.mean(features, axis=0)

    order = np.argsort(np.abs(features))[::-1]

    feature_changes = np.array(get_difference_bt_means(model, i, j))
    if len(order) != 0:
        return np.array(cols)[order].tolist(), feature_changes[order].tolist()
    else:
        return [], []


def characterize_all_transitions(model, data, best_predictions, cols, lengths, phases):

    # phases = list(set(best_predictions))

    transitions = {}
    
    n_phases = model.transmat_.shape[0]

    for i in range(n_phases):
        for j in range(n_phases):
            # if i != j:
            sorted_cols, feature_changes = characterize_transition_between_phases(
                model, data, best_predictions, cols, lengths, i, j)

            transition_key = str(i) + '>>' + str(j)
            transitions[transition_key] = {}
            transitions[transition_key]['cols'] = sorted_cols
            transitions[transition_key]['feature_changes'] = feature_changes

    return transitions


def training_run_json_to_csv(save_dir, is_transformer, has_loss, lr, optimizer,
                             init_scaling, input_dir=None, n_seeds=40):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for seed in range(n_seeds):
        # optimizer = "adamw"
        # lr = 0.001
        # init_scaling = 1.0
        d = (
            input_dir + "/"
            + f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}/*.json"
        )
        pths = glob.glob(d)
        vals = get_stats_for_run(pths, is_transformer, has_loss)
        df = pd.DataFrame(vals)
        df = df.reset_index()
        df.rename(columns={"index": "epoch"}, inplace=True)

        file_name = f"lr{lr}_{optimizer}_seed{seed}_scaling{init_scaling}.csv"

        df.to_csv(f"{save_dir}/{file_name}")
