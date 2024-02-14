import glob
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import numpy as np
from hmmlearn import hmm, _hmmc
from tqdm import trange
from utils import break_list_by_lengths, find_i_followed_by_j, softmax_with_overflow


class HMM():

    def __init__(self, max_components, cov_type, n_seeds, n_iter):
        self.max_components = max_components
        self.cov_type = cov_type
        self.n_seeds = n_seeds
        self.n_iter = n_iter

    def _make_hmm_data(self, data_dir, cols, sort, sort_col, first_n):

        print(data_dir)
        print(glob.glob(data_dir + "*"))
        if sort:
            # restrict to cols of interest
            dfs = [
                pd.read_csv(file)
                .sort_values(sort_col)
                .reset_index(drop=True)[cols]
                .head(first_n)
                for file in glob.glob(data_dir + "*")
            ]
        else:
            # restrict to cols of interest
            dfs = [
                pd.read_csv(file)[cols].head(first_n)
                for file in glob.glob(data_dir + "*")
            ]
        # remove invalid dfs
        dfs = [df for df in dfs if not df.isnull().values.any()]

        return dfs

    def _prep_train_data(self, dfs, test_size, seed):

        train_dfs, test_dfs = train_test_split(dfs, test_size=test_size,
                                               random_state=seed)

        train_data = np.vstack(
            [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in train_dfs]
        )
        test_data = np.vstack(
            [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in test_dfs]
        )

        return train_dfs, test_dfs, train_data, test_data

    def _train(self, n_components, train_data, test_data, train_dfs, test_dfs):

        scores_buf = []
        aics_buf = []
        bics_buf = []
        best_score = -np.inf
        best_model = None

        train_lengths = [len(df) for df in train_dfs]
        test_lengths = [len(df) for df in test_dfs]

        for seed in range(self.n_seeds):
            model = hmm.GaussianHMM(
                n_components=n_components, covariance_type=self.cov_type,
                n_iter=self.n_iter
            )
            model.fit(train_data, lengths=train_lengths)
            score = model.score(test_data, lengths=test_lengths)
            aics_buf.append(model.aic(test_data, lengths=test_lengths))
            bics_buf.append(model.bic(test_data, lengths=test_lengths))
            scores_buf.append(score)
            if score > best_score:
                best_score = score
                best_model = model

        return best_score, best_model, aics_buf, bics_buf, scores_buf

    def save_model(self, model_path, model):

        with open(model_path + ".pkl", "wb") as f:
            pickle.dump(model, f)

    def get_avg_log_likelihood(self, data_dir, cols, sort=True,
                               sort_col="epoch", first_n=None, test_size=0.2,
                               seed=0):

        best_scores = []
        mean_scores = []
        scores_stdev = []
        best_models = []
        aics = []
        bics = []

        dfs = self._make_hmm_data(data_dir, cols, sort, sort_col, first_n)

        train_dfs, test_dfs, train_data, \
            test_data = self._prep_train_data(dfs, test_size, seed)

        for i in trange(1, self.max_components+1):

            best_score, best_model, aics_buf, bics_buf, \
                scores_buf = self._train(i, train_data, test_data, train_dfs,
                                         test_dfs)

            best_scores.append(best_score)
            mean_scores.append(np.mean(scores_buf))
            scores_stdev.append(np.std(scores_buf))
            aics.append(np.mean(aics_buf))
            bics.append(np.mean(bics_buf))
            best_models.append(best_model)

        return {
            "best_scores": best_scores,
            "mean_scores": mean_scores,
            "scores_stdev": scores_stdev,
            "aics": aics,
            "bics": bics,
            "best_models": best_models,
        }

    def get_features_for_transition(self, model, data, best_predictions,
                                    lengths, phase_1, phase_2):
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
                derivatives = np.array(self.get_derivatives(datum, model))
                for idx in indexes:
                    features.append(derivatives[idx, phase_2])
        return features

    def get_difference_bt_means(self, model, phase_1, phase_2):
        """
        Returns the difference between mean values of two different phases

        Args:
            model : Trained HMM Model 
            phase_1 (int): Source Phase of a transition
            phase_2 (int): Destination phase of a transition

        """
        return model.means_[phase_2] - model.means_[phase_1]

    def get_derivatives(self, X, model):
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

    def munge_data(self, model_pth, data_dir, cols, n_components,
                   first_n=1000):

        dfs = self._make_hmm_data(data_dir, cols, sort=True, sort_col="epoch",
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

    def characterize_transition_between_phases(self, model, data,
                                               best_predictions, cols,
                                               lengths, i, j):
        '''
        Compute the average derivative for each feature, sort features by
        highest absolute value
        '''
        features = self.get_features_for_transition(model,
                                                    data, best_predictions,
                                                    lengths, i, j)

        print(f"Number of times transition happened: {len(features)}")
        features = np.mean(features, axis=0)
        order = np.argsort(np.abs(features))[::-1]

        feature_changes = np.array(self.get_difference_bt_means(model, i, j))

        return cols[order], feature_changes[order]

    def characterize_all_transitions(self, model, data, best_predictions, cols,
                                     lengths):

        phases = list(set(best_predictions))

        transitions = {}

        for i in phases:
            for j in phases:
                if i != j:
                    cols, feature_changes = self.characterize_transition_between_phases(model, data, best_predictions, cols, lengths, i, j)
                    transition_key = str(i) + '>>' + str(j)
                    transitions[transition_key]['cols'] = cols
                    transitions[transition_key]['feature_changes'] = feature_changes

        return transitions
