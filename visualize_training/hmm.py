import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import numpy as np
from hmmlearn import hmm
from tqdm import trange
from visualize_training.utils import characterize_all_transitions

class HMM():

    def __init__(self, max_components, cov_type, n_seeds, n_iter, seeds=None):
        self.max_components = max_components
        self.cov_type = cov_type
        self.n_seeds = n_seeds
        self.n_iter = n_iter
        self.seeds = seeds

    def _make_hmm_data(self, data_dir, cols, sort, sort_col, first_n):

        print(data_dir)
        print(glob.glob(data_dir + "*.csv"))
        if sort:
            # restrict to cols of interest
            dfs = [
                pd.read_csv(file)
                .sort_values(sort_col)
                # .sort_index()
                .reset_index(drop=True)[cols]
                .head(first_n)
                for file in glob.glob(data_dir + "*.csv")
            ]
        else:
            # restrict to cols of interest
            dfs = [
                pd.read_csv(file)[cols].head(first_n)
                for file in glob.glob(data_dir + "*.csv")
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

        seeds = self.seeds if self.seeds else list(range(self.n_seeds))

        for seed in seeds:
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

                self.best_model = model
                self.best_score = best_score

        return best_score, best_model, aics_buf, bics_buf, scores_buf

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

    def feature_importance(self, cols, data, best_predictions, phases,lengths):
        """
        Return Feature Importance of all transitions for the best model

        Args:
            cols (list): list of columns of interest
            data (list): list of dataframes
            best_predictions (list): list of best predictions over different
            dataframes

        Returns:
            transitions: feature importance of all transitions and avg mean
            difference due to them
        """

        transitions = characterize_all_transitions(self.best_model, data,
                                                   best_predictions, cols,
                                                   lengths, phases)

        return transitions
