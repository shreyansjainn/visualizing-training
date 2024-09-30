import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import numpy as np
from hmmlearn import hmm
from tqdm import trange
from visualize_training.utils import characterize_all_transitions


class HMM():
    """
    The :class:`HMM` is one of the core modules of Visualization Training.
    It is a wrapper that stores all the functionalities required to train,
    process and infer from the HMM models. It contains methods for data preparation,
    calculating average log likelood and feature importances to name a few.
    """

    def __init__(self, max_components, cov_type, n_seeds, n_iter, seeds=None):
        self.max_components = max_components
        self.cov_type = cov_type
        self.n_seeds = n_seeds
        self.n_iter = n_iter
        self.seeds = seeds

    def _make_hmm_data(self, data_dir, cols, sort, sort_col, first_n):
        """Reads data files from `data_dir` containing specified `cols`
        and `first_n` rows.

        Args:
            data_dir (str): Path to data files.
            cols (list): List of columns to be returned.
            sort (bool): Whether to sort the rows based on `sort_col` or not. 
            sort_col (str): Column name based on which sorting needs to be done.
            first_n (int): No of rows to be returned for each data file.

        Returns:
            dfs (list): List of dataframes
        """
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
        """_summary_       
        Prepares the training data into training and test set
        from the given list of dataframes

        Args:
            dfs (list): List of dataframes for HMM model training
            test_size (float): Size of test set as fraction of the total dataset
            seed (int): Random Seed

       Returns:
            train_dfs (list): List of dataframes for training the HMM
            test_dfs (list): List of dataframes for testing the HMM
            train_data (array): All the dataframes in `train_dfs` stacked vertically
            test_data (array): All the dataframes in `test_dfs` stacked vertically
        """

        train_dfs, test_dfs = train_test_split(dfs, test_size=test_size,
                                               random_state=seed)

        train_data = np.vstack(
            [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in train_dfs]
        )
        test_data = np.vstack(
            [np.apply_along_axis(zscore, 0, df.to_numpy()) for df in test_dfs]
        )

        return train_dfs, test_dfs, train_data, test_data

    def _train(self, n_components, train_data, test_data, train_dfs, test_dfs, best_score, best_model):
        """
        Method for training the HMM model for a given no of `n_components`.

        Args:
            n_components (float): No of components for the HMM model.
            train_data (array): Training data for training HMM model.
            test_data (array): Test data for testing HMM model.
            train_dfs (list): List of dataframes containing training data.
            test_dfs (list): List of dataframes containing testing data.
            best_score (float): Best score out of all the models trained till this point.
            best_model: Model corresponding to `best_score` till this point. 

        Returns:
            best_score (float): Best score out of all the models trained including the latest model.
            best_model (array): Model corresponding to `best_score`.
            aics_buf (list): List of AIC values for all the models trained in this run.
            bics_buf (list): List of BIC values for all the models trained in this run.
            scores_buf (list): List of Score values for all the models trained in this run.
        """

        scores_buf = []
        aics_buf = []
        bics_buf = []
        # best_score = -np.inf
        # best_model = None

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
                # print("score: ",score)
                # print("best_score: ",best_score)
                # print("best_model: ",model)
                best_score = score
                best_model = model

                self.best_model = model
                self.best_score = best_score

        return best_score, best_model, aics_buf, bics_buf, scores_buf

    def get_avg_log_likelihood(self, data_dir, cols, sort=True,
                               sort_col="epoch", first_n=None, test_size=0.2,
                               seed=0):
        """
        Wrapper function which reads, prepares data for model training, trains all the models
        for all the possible `n_components` values.

        Args:
            data_dir (str): Path to data files.
            cols (list): List of columns to be returned.
            sort (bool): Whether to sort the rows based on `sort_col` or not. Defaults to True.
            sort_col (str): Column name based on sorting needs to be done. Defaults to "epoch".
            first_n (int): No of rows to be returned for each data file. Defaults to None.
            test_size (float, optional): Size of test set as fraction of the total dataset. Defaults to 0.2.
            seed (int, optional): Random Seed. Defaults to 0.

        Returns:
            Dictionary containing 
            - best_scores: List of best scores for all the components
            - mean_scores: List of mean scores (average across all seeds) for all the components
            - scores_stdev: List of std dev (calculated across all seeds) for all the components
            - aics: List of mean AIC values (calculated across all seeds) for all the components
            - bics: List of mean BIC values (calculated across all seeds) for all the components
            - best_models: List of best models for all the components
            - best_model: Best model out of all the models
        """
        best_score = -np.inf
        best_model = None
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
                                         test_dfs, best_score, best_model)

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
            "best_model": self.best_model
        }

    def feature_importance(self, cols, data, best_predictions, phases,lengths,top_n=3):
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
                                                   lengths, phases,top_n)

        return transitions
