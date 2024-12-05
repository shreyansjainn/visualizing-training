# HMM

`class src.hmm.HMM(max_components: int, cov_type: str, n_seeds: int, n_iter: int)`

     PARAMETERS
     * `max_components`: The maximum number of components to consider for the HMM.
     * `cov_type`: The type of covariance matrix to use for the HMM. eg: 'diag'.
     * `n_seeds`: The number of random seeds to use for the HMM.
     * `n_iter`: The number of iterations to train the HMM for.

- `get_avg_log_likelihood(data_dir: str, cols: List[str], sort: bool=True, sort_col: str='epoch', first_n: int=None, test_size: float=0.2, seed: int=0) -> dict`

  Computes the different values like best scores, mean_scores, std_scores, aics, bics, and best models for different number of components for the HMM.

  PARAMETERS

  - `data_dir`: The directory containing the consolidated data from the training runs.
  - `cols`: The columns to consider for the HMM.
  - `sort`: Whether to sort the columns.
  - `sort_col`: The column to sort the data by.
  - `first_n`: The number of data points to consider.
  - `test_size`: The size of the test data for the HMM training.
  - `seed`: The random seed to use for the HMM training.

Example:

```python
from src.hmm import HMM

max_components = 8
cov_type = "diag"
n_seeds = 4
n_iter = 10
cols = ['col1', 'col2', 'col3']
first_n = 100
hmm_model = HMM(max_components, cov_type, n_seeds, n_iter)
data_dir = 'data_dir/'

hmm_output = hmm_model.get_avg_log_likelihood(data_dir, cols)
```

- `feature_importance(cols: List[str], data: List[pd.DataFrame], best_predictions: List[np.ndarray], phases: List[str], lengths: List[int])`

  Computes the feature importance for the HMM model.

  PARAMETERS

  - `cols`: The columns to consider for the HMM.
  - `data`: The data to consider for the HMM.
  - `best_predictions`: The best predictions from the HMM model.
  - `phases`: The phases of the HMM model.
  - `lengths`: The lengths of the data.

Example:

```python
from src.utils import munge_data

n_components = 8
model_path = 'model_path.pkl'

model, data, best_predictions, lengths = munge_data(hmm_model, model_path, data_dir, cols, n_components)

phases = list(set(hmm_model.best_model.predict(data, lengths=lengths)))

state_transitions = hmm_model.feature_importance(cols, data, best_predictions,phases,lengths)
```
