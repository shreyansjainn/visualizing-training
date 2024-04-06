# Submodules

## train
`class ModelManager(model, train_dataloader, test_dataloader, config: Dict)`

    PARAMETERS
    * `model`: The model to be trained.
    * `train_dataloader`: The dataloader for the training data.
    * `test_dataloader`: The dataloader for the test data.
    * `config`: A dictionary for the training configuration
      
* `attach_hooks(layernames: List[str])`
  
    Attaches hooks to the specified layers. The hooks are used to store the weights and biases of the layers during training in the `weights_biases_cache` attribute of the model manager.
    
    PARAMETERS
    * `layernames`: List of layer names to attach hooks to.
  
* `remove_hooks`
  
    Removes all hooks attached to the model.

* `clear_weights_biases_cache`

    Clears the `weights_biases_cache` of the model manager.

* `clear_metrics_cache`
  
    Clears the `metrics_cache` of the model manager.

* `train`
  
    Trains the model using the training data and evaluates it using the test data. The training metrics like loss, accuracy, etc. are stored in the `training_metrics` attribute of the model manager and used in the `compute_metrics` method.

* `compute_metrics`
  
    Computes the metrics required for HMM training using the data from the `weights_biases_cache`. The metrics are stored in the `metrics_cache` attribute of the model manager.

* `save_metrics`
  
    Saves the metrics in the `metrics_cache` to a files inside the output directory specified in the configuration. The files are saved under the folder `lr{learning_rate}_{optimizer}_seed{seed}_scaling{scaling}`  inside the output directory with a separate metrics file for each epoch (for epochs as specifed in the configuration `eval_every`).

* `train_and_save_metrics`
   
    Trains the model, computes the metrics and saves them.

## hmm
`class HMM(max_components: int, cov_type: str, n_seeds: int, n_iter: int)`
      
     PARAMETERS
     * `max_components`: The maximum number of components to consider for the HMM.
     * `cov_type`: The type of covariance matrix to use for the HMM. eg: 'diag'.
     * `n_seeds`: The number of random seeds to use for the HMM.
     * `n_iter`: The number of iterations to train the HMM for.
  
* `get_avg_log_likelihood(data_dir: str, cols: List[str], sort: bool=True, sort_col: str='epoch', first_n: int=None, test_size: float=0.2, seed: int=0) -> dict`

    Computes the different values like best scores, mean_scores, std_scores, aics, bics, and best models for different number of components for the HMM.
    
  PARAMETERS
    * `data_dir`: The directory containing the consolidated data from the training runs.
    * `cols`: The columns to consider for the HMM.
    * `sort`: Whether to sort the columns.
    * `sort_col`: The column to sort the data by.
    * `first_n`: The number of data points to consider.
    * `test_size`: The size of the test data for the HMM training.
    * `seed`: The random seed to use for the HMM training.
  
* `feature_importance(cols: List[str], data: List[pd.DataFrame], best_predictions: List[np.ndarray], phases: List[str], lengths: List[int])`
  
    Computes the feature importance for the HMM model.
    
    PARAMETERS
    * `cols`: The columns to consider for the HMM.
    * `data`: The data to consider for the HMM.
    * `best_predictions`: The best predictions from the HMM model.
    * `phases`: The phases of the HMM model.
    * `lengths`: The lengths of the data.

## visualize
* `visualize_hmm_loss(data: pd.DataFrame, phase_col: str, epoch_col: str, loss_col: str, hover_data: dict=None)`
   
   Visualizes the loss of the HMM model over epochs

   PARAMETERS
    * `data`: Input data containing the training runs
    * `phase_col`: Column name containing the HMM phases info
    * `epoch_col`: Column name containing the epoch info
    * `loss_col`: Column name for loss metric of the model
    * `hover_data`: dictionary with column names as fields and bool value against them for visibility in tooltip during hover
  
* `visualize_dag(transmat: np.array, hover_dict: dict=None, edge_hover_dict: dict=None, hex: List[str]=None)`
    
    Visualizes HMM State Transition Diagram with a toggle menu on the right to control graph features

    PARAMETERS
    * `transmat`: State Transition matrix of the HMM
    * `hover_dict`: phase wise list of important features to be shown in tooltip on hover for nodes
    * `edge_hover_dict`: phase wise list of important features to be shown in tooltip on hover for edges
    * `hex`: List of hex values of the color pallette 

* `visualize_avg_log_likelihood(data: dict, dataset_name: str, max_components: int=8)`
  
    Visualizes the average log likelihood of the HMM model for different number of components

    PARAMETERS
    * `data`: Dictionary containing the average log likelihood values for different number of components
    * `dataset_name`: Name of the dataset
    * `max_components`: Maximum number of components to consider for the HMM
