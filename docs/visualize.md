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
