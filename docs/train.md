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
