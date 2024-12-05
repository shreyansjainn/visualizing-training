# [Latent State Models of Training Dynamics](https://arxiv.org/abs/2308.09543)

[![Read the Docs
Here](https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://shreyansjainn.github.io/visualizing-training/)](https://shreyansjainn.github.io/visualizing-training/)

Directly model training dynamics, then interpret the dynamics model

# Setup

### Installation

Package can be installed using

```bash
pip install visualizing-training
```

For installing the package in editable mode, directly clone the repo and install the package.

```bash
git clone https://github.com/michahu/visualizing-training.git

pip install -e .
```

# Usage

Below set of commands will walk you through the usage of the package, [demo_run](https://github.com/michahu/visualizing-training/blob/main/demo_run.ipynb) notebook can be referred for a notebook version of the same.

## Step 0: Config Setup.

Setting up config required to collect training data and evaluation of the model

```python
num_epochs = 100
lr = 1e-3
train_bsz = 256
cpu = True
weight_decay = 1.0
eval_every = 10
n_heads = 4
init_scaling = 1.0
optimizer = "adamw" # type of optimizer
output_dir = 'modular_demo_run' # output directory for writing collected metrics
seed = 0
use_ln = True
test_bsz = 2048
clip_grad = True
dataset_name = "modular" # specifying the type of dataset to be trained on
n_seeds = 4 # no of seeds for which the data needs to be collected

config = {
    "lr": lr,
    "cpu": cpu,
    "num_epochs": num_epochs,
    "eval_every": eval_every,
    "run_output_dir": output_dir,
    "use_accelerator": False,
    "init_scaling": init_scaling,
    "optimizer": optimizer,
    "weight_decay": weight_decay,
    "clip_grad": clip_grad,
    "is_transformer": True,
    "train_bsz": train_bsz,
    "test_bsz": test_bsz,
    "n_heads": n_heads,
    "dataset_name": dataset_name,
    "clock_pizza_metrics": False # Set to True if you want to collect clock_pizza metrics - gradient symmetricity and distance irrelevance
}
```

## Step 1: Training a model and collecting metrics

```python
from visualizing_training.training.modular_addition import get_dataloaders
from visualizing_training.model import Transformer
from visualizing_training.train import ModelManager

for seed in range(n_seeds): # Saving the metrics for all the seeds
    config["seed"] = seed

    model = Transformer(
        d_model=128,
        d_head=32,
        d_vocab=114,
        num_heads=n_heads,
        num_layers=1,
        n_ctx=3,
        use_ln=use_ln,
    )

    train_loader, test_loader = get_dataloaders(train_bsz=train_bsz, test_bsz=test_bsz)

    #Initialize the ModelManager class which will take care of training and collecting the metrics
    model_manager = ModelManager(model,train_loader,test_loader, config)

    # Specify the layers where hooks needs to be attached
    model_manager.attach_hooks(['blocks.0.mlp','blocks.0.attn'])

    # method for training and saving the metrics in output_dir
    model_manager.train_and_save_metrics()
print("Finished training and saving metrics for all seeds")
```

## Step 2: Collate statistics into 1 file.

Take the stats computed in step 1 and organize them into CSVs suitable for training the HMM.

```python
from visualizing_training.utils import training_run_json_to_csv

training_run_json_to_csv(config['run_output_dir'], is_transformer=True, has_loss=False, lr=lr, optimizer=config['optimizer'], init_scaling=config['init_scaling'], input_dir=config['run_output_dir'], n_seeds=n_seeds, clock_pizza_metrics=config['clock_pizza_metrics'])
```

## Step 4: Train HMM.

Model selection computes the AIC-BIC-log-likelihood curves for varying number of hidden states in the HMM and saves out the best model for each number of hidden states.

```python
from visualizing_training.hmm import HMM

max_components = 8 # max no of components for which HMM will be trained
cov_type = "diag" # type of covariance for HMM model
n_seeds = 4 # no of seeds HMM needs to be trained for
n_iter = 10
cols = ['var_w', 'l1', 'l2'] # columnns of interest
first_n = 100 # no of rows to consider in the dataset
hmm_model = HMM(max_components, cov_type, n_seeds, n_iter)
data_dir = 'modular_demo_run/'

hmm_output = hmm_model.get_avg_log_likelihood(data_dir, cols)
```

## Step 5: HMM Model Selection

Visualizing average log-likelihood, along with AIC and BIC helps with the model selection for the different HMM models we have trained. Currently we are selecting the HMM model with the lowest BIC.

```python
from visualizing_training.visualize import visualize_avg_log_likelihood,

visualize_avg_log_likelihood(hmm_output,'modular_demo_run')
```

## Step 6: Saving the model

```python
model_path = 'model_path'
save_model(model_path,hmm_output)
```

## Step 7: Calculating Feature Importance

Calculating feature importance to shortlist top n most important features contributing to a state transition

```python
from visualizing_training.utils import munge_data

n_components = 8 # best model chosen from the visualization
model_path = 'model_path.pkl'

model, data, best_predictions, lengths = munge_data(hmm_model, model_path, data_dir, cols, n_components)

phases = list(set(hmm_model.best_model.predict(data, lengths=lengths)))

state_transitions = hmm_model.feature_importance(cols, data, best_predictions,phases,lengths) # dictionary storing state transitions
```

## Step 8: Visualizing State Transitions

Visualizing state transitions using a DAG visualization which allows the user to interact with it for deeper insight into the training dynamics.

```python
from visualizing_training.visualize import visualize_states

best_model_transmat = model.transmat_

visualize_states(best_model_transmat, edge_hover_dict = state_transitions)
```

# Citation

Thank you for your interest in our work! If you use this repo, please cite:

```
@article{
hu2023latent,
title={Latent State Models of Training Dynamics},
author={Michael Y. Hu and Angelica Chen and Naomi Saphra and Kyunghyun Cho},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=NE2xXWo0LF},
note={}
}
```
