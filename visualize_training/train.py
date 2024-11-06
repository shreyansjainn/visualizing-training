import torch
import json
import os
import csv
import numpy as np

from visualize_training.metrics import calculate_metrics

class ModelManager:
    def __init__(self, model, output_dir, layer_names= None, interval=100, standard_metrics=None, custom_metrics=None):
        """
        Initialize the ModelManager.

        Args:
            model (torch.nn.Module): The model to analyze.
            layer_names (list): List of layer names to log metrics from.
            output_dir (str): Directory to save logged data.
            interval (int): Interval for logging.
            standard_metrics (list): List of standard metric names to calculate.
            custom_metrics (dict): Dictionary of custom metric functions.
        """
        self.model = model
        self.layer_names = layer_names
        self.output_dir = output_dir
        self.interval = interval
        self.standard_metrics = standard_metrics or []
        self.custom_metrics = custom_metrics or {}
        self.current_data = {}
        self.epoch = 0

        # Prepare output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate specified layers
        self.validated_layers = self._validate_layers()
        
        # Register hooks on validated layers
        self.hooks = []
        if self.validated_layers:
            for name, layer in self.validated_layers.items():
                hook = layer.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)

    def _validate_layers(self):
        """
        Validate that user-specified layers (or all layers if none specified) exist in the model, 
        and extract all trainable parameters.

        Returns:
            dict: Dictionary with layer names as keys and their layer objects as values.

        Raises:
            ValueError: If any specified layer does not exist in the model.
        """
        # If no layer names are provided, list all available layers with trainable parameters
        if self.layer_names is None:
            available_layers = {name: module for name, module in self.model.named_modules()
                                if any(p.requires_grad for p in module.parameters(recurse=False))}

            print("No layer names were specified.")
            print("Available layers with trainable parameters in the model:")
            for layer, layer_type in available_layers.items():
                print(f"  - {layer} ({layer_type.__class__.__name__})")
            
            print("\nPlease specify the layers you want by referring to the model architecture.")
            return None  # Exit without proceeding, prompting the user to specify layers

        validated_layers = {}
        invalid_layers = []

        for layer_name in self.layer_names:
            # Split layer name by dots to navigate nested modules
            submodules = layer_name.split('.')
            current_module = self.model

            try:
                # Traverse through submodules to get to the target layer
                for submodule_name in submodules:
                    current_module = getattr(current_module, submodule_name)

                validated_layers[layer_name] = current_module  # Store the actual layer object
            except AttributeError:
                invalid_layers.append(layer_name)

        # Raise an error if any layers were not found, with detailed messages
        if invalid_layers:
            available_layers = ", ".join(
                f"{name} ({type(module).__name__})" for name, module in self.model.named_modules()
                if any(p.requires_grad for p in module.parameters(recurse=False))
            )
            raise ValueError(
                f"The following layers were not found in the model: {', '.join(invalid_layers)}. "
                f"Available layers with trainable parameters: {available_layers}."
            )

        return validated_layers

    def _hook_fn(self, layer_name):
        def fn(module, input, output):
            """Cache weights and biases of the specified module."""
            cache_entry = {"epoch": self.epoch, "layer_name": layer_name}

            # Cache all named parameters generically
            for param_name, param_value in module.named_parameters(recurse=False):
                cache_entry[param_name] = param_value.clone().detach().cpu().numpy().tolist()
                
            # Store cache entry in current_data
            self.current_data[layer_name] = cache_entry
        return fn

    def start_epoch(self, epoch):
        """Set the current epoch and initialize logging data."""
        self.epoch = epoch
        self.current_data.clear()  # Clear previous epoch data to prevent conflicts

    def save_epoch_data(self, seed, train_loss=None, train_accuracy=None, eval_loss=None, eval_accuracy=None):
        """Save logged data at the end of each epoch."""
        if self.epoch % self.interval == 0:
            seed_dir = os.path.join(self.output_dir, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            data_output = {
                'epoch': self.epoch,
                'seed': seed,
                'layers': self.current_data,
            }
            if train_loss is not None:
                data_output['train_loss'] = train_loss
            if train_accuracy is not None:
                data_output['train_accuracy'] = train_accuracy
            if eval_loss is not None:
                data_output['eval_loss'] = eval_loss
            if eval_accuracy is not None:
                data_output['eval_accuracy'] = eval_accuracy

            epoch_file = os.path.join(seed_dir, f"epoch_{self.epoch}.json")
            with open(epoch_file, 'w') as f:
                json.dump(data_output, f, indent=2)

    def calculate_metrics_post_training(self):
        for seed_dir in os.listdir(self.output_dir):
            seed_path = os.path.join(self.output_dir, seed_dir)
            if not os.path.isdir(seed_path):
                continue

            csv_file_path = os.path.join(self.output_dir, f"metrics_summary_{seed_dir}.csv")

            # Initialize buffers for metrics
            buf = {
                "epoch": [], "l1": [], "l2": [], "code_sparsity": [], "trace": [], "spectral": [],
                "computational_sparsity": [], "mean_singular_value": [], "var_singular_value": [],
                "mean_w": [], "median_w": [], "var_w": [], "mean_b": [], "median_b": [], "var_b": [],
                "train_loss": [], "eval_loss": [], "train_accuracy": [], "eval_accuracy": []
            }

            for epoch_file in sorted(
                    [f for f in os.listdir(seed_path) if f.startswith("epoch") and f.endswith(".json")],
                    key=lambda x: int(x.split('_')[1].split('.')[0])
                ):
                epoch_data_path = os.path.join(seed_path, epoch_file)
                if not epoch_file.endswith(".json"):
                    continue

                with open(epoch_data_path, 'r') as f:
                    epoch_data = json.load(f)

                # Aggregation buffers
                l1_buf, l2_buf, trace_buf, spectral_buf, code_sparsity_buf = [], [], [], [], []
                computational_sparsity_buf, mean_lambda_buf, variance_lambda_buf = [], [], []
                aggregated_weights, aggregated_biases = [], []

                # Gather metrics for each layer
                for layer_name, params in epoch_data['layers'].items():
                    for param_name, param_data in params.items():
                        if param_name in ["epoch", "layer_name"]:
                            continue

                        # Convert param_data to tensor
                        param_tensor = torch.tensor(param_data, dtype=torch.float32)

                        if "weight" in param_name or param_name.startswith("W_") or param_name.startswith("w_"):
                            metrics = calculate_metrics(param_tensor)
                            
                            # Append layer metrics to corresponding buffers
                            l1_buf.append(metrics["l1"])
                            l2_buf.append(metrics["l2"])
                            trace_buf.append(metrics["trace"])
                            spectral_buf.append(metrics["spectral"])
                            code_sparsity_buf.append(metrics["code_sparsity"])
                            computational_sparsity_buf.append(metrics["computational_sparsity"])
                            mean_lambda_buf.append(metrics["mean_singular_value"])
                            variance_lambda_buf.append(metrics["var_singular_value"])

                            aggregated_weights.append(param_tensor.flatten())
                        elif "bias" in param_name or param_name.startswith("B_") or param_name.startswith("b_"):
                            aggregated_biases.append(param_tensor.flatten())

                # Aggregate layer metrics and add to buffer
                buf["epoch"].append(epoch_data['epoch'])
                buf["l1"].append(np.mean(l1_buf))
                buf["l2"].append(np.mean(l2_buf))
                buf["trace"].append(np.mean(trace_buf))
                buf["spectral"].append(np.mean(spectral_buf))
                buf["code_sparsity"].append(np.mean(code_sparsity_buf))
                buf["computational_sparsity"].append(np.mean(computational_sparsity_buf))
                buf["mean_singular_value"].append(np.mean(mean_lambda_buf))
                buf["var_singular_value"].append(np.nanmean(variance_lambda_buf))  

                # Global weight and bias stats
                if aggregated_weights:
                    aggregated_weights = torch.cat(aggregated_weights)
                    buf["mean_w"].append(aggregated_weights.mean().item())
                    buf["median_w"].append(aggregated_weights.median().item())
                    buf["var_w"].append(aggregated_weights.var().item())
                if aggregated_biases:
                    aggregated_biases = torch.cat(aggregated_biases)
                    buf["mean_b"].append(aggregated_biases.mean().item())
                    buf["median_b"].append(aggregated_biases.median().item())
                    buf["var_b"].append(aggregated_biases.var().item())

                # Training/evaluation metrics
                buf["train_loss"].append(epoch_data.get("train_loss", None))
                buf["eval_loss"].append(epoch_data.get("eval_loss", None))
                buf["train_accuracy"].append(epoch_data.get("train_accuracy", None))
                buf["eval_accuracy"].append(epoch_data.get("eval_accuracy", None))

            # Write final buffer to CSV
            with open(csv_file_path, mode="w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(buf.keys())
                writer.writerows(zip(*buf.values()))

            print(f"Metrics summary saved to {csv_file_path}")
        print(f"Calculated and saved metrics for all training data")
        
    def close_hooks(self):
        """Remove hooks from the model after training."""
        for hook in self.hooks:
            hook.remove()
