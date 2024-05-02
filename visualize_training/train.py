import json
import os

import evaluate
import torch
from accelerate import Accelerator
from visualize_training.metrics import get_distribution_stats, get_matrix_metrics, get_tensor_metrics


class ModelManager:
    def __init__(self, model, train_dataloader, test_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.accelerator = None

        if self.config.get("optimizer") == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.config.get("lr", 1e-3), weight_decay=self.config.get("weight_decay")
            )
        elif self.config.get("optimizer") == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config.get("lr", 1e-3), weight_decay=self.config.get("weight_decay")
            )
        else:
            raise ValueError(f"Optimizer {self.config.get('optimizer')} not supported")

        self.criterion = torch.nn.CrossEntropyLoss()

        if "init_scaling" in self.config:
            scaling_factor = self.config["init_scaling"]
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data *= scaling_factor

        if config.get("use_accelerator", True):
            self.accelerator = Accelerator(cpu=config.get("cpu", False))
            self.model, self.optimizer, self.train_dataloader, self.test_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.test_dataloader
            )

        self.save_path = os.path.join(
            self.config["run_output_dir"],
            f'lr{self.config["lr"]}_{self.config["optimizer"]}_seed{self.config["seed"]}_scaling{self.config["init_scaling"]}',
        )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self._load_eval()

        if not config.get("is_transformer"):
            self._set_miscs()

        self.hooks = []
        self.metrics_cache = []
        self.weights_biases_cache = {}
        self.training_metrics = {"loss": [], "accuracy": []}

    def _load_eval(self):
        self.train_accuracy = evaluate.load(
            "accuracy", experiment_id=f"{self.config['dataset_name']}{self.config['seed']}", keep_in_memory=True
        )
        self.test_accuracy = evaluate.load(
            "accuracy", experiment_id=f"{self.config['dataset_name']}{self.config['seed']}", keep_in_memory=True
        )

    def _set_miscs(self):
        if self.config["dataset_name"] == "mnist":
            self.num_classes = 10
        else:
            self.num_classes = 100

    def make_hook_function(self, name):
        def hook_fn(module, input, output):
            self._cache_weights_biases(name, module)

        return hook_fn

    def attach_hooks(self, layers):
        for layer_name in layers:
            for name, module in self.model.named_modules():
                if layer_name == name:
                    hook_fn = self.make_hook_function(name)
                    self.hooks.append(module.register_forward_hook(hook_fn))
                    self.weights_biases_cache[name] = []

    def _cache_weights_biases(self, name, module):
        if self.epoch % self.config["eval_every"] == 0:
            # Initialize cache entry with epoch, steps, and layer name
            cache_entry = {"epoch": self.epoch, "steps": self.steps, "layer_name": name}

            # Iterate through all named parameters of the module
            for param_name, param_value in module.named_parameters():
                cache_entry[param_name] = param_value.clone().detach().cpu().numpy()

            # Check if we already have an entry for this layer and step
            if not (
                self.weights_biases_cache.get(name) and self.weights_biases_cache[name][-1]["steps"] == self.steps
            ):
                self.weights_biases_cache.setdefault(name, []).append(cache_entry)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def clear_weights_biases_cache(self):
        self.weights_biases_cache = {}

    def clear_metrics_cache(self):
        self.metrics_cache = []

    def train(self):
        self.steps = 0
        torch.manual_seed(self.config["seed"])
        self.model.train()

        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch
            for batch in self.train_dataloader:
                x, y = batch

                output = self.model(x)[:, -1, :] if self.config.get("is_transformer") else self.model(x)
                loss = self.criterion(output, y)

                if self.config.get("clip_grad"):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["lr"] * 10)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss = loss.detach().cpu().item()

                predictions = torch.argmax(output, dim=1)
                self.train_accuracy.add_batch(predictions=predictions, references=y)

                self.steps += 1

                if self.epoch % self.config["eval_every"] == 0:
                    eval_loss, eval_accuracy = self._evaluate()
                    train_accuracy_metric = self.train_accuracy.compute()

                    self.training_metrics["loss"].append(
                        {"epoch": epoch, "steps": self.steps, "train_loss": train_loss, "eval_loss": eval_loss}
                    )
                    self.training_metrics["accuracy"].append(
                        {
                            "epoch": epoch,
                            "steps": self.steps,
                            "train_accuracy": train_accuracy_metric.get("accuracy"),
                            "eval_accuracy": eval_accuracy.get("accuracy"),
                        }
                    )

                    self.model.train()

            print(
                f"Epoch {epoch}, Train Loss: {train_loss}, Eval Loss: {eval_loss}, "
                f"Train Accuracy: {train_accuracy_metric.get('accuracy')}, Eval Accuracy: {eval_accuracy.get('accuracy')}"
            )

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            model_path = os.path.join(self.save_path, "model.pt")
            self.accelerator.save(self.model.state_dict(), model_path)
        else:
            model_path = os.path.join(self.save_path, "model.pt")
            torch.save(self.model.state_dict(), model_path)

    def _evaluate(self):
        self.model.eval()
        eval_loss = 0.0
        eval_steps = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                x, y = batch
                output = self.model(x)[:, -1, :] if self.config.get("is_transformer") else self.model(x)
                loss = self.criterion(output, y)  # Use class indices instead of one-hot
                eval_loss += loss.item()

                predictions = torch.argmax(output, dim=1)
                self.test_accuracy.add_batch(predictions=predictions, references=y)
                eval_steps += 1

        eval_loss /= eval_steps
        eval_accuracy = self.test_accuracy.compute()
        return eval_loss, eval_accuracy

    def compute_metrics(self):
        # Initialize a dictionary to hold the metrics per step and epoch
        metrics_per_step = {}

        # Temporary storage for global weight and bias statistics
        aggregated_weights = {}
        aggregated_biases = {}

        # Aggregate layer-specific metrics and prepare for global stats calculation
        for layer_name, data_list in self.weights_biases_cache.items():
            for data in data_list:
                step = data["steps"]
                epoch = data["epoch"]
                key = (step, epoch)

                # Ensure the structure is initialized for each unique step and epoch
                # Find corresponding training and evaluation data
                train_data = next(
                    (
                        item
                        for item in self.training_metrics["loss"]
                        if item["steps"] == step and item["epoch"] == epoch
                    ),
                    None,
                )
                accuracy_data = next(
                    (
                        item
                        for item in self.training_metrics["accuracy"]
                        if item["steps"] == step and item["epoch"] == epoch
                    ),
                    None,
                )

                if self.config.get("is_transformer"):
                    if key not in metrics_per_step:
                        metrics_per_step[key] = {
                            "step": step,
                            "epoch": epoch,
                            "k": [],
                            "q": [],
                            "v": [],
                            "in_proj": [],
                            "ffn_in": [],
                            "ffn_out": [],
                            "train_loss": train_data["train_loss"] if train_data else None,
                            "eval_loss": train_data["eval_loss"] if train_data else None,
                            "train_accuracy": accuracy_data["train_accuracy"] if accuracy_data else None,
                            "eval_accuracy": accuracy_data["eval_accuracy"] if accuracy_data else None,
                        }

                        aggregated_weights[key] = []
                        aggregated_biases[key] = []

                    if "W_K" in data:
                        k = torch.from_numpy(data["W_K"])
                        metrics_data = [get_matrix_metrics(k[i, :, :]) for i in range(self.config.get("n_heads"))]

                        metrics_per_step[key]["k"].append({"0": metrics_data})

                        aggregated_weights[key].append(k.view(-1))

                    if "W_Q" in data:
                        q = torch.from_numpy(data["W_Q"])
                        metrics_data = [get_matrix_metrics(q[i, :, :]) for i in range(self.config.get("n_heads"))]

                        metrics_per_step[key]["q"].append({"0": metrics_data})

                        aggregated_weights[key].append(q.view(-1))

                    if "W_V" in data:
                        v = torch.from_numpy(data["W_V"])
                        metrics_data = [get_matrix_metrics(v[i, :, :]) for i in range(self.config.get("n_heads"))]

                        metrics_per_step[key]["v"].append({"0": metrics_data})

                        aggregated_weights[key].append(v.view(-1))

                    if "W_O" in data:
                        in_proj = torch.from_numpy(data["W_O"])
                        metrics_data = get_matrix_metrics(in_proj)

                        metrics_per_step[key]["in_proj"].append({"0": metrics_data})

                        aggregated_weights[key].append(in_proj.view(-1))

                    if "W_in" in data:
                        ffn_in = torch.from_numpy(data["W_in"])
                        metrics_data = get_matrix_metrics(ffn_in)

                        metrics_per_step[key]["ffn_in"].append({"0": metrics_data})

                        aggregated_weights[key].append(ffn_in.view(-1))

                    if "W_out" in data:
                        ffn_out = torch.from_numpy(data["W_out"])
                        metrics_data = get_matrix_metrics(ffn_out)

                        metrics_per_step[key]["ffn_out"].append({"0": metrics_data})

                        aggregated_weights[key].append(ffn_out.view(-1))

                    if "b_in" in data:
                        b_in = torch.from_numpy(data["b_in"])

                        aggregated_biases[key].append(b_in.view(-1))

                    if "b_out" in data:
                        b_out = torch.from_numpy(data["b_out"])

                        aggregated_biases[key].append(b_out.view(-1))

                else:
                    if key not in metrics_per_step:
                        metrics_per_step[key] = {
                            "step": step,
                            "epoch": epoch,
                            "w": [],
                            "train_loss": train_data["train_loss"] if train_data else None,
                            "eval_loss": train_data["eval_loss"] if train_data else None,
                            "train_accuracy": accuracy_data["train_accuracy"] if accuracy_data else None,
                            "eval_accuracy": accuracy_data["eval_accuracy"] if accuracy_data else None,
                        }

                        aggregated_weights[key] = []
                        aggregated_biases[key] = []

                    # Layer-specific metrics calculation
                    if data["weight"] is not None:
                        weights = torch.from_numpy(data["weight"])
                        # Depending on tensor dimension, compute metrics
                        if weights.dim() == 4:  # Conv2D layers
                            metrics_data = get_tensor_metrics(weights)
                        elif weights.dim() == 2:  # Linear layers
                            metrics_data = get_matrix_metrics(weights)

                        metrics_per_step[key]["w"].append(metrics_data)

                        # Aggregate weights for global stats
                        aggregated_weights[key].append(weights.view(-1))

                    if data["bias"] is not None:
                        biases = torch.from_numpy(data["bias"])
                        # Aggregate biases for global stats
                        aggregated_biases[key].append(biases.view(-1))

        # Calculate and append global statistics for weights and biases
        for (step, epoch), _ in metrics_per_step.items():
            key = (step, epoch)

            # Global weights statistics
            if aggregated_weights[key]:
                all_weights_combined = torch.cat(aggregated_weights[key])
                w_all_stats = get_distribution_stats(all_weights_combined)
            else:
                w_all_stats = {}

            # Global biases statistics
            if aggregated_biases[key]:
                all_biases_combined = torch.cat(aggregated_biases[key])
                b_all_stats = get_distribution_stats(all_biases_combined)
            else:
                b_all_stats = {}

            # Update the metrics for the step and epoch with global stats
            metrics_per_step[key].update({"w_all": w_all_stats, "b_all": b_all_stats})

        # Update the metrics cache with the aggregated metrics
        self.metrics_cache = list(metrics_per_step.values())

    def save_metrics(self):
        metrics_by_epoch = {}
        for metric in self.metrics_cache:
            epoch = metric["epoch"]
            if epoch not in metrics_by_epoch:
                metrics_by_epoch[epoch] = []
            metrics_by_epoch[epoch].append(metric)

        for epoch, metrics in metrics_by_epoch.items():
            highest_step_metric = max(metrics, key=lambda x: x["step"])

            if self.config.get("is_transformer"):
                filtered_metrics = {
                    "k": highest_step_metric.get("k", {}),
                    "q": highest_step_metric.get("q", {}),
                    "v": highest_step_metric.get("v", {}),
                    "in_proj": highest_step_metric.get("in_proj", {}),
                    "ffn_in": highest_step_metric.get("ffn_in", {}),
                    "ffn_out": highest_step_metric.get("ffn_out", {}),
                    "w_all": highest_step_metric.get("w_all", {}),
                    "b_all": highest_step_metric.get("b_all", {}),
                    "train_loss": highest_step_metric.get("train_loss"),
                    "eval_loss": highest_step_metric.get("eval_loss"),
                    "train_accuracy": highest_step_metric.get("train_accuracy"),
                    "eval_accuracy": highest_step_metric.get("eval_accuracy"),
                }
            else:
                filtered_metrics = {
                    "w": highest_step_metric.get("w", {}),
                    "w_all": highest_step_metric.get("w_all", {}),
                    "b_all": highest_step_metric.get("b_all", {}),
                    "train_loss": highest_step_metric.get("train_loss"),
                    "eval_loss": highest_step_metric.get("eval_loss"),
                    "train_accuracy": highest_step_metric.get("train_accuracy"),
                    "eval_accuracy": highest_step_metric.get("eval_accuracy"),
                }

            metrics_filename = os.path.join(self.save_path, f"epoch{epoch}.json")
            with open(metrics_filename, "w") as f:
                json.dump(filtered_metrics, f, indent=4)

    def train_and_save_metrics(self):
        print(f"Training with seed {self.config['seed']}")
        self.train()
        print("Computing metrics")
        self.compute_metrics()
        print("Saving metrics")
        self.save_metrics()
        self.clear_weights_biases_cache()
        self.clear_metrics_cache()
        self.remove_hooks()
        print(f"Finished training and saving metrics with seed {self.config['seed']}")
