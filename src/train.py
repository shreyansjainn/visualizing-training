import json
import os

import evaluate
import torch
from accelerate import Accelerator
from src.metrics import get_distribution_stats, get_matrix_metrics, get_tensor_metrics


class ModelManager:
    def __init__(self, model, train_dataloader, test_dataloader, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.accelerator = None

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.get("lr", 1e-3))
        self.criterion = torch.nn.CrossEntropyLoss()

        if config.get("use_accelerator", True):
            self.accelerator = Accelerator(cpu=config.get("cpu", False))
            self.model, self.optimizer, self.train_dataloader, self.test_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.test_dataloader
            )

        self._load_eval()
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
        for layer_info in layers:
            layer_name, layer_type = layer_info

            for name, module in self.model.named_modules():
                if layer_name == name and isinstance(module, layer_type):
                    hook_fn = self.make_hook_function(name)
                    self.hooks.append(module.register_forward_hook(hook_fn))
                    self.weights_biases_cache[name] = []

    def _cache_weights_biases(self, name, module):
        if self.steps % self.config["eval_every"] == 0:
            cache_entry = {
                "epoch": self.epoch,
                "steps": self.steps,
                "layer_name": name,
                "weights": module.weight.clone().detach().cpu().numpy() if module.weight is not None else None,
                "biases": module.bias.clone().detach().cpu().numpy() if module.bias is not None else None,
            }

            # Check if we already have an entry for this step
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
        self.model.train()

        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch
            for batch in self.train_dataloader:
                x, y = batch

                output = self.model(x)
                loss = self.criterion(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss = loss.detach().cpu().item()

                predictions = torch.argmax(output, dim=1)
                self.train_accuracy.add_batch(predictions=predictions, references=y)

                self.steps += 1

                if self.steps % self.config["eval_every"] == 0:
                    eval_loss, eval_accuracy = self._evaluate()
                    train_accuracy_metric = self.train_accuracy.compute()

                    print(
                        f"Epoch {epoch}, Step {self.steps}, Train Loss: {train_loss}, Eval Loss: {eval_loss}, "
                        f"Train Accuracy: {train_accuracy_metric.get('accuracy')}, Eval Accuracy: {eval_accuracy.get('accuracy')}"
                    )

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

        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            model_path = os.path.join(self.config["run_output_dir"], "model.pt")
            self.accelerator.save(self.model.state_dict(), model_path)
        else:
            model_path = os.path.join(self.config["run_output_dir"], "model.pt")
            torch.save(self.model.state_dict(), model_path)

    def _evaluate(self):
        self.model.eval()
        eval_loss = 0.0
        eval_steps = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                x, y = batch
                output = self.model(x)
                loss = self.criterion(output, y)  # Use class indices instead of one-hot
                eval_loss += loss.item()

                predictions = torch.argmax(output, dim=1)
                self.test_accuracy.add_batch(predictions=predictions, references=y)
                eval_steps += 1

        eval_loss /= eval_steps
        eval_accuracy = self.test_accuracy.compute()
        return eval_loss, eval_accuracy

    def compute_metrics(self):
        metrics_per_step = {}

        aggregated_weights = {}
        aggregated_biases = {}

        for layer_name, data_list in self.weights_biases_cache.items():
            for data in data_list:
                step = data["steps"]
                epoch = data["epoch"]
                key = (step, epoch)

                if key not in metrics_per_step:
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
                if data["weights"] is not None:
                    weights = torch.from_numpy(data["weights"])
                    # Depending on tensor dimension, compute metrics
                    if weights.dim() == 4:  # Conv2D layers
                        metrics_data = get_tensor_metrics(weights)
                    elif weights.dim() == 2:  # Linear layers
                        metrics_data = get_matrix_metrics(weights)

                    metrics_per_step[key]["w"].append(metrics_data)

                    aggregated_weights[key].append(weights.view(-1))

                if data["biases"] is not None:
                    biases = torch.from_numpy(data["biases"])
                    aggregated_biases[key].append(biases.view(-1))

        # Aggregate statistics
        for (step, epoch), _ in metrics_per_step.items():
            key = (step, epoch)

            if aggregated_weights[key]:
                all_weights_combined = torch.cat(aggregated_weights[key])
                w_all_stats = get_distribution_stats(all_weights_combined)
            else:
                w_all_stats = {}

            if aggregated_biases[key]:
                all_biases_combined = torch.cat(aggregated_biases[key])
                b_all_stats = get_distribution_stats(all_biases_combined)
            else:
                b_all_stats = {}

            metrics_per_step[key].update({"w_all": w_all_stats, "b_all": b_all_stats})

        self.metrics_cache = list(metrics_per_step.values())

    def save_metrics(self):
        metrics_path = os.path.join(
            self.config["run_output_dir"],
            f'lr{self.config["lr"]}_{self.config["optim"]}_seed{self.config["seed"]}_scaling{self.config["init_scaling"]}.json',
        )
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_cache, f)
