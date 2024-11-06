import numpy as np
import random
import torch

# Core metric functions
def l1_norm(tensor):
    """Calculate the L1 norm of a tensor."""
    return torch.linalg.vector_norm(tensor, ord=1).item()

def l2_norm(tensor):
    """Calculate the L2 norm of a tensor."""
    return torch.linalg.vector_norm(tensor, ord=2).item()

def trace(tensor):
    """Calculate the trace of a tensor, if 2D."""
    if tensor.dim() == 2:
        return torch.trace(tensor).item()
    else:
        return 0.0

def spectral_norm(tensor):
    """Calculate the spectral norm of a tensor, if 2D."""
    if tensor.dim() == 2:
        return torch.linalg.matrix_norm(tensor, ord=2).item()
    else:
        return 0.0

def singular_value_metrics(tensor):
    """Calculate mean and variance of singular values, if the tensor is 2D."""
    if tensor.dim() == 2:
        singular_vals = torch.svd(tensor, compute_uv=False).S
        singular_vals[singular_vals < 1e-5] = 0.0  # Remove small values
        mean_singular_value = torch.mean(singular_vals).item()
        var_singular_value = torch.var(singular_vals).item()
        
        return {
            "mean_singular_value": mean_singular_value,
            "var_singular_value": var_singular_value,
        }
    else:
        return {"mean_singular_value": 0.0, "var_singular_value": 0.0}

def code_sparsity(tensor):
    """Calculate L1 norm, L2 norm, and code sparsity for higher-dimensional tensors."""
    l1 = l1_norm(tensor)
    l2 = l2_norm(tensor)
    return {"code_sparsity": l1 / l2 if l2 > 0 else 0.0}

def computational_sparsity(tensor):
    """Calculate trace and spectral norm, if available, for computational sparsity."""
    trace_val = trace(tensor)
    spectral = spectral_norm(tensor)
    return {"computational_sparsity": trace_val / spectral if spectral > 0 else 0.0}

# Dictionary mapping metric names to functions
METRICS_MAP = {
    'l1': l1_norm,
    'l2': l2_norm,
    'trace': trace,
    'spectral': spectral_norm,
    'mean_singular_value': lambda tensor: singular_value_metrics(tensor).get("mean_singular_value"),
    'var_singular_value': lambda tensor: singular_value_metrics(tensor).get("var_singular_value"),
    'code_sparsity': lambda tensor: code_sparsity(tensor).get("code_sparsity"),
    'computational_sparsity': lambda tensor: computational_sparsity(tensor).get("computational_sparsity"),
}

def calculate_metrics(tensor, metric_names=None, custom_metrics=None):
    """
    Calculate specified metrics for a given tensor.
    
    Args:
        tensor (torch.Tensor): The tensor on which to calculate metrics.
        metric_names (list): List of metric names to calculate. If None or empty, defaults to ['l1', 'l2', 'code_sparsity'].
        custom_metrics (dict): Optional dictionary of custom metric functions.
    
    Returns:
        dict: Calculated metrics in a flattened dictionary.
    """
    # Set default metrics if no specific metric names are provided
    if not metric_names:
        metric_names = ['l1', 'l2', 'trace', 'spectral', 'mean_singular_value', 'var_singular_value', 'code_sparsity', 'computational_sparsity']

    metrics = {}
    for name in metric_names:
        if name in METRICS_MAP:
            metric_func = METRICS_MAP[name]
            try:
                result = metric_func(tensor)
                if isinstance(result, dict):
                    # Flatten nested dictionary values
                    metrics.update(result)
                else:
                    metrics[name] = result
            except Exception as e:
                print(f"Error calculating metric '{name}': {e}")

    # Calculate custom metrics if provided
    if custom_metrics:
        for name, func in custom_metrics.items():
            try:
                result = func(tensor)
                if isinstance(result, dict):
                    metrics.update(result)
                else:
                    metrics[name] = result
            except Exception as e:
                print(f"Error calculating custom metric '{name}': {e}")
    
    return metrics

def get_distribution_stats(tensor):
    """Calculate distribution statistics like mean, variance, and median."""
    return {
        "mean": torch.mean(tensor).item(),
        "var": torch.var(tensor).item(),
        "median": torch.median(tensor).item()
    }


def gradient_symmetricity(model, mod_no: int, n_sample: int = 100,
                          device: str = 'cpu'):

    sample_data = [(a, b, c) for a in range(mod_no) for b in range(mod_no) for c in range(mod_no)]
    random.shuffle(sample_data)
    sample_data = sample_data[:n_sample]

    total = 0
    for a, b, c in sample_data:
        x = torch.tensor([[a, b]], device=device)
        temp, output = model.forward_h(x)
        model.zero_grad()
        model.remove_all_hooks()
        o = output[0, -1, :]
        temp.retain_grad()
        o[c].backward(retain_graph=True)
        gradient = temp.grad[0].detach().cpu().numpy()

        cos_sim = np.sum(gradient[0]*gradient[1])/np.sqrt(np.sum(gradient[0]**2))/np.sqrt(np.sum(gradient[1]**2))
        total += cos_sim

    return total / len(sample_data)


def distance_irrelevance(model, dataloader, mod_no: int):
    correct_logits = [[0]*mod_no for _ in range(mod_no)]
    for x, y in dataloader:
        with torch.inference_mode():
            model.eval()
            model.remove_all_hooks()
            output = model(x)[:, -1, :]

            # selecting diagonal elements of the tensor to get correct logits
            diag_output = output[list(range(len(x))), y]
            diag_output = diag_output.cpu()
            x = x.cpu()
            for p, q in zip(diag_output, x):
                A, B = int(q[0].item()), int(q[1].item())
                correct_logits[(A+B) % mod_no][(A-B) % mod_no] = p.item()

    correct_logits = np.array(correct_logits)
    distance_irrelevance = np.mean(np.std(correct_logits, axis=0)
                                   )/np.std(correct_logits.flatten())
    return distance_irrelevance
