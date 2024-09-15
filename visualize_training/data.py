import torch
from torch.utils.data import Dataset
import numpy as np
import random

from itertools import product


def distance(x, y):
    assert x.shape == y.shape
    return torch.norm(x - y, 2)


class ModularArithmetic(Dataset):
    @property
    def fns_dict(self):
        return {
            "add": lambda x, y, z: (x + y) % z,
            "sub": lambda x, y, z: (x - y) % z,
            "mul": lambda x, y, z: (x * y) % z,
            "div": lambda x, y, z: (x / y) % z,
        }

    def __init__(self, operation, num_to_generate: int = 113):
        """Generate train and test split"""
        result_fn = self.fns_dict[operation]
        self.x = [
            [i, j, num_to_generate]
            for i in range(num_to_generate)
            for j in range(num_to_generate)
        ]
        self.y = [result_fn(i, j, k) for (i, j, k) in self.x]

    def __getitem__(self, index):
        return torch.tensor(self.x[index]), self.y[index]

    def __len__(self):
        return len(self.x)


class SparseParity(Dataset):
    def __init__(self, num_samples, total_bits, parity_bits):
        self.x = torch.randint(0, 2, (num_samples, total_bits)) * 2 - 1.0
        # self.x = torch.tensor([[random.choice([-1, 1]) for j in range(total_bits)] for i in range(num_samples)])
        # self.s = np.random.choice(total_bits, parity_bits, replace=False)
        self.y = torch.prod(self.x[:, :parity_bits], dim=1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]
    
class MyAddDataSet(torch.utils.data.Dataset):
    def __init__(self, func, C, diff_vocab=False, eqn_sign=False, device='cpu'):
        self.func = func
        dim = 2
        self.dim = dim
        self.C = C
        self.inputs = []
        self.outputs = []
        self.vocab=C
        self.device=device
        if diff_vocab:
            self.vocab*=2
        if eqn_sign:
            self.vocab+=1
            self.dim+=1
        self.vocab_out=0
        for p in range(C**dim):
            x = np.unravel_index(p, (C,)*dim)
            o=self.func(x)
            s=[x[0],x[1]]
            if diff_vocab:
                s[1]+=C
            if eqn_sign:
                s.append(self.vocab-1)
            self.inputs.append(s)
            self.outputs.append(o)
            self.vocab_out=max(self.vocab_out, o+1)
        if self.vocab_out!=C:
            print(f'warning {self.vocab_out=} neq to {C=}')
        self.inputs = torch.tensor(self.inputs, dtype=torch.long, device=self.device)
        self.outputs = torch.tensor(self.outputs, dtype=torch.long, device=self.device)
        print(self.inputs,self.outputs)
    def __len__(self):
        return len(self.outputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
