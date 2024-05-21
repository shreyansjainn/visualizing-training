import torch
import numpy as np
from torch.utils.data import DataLoader

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'

class MyAddDataSet(torch.utils.data.Dataset):
    def __init__(self, func, C, diff_vocab=False, eqn_sign=False):
        self.func = func
        dim = 2
        self.dim = dim
        self.C = C
        self.inputs = []
        self.outputs = []
        self.vocab=C
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
        self.inputs = torch.tensor(self.inputs, dtype=torch.long, device=DEVICE)
        self.outputs = torch.tensor(self.outputs, dtype=torch.long, device=DEVICE)
        print(self.inputs,self.outputs)
    def __len__(self):
        return len(self.outputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


def get_dataloaders(config):
    config['func']=eval(config['funcs'])
    useLinear=config.get('use_linear',False)
    full_dataset = MyAddDataSet(func=config['func'],C=config['C'],diff_vocab=config['diff_vocab'],eqn_sign=config['eqn_sign'])
    train_size = int(config['frac'] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print('random split',len(train_dataset),len(test_dataset))
    batch_size = config.get('batch_size',len(full_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader