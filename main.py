#!/usr/bin/env python3

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cuda:0"

class LinearFunctionFamily:
    @staticmethod
    def sample_instance(n_functions, input_dim, output_dim):
        functions = []
        for _ in range(n_functions):
            weights = np.random.randn(input_dim, output_dim).astype(np.float32)
            def f(x):
                return weights.T @ x
            functions.append(f)
        return functions


class ComposedFamily:
    def __init__(self, functions, combiner, input_dim):
        self.functions = functions
        self.combiner = combiner
        self.input_dim = input_dim

    def sample_instance(self):
        i1, i2 = np.random.choice(len(self.functions), size=2)
        f1, f2 = self.functions[i1], self.functions[i2]
        x = np.random.randn(self.input_dim).astype(np.float32)
        y = self.combiner(f1, f2)(x)
        return (i1, i2, x, y)

    def sample_batch(self, batch_size):
        return [self.sample_instance() for _ in range(batch_size)]


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)) + x)

class ResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.unembed = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)


def run_experiment(function_family, combiner, n_functions, dim, n_samples):
    functions = function_family.sample_instance(n_functions, dim, dim)
    family = ComposedFamily(functions, combiner, dim)

    dataset = family.sample_batch(n_samples)
    test_function = (0, 0)
    train_dataset = [d for d in dataset if d[:2] != test_function]
    test_dataset = [d for d in dataset if d[:2] == test_function]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    model_input_dim = len(functions) * 2 + dim

    network = ResidualNetwork(model_input_dim, hidden_dim=64, output_dim=dim, n_blocks=2).to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    objective = nn.MSELoss()
    for i_epoch in range(512 * int(np.ceil(np.log2(n_functions)))):
        train_loss = 0
        for i1, i2, x, y in train_loader:
            e1 = F.one_hot(i1, n_functions).float()
            e2 = F.one_hot(i2, n_functions).float()
            inp = torch.cat([e1, e2, x], dim=1).to(DEVICE)
            y_pred = network(inp)
            optimizer.zero_grad()
            loss = objective(y_pred, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        #print(train_loss)

    with torch.no_grad():
        test_loss = 0
        for i1, i2, x, y in test_loader:
            e1 = F.one_hot(i1, n_functions).float()
            e2 = F.one_hot(i2, n_functions).float()
            inp = torch.cat([e1, e2, x], dim=1).to(DEVICE)
            y_pred = network(inp)
            loss = objective(y_pred, y.to(DEVICE))
            test_loss += loss.item()

    return train_loss, test_loss

if __name__ == "__main__":
    function_family = LinearFunctionFamily
    combiners = [
        ("add", lambda f, g: lambda x: f(x) + g(x)),
        ("mul", lambda f, g: lambda x: f(x) * g(x)),
        ("comp", lambda f, g: lambda x: f(g(x))),
    ]
    for combiner_name, combiner in combiners:
        for dim in [8, 16, 32, 64]:
            for n_functions in [2, 4, 8, 16]:
                for replicate in range(3):
                    train_loss, test_loss = run_experiment(function_family, combiner, n_functions, dim, 1000)
                    #print(f"input_dim={input_dim}, n_functions={n_functions}, test_loss={test_loss}")
                    print(f"{combiner_name},dim={dim},n={n_functions},train_loss={train_loss:.4f},test_loss={test_loss:.4f}")
