import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import *
import numpy as np
import pandas as pd
import math
import csv
from collections import Counter
from sklearn.metrics import f1_score

# --- Utilities ---
millnames = ['', 'K', 'M', 'B', 'T']

def millify(n):
    n = float(n)
    if n == 0:
        return '0'
    millidx = max(0, min(len(millnames)-1, int(math.floor(math.log10(abs(n))/3))))
    scaled = n / 10**(3*millidx)
    # Format with 1 decimal if needed, remove trailing .0
    if scaled % 1 == 0:
        return f'{int(scaled)}{millnames[millidx]}'
    else:
        return f'{scaled:.1f}{millnames[millidx]}'

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i*i <= n:
        if n % i == 0 or n % (i+2) == 0:
            return False
        i += 6
    return True

def r2_score_torch(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred)**2)
    ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
    score = 1 - ss_res/(ss_tot + 1e-6)
    if torch.isnan(score) or torch.isinf(score):
        return 0.0
    return score.item()

# --- Safe Division ---
def safe_div(numerator, denominator, mode="very_close", eps=1e-6):
    e_const = torch.tensor(torch.e, device=denominator.device, dtype=denominator.dtype)
    if mode == "too_near":
        safe_den = torch.where(torch.abs(denominator) < eps*0.1, e_const, denominator)
    elif mode == "too_close":
        safe_den = torch.where(torch.abs(denominator) < eps, e_const, denominator)
    elif mode in ["close","very_close"]:
        sign = torch.sign(denominator)
        sign = torch.where(sign==0, torch.ones_like(sign), sign)
        factor = 10 if mode=="very_close" else 1
        safe_den = denominator + factor*eps*sign
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return numerator / safe_den

# --- Custom Activation Functions ---
class X_EReLU(nn.Module):
    def __init__(self, mode="too_close", eps=1e-6):
        super().__init__()
        self.a1 = nn.Parameter(torch.tensor(1.0)); self.a2 = nn.Parameter(torch.tensor(1.0))
        self.b1 = nn.Parameter(torch.tensor(1.0)); self.b2 = nn.Parameter(torch.tensor(1.0))
        self.c1 = nn.Parameter(torch.tensor(1.0)); self.c2 = nn.Parameter(torch.tensor(1.0))
        self.d1 = nn.Parameter(torch.tensor(1.0)); self.d2 = nn.Parameter(torch.tensor(1.0))
        self.i = nn.Parameter(torch.tensor(1.0))
        self.mode, self.eps = mode, eps
    def forward(self, x):
        a = safe_div(self.a1,self.a2,self.mode,self.eps)
        b = safe_div(self.b1,self.b2,self.mode,self.eps)
        c = safe_div(self.c1,self.c2,self.mode,self.eps)
        d = safe_div(self.d1,self.d2,self.mode,self.eps)
        numerator = a + b*x - c*torch.abs(x) + d
        return safe_div(numerator, self.i, self.mode, self.eps)

class EReLU(nn.Module):
    def __init__(self, init_a=0.0, init_b=1.0, init_c=0.5, init_d=0.0, init_i=1.0, eps=1e-6, safety_mode="too_close"):
        super().__init__()
        self.a,self.b,self.c,self.d,self.i = [nn.Parameter(torch.tensor(p)) for p in [init_a,init_b,init_c,init_d,init_i]]
        self.eps,self.safety_mode = eps,safety_mode
    def forward(self,x):
        numerator = self.a + self.b*x - self.c*torch.abs(x) + self.d
        return safe_div(numerator,self.i,mode=self.safety_mode,eps=self.eps)

class UltraSMHT(nn.Module):
    def __init__(self, a=2.0, b=1.2, c=3.0, d=0.8, g=2.0, f=1.1, h=2.0, i=0.9):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))
        self.b = nn.Parameter(torch.tensor(b))
        self.c = nn.Parameter(torch.tensor(c))
        self.d = nn.Parameter(torch.tensor(d))
        self.g = nn.Parameter(torch.tensor(g))
        self.f = nn.Parameter(torch.tensor(f))
        self.h = nn.Parameter(torch.tensor(h))
        self.i = nn.Parameter(torch.tensor(i))

    def forward(self, x):
        base = lambda p: torch.clamp(p, min=1e-3)
        num = torch.pow(base(self.a), self.b * x) - torch.pow(base(self.c), self.d * x)
        denom = torch.pow(base(self.g), self.f * x) + torch.pow(base(self.h), self.i * x)
        return safe_div(num, denom)

# --- Dataset Generator ---
def generate_dataset(name, n=400):
    """
    Generates a variety of 1D and 2D datasets for regression and classification tasks.
    """
    noise = 0.1
    x = torch.linspace(-5, 5, n).unsqueeze(1)

    if name == 'simplelinear': return x+noise * torch.randn_like(x),x
    if name == 'quadratic': return x, x**2 + torch.randn_like(x) * 2
    if name == 'cubic': return x, x**3 + torch.randn_like(x) * 5
    if name == 'quartic': return x, x**4 - 10 * x**2 + torch.randn_like(x) * 10
    if name == 'quintic': return x, x**5 - 20 * x**3 + 50 * x + torch.randn_like(x) * 20
    if name == 'sin': return x, torch.sin(x) + noise * torch.randn_like(x)
    if name == 'cos': return x, torch.cos(x) + noise * torch.randn_like(x)
    if name == 'tan':
        x = torch.linspace(-1.4, 1.4, n).unsqueeze(1)
        return x, torch.tan(x) + noise * torch.randn_like(x)
    if name == 'cot':
        x = torch.linspace(0.1, 3.0, n).unsqueeze(1)
        return x, 1 / torch.tan(x) + noise * torch.randn_like(x)
    if name == 'sec':
        x = torch.linspace(-1.4, 1.4, n).unsqueeze(1)
        return x, 1 / torch.cos(x) + noise * torch.randn_like(x)
    if name == 'csc':
        x = torch.linspace(0.1, 3.0, n).unsqueeze(1)
        return x, 1 / torch.sin(x) + noise * torch.randn_like(x)
    if name == 'cube_root':
        x = torch.linspace(-10, 10, n).unsqueeze(1)
        y = torch.sign(x) * torch.abs(x)**(1/3)
        return x, y + noise * torch.randn_like(x)
    if name == 'reciprocal':
        x = torch.cat((torch.linspace(-5, -0.1, n // 2), torch.linspace(0.1, 5, n // 2))).unsqueeze(1)
        return x, 1 / x + noise * torch.randn_like(x)
    if name == 'prime':
        x_vals = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(1)
        y_vals = torch.tensor([1.0 if is_prime(i) else 0.0 for i in range(1, n + 1)]).unsqueeze(1)
        return x_vals, y_vals
    if name == 'abs': return x, torch.abs(x) + noise * torch.randn_like(x)
    if name == 'gaussian': return x, torch.exp(-x**2) + noise * torch.randn_like(x)
    if name == 'sigmoid':
        x = torch.linspace(-6, 6, n).unsqueeze(1)
        return x, torch.sigmoid(x) + noise * torch.randn_like(x)
    if name == 'relu':
        x = torch.linspace(-6, 6, n).unsqueeze(1)
        return x, torch.relu(x) + noise * torch.randn_like(x)
    if name == 'tanh':
        x = torch.linspace(-6, 6, n).unsqueeze(1)
        return x, torch.tanh(x) + noise * torch.randn_like(x)
    if name == 'log':
        x = torch.linspace(0.1, 5, n).unsqueeze(1)
        return x, torch.log(x) + noise * torch.randn_like(x)
    if name == 'exp':
        x = torch.linspace(-3, 3, n).unsqueeze(1)
        return x, torch.exp(x) / 10 + noise * torch.randn_like(x)
    if name == 'sawtooth':
        period = 5
        y = 2 * (x / period - torch.floor(0.5 + x / period))
        return x, y + noise * torch.randn_like(x)
    if name == 'step': return x, (x > 0).float() + noise/2 * torch.randn_like(x)
    if name == 'exponential_decay':
        y = x / (torch.exp(torch.ones(1)) - torch.abs(x))
        return x, y + noise * torch.randn_like(x)

    # --- Step Function Variants ---
    if name == 'staircase':
        y = torch.floor(x)
        return x, y + noise/2 * torch.randn_like(x)
    if name == 'step_sine':
        y = torch.round(3 * torch.sin(x))
        return x, y + noise/2 * torch.randn_like(x)
    if name == 'step_quadratic':
        x = torch.linspace(-3.5, 3.5, n).unsqueeze(1)
        y = torch.floor(x**2)
        return x, y + noise/2 * torch.randn_like(x)
    if name == 'random_steps':
        jumps = torch.tensor([-4, -2.5, 0, 1.5, 3])
        y = torch.zeros_like(x)
        for jump_point in jumps:
            y += (x > jump_point).float()
        return x, y + noise/2 * torch.randn_like(x)

    # --- 2D Classification Datasets ---
    X, Y = None, None
    is_torch_native = False
    if name == 'moon':
        X, Y = make_moons(n_samples=n, noise=0.2, random_state=42)
    elif name == 'circle':
        X, Y = make_circles(n_samples=n, noise=0.1, factor=0.5, random_state=42)
    elif name == 'blobs':
        X, Y = make_blobs(n_samples=n, centers=4, n_features=2, random_state=42, cluster_std=1.5)
    elif name == 'checkerboard':
        x1 = np.random.rand(n) * 4 - 2
        x2 = np.random.rand(n) * 4 - 2
        Y = ((x1 > 0) ^ (x2 > 0)).astype(int)
        X = np.vstack((x1, x2)).T + np.random.randn(n, 2) * 0.2
    elif name == 'gaussian_quantiles':
        X, Y = make_gaussian_quantiles(n_samples=n, n_features=2, n_classes=4, random_state=42)
    elif name == 'xor':
        X_base = torch.randn(n, 2) * 0.8
        Y_base = ((X_base[:, 0] > 0) ^ (X_base[:, 1] > 0)).long()
        return X_base, Y_base.unsqueeze(1).float()
    elif name == 'spiral':
        N = n // 2
        theta = torch.sqrt(torch.rand(N)) * 2 * np.pi
        r_a = 2 * theta + np.pi
        data_a = torch.stack([r_a * torch.cos(theta), r_a * torch.sin(theta)], 1) + torch.randn(N, 2) * 0.2
        r_b = -2 * theta - np.pi
        data_b = torch.stack([r_b * torch.cos(theta), r_b * torch.sin(theta)], 1) + torch.randn(N, 2) * 0.2
        X_tensor = torch.cat((data_a, data_b))
        Y_tensor = torch.cat((torch.zeros(N), torch.ones(N)))
        return X_tensor, Y_tensor.long().unsqueeze(1).float()
    elif name == 'anisotropic':
        X_raw, Y = make_blobs(n_samples=n, centers=3, random_state=170, cluster_std=0.8)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X_raw, transformation)
    elif name == 'linear':
        X, Y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2,
                                     n_clusters_per_class=1, flip_y=0.01, random_state=42)
    elif name == 'swiss_roll':
        X_3d, t = make_swiss_roll(n_samples=n, noise=0.3, random_state=42)
        X = X_3d[:, [0, 2]]
        Y = (t > t.mean()).astype(int)
    else:
        raise ValueError(f"Dataset '{name}' not found.")

    y_dtype = torch.float32
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=y_dtype).unsqueeze(1)

# --- Simple Neural Network ---
class SimpleNet(nn.Module):
    def __init__(self,input_dim,act,hidden_dims=[32,16,8,4],output_dim=1):
        super().__init__()
        self.layers = nn.ModuleList()
        dims=[input_dim]+hidden_dims
        for in_dim,out_dim in zip(dims[:-1],dims[1:]): self.layers.append(nn.Linear(in_dim,out_dim))
        self.act,self.output_layer = act, nn.Linear(dims[-1],output_dim)
    def forward(self,x):
        for layer in self.layers: x = self.act(layer(x))
        return self.output_layer(x)

# --- Main Experiment Runner ---
def run_all_experiments(dataset_list, num_epochs=1000):
    """
    Runs training and analysis for a list of datasets and saves a summary CSV.
    """
    all_results_data = []

    for dataset_name in dataset_list:
        print(f"\n{'='*25} Running Dataset: {dataset_name.capitalize()} {'='*25}")
        try:
            x_data, y_data = generate_dataset(dataset_name)
            input_dim = x_data.shape[1]
        except Exception as e:
            print(f"Could not generate dataset '{dataset_name}'. Error: {e}")
            continue

        is_class = dataset_name.lower() in ['moon','circle','xor','checkerboard','blobs','gaussian_quantiles','spiral','anisotropic','linear','swiss_roll', 'prime']
        criterion = nn.BCEWithLogitsLoss() if is_class else nn.MSELoss()
        
        act_functions = {
            'E': EReLU(), 'XER': X_EReLU(), 'R': nn.ReLU(), 'LR': nn.LeakyReLU(), 'PR': nn.PReLU(),
            'RR': nn.RReLU(), 'G': nn.GELU(), 'S': nn.SiLU(), 'C': nn.CELU(), 'ELU': nn.ELU(),
            'SELU': nn.SELU(), 'T': nn.Tanh(), 'HT': nn.Hardtanh(), 'ST': nn.Softsign(),
            'SS': nn.Softshrink(), 'HS': nn.Hardshrink(), 'ReLU6': nn.ReLU6(), 'Sw': nn.Softplus(),
            'LgS': nn.LogSigmoid(), 'Si': nn.Sigmoid(), 'Mish': nn.Mish(), "U": UltraSMHT()
        }

        models, scores_history = {}, {}

        for name, act_fn in act_functions.items():
            model = SimpleNet(input_dim, act_fn)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            models[name] = (model, optimizer)
            scores_history[name] = []
        
        best_activation_history = []

        # --- Training Loop ---
        for epoch in range(num_epochs):
            current_scores = {}
            for name, (model, optim) in models.items():
                model.train()
                optim.zero_grad()
                output = model(x_data)
                loss = criterion(output, y_data)
                loss.backward()
                optim.step()

                model.eval()
                with torch.no_grad():
                    if is_class:
                        preds = (torch.sigmoid(output) > 0.5).float()
                        score = f1_score(y_data.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
                    else:
                        score = r2_score_torch(y_data, output)
                    
                    scores_history[name].append(score)
                    current_scores[name] = score
            
            if current_scores:
                best_score = max(current_scores.values())
                best_activations = [name for name, score in current_scores.items() if score == best_score]
                best_activation_history.append(best_activations[0])
            else:
                best_activation_history.append(None)

        # --- Analysis and Printing for the Current Dataset ---
        print(f"--- Analysis for {dataset_name.capitalize()} after {num_epochs} epochs ---")
        
        leader_changes = sum(1 for i in range(1, len(best_activation_history)) if best_activation_history[i] != best_activation_history[i-1])
        print(f"\n🔄 Total change of best activation: {leader_changes}")

        win_counts = Counter(best_activation_history)
        if win_counts:
            most_common_winner = win_counts.most_common(1)[0]
            print(f"📌 Mostly best activation: '{most_common_winner[0]}' (won {most_common_winner[1]} epochs)")
            
            print("\n⏳ Mostly staying on top (Top 3):")
            for act, count in win_counts.most_common(3):
                print(f"   - {act}: {count} epochs")
        else:
             print("📌 No best activation found.")

        dataset_stats = []
        for name in act_functions.keys():
            scores = scores_history.get(name, [0.0])
            dataset_stats.append({
                "Activation": name,
                "Avg_Score": np.mean(scores),
                "Max_Score": np.max(scores),
                "Min_Score": np.min(scores),
                "Epochs_As_Best": win_counts.get(name, 0)
            })

        df_stats = pd.DataFrame(dataset_stats).sort_values(by="Avg_Score", ascending=False).round(4)
        print("\n📈 Overall Performance Metrics:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(df_stats.to_string(index=False))

        # Prepare data for the final CSV
        for row in dataset_stats:
            row['Dataset'] = dataset_name
            row['Task'] = 'Classification' if is_class else 'Regression'
            row['Metric'] = 'F1' if is_class else 'R2'
            row['Leader_Changes'] = leader_changes
            row['Overall_Winner'] = win_counts.most_common(1)[0][0] if win_counts else 'N/A'
        
        all_results_data.extend(dataset_stats)

    # --- Save all results to a single CSV file ---
    if all_results_data:
        fieldnames = [
            'Dataset', 'Task', 'Metric', 'Activation', 'Avg_Score', 'Max_Score', 
            'Min_Score', 'Epochs_As_Best', 'Overall_Winner', 'Leader_Changes'
        ]
        final_df = pd.DataFrame(all_results_data)[fieldnames]
        
        try:
            final_df.to_csv('activation_performance_summary.csv', index=False)
            print(f"\n\n{'='*70}")
            print("✅ Successfully saved detailed results to 'activation_performance_summary.csv'")
            print(f"{'='*70}")
        except Exception as e:
            print(f"\n\n❌ Failed to save CSV file. Error: {e}")

# --- Run datasets ---
if __name__ == '__main__':
    # A comprehensive list covering various function types and classification problems.
    dataset_types = [
        # Regression
        "simplelinear", "quadratic", "cubic", "quartic", "quintic",
        "sin", "cos", "tan", "cot", "sec", "csc",
        "cube_root", "reciprocal", "abs", "gaussian", "sigmoid",
        "relu", "tanh", "log", "exp", "sawtooth", "step",
        "exponential_decay", "staircase", "step_sine", "step_quadratic",
        "random_steps",
        # Classification
        "prime", "moon", "circle", "xor", "blobs", "checkerboard",
        "gaussian_quantiles", "spiral", "anisotropic", "linear", "swiss_roll"
    ]

    run_all_experiments(dataset_types, num_epochs=500)
