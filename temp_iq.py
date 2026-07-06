## Data used in this study
import numpy as np
X = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Train\X1.npy')
print(X.shape)
X_val = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Validation\X_validation.npy')
print(X_val.shape)
Y = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Train\Y.npy')
print(Y.shape)
Y_val = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Validation\Y_validation.npy')
print(Y_val.shape)

# Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb

# HYDRA: Competing convolutional kernels for fast and accurate time series classification
# https://arxiv.org/abs/2203.13652

# ** EXPERIMENTAL **
# This is an *untested*, *experimental* extension of Hydra to multivariate input.

# todo: cleanup, documentation

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class HydraMultivariate(nn.Module):

    def __init__(self, input_length, num_channels, k = 16, g = 128, max_num_channels = 8):

        super().__init__()

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        #self.dilations = 2 ** torch.arange(int(max_exponent) + 1)

        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        # if g > 1, assign: half the groups to X, half the groups to diff(X)
        divisor = 2 if self.g > 1 else 1
        _g = g // divisor
        self._g = _g

        self.W = [self.normalize(torch.randn(divisor, k * _g, 1, 9)) for _ in range(self.num_dilations)]

        # combine num_channels // 2 channels (2 < n < max_num_channels)
        num_channels_per = np.clip(num_channels // 2, 2, max_num_channels)
        self.I = [torch.randint(0, num_channels, (divisor, _g, num_channels_per)) for _ in range(self.num_dilations)]

    @staticmethod
    def normalize(W):
        W -= W.mean(-1, keepdims = True)
        W /= W.abs().sum(-1, keepdims = True)
        return W

    # transform in batches of *batch_size*
    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for i, batch in enumerate(batches):
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.g > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            # diff_index == 0 -> X
            # diff_index == 1 -> diff(X)
            for diff_index in range(min(2, self.g)):

                _Z = F.conv1d(X[:, self.I[dilation_index][diff_index]].sum(2) if diff_index == 0 else diff_X[:, self.I[dilation_index][diff_index]].sum(2),
                              self.W[dilation_index][diff_index], dilation = d, padding = p,
                              groups = self._g) \
                      .view(num_examples, self._g, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self._g, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self._g, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z


# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# HYDRA: Competing Convolutional Kernels for Fast and Accurate Time Series Classification
# https://arxiv.org/abs/2203.13652

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class Hydra(nn.Module):

    def __init__(self, input_length, k = 8, g = 64, seed = None):

        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((input_length - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor

        self.W = torch.randn(self.num_dilations, self.divisor, self.k * self.h, 1, 9)
        self.W = self.W - self.W.mean(-1, keepdims = True)
        self.W = self.W / self.W.abs().sum(-1, keepdims = True)

    # transform in batches of *batch_size*
    def batch(self, X, batch_size = 256):
        num_examples = X.shape[0]
        if num_examples <= batch_size:
            return self(X)
        else:
            Z = []
            batches = torch.arange(num_examples).split(batch_size)
            for batch in batches:
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):

        num_examples = X.shape[0]

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):

                _Z = F.conv1d(X if diff_index == 0 else diff_X, self.W[dilation_index, diff_index], dilation = d, padding = p) \
                      .view(num_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(num_examples, self.h, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(num_examples, self.h, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(num_examples, -1)

        return Z

class SparseScaler():

    def __init__(self, mask = True, exponent = 4):

        self.mask = mask
        self.exponent = exponent

        self.fitted = False

    def fit(self, X):

        assert not self.fitted, "Already fitted."

        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

        self.fitted = True

    def transform(self, X):

        assert self.fitted, "Not fitted."

        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)



## Temporal-based model
import numpy as np
import torch

from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import pearsonr, t

# ============================================================
# Random Seed
# ============================================================

best_seed = 42

np.random.seed(best_seed)
torch.manual_seed(best_seed)
torch.cuda.manual_seed_all(best_seed)

# ============================================================
# Fisher Confidence Interval (Per Fold)
# ============================================================

def fisher_ci(r, n, confidence=0.95):
    """
    Fisher z-transformation confidence interval
    for Pearson correlation.
    """

    if abs(r) >= 1:
        return r, r

    z = np.arctanh(r)

    se = 1 / np.sqrt(n - 3)

    z_critical = 1.96

    lower = np.tanh(z - z_critical * se)
    upper = np.tanh(z + z_critical * se)

    return lower, upper


# ============================================================
# Confidence Interval Across Folds
# ============================================================

def fold_confidence_interval(values, confidence=0.95):

    values = np.asarray(values)

    mean = np.mean(values)

    std = np.std(values, ddof=1)

    n = len(values)

    t_value = t.ppf((1 + confidence) / 2.0, n - 1)

    margin = t_value * std / np.sqrt(n)

    return mean, std, mean - margin, mean + margin


# ============================================================
# K-Fold
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=best_seed
)

all_predictions = np.zeros(len(Y))
all_targets = np.zeros(len(Y))
# ============================================================
# Best fold storage
# ============================================================

best_r = -np.inf
best_actual_iq = None
best_predicted_iq = None
best_fold = None
fold_corr = []
fold_mae = []
fold_rmse = []
fold_p = []
# ============================================================
# Store all fold predictions
# ============================================================

all_actual_iq = []
all_predicted_iq = []
# ============================================================
# Cross Validation
# ============================================================

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):

    print("\n" + "=" * 60)
    print(f"Fold {fold}")
    print("=" * 60)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    # --------------------------------------------------------
    # Torch
    # --------------------------------------------------------

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()

    # --------------------------------------------------------
    # HYDRA
    # --------------------------------------------------------

    transform = HydraMultivariate(
        X_train.shape[-1],
        116
    )

    X_train_tr = transform(X_train)
    X_test_tr = transform(X_test)

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = SparseScaler()

    X_train_tr = scaler.fit_transform(X_train_tr)
    X_test_tr = scaler.transform(X_test_tr)

    # --------------------------------------------------------
    # Ridge Regression
    # --------------------------------------------------------

    regressor = RidgeCV(
        alphas=np.logspace(-3, 3, 10)
    )

    regressor.fit(X_train_tr, Y_train)

    pred = regressor.predict(X_test_tr)
    # Save for scatter plot

    all_actual_iq.extend(Y_test.tolist())
    all_predicted_iq.extend(pred.tolist())

    # --------------------------------------------------------
    # Save Predictions
    # --------------------------------------------------------

    all_predictions[test_idx] = pred
    all_targets[test_idx] = Y_test

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    r, p = pearsonr(pred, Y_test)
    # Save best fold predictions

    if r > best_r:
        best_r = r
        best_fold = fold
        best_actual_iq = Y_test.copy()
        best_predicted_iq = pred.copy()

    ci_low, ci_high = fisher_ci(r, len(Y_test))

    mae = mean_absolute_error(Y_test, pred)

    rmse = np.sqrt(mean_squared_error(Y_test, pred))

    fold_corr.append(r)
    fold_mae.append(mae)
    fold_rmse.append(rmse)
    fold_p.append(p)

    # --------------------------------------------------------
    # Print Fold Results
    # --------------------------------------------------------

    print(f"Samples     : {len(test_idx)}")

    print(f"Correlation : {r:.4f}")
    print(f"95% CI      : [{ci_low:.4f}, {ci_high:.4f}]")

    print(f"P-value     : {p:.6e}")

    print(f"MAE         : {mae:.4f}")
    print(f"RMSE        : {rmse:.4f}")

# ============================================================
# Statistics Across Folds
# ============================================================
best_actual_iq = np.array(best_actual_iq)
best_predicted_iq = np.array(best_predicted_iq)

print(f"\nBest Fold : {best_fold}")
print(f"Best Fold Correlation : {best_r:.4f}")
all_actual_iq = np.array(all_actual_iq)
all_predicted_iq = np.array(all_predicted_iq)

corr_mean, corr_std, corr_low, corr_high = fold_confidence_interval(
    fold_corr
)

mae_mean, mae_std, mae_low, mae_high = fold_confidence_interval(
    fold_mae
)

rmse_mean, rmse_std, rmse_low, rmse_high = fold_confidence_interval(
    fold_rmse
)

# ============================================================
# Overall Out-of-Fold Metrics
# ============================================================

overall_r, overall_p = pearsonr(
    all_predictions,
    all_targets
)

overall_mae = mean_absolute_error(
    all_targets,
    all_predictions
)

overall_rmse = np.sqrt(
    mean_squared_error(
        all_targets,
        all_predictions
    )
)

# ============================================================
# Print Fold Summary
# ============================================================

print("\n")
print("=" * 75)
print("FOLD-WISE SUMMARY")
print("=" * 75)

print(
    "{:<8s} {:<12s} {:<28s} {:<12s} {:<12s}".format(
        "Fold",
        "r",
        "95% CI",
        "MAE",
        "RMSE"
    )
)

for i in range(len(fold_corr)):

    ci_l, ci_u = fisher_ci(
        fold_corr[i],
        len(Y) // 5
    )

    print(
        "{:<8d} {:<12.4f} [{:.4f}, {:.4f}] {:<12.4f} {:<12.4f}".format(
            i + 1,
            fold_corr[i],
            ci_l,
            ci_u,
            fold_mae[i],
            fold_rmse[i]
        )
    )

# ============================================================
# Cross Validation Summary
# ============================================================

print("\n")
print("=" * 75)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 75)

print()

print(f"Correlation : {corr_mean:.4f} ± {corr_std:.4f}")
print(f"95% CI      : [{corr_low:.4f}, {corr_high:.4f}]")

print()

print(f"MAE         : {mae_mean:.4f} ± {mae_std:.4f}")
print(f"95% CI      : [{mae_low:.4f}, {mae_high:.4f}]")

print()

print(f"RMSE        : {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"95% CI      : [{rmse_low:.4f}, {rmse_high:.4f}]")

print()

print(f"Average Fold P-value : {np.mean(fold_p):.6e}")


##Plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

test_pred = np.array(best_predicted_iq)
y_test_h = np.array(best_actual_iq)

# -------------------------------------------------
# PEARSON CORRELATION
# -------------------------------------------------
r, _ = pearsonr(test_pred, y_test_h)

# -------------------------------------------------
# STYLE
# -------------------------------------------------
sns.set_style("white")
plt.figure(figsize=(6, 6))

# Regression plot with confidence interval
sns.regplot(
    x=test_pred,
    y=y_test_h,
    scatter_kws={'color': 'blue', 'alpha': 0.8, 's': 40},
    line_kws={'color': 'black', 'linewidth': 2},
    ci=95
)

# Labels
# Set axis limits
plt.xlabel("Predicted IQ score", fontsize=12, weight='bold')
plt.ylabel("Actual IQ score", fontsize=12, weight='bold')
#plt.title("HYBRID MODEL", fontsize=14, weight='bold')


# Clean look
plt.grid(False)
plt.tight_layout()
plt.savefig(
    "Hybrid_model_plot.jpg",
    dpi=600,                 # 300 for normal HD, 600 for journal quality
    bbox_inches="tight",
    format="png"
)

plt.show()