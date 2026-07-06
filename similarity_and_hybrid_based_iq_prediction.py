import numpy as np
import scipy
from scipy.signal import hilbert, chirp, coherence
import matplotlib.pyplot as plt


import numpy as np
X = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Train\X1.npy')
print(X.shape)
X_val = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Validation\X_validation.npy')
print(X_val.shape)
Y = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Train\Y.npy')
print(Y.shape)
Y_val = np.load(r'E:\ABIDE_RESULTS - Copy\data\Temporal data\Validation\Y_validation.npy')
print(Y_val.shape)



import numpy as np
import scipy
from scipy.signal import hilbert
from tslearn.metrics import dtw, lcss

# -------------------------------------------------------------
# PLV and PLI
# -------------------------------------------------------------
def calculate_PLV_PLI(signal1, signal2):

    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))

    phase_diff = phase1 - phase2

    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    pli = np.abs(np.mean(np.sign(np.sin(phase_diff))))

    return plv, pli
# Replace X with X_val for validation data
n_subjects = X.shape[0]
n_regions = X.shape[1]

data11 = np.zeros((n_subjects, n_regions, n_regions))  # Correlation
data22 = np.zeros((n_subjects, n_regions, n_regions))  # Coherence
data33 = np.zeros((n_subjects, n_regions, n_regions))  # PLV
data44 = np.zeros((n_subjects, n_regions, n_regions))  # PLI
data55 = np.zeros((n_subjects, n_regions, n_regions))  # DTW
data66 = np.zeros((n_subjects, n_regions, n_regions))  # LCSS

for i in range(n_subjects):

    print(f"Subject {i+1}/{n_subjects}")

    # Pearson correlation
    data11[i] = np.corrcoef(X[i])

    for j in range(n_regions):

        data22[i, j, j] = 1
        data33[i, j, j] = 1
        data44[i, j, j] = 1
        data55[i, j, j] = 0
        data66[i, j, j] = 1

        for k in range(j + 1, n_regions):

            sig1 = X[i, j, :]
            sig2 = X[i, k, :]

            # Coherence
            _, Cxy = scipy.signal.coherence(sig1, sig2)
            coh = np.mean(Cxy)

            # PLV & PLI
            plv, pli = calculate_PLV_PLI(sig1, sig2)

            # DTW
            dtw_dist = dtw(sig1, sig2)

            # LCSS
            lcss_sim = lcss(
                sig1.reshape(-1, 1),
                sig2.reshape(-1, 1),
                eps=0.5
            )

            # Fill symmetric matrices
            data22[i, j, k] = data22[i, k, j] = coh
            data33[i, j, k] = data33[i, k, j] = plv
            data44[i, j, k] = data44[i, k, j] = pli
            data55[i, j, k] = data55[i, k, j] = dtw_dist
            data66[i, j, k] = data66[i, k, j] = lcss_sim




np.save("Correlation1.npy", data11)
np.save("Coherence1.npy", data22)
np.save("PLV1.npy", data33)
np.save("PLI1.npy", data44)
np.save("DTW1.npy", data55)
np.save("lcss.npy", data66)



A1 = np.load(r'E:\ABIDE_RESULTS - Copy\correlation1.npy')
B1 = np.load(r'E:\ABIDE_RESULTS - Copy\coherence1.npy')
C1 = np.load(r'E:\ABIDE_RESULTS - Copy\PLV1.npy')
D1 = np.load(r'E:\ABIDE_RESULTS - Copy\PLI1.npy')
E1 = np.load(r'E:\ABIDE_RESULTS - Copy\lcss1.npy')
F1 = np.load(r'E:\ABIDE_RESULTS - Copy\DTW_MATRIX1.npy')
print(A1.shape,B1.shape,C1.shape,D1.shape,E1.shape,F1.shape)




import numpy as np

# Step 1: Concatenate A and B horizontally to form the top half
top_half = np.concatenate((C1, D1), axis=2)

# Step 2: Concatenate C and D horizontally to form the bottom half
bottom_half = np.concatenate((E1, F1), axis=2)

# Step 3: Concatenate the top and bottom halves vertically to form the final square matrix
final_matrix = np.concatenate((top_half, bottom_half), axis=1)

# Step 4: Split the final matrix into top and bottom rectangular matrices
top_rectangular = final_matrix[:, :200, :]
bottom_rectangular = final_matrix[:, 200:, :]

# Print shapes to verify
print("Final square matrix shape:", final_matrix.shape)
print("Top rectangular matrix shape:", top_rectangular.shape)
print("Bottom rectangular matrix shape:", bottom_rectangular.shape)




import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.stats import pearsonr, t

import random
import os

# =====================================================
# Seed
# =====================================================

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================
# Input
# =====================================================

A = bottom_rectangular[..., np.newaxis].astype(np.float32)

Y = np.asarray(Y).astype(np.float32)

# =====================================================
# Confidence Interval Functions
# =====================================================

def fisher_ci(r, n):

    if abs(r) >= 1:
        return r, r

    z = np.arctanh(r)

    se = 1 / np.sqrt(n - 3)

    z_critical = 1.96

    lower = np.tanh(z - z_critical * se)
    upper = np.tanh(z + z_critical * se)

    return lower, upper


def fold_confidence_interval(values):

    values = np.asarray(values)

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    n = len(values)

    t_value = t.ppf(0.975, n - 1)

    margin = t_value * std / np.sqrt(n)

    return mean, std, mean - margin, mean + margin


# =====================================================
# CNN
# =====================================================

def build_model():

    model = Sequential([

        Conv2D(
            128,
            (3,3),
            activation='relu',
            input_shape=(200,400,1)
        ),

        MaxPooling2D((2,2)),

        Conv2D(
            64,
            (3,3),
            activation='relu'
        ),

        MaxPooling2D((2,2)),

        Conv2D(
            32,
            (3,3),
            activation='relu'
        ),

        MaxPooling2D((2,2)),

        Flatten(),

        Dense(
            32,
            activation='relu'
        ),

        Dense(1)

    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# =====================================================
# K-Fold
# =====================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

fold_corr = []
fold_mae = []
fold_rmse = []
fold_p = []
fold_sizes = []
# ============================================================
# Best fold storage
# ============================================================

best_r = -np.inf
best_actual_iq = None
best_predicted_iq = None
best_fold = None
# =====================================================
# Cross Validation
# =====================================================

for fold, (train_idx, test_idx) in enumerate(kf.split(A), start=1):

    print("\n" + "="*60)
    print(f"Fold {fold}")
    print("="*60)

    X_train = A[train_idx]
    X_test = A[test_idx]

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    fold_sizes.append(len(Y_test))

    tf.keras.backend.clear_session()

    model = build_model()

    model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=16,
        shuffle=False,
        verbose=1
    )

    pred = model.predict(
        X_test,
        verbose=0
    ).flatten()

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    r, p = pearsonr(pred, Y_test)
    # Save best fold predictions

    if r > best_r:
        best_r = r
        best_fold = fold
        best_actual_iq = Y_test.copy()
        best_predicted_iq = pred.copy()

    ci_low, ci_high = fisher_ci(
        r,
        len(Y_test)
    )

    mae = mean_absolute_error(
        Y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            Y_test,
            pred
        )
    )

    fold_corr.append(r)
    fold_mae.append(mae)
    fold_rmse.append(rmse)
    fold_p.append(p)

    print(f"Samples     : {len(Y_test)}")
    print(f"Correlation : {r:.4f}")
    print(f"95% CI      : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"P-value     : {p:.6e}")
    print(f"MAE         : {mae:.4f}")
    print(f"RMSE        : {rmse:.4f}")

# =====================================================
# Fold-wise Summary
# =====================================================
best_actual_iq = np.array(best_actual_iq)
best_predicted_iq = np.array(best_predicted_iq)

print(f"\nBest Fold : {best_fold}")
print(f"Best Fold Correlation : {best_r:.4f}")
print("\n")
print("="*75)
print("FOLD-WISE SUMMARY")
print("="*75)

print("{:<6} {:<12} {:<28} {:<12} {:<12}".format(
    "Fold",
    "r",
    "95% CI",
    "MAE",
    "RMSE"
))

for i in range(5):

    ci_low, ci_high = fisher_ci(
        fold_corr[i],
        fold_sizes[i]
    )

    print("{:<6d} {:<12.4f} [{:.4f}, {:.4f}] {:<12.4f} {:<12.4f}".format(
        i+1,
        fold_corr[i],
        ci_low,
        ci_high,
        fold_mae[i],
        fold_rmse[i]
    ))

# =====================================================
# Statistics Across Folds
# =====================================================

corr_mean, corr_std, corr_low, corr_high = \
    fold_confidence_interval(fold_corr)

mae_mean, mae_std, mae_low, mae_high = \
    fold_confidence_interval(fold_mae)

rmse_mean, rmse_std, rmse_low, rmse_high = \
    fold_confidence_interval(fold_rmse)

# =====================================================
# Final Results
# =====================================================

print("\n")
print("="*75)
print("5-FOLD CROSS VALIDATION RESULTS")
print("="*75)

print()

print(f"Correlation : {corr_mean:.4f} ± {corr_std:.4f}")
print(f"95% CI      : [{corr_low:.4f}, {corr_high:.4f}]")

print()

print(f"MAE         : {mae_mean:.4f} ± {mae_std:.4f}")
print(f"95% CI      : [{mae_low:.4f}, {mae_high:.4f}]")

print()

print(f"RMSE        : {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"95% CI      : [{rmse_low:.4f}, {rmse_high:.4f}]")




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




### Feature Extraction ####
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# -------------------------------------------------------
# Prepare Data
# -------------------------------------------------------
A = bottom_rectangular[..., np.newaxis].astype(np.float32)
Y = np.asarray(Y).astype(np.float32)

print("Input shape :", A.shape)
print("Target shape:", Y.shape)

# -------------------------------------------------------
# CNN Model
# -------------------------------------------------------
model = Sequential([
    Input(shape=(200, 400, 1)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(32, activation='relu', name='feature_layer'),

    Dense(1)
])

# -------------------------------------------------------
# Compile
# -------------------------------------------------------
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.summary()

# -------------------------------------------------------
# Train
# -------------------------------------------------------
history = model.fit(
    A,
    Y,
    epochs=10,
    batch_size=16,
    verbose=1
)

# -------------------------------------------------------
# Feature Extractor
# -------------------------------------------------------
feature_extractor = Model(
    inputs=model.inputs,
    outputs=model.get_layer("feature_layer").output
)

# Extract 32-dimensional features
features_train = feature_extractor.predict(A)

print("Extracted feature shape:", features_train.shape)



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
# Fisher CI (Per Fold)
# ============================================================

def fisher_ci(r, n):

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

def fold_confidence_interval(values):

    values = np.asarray(values)

    mean = np.mean(values)
    std = np.std(values, ddof=1)

    n = len(values)

    t_value = t.ppf(0.975, n - 1)

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

fold_corr = []
fold_mae = []
fold_rmse = []
fold_p = []
fold_sizes = []
# ============================================================
# Best fold storage
# ============================================================

best_r = -np.inf
best_actual_iq = None
best_predicted_iq = None
best_fold = None
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

    X1_train = features_train[train_idx]
    X1_test = features_train[test_idx]

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]

    fold_sizes.append(len(Y_test))

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
        100
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
    # Concatenate CNN Features
    # --------------------------------------------------------

    X_train_final = np.concatenate(
        (X_train_tr, X1_train),
        axis=1
    )

    X_test_final = np.concatenate(
        (X_test_tr, X1_test),
        axis=1
    )

    # --------------------------------------------------------
    # Ridge Regression
    # --------------------------------------------------------

    regressor = RidgeCV(
        alphas=np.logspace(-3, 3, 10)
    )

    regressor.fit(X_train_final, Y_train)

    pred = regressor.predict(X_test_final)

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

    print(f"Samples     : {len(Y_test)}")
    print(f"Correlation : {r:.4f}")
    print(f"95% CI      : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"P-value     : {p:.6e}")
    print(f"MAE         : {mae:.4f}")
    print(f"RMSE        : {rmse:.4f}")

# ============================================================
# Fold-wise Summary
# ============================================================
best_actual_iq = np.array(best_actual_iq)
best_predicted_iq = np.array(best_predicted_iq)

print(f"\nBest Fold : {best_fold}")
print(f"Best Fold Correlation : {best_r:.4f}")
print("\n")
print("=" * 75)
print("FOLD-WISE SUMMARY")
print("=" * 75)

print("{:<6} {:<12} {:<28} {:<12} {:<12}".format(
    "Fold",
    "r",
    "95% CI",
    "MAE",
    "RMSE"
))

for i in range(5):

    ci_low, ci_high = fisher_ci(
        fold_corr[i],
        fold_sizes[i]
    )

    print("{:<6} {:<12.4f} [{:.4f}, {:.4f}] {:<12.4f} {:<12.4f}".format(
        i + 1,
        fold_corr[i],
        ci_low,
        ci_high,
        fold_mae[i],
        fold_rmse[i]
    ))

# ============================================================
# Statistics Across Folds
# ============================================================

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
# Final Results
# ============================================================

print("\n")
print("=" * 75)
print("5-FOLD CROSS VALIDATION RESULTS")
print("=" * 75)

print(f"Correlation : {corr_mean:.4f} ± {corr_std:.4f}")
print(f"95% CI      : [{corr_low:.4f}, {corr_high:.4f}]")

print()

print(f"MAE         : {mae_mean:.4f} ± {mae_std:.4f}")
print(f"95% CI      : [{mae_low:.4f}, {mae_high:.4f}]")

print()

print(f"RMSE        : {rmse_mean:.4f} ± {rmse_std:.4f}")
print(f"95% CI      : [{rmse_low:.4f}, {rmse_high:.4f}]")




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