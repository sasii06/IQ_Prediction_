import numpy as np
import scipy
from scipy.signal import hilbert, chirp, coherence
import matplotlib.pyplot as plt


import numpy as np

# Concatenate
X_val = np.load(r'E:\ABIDE_RESULTS - Copy\X_validation_mixed.npy')
Y_val = np.load(r'E:\ABIDE_RESULTS - Copy\Y_validation_mixed.npy')

print("X shape:", X_val.shape)
print("Y shape:", Y_val.shape)


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
X = X_val
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


# -------------------------------------------------------------
# Optional: Save
# -------------------------------------------------------------
np.save("Correlation_val.npy", data11)
np.save("Coherence_val.npy", data22)
np.save("PLV_val.npy", data33)
np.save("PLI_val.npy", data44)
np.save("DTW_val.npy", data55)
np.save("lcss_val.npy", data66)




import numpy as np

# Concatenate
X = np.load(r'E:\ABIDE_RESULTS - Copy\X1.npy')
Y = np.load(r'E:\ABIDE_RESULTS - Copy\Y.npy')

print("X shape:", X.shape)
print("Y shape:", Y.shape)




A1 = np.load(r'E:\ABIDE_RESULTS - Copy\correlation1.npy')
B1 = np.load(r'E:\ABIDE_RESULTS - Copy\coherence1.npy')
C1 = np.load(r'E:\ABIDE_RESULTS - Copy\PLV1.npy')
D1 = np.load(r'E:\ABIDE_RESULTS - Copy\PLI1.npy')
E1 = np.load(r'E:\ABIDE_RESULTS - Copy\lcss1.npy')
F1 = np.load(r'E:\ABIDE_RESULTS - Copy\DTW_MATRIX1.npy')
print(A1.shape,B1.shape,C1.shape,D1.shape,E1.shape,F1.shape)




A1_val = np.load(r'E:\ABIDE_RESULTS - Copy\correlation_val.npy')
B1_val = np.load(r'E:\ABIDE_RESULTS - Copy\coherence_val.npy')
C1_val = np.load(r'E:\ABIDE_RESULTS - Copy\PLV_val.npy')
D1_val = np.load(r'E:\ABIDE_RESULTS - Copy\PLI_val.npy')
E1_val = np.load(r'E:\ABIDE_RESULTS - Copy\lcss_val.npy')
F1_val = np.load(r'E:\ABIDE_RESULTS - Copy\DTW_MATRIX_val.npy')
print(A1_val.shape,B1_val.shape,C1_val.shape,D1_val.shape,E1_val.shape,F1_val.shape)





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

# Step 1: Concatenate A and B horizontally to form the top half
top_half_val = np.concatenate((C1_val, D1_val), axis=2)

# Step 2: Concatenate C and D horizontally to form the bottom half
bottom_half_val = np.concatenate((E1_val, F1_val), axis=2)

# Step 3: Concatenate the top and bottom halves vertically to form the final square matrix
final_matrix_val = np.concatenate((top_half_val, bottom_half_val), axis=1)

# Step 4: Split the final matrix into top and bottom rectangular matrices
top_rectangular_val = final_matrix_val[:, :200, :]
bottom_rectangular_val = final_matrix_val[:, 200:, :]

# Print shapes to verify
print("Final square matrix shape:", final_matrix_val.shape)
print("Top rectangular matrix shape:", top_rectangular_val.shape)
print("Bottom rectangular matrix shape:", bottom_rectangular_val.shape)




np.save(r'E:\ABIDE_RESULTS - Copy\data\Spatial data\Validation\bottomx_validation_mixed.npy',bottom_rectangular_val)




import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
import os
import random
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
bottom_rectangular_val = np.load(r'E:\ABIDE_RESULTS - Copy\data\Spatial data\Validation\bottomx_validation_mixed.npy')
# -------------------------------------------------------
# Prepare Data
# -------------------------------------------------------
A = bottom_rectangular[..., np.newaxis].astype(np.float32)
XX = bottom_rectangular_val[..., np.newaxis].astype(np.float32)
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
features_val = feature_extractor.predict(XX)

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

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

best_seed = 69472
np.random.seed(best_seed)
torch.manual_seed(best_seed)
torch.cuda.manual_seed_all(best_seed)

# ------------------------------------------------------------
# Parameter combinations
# ------------------------------------------------------------
Ng_list = [32, 64, 128]
Nk_list = [16, 32, 64]

results = []

for Ng in Ng_list:
    for Nk in Nk_list:

        print("=" * 60)
        print(f"Testing Ng={Ng}, Nk={Nk}")
        print("=" * 60)

        # ------------------------------------------------------------
        # Hydra Features
        # ------------------------------------------------------------
        transform = HydraMultivariate(
            X.shape[-1],
            Ng,
            Nk
        )

        X_train = torch.tensor(X).float()
        X_test = torch.tensor(X_val).float()

        X_train_tr = transform(X_train)
        X_test_tr = transform(X_test)

        # ------------------------------------------------------------
        # Scaling
        # ------------------------------------------------------------
        scaler = SparseScaler()

        X_train_tr = scaler.fit_transform(X_train_tr)
        X_test_tr = scaler.transform(X_test_tr)

        # ------------------------------------------------------------
        # Ridge Regression
        # ------------------------------------------------------------
        regressor = RidgeCV(alphas=np.logspace(-3, 3, 10))
        regressor.fit(X_train_tr, Y)

        pred = regressor.predict(X_test_tr)

        # ------------------------------------------------------------
        # Metrics
        # ------------------------------------------------------------
        r, _ = pearsonr(pred, Y_val)
        mae = mean_absolute_error(Y_val, pred)
        rmse = np.sqrt(mean_squared_error(Y_val, pred))

        results.append([Ng, Nk, r, mae, rmse])

        print(f"Correlation : {r:.4f}")
        print(f"MAE         : {mae:.4f}")
        print(f"RMSE        : {rmse:.4f}")

# ------------------------------------------------------------
# Final Summary
# ------------------------------------------------------------
print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"{'Ng':>6} {'Nk':>6} {'Correlation':>15} {'MAE':>12} {'RMSE':>12}")

for row in results:
    print(f"{row[0]:>6} {row[1]:>6} {row[2]:>15.4f} {row[3]:>12.4f} {row[4]:>12.4f}")

# Best configuration
best = max(results, key=lambda x: x[2])

print("\nBest Configuration")
print(f"Ng = {best[0]}")
print(f"Nk = {best[1]}")
print(f"Correlation = {best[2]:.4f}")
print(f"MAE = {best[3]:.4f}")
print(f"RMSE = {best[4]:.4f}")




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Input
import random
import os


X_val = np.load(r'E:\ABIDE_RESULTS - Copy\bottomx_validation_mixed.npy')
Y_val = np.load(r'E:\ABIDE_RESULTS - Copy\bottom_validation_mixed.npy')
# Define the CNN model
model = Sequential([
    Conv2D(512, (3, 3), activation='relu', input_shape=(200, 400,1)),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)  # Single output for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Print the model summary
model.summary()

# Train the model
model.fit(bottom_rectangular, Y, epochs=10, batch_size=16)

# Note: Adjust the number of epochs and batch size as needed

p = model.predict(X_val)
p = p.reshape(22)
correlation = np.corrcoef(p,Y_val)
print(correlation)




import numpy as np
import torch

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

# ------------------------------------------------------------
# Seed
# ------------------------------------------------------------
best_seed = 69472

np.random.seed(best_seed)
torch.manual_seed(best_seed)
torch.cuda.manual_seed_all(best_seed)

# ------------------------------------------------------------
# Hydra Features
# ------------------------------------------------------------
transform = HydraMultivariate(X.shape[-1], 116,k=32,g=16)

X_train = torch.tensor(X).float()
X_test = torch.tensor(X_val).float()

X_train_tr = transform(X_train)
X_test_tr = transform(X_test)

# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------
scaler = SparseScaler()

X_train_tr = scaler.fit_transform(X_train_tr)
X_test_tr = scaler.transform(X_test_tr)

# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
models = {
    "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 10)),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=best_seed,
        n_jobs=-1
    ),

    "SVR": SVR(
    kernel='poly',
    C=50.0),
    
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.0001,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=best_seed
    )
}

results = []

# ------------------------------------------------------------
# Train and Evaluate
# ------------------------------------------------------------
for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model.fit(X_train_tr, Y)

    pred = model.predict(X_test_tr)

    r, _ = pearsonr(pred, Y_val)
    mae = mean_absolute_error(Y_val, pred)
    rmse = np.sqrt(mean_squared_error(Y_val, pred))

    results.append([name, np.abs(r), mae, rmse])

    print(f"Correlation : {np.abs(r):.4f}")
 

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n")
print("="*70)
print("FINAL RESULTS")
print("="*70)

print(f"{'Model':<20}{'Correlation':>15}{'MAE':>12}{'RMSE':>12}")

for name, r, mae, rmse in results:
    print(f"{name:<20}{r:>15.4f}{mae:>12.4f}{rmse:>12.4f}")

best = max(results, key=lambda x: x[1])

print("\nBest Model")
print(f"Model       : {best[0]}")
print(f"Correlation : {best[1]:.4f}")





import numpy as np
import torch

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

# ------------------------------------------------------------
# Seed
# ------------------------------------------------------------
best_seed = 69472

np.random.seed(best_seed)
torch.manual_seed(best_seed)
torch.cuda.manual_seed_all(best_seed)

# ------------------------------------------------------------
# Hydra Features
# ------------------------------------------------------------
transform = HydraMultivariate(X.shape[-1], 116,k=32,g=16)

X_train = torch.tensor(X).float()
X_test = torch.tensor(X_val).float()

X_train_tr = transform(X_train)
X_test_tr = transform(X_test)

# ------------------------------------------------------------
# Scaling
# ------------------------------------------------------------
scaler = SparseScaler()

X_train_tr = scaler.fit_transform(X_train_tr)
X_test_tr = scaler.transform(X_test_tr)
X1 = np.load(r'E:\ABIDE_RESULTS - Copy\data\Spatial data\Validation\bottomx_validation_mixed.npy')
X_train_tr11 = np.concatenate((X_train_tr,features_train),axis=1)
X_test_tr11 = np.concatenate((X_test_tr,features_val),axis=1)
# ------------------------------------------------------------
# Models
# ------------------------------------------------------------
models = {
    "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 10)),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=best_seed,
        n_jobs=-1
    ),

    "SVR": SVR(
    kernel='poly',
    C=50.0),
    
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.0001,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=best_seed
    )
}

results = []

# ------------------------------------------------------------
# Train and Evaluate
# ------------------------------------------------------------
for name, model in models.items():

    print("\n" + "="*60)
    print(name)
    print("="*60)

    model.fit(X_train_tr11, Y)

    pred = model.predict(X_test_tr11)

    r, _ = pearsonr(pred, Y_val)
    mae = mean_absolute_error(Y_val, pred)
    rmse = np.sqrt(mean_squared_error(Y_val, pred))

    results.append([name, np.abs(r), mae, rmse])

    print(f"Correlation : {np.abs(r):.4f}")
 

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------
print("\n")
print("="*70)
print("FINAL RESULTS")
print("="*70)

print(f"{'Model':<20}{'Correlation':>15}{'MAE':>12}{'RMSE':>12}")

for name, r, mae, rmse in results:
    print(f"{name:<20}{r:>15.4f}{mae:>12.4f}{rmse:>12.4f}")

best = max(results, key=lambda x: x[1])

print("\nBest Model")
print(f"Model       : {best[0]}")
print(f"Correlation : {best[1]:.4f}")





