import csv
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(" Device:", DEVICE)

SEEDS        = [1, 7, 21, 42, 100]
EPOCHS       = 15
N_TEST_ATTACK = 6000
EPSILONS     = [0.1, 0.3, 0.5]


# CLASS 1 – IoTDatasetBuilder
# Responsible for: raw CSV reading, block detection, label assignment

class IoTDatasetBuilder:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._df = None
        self._block_lengths = []

    def _parse_blocks(self):
        rows, block_lengths, current = [], [], 0

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader)
            expected_cols = len(header)

            for row in reader:
                if row == header:
                    block_lengths.append(current)
                    current = 0
                    continue

                if len(row) == expected_cols:
                    rows.append(row)
                    current += 1

        block_lengths.append(current)
        return rows, header, block_lengths

    def _clean_numeric(self, df):
        df = df.apply(pd.to_numeric, errors="coerce")

        for col in df.columns:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)

        df.replace([np.inf, -np.inf], [1e9, -1e9], inplace=True)
        return df

    def _split_attack_benign(self, df_all, block_lengths):
        if len(block_lengths) == 2:
            b1, b2 = block_lengths

            attack_df = df_all.iloc[:b1].copy()
            benign_df = df_all.iloc[b1:b1 + b2].copy()

            print(" Detected 2-block structure: [Attack, Benign]")

        elif len(block_lengths) == 3:
            b1, b2, b3 = block_lengths

            attack_df = pd.concat([
                df_all.iloc[:b1],
                df_all.iloc[b1:b1 + b2]
            ], axis=0).copy()

            benign_df = df_all.iloc[b1 + b2:b1 + b2 + b3].copy()

            print(" Detected 3-block structure: [Attack, Attack, Benign]")

        else:
            raise ValueError(
                f"Unsupported block structure: {block_lengths}. "
                "Expected either [attack, benign] or [attack, attack, benign]."
            )

        attack_df["label"] = 1
        benign_df["label"] = 0

        return attack_df, benign_df

    def load(self):
        rows, header, block_lengths = self._parse_blocks()

        df_all = pd.DataFrame(rows, columns=header)
        df_all = self._clean_numeric(df_all)

        attack_df, benign_df = self._split_attack_benign(df_all, block_lengths)

        df = pd.concat([attack_df, benign_df], axis=0)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        self._df = df
        self._block_lengths = block_lengths

        print(" Dataset loaded")
        print(f"   Blocks detected : {block_lengths}")
        print(f"   Attack samples  : {len(attack_df)}")
        print(f"   Benign samples  : {len(benign_df)}")
        print(f"   Final shape     : {df.shape}")
        print(df["label"].value_counts().rename({
            0: "Benign",
            1: "Gafgyt Attack"
        }))

        return df, block_lengths

    @property
    def dataframe(self):
        if self._df is None:
            raise RuntimeError("Call .load() first.")
        return self._df

# CLASS 2 – DataPreprocessor
# Responsible for: train/test splitting, scaling strategy, class-weight calc,
#                  outlier removal, optional sub-sampling

class DataPreprocessor:

    SCALER_MAP = {
        "standard": StandardScaler,
        "robust":   RobustScaler,
        "minmax":   MinMaxScaler,
    }

    def __init__(self,
                 scaler_type: str = "standard",
                 test_size: float = 0.20,
                 remove_outliers: bool = False):
        self.scaler_type = scaler_type
        self.test_size   = test_size
        self.remove_outliers = remove_outliers

        self.scaler = self.SCALER_MAP[scaler_type]()
        self.feature_min = None
        self.feature_max = None
        self._X_train_raw = None

    # Internal helpers

    def _clip_outliers_iqr(self, X: np.ndarray, factor: float = 3.0) -> np.ndarray:
        """Clip feature values beyond factor*IQR from the median (per feature)."""
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr = q75 - q25
        lower = q25 - factor * iqr
        upper = q75 + factor * iqr
        return np.clip(X, lower, upper)

    def _compute_class_weight(self, y: np.ndarray) -> float:
        """Return scale_pos_weight = neg_count / pos_count (for XGBoost)."""
        neg = np.sum(y == 0)
        pos = np.sum(y == 1)
        return float(neg / pos) if pos > 0 else 1.0

    def _feature_bounds(self, X_scaled: np.ndarray):
        """Per-feature min/max on scaled training data (for adversarial clamping)."""
        return X_scaled.min(axis=0), X_scaled.max(axis=0)

    def _validate_split(self, y_train, y_test):
        """Sanity-check that both classes appear in train & test."""
        for split_name, y_split in [("train", y_train), ("test", y_test)]:
            unique = np.unique(y_split)
            if len(unique) < 2:
                raise ValueError(f"Split '{split_name}' has only classes: {unique}. "
                                 "Check class balance or reduce test_size.")

    # Public API

    def split_and_scale(self, X: np.ndarray, y: np.ndarray, seed: int = 42):
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=seed
        )
        self._validate_split(y_tr, y_te)

        if self.remove_outliers:
            X_tr_raw = self._clip_outliers_iqr(X_tr_raw)

        X_tr = self.scaler.fit_transform(X_tr_raw)
        X_te = self.scaler.transform(X_te_raw)

        self._X_train_raw = X_tr_raw
        self.feature_min, self.feature_max = self._feature_bounds(X_tr)
        self.scale_pos_weight = self._compute_class_weight(y_tr)

        return X_tr, X_te, y_tr, y_te

    def get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """sklearn-style per-sample weight array (balanced)."""
        return compute_sample_weight("balanced", y)

    def inverse_scale(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(X_scaled)


# CLASS 3 – FeatureEngineer
# Responsible for: statistical aggregates, frequency domain, entropy features

class FeatureEngineer:

    EPS = 1e-9

    def __init__(self, base_names: list):
        self.base_names = base_names
        self._extra_names = []

    # Feature group methods

    def _central_tendency(self, X):
        return {
            "row_mean":   X.mean(axis=1),
            "row_median": np.median(X, axis=1),
            "row_q25":    np.percentile(X, 25, axis=1),
            "row_q75":    np.percentile(X, 75, axis=1),
        }

    def _dispersion(self, X, mean, std, median):
        var    = X.var(axis=1)
        xmin   = X.min(axis=1)
        xmax   = X.max(axis=1)
        frange = xmax - xmin
        iqr    = np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1)
        mad    = np.mean(np.abs(X - median[:, None]), axis=1)
        cv     = std / (np.abs(mean) + self.EPS)
        return {
            "row_std":   std,
            "row_var":   var,
            "row_min":   xmin,
            "row_max":   xmax,
            "row_range": frange,
            "row_iqr":   iqr,
            "row_mad":   mad,
            "row_cv":    cv,
        }

    def _shape(self, X, mean, std):
        centered = X - mean[:, None]
        skew = np.mean(centered ** 3, axis=1) / (std ** 3 + self.EPS)
        kurt = np.mean(centered ** 4, axis=1) / (std ** 4 + self.EPS)
        return {"row_skew": skew, "row_kurtosis": kurt}

    def _norms(self, X):
        l1     = np.sum(np.abs(X), axis=1)
        l2     = np.sqrt(np.sum(X ** 2, axis=1))
        energy = np.sum(X ** 2, axis=1)
        rms    = np.sqrt(np.mean(X ** 2, axis=1))
        return {"row_l1": l1, "row_l2": l2, "row_energy": energy, "row_rms": rms}

    def _tail_stats(self, X, k=10):
        k = min(k, X.shape[1])
        sorted_abs = np.sort(np.abs(X), axis=1)
        topk = sorted_abs[:, -k:]
        return {
            "top10_abs_mean": topk.mean(axis=1),
            "top10_abs_std":  topk.std(axis=1),
            "top10_abs_max":  topk.max(axis=1),
        }

    def _spectral_proxy(self, X):
        """Split features into 3 equal bands; sum squared values as proxy energy."""
        n = X.shape[1]
        b = n // 3
        low  = np.sum(X[:, :b] ** 2, axis=1)
        mid  = np.sum(X[:, b:2*b] ** 2, axis=1)
        high = np.sum(X[:, 2*b:] ** 2, axis=1)
        total = low + mid + high + self.EPS
        return {
            "spectral_low_ratio":  low  / total,
            "spectral_mid_ratio":  mid  / total,
            "spectral_high_ratio": high / total,
        }

    def _entropy_approx(self, X, bins=16):
        """Per-row approximate Shannon entropy via histogram over feature values."""
        out = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            hist, _ = np.histogram(X[i], bins=bins, density=True)
            hist = hist + self.EPS
            hist = hist / hist.sum()
            out[i] = -np.sum(hist * np.log2(hist))
        return {"row_entropy": out}

    def _sparsity(self, X):
        return {"row_sparsity": np.mean(np.abs(X) < self.EPS, axis=1)}

    # Public API

    def transform(self, X: np.ndarray) -> np.ndarray:
        mean   = X.mean(axis=1)
        std    = X.std(axis=1)
        median = np.median(X, axis=1)

        groups = {}
        groups.update(self._central_tendency(X))
        groups.update(self._dispersion(X, mean, std, median))
        groups.update(self._shape(X, mean, std))
        groups.update(self._norms(X))
        groups.update(self._tail_stats(X))
        groups.update(self._spectral_proxy(X))
        groups.update(self._entropy_approx(X))
        groups.update(self._sparsity(X))

        extra = np.vstack(list(groups.values())).T
        self._extra_names = list(groups.keys())

        return np.hstack([X, extra])

    def names(self) -> list:
        return self.base_names + self._extra_names


# CLASS 4 – EnhancedMLP
# Responsible for: deep binary classifier with BN, dropout, residual-style path

class ResidualBlock(nn.Module):
    """Mini residual block: Linear → BN → ReLU → Linear → BN, with skip."""
    def __init__(self, dim, dropout=0.20):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class EnhancedMLP(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.entry = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.30),
        )

        self.down1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.res1 = ResidualBlock(256, dropout=0.25)

        self.down2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.res2 = ResidualBlock(128, dropout=0.20)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.down1(x)
        x = self.res1(x)
        x = self.down2(x)
        x = self.res2(x)
        return self.head(x).squeeze(1)

    # Training utilities

    def fit(self, X_train, y_train, X_val, y_val,
            seed=42, epochs=15, lr=1e-3, batch_size=512):
        torch.manual_seed(seed)
        self.to(DEVICE)

        ds     = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                               torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_f1, best_state = -1.0, None

        for epoch in range(epochs):
            self.train()
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                nn.BCEWithLogitsLoss()(self(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()

            # validate
            preds = self.predict(X_val)
            val_f1 = f1_score(y_val, preds, average="weighted", zero_division=0)
            if val_f1 > best_f1:
                best_f1   = val_f1
                best_state = copy.deepcopy(self.state_dict())

        if best_state is not None:
            self.load_state_dict(best_state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x      = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            logits = self(x)
            probs  = torch.sigmoid(logits)
            return (probs >= 0.5).long().cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x    = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            prob = torch.sigmoid(self(x)).cpu().numpy()
            return np.stack([1 - prob, prob], axis=1)

# CLASS 5 – ModelEnsemble
# Responsible for: wrapping RF / XGB / MLP under unified predict interface,
#                  soft-vote ensemble, per-model importance

class ModelEnsemble:

    def __init__(self, input_dim: int, scale_pos_weight: float = 1.0, seed: int = 42):
        self.seed = seed
        self.input_dim = input_dim

        self.rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        )

        self.xgb = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            verbosity=0,
            use_label_encoder=False
        )

        self.mlp = EnhancedMLP(input_dim)

        self._models = {
            "Random Forest": self.rf,
            "XGBoost":       self.xgb,
            "Enhanced MLP":  self.mlp,
        }

        self._is_fitted = False

    # Internal helpers

    def _predict_single(self, name: str, model, X: np.ndarray) -> np.ndarray:
        if name == "Enhanced MLP":
            return model.predict(X)
        return model.predict(X)

    def _proba_single(self, name: str, model, X: np.ndarray) -> np.ndarray:
        if name == "Enhanced MLP":
            return model.predict_proba(X)
        return model.predict_proba(X)

    def _feature_importance_rf(self, feature_names: list) -> pd.DataFrame:
        imp = self.rf.feature_importances_
        return (pd.DataFrame({"feature": feature_names, "importance": imp})
                .sort_values("importance", ascending=False)
                .head(20))

    # Public API

    def fit(self, X_train, y_train, X_val, y_val, epochs=15):
        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        self.mlp.fit(X_train, y_train, X_val, y_val,
                     seed=self.seed, epochs=epochs)
        self._is_fitted = True
        return self

    def predict(self, name: str, X: np.ndarray) -> np.ndarray:
        return self._predict_single(name, self._models[name], X)

    def predict_all(self, X: np.ndarray) -> dict:
        return {n: self._predict_single(n, m, X) for n, m in self._models.items()}

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Soft-vote across all three models."""
        probas = np.mean(
            [self._proba_single(n, m, X)[:, 1] for n, m in self._models.items()],
            axis=0
        )
        return (probas >= 0.5).astype(int)

    def model_names(self) -> list:
        return list(self._models.keys())

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        return self._feature_importance_rf(feature_names)


# CLASS 6 – AdversarialAttacker
# Responsible for: FGSM, PGD, CW-like soft-label attack generation

class AdversarialAttacker:

    def __init__(self, mlp_model: EnhancedMLP,
                 feature_min: np.ndarray,
                 feature_max: np.ndarray):
        self.model       = mlp_model
        self.feature_min = torch.tensor(feature_min, dtype=torch.float32).to(DEVICE)
        self.feature_max = torch.tensor(feature_max, dtype=torch.float32).to(DEVICE)

    # Internal helpers

    def _clamp(self, x):
        return torch.max(torch.min(x, self.feature_max), self.feature_min)

    def _to_tensor(self, X, y):
        return (torch.tensor(X, dtype=torch.float32).to(DEVICE),
                torch.tensor(y, dtype=torch.float32).to(DEVICE))

    def _loss(self, x_adv, y_t):
        return nn.BCEWithLogitsLoss()(self.model(x_adv), y_t)

    # Attack methods

    def fgsm(self, X: np.ndarray, y: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Single-step FGSM adversarial example."""
        self.model.eval()
        x, y_t = self._to_tensor(X, y)
        x_adv  = x.clone().detach().requires_grad_(True)

        loss = self._loss(x_adv, y_t)
        self.model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.sign()
        x_adv     = self._clamp(x_adv.detach() + epsilon * grad_sign)
        return x_adv.cpu().numpy()

    def pgd(self, X: np.ndarray, y: np.ndarray,
            epsilon: float = 0.1, alpha: float = 0.01,
            steps: int = 30, restarts: int = 1) -> np.ndarray:
        """
        Multi-step PGD with optional random restarts.
        BUG FIX: properly detach between iterations to avoid stale gradients.
        """
        self.model.eval()
        x_orig, y_t = self._to_tensor(X, y)

        best_loss   = torch.full((X.shape[0],), -1e9, device=DEVICE)
        best_x_adv  = x_orig.clone()

        for _ in range(restarts):
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
            x_adv = self._clamp(x_adv).detach()

            for _ in range(steps):
                x_adv.requires_grad_(True)

                loss_per_sample = nn.BCEWithLogitsLoss(reduction="none")(
                    self.model(x_adv), y_t
                )
                loss = loss_per_sample.sum()

                self.model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    x_adv = x_adv + alpha * x_adv.grad.sign()
                    x_adv = torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)
                    x_adv = self._clamp(x_adv)

                update_mask = loss_per_sample.detach() > best_loss
                best_loss  = torch.where(update_mask, loss_per_sample.detach(), best_loss)
                best_x_adv = torch.where(update_mask.unsqueeze(1), x_adv.detach(), best_x_adv)

        return best_x_adv.cpu().numpy()

    def noise_baseline(self, X: np.ndarray, epsilon: float = 0.1) -> np.ndarray:
        """Random Gaussian noise as a sanity-check baseline attack."""
        noise = np.random.normal(0, epsilon, X.shape)
        x_noisy = torch.tensor(X + noise, dtype=torch.float32).to(DEVICE)
        x_noisy = self._clamp(x_noisy)
        return x_noisy.cpu().numpy()


# CLASS 7 – AdversarialDefender
# Responsible for: feature squeezing, input smoothing, adversarial training

class AdversarialDefender:

    def __init__(self, squeeze_bits: int = 8, noise_sigma: float = 0.05):
        self.squeeze_bits  = squeeze_bits
        self.noise_sigma   = noise_sigma

    # Preprocessing defenses

    def feature_squeeze(self, X: np.ndarray) -> np.ndarray:
        """Reduce effective precision by rounding to squeeze_bits levels."""
        levels = 2 ** self.squeeze_bits
        X_min  = X.min(axis=0, keepdims=True)
        X_max  = X.max(axis=0, keepdims=True) + 1e-9
        X_norm = (X - X_min) / (X_max - X_min)
        X_sq   = np.round(X_norm * levels) / levels
        return X_sq * (X_max - X_min) + X_min

    def gaussian_smoothing(self, X: np.ndarray) -> np.ndarray:
        """Add calibrated Gaussian noise to smooth adversarial perturbations."""
        noise = np.random.normal(0, self.noise_sigma, X.shape)
        return X + noise

    def median_smooth(self, X: np.ndarray, kernel: int = 3) -> np.ndarray:
        """
        Apply median filter along feature axis (1D sliding window).
        Effective against high-frequency adversarial perturbations.
        """
        from scipy.ndimage import median_filter
        return median_filter(X, size=(1, kernel))

    # Training-time defense

    def augment_with_adversarial(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  attacker: "AdversarialAttacker",
                                  epsilon: float = 0.1,
                                  frac: float = 0.15) -> tuple:

        idx_attack = np.where(y_train == 1)[0]
        n_aug      = int(len(idx_attack) * frac)
        if n_aug == 0:
            return X_train, y_train

        idx_sub  = np.random.choice(idx_attack, n_aug, replace=False)
        X_sub    = X_train[idx_sub]
        y_sub    = y_train[idx_sub]
        X_adv    = attacker.fgsm(X_sub, y_sub, epsilon=epsilon)

        X_aug = np.vstack([X_train, X_adv])
        y_aug = np.concatenate([y_train, y_sub])

        shuffle_idx = np.random.permutation(len(y_aug))
        return X_aug[shuffle_idx], y_aug[shuffle_idx]

    # Detection

    def detect_adversarial(self, X_clean: np.ndarray,
                            X_query: np.ndarray,
                            threshold: float = 0.5) -> np.ndarray:
        """
        Simple perturbation magnitude detector.
        Returns boolean array: True = likely adversarial.
        """
        delta     = np.abs(X_query - X_clean)
        magnitude = np.mean(delta, axis=1)
        return magnitude > threshold


# CLASS 8 – RobustnessEvaluator
# Responsible for: all metric computation, ASR, cross-seed stability, ablation

class RobustnessEvaluator:

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            tpr = tp / (tp + fn + 1e-9)   # Recall / Detection Rate
            tnr = tn / (tn + fp + 1e-9)   # Specificity
            fpr = fp / (fp + tn + 1e-9)   # False Positive Rate
            fnr = fn / (fn + tp + 1e-9)   # False Negative Rate
        else:
            tn = fp = fn = tp = 0
            tpr = tnr = fpr = fnr = 0.0

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),

            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),

            "TPR": float(tpr),
            "TNR": float(tnr),
            "FPR": float(fpr),
            "FNR": float(fnr),
        }

    @staticmethod
    def compute_asr(y_true: np.ndarray,
                    pred_clean: np.ndarray,
                    pred_adv: np.ndarray) -> float:
        mask = (y_true == 1) & (pred_clean == 1)

        if mask.sum() == 0:
            return 0.0

        return float(np.mean(pred_adv[mask] == 0))

    @staticmethod
    def compute_robustness_score(clean_f1: float, adv_f1: float) -> float:
        if clean_f1 < 1e-9:
            return 0.0

        return adv_f1 / clean_f1

    @staticmethod
    def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            return {
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TPR": tp / (tp + fn + 1e-9),
                "TNR": tn / (tn + fp + 1e-9),
                "FPR": fp / (fp + tn + 1e-9),
                "FNR": fn / (fn + tp + 1e-9),
            }

        return {}

    def evaluate_under_attack(self,
                               ensemble: ModelEnsemble,
                               attacker: AdversarialAttacker,
                               defender: AdversarialDefender,
                               X_sub: np.ndarray,
                               y_sub: np.ndarray,
                               clean_results: dict,
                               config_name: str,
                               seed: int) -> list:
        rows = []

        attack_methods = {
            "FGSM":           lambda eps: attacker.fgsm(X_sub, y_sub, epsilon=eps),
            "PGD":            lambda eps: attacker.pgd(X_sub, y_sub, epsilon=eps,
                                                        alpha=eps/10, steps=30),
            "Noise_Baseline": lambda eps: attacker.noise_baseline(X_sub, epsilon=eps),
        }

        defense_methods = {
            "No_Defense":   lambda Xa: Xa,
            "Feat_Squeeze": lambda Xa: defender.feature_squeeze(Xa),
            "Gauss_Smooth": lambda Xa: defender.gaussian_smoothing(Xa),
        }

        for model_name in ensemble.model_names():
            pred_clean = ensemble.predict(model_name, X_sub)
            clean_m = self.compute_metrics(y_sub, pred_clean)

            for atk_name, atk_fn in attack_methods.items():
                for eps in EPSILONS:
                    X_adv_raw = atk_fn(eps)

                    for def_name, def_fn in defense_methods.items():
                        X_adv = def_fn(X_adv_raw)

                        pred_adv = ensemble.predict(model_name, X_adv)
                        adv_m = self.compute_metrics(y_sub, pred_adv)

                        asr = self.compute_asr(y_sub, pred_clean, pred_adv)
                        rob = self.compute_robustness_score(
                            clean_results[model_name]["f1"],
                            adv_m["f1"]
                        )

                        rows.append({
                            "config": config_name,
                            "seed": seed,
                            "model": model_name,
                            "attack": atk_name,
                            "defense": def_name,
                            "epsilon": eps,

                            # Clean metrics
                            "clean_accuracy": clean_m["accuracy"],
                            "clean_precision": clean_m["precision"],
                            "clean_recall": clean_m["recall"],
                            "clean_f1": clean_m["f1"],

                            # Adversarial metrics
                            "adv_accuracy": adv_m["accuracy"],
                            "adv_precision": adv_m["precision"],
                            "adv_recall": adv_m["recall"],
                            "adv_f1": adv_m["f1"],

                            # Confusion matrix under attack
                            "TP": adv_m["TP"],
                            "TN": adv_m["TN"],
                            "FP": adv_m["FP"],
                            "FN": adv_m["FN"],

                            # IDS-specific metrics under attack
                            "TPR": adv_m["TPR"],
                            "TNR": adv_m["TNR"],
                            "FPR": adv_m["FPR"],
                            "FNR": adv_m["FNR"],

                            # Robustness metrics
                            "f1_drop": clean_m["f1"] - adv_m["f1"],
                            "asr": asr,
                            "robustness_score": rob,
                        })

        return rows

    @staticmethod
    def stability_summary(results_df: pd.DataFrame) -> pd.DataFrame:
        agg_cols = {
            "clean_accuracy": ["mean", "std"],
            "clean_precision": ["mean", "std"],
            "clean_recall": ["mean", "std"],
            "clean_f1": ["mean", "std"],

            "adv_accuracy": ["mean", "std"],
            "adv_precision": ["mean", "std"],
            "adv_recall": ["mean", "std"],
            "adv_f1": ["mean", "std"],

            "TP": ["mean", "std"],
            "TN": ["mean", "std"],
            "FP": ["mean", "std"],
            "FN": ["mean", "std"],

            "TPR": ["mean", "std"],
            "TNR": ["mean", "std"],
            "FPR": ["mean", "std"],
            "FNR": ["mean", "std"],

            "f1_drop": ["mean", "std"],
            "asr": ["mean", "std"],
            "robustness_score": ["mean", "std"],
        }

        summary = (
            results_df
            .groupby(["config", "model", "attack", "defense", "epsilon"])
            .agg(agg_cols)
            .reset_index()
        )

        summary.columns = ["_".join(c).strip("_") for c in summary.columns]

        return summary

# CLASS 9 – ExperimentLogger
# Responsible for: saving CSVs, plots, .pkl / .pth artifacts

class ExperimentLogger:

    def __init__(self, output_dir: str = "."):
        import os
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _path(self, filename):
        import os
        return os.path.join(self.output_dir, filename)

    # CSV saving

    def save_results(self, df: pd.DataFrame, name: str = "robustness_full_results.csv"):
        path = self._path(name)
        df.to_csv(path, index=False)
        print(f" Saved: {path}")

    def save_summary(self, df: pd.DataFrame, name: str = "robustness_summary_mean_std.csv"):
        path = self._path(name)
        df.to_csv(path, index=False)
        print(f" Saved: {path}")

    # Plots

    def plot_pgd_f1(self, summary_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, config in zip(axes, summary_df["config"].unique()[:2]):
            data = summary_df[
                (summary_df["config"]  == config) &
                (summary_df["attack"]  == "PGD")  &
                (summary_df["defense"] == "No_Defense")
            ]
            sns.lineplot(data=data, x="epsilon", y="adv_f1_mean",
                         hue="model", marker="o", ax=ax)
            ax.set_title(f"PGD Robustness – {config}")
            ax.set_ylabel("Mean Adversarial F1")
            ax.set_xlabel("Epsilon")
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self._path("pgd_robustness_mean_f1.png"), dpi=200)
        plt.show()
        print(" Saved: pgd_robustness_mean_f1.png")

    def plot_asr(self, summary_df: pd.DataFrame):
        data = summary_df[
            (summary_df["attack"]  == "PGD") &
            (summary_df["defense"] == "No_Defense")
        ]
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=data, x="epsilon", y="asr_mean",
                     hue="model", style="config", marker="s")
        plt.title("PGD Attack Success Rate (No Defense)")
        plt.ylabel("Mean ASR")
        plt.xlabel("Epsilon")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self._path("pgd_asr_mean.png"), dpi=200)
        plt.show()
        print(" Saved: pgd_asr_mean.png")

    def plot_defense_comparison(self, summary_df: pd.DataFrame):
        """Bar chart comparing defense strategies by mean adv F1 at eps=0.3."""
        data = summary_df[
            (summary_df["attack"]  == "PGD") &
            (summary_df["epsilon"] == 0.3)   &
            (summary_df["config"]  == "Original_Plus_Engineered")
        ]
        plt.figure(figsize=(12, 6))
        sns.barplot(data=data, x="model", y="adv_f1_mean",
                    hue="defense", palette="Set2")
        plt.title("Defense Comparison: Mean Adv F1 (PGD ε=0.3)")
        plt.ylabel("Mean Adversarial F1")
        plt.xlabel("Model")
        plt.legend(title="Defense")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(self._path("defense_comparison.png"), dpi=200)
        plt.show()
        print(" Saved: defense_comparison.png")

    # Model artifact saving

    def save_model_artifacts(self, artifacts: dict, feature_names, X_sample, y_sample):
        joblib.dump(artifacts["models"]["Random Forest"], self._path("rf_model.pkl"))
        joblib.dump(artifacts["models"]["XGBoost"],       self._path("xgb_model.pkl"))
        joblib.dump(artifacts["scaler"],                  self._path("scaler.pkl"))
        joblib.dump(feature_names,                        self._path("feature_names.pkl"))
        joblib.dump(artifacts["feature_min"],             self._path("feature_min.pkl"))
        joblib.dump(artifacts["feature_max"],             self._path("feature_max.pkl"))

        torch.save(artifacts["models"]["Enhanced MLP"].state_dict(),
                   self._path("enhanced_mlp.pth"))

        demo = pd.DataFrame(X_sample, columns=feature_names)
        demo["label"] = y_sample
        demo.to_csv(self._path("demo_samples.csv"), index=False)

        print("\n Demo artifacts saved:")
        for fname in ["rf_model.pkl", "xgb_model.pkl", "enhanced_mlp.pth",
                      "scaler.pkl", "feature_names.pkl",
                      "feature_min.pkl", "feature_max.pkl", "demo_samples.csv"]:
            print(f"   {fname}")


# MAIN EXPERIMENT RUNNER

def run_experiment(X: np.ndarray,
                   y: np.ndarray,
                   feature_names: list,
                   seed: int = 42,
                   config_name: str = "Default") -> tuple:

    # --- Preprocessing ---
    preprocessor = DataPreprocessor(scaler_type="standard",
                                    test_size=0.20,
                                    remove_outliers=False)
    X_train, X_test, y_train, y_test = preprocessor.split_and_scale(X, y, seed=seed)

    # --- Train ensemble ---
    ensemble = ModelEnsemble(
        input_dim=X_train.shape[1],
        scale_pos_weight=preprocessor.scale_pos_weight,
        seed=seed
    )
    ensemble.fit(X_train, y_train, X_test, y_test, epochs=EPOCHS)

    # --- Adversarial objects ---
    attacker = AdversarialAttacker(
        mlp_model=ensemble.mlp,
        feature_min=preprocessor.feature_min,
        feature_max=preprocessor.feature_max
    )
    defender = AdversarialDefender(squeeze_bits=8, noise_sigma=0.05)

    # --- Clean evaluation ---
    evaluator    = RobustnessEvaluator()
    clean_results = {}
    for name in ensemble.model_names():
        pred = ensemble.predict(name, X_test)
        clean_results[name] = evaluator.compute_metrics(y_test, pred)

    # --- Adversarial evaluation ---
    n_sub = min(N_TEST_ATTACK, len(X_test))
    X_sub = X_test[:n_sub]
    y_sub = y_test[:n_sub]

    rows = evaluator.evaluate_under_attack(
        ensemble, attacker, defender,
        X_sub, y_sub,
        clean_results, config_name, seed
    )

    result_df = pd.DataFrame(rows)

    return result_df, clean_results, ensemble, preprocessor


# ABLATION STUDY DRIVER

# --- Load data ---
builder = IoTDatasetBuilder("combo.csv")
df, block_lengths = builder.load()

attack_df = df[df["label"] == 1]
benign_df = df[df["label"] == 0]
n_min = min(len(attack_df), len(benign_df))  # = 40,289

df = pd.concat([
    attack_df.sample(n=n_min, random_state=42),
    benign_df.sample(n=n_min, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n After balancing:")
print(df["label"].value_counts().rename({0: "Benign", 1: "Gafgyt Attack"}))

base_feature_names = [c for c in df.columns if c != "label"]
X_raw = df[base_feature_names].values
y     = df["label"].values.astype(int)
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e9, neginf=-1e9)

# --- Feature engineering ---
fe           = FeatureEngineer(base_feature_names)
X_engineered = fe.transform(X_raw)
eng_names    = fe.names()

print(f" Feature engineering: {X_raw.shape[1]} → {X_engineered.shape[1]} features")

# --- Ablation configs ---
ablation_configs = {
    "Original_115":          (X_raw,        base_feature_names),
    "Original_Plus_Engineered": (X_engineered, eng_names),
}

# --- Run ---
logger      = ExperimentLogger(".")
all_results = []
saved_artifacts = {}

for config_name, (X_cfg, names_cfg) in ablation_configs.items():
    print(f"\n{'='*50}")
    print(f"Config: {config_name}  |  Features: {X_cfg.shape[1]}")
    print(f"{'='*50}")

    for seed in SEEDS:
        print(f"  Seed = {seed} ...", end=" ")

        result_df, clean_results, ensemble, preprocessor = run_experiment(
            X_cfg, y, names_cfg,
            seed=seed,
            config_name=config_name
        )

        all_results.append(result_df)
        print("done")

        if seed == 42:
            saved_artifacts[config_name] = {
                "models": {
                    "Random Forest": ensemble.rf,
                    "XGBoost":       ensemble.xgb,
                    "Enhanced MLP":  ensemble.mlp,
                },
                "scaler":      preprocessor.scaler,
                "feature_min": preprocessor.feature_min,
                "feature_max": preprocessor.feature_max,
            }

# --- Aggregate & save ---
results_df = pd.concat(all_results, ignore_index=True)
summary_df = RobustnessEvaluator.stability_summary(results_df)

logger.save_results(results_df)
logger.save_summary(summary_df)

print("\n Summary (first rows):")
display(summary_df.head(10))

# --- Plots ---
logger.plot_pgd_f1(summary_df)
logger.plot_asr(summary_df)
logger.plot_defense_comparison(summary_df)

# --- Extra report figures

print("\n Generating extra report figures...")

# CLASS DISTRIBUTION

plt.figure(figsize=(6, 5))
label_counts = pd.Series(y).value_counts().sort_index()
plt.pie(
    label_counts.values,
    labels=["Benign", "Gafgyt Attack"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Class Distribution in N-BaIoT Dataset", fontweight="bold")
plt.tight_layout()
plt.savefig("data_distribution_pie.png", dpi=200, bbox_inches="tight")
plt.show()

print(" Saved: data_distribution_pie.png")


# 2. CLEAN F1 COMPARISON

clean_plot = (
    summary_df
    .groupby(["config", "model"])
    .agg(clean_f1_mean=("clean_f1_mean", "mean"),
         clean_f1_std=("clean_f1_std", "mean"))
    .reset_index()
)

plt.figure(figsize=(12, 6))
sns.barplot(
    data=clean_plot,
    x="model",
    y="clean_f1_mean",
    hue="config"
)
plt.title("Clean Data Performance Across Models", fontweight="bold")
plt.ylabel("Mean Clean F1-score")
plt.xlabel("Model")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("clean_f1_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

print(" Saved: clean_f1_comparison.png")

# 3. MAIN ROBUSTNESS ANALYSIS: F1 / ASR / F1 DROP

main_summary = summary_df[
    (summary_df["defense"] == "No_Defense") &
    (summary_df["attack"].isin(["FGSM", "PGD"])) &
    (summary_df["config"] == "Original_Plus_Engineered")
].copy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

fig.suptitle(
    "Robustness Analysis under FGSM and PGD Attacks",
    fontsize=14,
    fontweight="bold"
)

for row, attack_name in enumerate(["FGSM", "PGD"]):
    data = main_summary[main_summary["attack"] == attack_name]

    # F1-score
    sns.lineplot(
        data=data,
        x="epsilon",
        y="adv_f1_mean",
        hue="model",
        marker="o",
        ax=axes[row][0]
    )
    axes[row][0].set_title(f"{attack_name}: Adversarial F1-score")
    axes[row][0].set_ylabel("Mean Adv F1")
    axes[row][0].set_xlabel("Epsilon")
    axes[row][0].grid(alpha=0.3)

    # ASR
    sns.lineplot(
        data=data,
        x="epsilon",
        y="asr_mean",
        hue="model",
        marker="s",
        ax=axes[row][1],
        legend=False
    )
    axes[row][1].set_title(f"{attack_name}: Attack Success Rate")
    axes[row][1].set_ylabel("Mean ASR")
    axes[row][1].set_xlabel("Epsilon")
    axes[row][1].set_ylim(0, 1)
    axes[row][1].grid(alpha=0.3)

    # F1 Drop
    sns.lineplot(
        data=data,
        x="epsilon",
        y="f1_drop_mean",
        hue="model",
        marker="^",
        ax=axes[row][2],
        legend=False
    )
    axes[row][2].set_title(f"{attack_name}: F1-score Drop")
    axes[row][2].set_ylabel("Mean F1 Drop")
    axes[row][2].set_xlabel("Epsilon")
    axes[row][2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("robustness_analysis_full.png", dpi=200, bbox_inches="tight")
plt.show()

print(" Saved: robustness_analysis_full.png")

# 4. DEFENSE COMPARISON

defense_plot = summary_df[
    (summary_df["config"] == "Original_Plus_Engineered") &
    (summary_df["attack"] == "PGD") &
    (summary_df["epsilon"] == 0.3)
].copy()

plt.figure(figsize=(14, 6))
sns.barplot(
    data=defense_plot,
    x="model",
    y="adv_f1_mean",
    hue="defense"
)
plt.title("Defense Comparison under PGD Attack (ε = 0.3)", fontweight="bold")
plt.ylabel("Mean Adversarial F1-score")
plt.xlabel("Model")
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("defense_comparison_full.png", dpi=200, bbox_inches="tight")
plt.show()

print(" Saved: defense_comparison_full.png")

# 5. FEATURE IMPORTANCE

try:
    rf_model = saved_artifacts["Original_Plus_Engineered"]["models"]["Random Forest"]
    feat_names = saved_artifacts["Original_Plus_Engineered"].get("feature_names", eng_names)

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(
        range(len(indices)),
        [feat_names[i] for i in indices],
        rotation=60,
        ha="right",
        fontsize=8
    )
    plt.title("Top 20 Important Features from Random Forest", fontweight="bold")
    plt.ylabel("Feature Importance")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance_full.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(" Saved: feature_importance_full.png")

except Exception as e:
    print(" Feature importance plot skipped:", e)


# 6. CONFUSION MATRICES — CLEAN VS PGD

try:
    X_cfg = X_engineered
    names_cfg = eng_names

    preprocessor = DataPreprocessor(
        scaler_type="standard",
        test_size=0.20,
        remove_outliers=False
    )

    X_train_cm, X_test_cm, y_train_cm, y_test_cm = preprocessor.split_and_scale(
        X_cfg,
        y,
        seed=42
    )

    ensemble_cm = ModelEnsemble(
        input_dim=X_train_cm.shape[1],
        scale_pos_weight=preprocessor.scale_pos_weight,
        seed=42
    )

    ensemble_cm.fit(
        X_train_cm,
        y_train_cm,
        X_test_cm,
        y_test_cm,
        epochs=EPOCHS
    )

    attacker_cm = AdversarialAttacker(
        mlp_model=ensemble_cm.mlp,
        feature_min=preprocessor.feature_min,
        feature_max=preprocessor.feature_max
    )

    N_CM = min(6000, len(X_test_cm))
    X_cm = X_test_cm[:N_CM]
    y_cm = y_test_cm[:N_CM]

    X_pgd_cm = attacker_cm.pgd(
        X_cm,
        y_cm,
        epsilon=0.3,
        alpha=0.03,
        steps=30
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    fig.suptitle(
        "Confusion Matrices: Clean vs PGD Attack (ε = 0.3)",
        fontsize=14,
        fontweight="bold"
    )

    for col, model_name in enumerate(ensemble_cm.model_names()):
        pred_clean = ensemble_cm.predict(model_name, X_cm)
        pred_pgd = ensemble_cm.predict(model_name, X_pgd_cm)

        cm_clean = confusion_matrix(y_cm, pred_clean)
        cm_pgd = confusion_matrix(y_cm, pred_pgd)

        sns.heatmap(
            cm_clean,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign", "Attack"],
            yticklabels=["Benign", "Attack"],
            ax=axes[0][col]
        )
        axes[0][col].set_title(f"{model_name}\nClean")
        axes[0][col].set_xlabel("Predicted")
        axes[0][col].set_ylabel("Actual")

        sns.heatmap(
            cm_pgd,
            annot=True,
            fmt="d",
            cmap="Reds",
            cbar=False,
            xticklabels=["Benign", "Attack"],
            yticklabels=["Benign", "Attack"],
            ax=axes[1][col]
        )
        axes[1][col].set_title(f"{model_name}\nPGD Attack")
        axes[1][col].set_xlabel("Predicted")
        axes[1][col].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("confusion_matrices_full.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(" Saved: confusion_matrices_full.png")

except Exception as e:
    print(" Confusion matrix plot skipped:", e)

# 7. SAVE MAIN

main_summary.to_csv("MAIN_TABLE_chapter4.csv", index=False)
defense_plot.to_csv("DEFENSE_TABLE_chapter4.csv", index=False)

print(" Saved: MAIN_TABLE_chapter4.csv")
print(" Saved: DEFENSE_TABLE_chapter4.csv")

print("\n Extra report figures completed.")

# --- Save artifacts ---
if "Original_Plus_Engineered" in saved_artifacts:
    logger.save_model_artifacts(
        saved_artifacts["Original_Plus_Engineered"],
        eng_names,
        X_engineered[:200],
        y[:200]
    )

print("\n Pipeline complete.")