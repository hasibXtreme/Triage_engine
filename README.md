# Triage_engine

# Mixed Naïve Bayes Classifier for Medical Triage

A custom, high-performance **Mixed Naïve Bayes Classifier** implemented from scratch using **NumPy** and **Pandas**. This model is designed specifically for clinical decision-support tasks, combining **Gaussian Probability Density Functions** for continuous vital signs and **Categorical Likelihood Tables with Laplace Smoothing** for discrete clinical symptoms.

---

## Model Architecture & Mathematical Foundations

Standard Naïve Bayes implementations often force continuous bell curves on discrete binary or multiclass variables. This custom model partitions features based on their statistical distributions:

$$\text{Score}(c) = \log P(Y=c) + \sum_{i \in \text{Cont}} \log P_{\text{Gaussian}}(X_i \mid Y=c) + \sum_{j \in \text{Cat}} \log P_{\text{Categorical}}(X_j \mid Y=c)$$

### 1. Continuous Features (Gaussian Naïve Bayes)
For continuous physiological vitals ($X_i$), the likelihood is computed using the Gaussian Probability Density Function (PDF):

$$P(X_i = x \mid Y = c) = \frac{1}{\sqrt{2\pi\sigma_{c,i}^2}} \exp\left(-\frac{(x - \mu_{c,i})^2}{2\sigma_{c,i}^2}\right)$$

* **Parameters:** Class-specific mean ($\mu_{c,i}$) and variance ($\sigma_{c,i}^2$).
* **Variance Smoothing:** An $\epsilon = 10^{-9}$ threshold is added to calculated variances to prevent division-by-zero errors.

### 2. Categorical Features (Laplace-Smoothed Categorical Naïve Bayes)
For binary or multiclass discrete variables ($X_j$) with $K$ unique categories, likelihoods are estimated via empirical frequency ratios with Laplace smoothing ($\alpha = 1.0$):

$$P(X_j = k \mid Y = c) = \frac{\text{Count}(X_j = k \text{ in class } c) + \alpha}{N_c + \alpha \cdot K}$$

---

## Dataset Breakdown (`heart.csv`)

The model processes all 13 clinical features from the UCI / Kaggle Heart Attack Analysis Dataset:

| Feature Name | Category | Type | Description / Units |
| :--- | :--- | :--- | :--- |
| `trtbps` | Continuous | Floating/Int | Resting blood pressure ($\text{mmHg}$) |
| `chol` | Continuous | Floating/Int | Serum cholesterol ($\text{mg/dl}$) |
| `thalachh` | Continuous | Floating/Int | Maximum heart rate achieved ($\text{bpm}$) |
| `oldpeak` | Continuous | Floating | ST depression induced by exercise |
| `age` | Continuous | Integer | Patient age in years |
| `sex` | Categorical | Binary ($0/1$) | $0$: Female, $1$: Male |
| `cp` | Categorical | Multiclass ($0-3$) | Chest Pain Type ($0$: Typical, $1$: Atypical, $2$: Non-anginal, $3$: Asymptomatic) |
| `fbs` | Categorical | Binary ($0/1$) | Fasting blood sugar $> 120 \text{ mg/dl}$ |
| `restecg` | Categorical | Multiclass ($0-2$) | Resting ECG results ($0$: Normal, $1$: Abnormality, $2$: Hypertrophy) |
| `exng` | Categorical | Binary ($0/1$) | Exercise-induced angina |
| `slp` | Categorical | Multiclass ($0-2$) | Slope of peak exercise ST segment |
| `caa` | Categorical | Multiclass ($0-4$) | Number of major vessels colored by fluoroscopy |
| `thall` | Categorical | Multiclass ($0-3$) | Thallium stress test result |

---

## Model Code (`mixed_naive_bayes.py`)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CompleteMixedNaiveBayes:
    """
    A custom Naïve Bayes implementation combining Gaussian distributions 
    for continuous features and Categorical distributions with Laplace smoothing 
    for discrete/multiclass features.
    """
    def __init__(self, laplace_alpha=1.0):
        self.alpha = laplace_alpha

    def fit(self, X_cont, X_cat, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples = len(y)

        # 1. Class Priors: P(Y = c)
        self.priors = np.array([np.sum(y == c) / float(n_samples) for c in self.classes])

        # 2. Gaussian Parameters (Mean & Variance) for Continuous Features
        self.means = np.zeros((n_classes, X_cont.shape[1]))
        self.variances = np.zeros((n_classes, X_cont.shape[1]))
        for idx, c in enumerate(self.classes):
            X_c = X_cont[y == c]
            self.means[idx, :] = X_c.mean(axis=0)
            self.variances[idx, :] = X_c.var(axis=0) + 1e-9  # Variance smoothing

        # 3. Categorical Probability Tables with Laplace Smoothing
        self.cat_probs = []
        for feat_idx in range(X_cat.shape[1]):
            feat_col = X_cat[:, feat_idx]
            categories = np.unique(feat_col)
            prob_dict = {}

            for idx, c in enumerate(self.classes):
                feat_c = feat_col[y == c]
                total_count_c = len(feat_c)
                num_categories = len(categories)
                prob_dict[c] = {}

                for cat in categories:
                    count = np.sum(feat_c == cat)
                    # Laplace formula: (Count + alpha) / (N_c + alpha * |Categories|)
                    prob_dict[c][cat] = (count + self.alpha) / (total_count_c + self.alpha * num_categories)

            self.cat_probs.append(prob_dict)

    def predict(self, X_cont, X_cat):
        predictions = []
        for i in range(len(X_cont)):
            x_cont_i = X_cont[i]
            x_cat_i = X_cat[i]

            log_posteriors = []
            for idx, c in enumerate(self.classes):
                # Start with log prior
                log_score = np.log(self.priors[idx])

                # Log-Gaussian PDF for Continuous Features
                mean = self.means[idx]
                var = self.variances[idx]
                log_gaussian_pdf = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((x_cont_i - mean) ** 2) / var)
                log_score += log_gaussian_pdf

                # Discrete Log-Likelihoods for Categorical Features
                for feat_idx, val in enumerate(x_cat_i):
                    prob = self.cat_probs[feat_idx][c].get(val, 1e-6)
                    log_score += np.log(prob)

                log_posteriors.append(log_score)

            predictions.append(self.classes[np.argmax(log_posteriors)])

        return np.array(predictions)
