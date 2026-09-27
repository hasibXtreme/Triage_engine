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
