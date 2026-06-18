# PCOS Prediction System using Machine Learning

A Machine Learning based PCOS (Polycystic Ovarian Syndrome) Prediction System developed to predict the likelihood of PCOS using patient medical data and multiple classification algorithms.

The system performs complete data preprocessing, model training, evaluation, prediction, and visualization.

---

# Project Objective

Develop a PCOS Prediction System that:

- Processes and cleans patient medical data
- Handles missing and inconsistent values
- Applies scaling and normalization
- Trains multiple machine learning models
- Evaluates model performance
- Selects and saves the best model
- Predicts PCOS likelihood from new patient inputs
- Visualizes model performance

---

# Dataset Information

Dataset Type:
PCOS Medical Dataset

Dataset Shape:

541 Rows × 45 Columns

Target Variable:

PCOS (Y/N)

Final Features Used:

41 Features

---

# Project Structure

```plaintext
PCOS_PREDICTION_SYSTEM/

dataset/
│
└── pcos.csv

models/
│
├── pcos_model.pkl
├── scaler.pkl
└── normalizer.pkl

src/
│
├── preprocessing.py
├── train.py
├── predict.py
└── visualize.py

README.md
requirements.txt
```

---

# Features Implemented

## Data Processing Module

✔ Handle missing values

✔ Remove duplicates

✔ Remove unnecessary columns

✔ Feature scaling using StandardScaler

✔ Feature normalization using MinMaxScaler

✔ Data type conversion

✔ Train/Test split

---

## Machine Learning Module

Models Used:

- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Random Forest | 91.74% | 96.55% | 77.78% | 86.15% |
| Logistic Regression | 88.07% | 81.08% | 83.33% | 82.19% |
| SVM | 88.99% | 92.86% | 72.22% | 81.25% |
| KNN | 87.16% | 92.31% | 66.67% | 77.42% |

Best Model:

Random Forest

---

# Prediction Module

Load saved model:

```python
pcos_model.pkl
```

Input:

Patient medical information

Output:

```plaintext
Prediction:
PCOS Detected / No PCOS Detected

Confidence:
XX %
```

---

# Visualization

Implemented:

- Accuracy Comparison Chart
- Confusion Matrix Visualization

---

# Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Joblib

VS Code

---

# Installation

Clone repository:

```bash
git clone <repository-url>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python src/preprocessing.py
```

Train models:

```bash
python src/train.py
```

Run prediction:

```bash
python src/predict.py
```

Run visualization:

```bash
python src/visualize.py
```

---

# Future Improvements

- Streamlit Web Interface
- Hyperparameter Optimization
- Cloud Deployment
- Additional Medical Features

---

# Author

Madhav V Menon

Data Science and AI Project
