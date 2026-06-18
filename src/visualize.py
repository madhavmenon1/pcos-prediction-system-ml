import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from pathlib import Path
from preprocessing import (

X_test,

y_test

)

models = [

"Random Forest",
"Logistic Regression",
"SVM",
"KNN"

]

accuracy = [

0.917,
0.881,
0.890,
0.872

]


plt.figure(
figsize=(8,6)
)


plt.bar(
models,
accuracy
)


plt.title(
"PCOD Model Accuracy Comparison"
)


plt.ylabel(
"Accuracy"
)


plt.xlabel(
"Models"
)


plt.ylim(
0,
1
)

plt.show()

# LOAD MODEL

# PROJECT ROOT

BASE_DIR = Path(
__file__
).resolve().parent.parent


# MODEL PATH

MODEL_PATH = (

BASE_DIR /

"models" /

"pcod_model.pkl"

)


# LOAD MODEL

model = joblib.load(
MODEL_PATH
)

# GENERATE MATRIX

ConfusionMatrixDisplay.from_predictions(

y_test,

model.predict(
X_test
)

)


plt.title(
"Random Forest Confusion Matrix"
)

plt.show()
