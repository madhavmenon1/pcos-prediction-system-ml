from preprocessing import (
X_train,
X_test,
y_train,
y_test,
scaler,
normalizer
)

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
confusion_matrix
)

# MACHINE LEARNING
# TRAIN MODELS

models = {

"Random Forest":
RandomForestClassifier(
random_state=42
),

"Logistic Regression":
LogisticRegression(
max_iter=5000
),

"SVM":
SVC(
probability=True
),

"KNN":
KNeighborsClassifier()

}


trained_models = {}


for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    print(f"\n{name} trained successfully")
    

# MODEL EVALUATION

scores = {}

for name, model in trained_models.items():

    pred = model.predict(
        X_test
    )

    scores[name] = {

        "Accuracy":
        accuracy_score(
            y_test,
            pred
        ),

        "Precision":
        precision_score(
            y_test,
            pred
        ),

        "Recall":
        recall_score(
            y_test,
            pred
        ),

        "F1":
        f1_score(
            y_test,
            pred
        )

    }

    print(f"\n{name}")

    print(scores[name])

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            pred
        )
    )

results = (
pd.DataFrame(scores)
.T
)

print(
"\nModel Summary:"
)

print(results)

# SAVE BEST MODEL

best_model_name = max(
scores,
key=lambda x:
scores[x]["F1"]
)

best_model = trained_models[
best_model_name
]

print(
"\nSelected Best Model:",
best_model_name
)

# PROJECT ROOT

BASE_DIR = Path(
__file__
).resolve().parent.parent


# MODEL DIRECTORY

MODEL_DIR = (
BASE_DIR /
"models"
)

MODEL_DIR.mkdir(
exist_ok=True
)


# SAVE FILES

joblib.dump(

best_model,

MODEL_DIR /

"pcos_model.pkl"

)

joblib.dump(

scaler,

MODEL_DIR /

"scaler.pkl"

)

joblib.dump(

normalizer,

MODEL_DIR /

"normalizer.pkl"

)

print("\nModel saved successfully")
print("Scaler saved successfully")
print("Normalizer saved successfully")
