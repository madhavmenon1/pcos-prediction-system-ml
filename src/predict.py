import joblib
import numpy as np
import pandas as pd
from pathlib import Path
# LOAD SAVED OBJECTS

# PROJECT ROOT

BASE_DIR = Path(
__file__
).resolve().parent.parent


# MODEL PATHS

MODEL_PATH = (
BASE_DIR /
"models" /
"pcos_model.pkl"
)

SCALER_PATH = (
BASE_DIR /
"models" /
"scaler.pkl"
)

NORMALIZER_PATH = (
BASE_DIR /
"models" /
"normalizer.pkl"
)


# LOAD OBJECTS

model = joblib.load(
MODEL_PATH
)

scaler = joblib.load(
SCALER_PATH
)

normalizer = joblib.load(
NORMALIZER_PATH
)

print("Objects Loaded")

# LOAD FEATURE ORDER

DATASET_PATH = (

BASE_DIR /

"dataset" /

"pcos.csv"

)


feature_columns = pd.read_csv(
DATASET_PATH
)

feature_columns = (
feature_columns
.drop(
[
'Sl. No',
'Patient File No.',
'PCOS (Y/N)',
'Unnamed: 44'
],
axis=1
)
.columns
)


# SAMPLE INPUT

sample = pd.DataFrame(

[[

28,
65,
160,
25,
13,
72,
20,
11,
2,
5,
0,
0,
1.99,
1.99,
5,
4,
1.2,
36,
32,
0.89,
2.5,
7,
10,
40,
3,
90,
0.7,
100,
1,
1,
0,
0,
1,
1,
120,
80,
12,
10,
18,
17,
9

]],

columns=feature_columns

)

# APPLY PREPROCESSING

# APPLY SCALING

sample_scaled = pd.DataFrame(

scaler.transform(sample),

columns=sample.columns

)


# APPLY NORMALIZATION

sample_normalized = pd.DataFrame(

normalizer.transform(
sample_scaled
),

columns=sample.columns

)

prediction = model.predict(
sample_scaled.values
)

probability = model.predict_proba(
sample_scaled.values
)

print("\nPrediction Result:")

if prediction[0] == 1:

    print(
        "PCOS Detected"
    )

else:

    print(
        "No PCOS Detected"
    )


print("\nConfidence:")

print(
round(
max(
probability[0]
)*100,
2
),
"%"
)
