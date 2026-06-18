import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

# PROJECT ROOT

BASE_DIR = Path(
__file__
).resolve().parent.parent


# DATASET LOCATION

DATASET_PATH = (

BASE_DIR /

"dataset" /

"pcod.csv"

)


# LOAD DATASET

df = pd.read_csv(
DATASET_PATH
)

# Show size
print("Dataset Shape:")
print(df.shape)

# Show columns
print("\nColumns:")
print(df.columns)

# Show missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove IDs
df.drop(
[
'Sl. No',
'Patient File No.'
],
axis=1,
inplace=True
)

print("\nShape After Removing IDs:")
print(df.shape)

# REMOVE EMPTY COLUMN

df.drop(
columns=["Unnamed: 44"],
inplace=True
)

print("\nShape after removing empty column:")
print(df.shape)

# HANDLE MISSING VALUES

# Numeric column
df["Marraige Status (Yrs)"] = (df["Marraige Status (Yrs)"].fillna(df["Marraige Status (Yrs)"].mean()))

# Categorical column
df["Fast food (Y/N)"] = (df["Fast food (Y/N)"].fillna(df["Fast food (Y/N)"].mode()[0]))

print("\nMissing values after cleaning:")

print(df.isnull().sum())    

# REMOVE DUPLICATES

before = len(df)

df.drop_duplicates(
inplace=True
)

after = len(df)

print("\nDuplicates Removed:")
print(before - after)

print("\nDataset Shape After Duplicate Removal:")
print(df.shape)

# FEATURES & TARGET

X = df.drop("PCOS (Y/N)",axis=1)

y = df["PCOS (Y/N)"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# FIX DATA TYPES

X = X.apply(
pd.to_numeric,
errors="coerce"
)

# Fill any values generated during conversion
X = X.fillna(
X.mean()
)

print("\nMissing After Conversion:")

print(
X.isnull().sum().sum()
)

# FEATURE SCALING

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nScaled Dataset Shape:")

print(X_scaled.shape)

# NORMALIZATION

normalizer = MinMaxScaler()

X_normalized = normalizer.fit_transform(X)

print("\nNormalized Dataset Shape:")

print(X_normalized.shape)

#encoding check
print("\nCategorical Columns:")

print(
X.select_dtypes(
include="object"
).columns
)

# TRAIN TEST SPLIT

X_train,\
X_test,\
y_train,\
y_test = (

train_test_split(
X_scaled,
y,
test_size=0.20,
random_state=42,
stratify=y
)

)

print("\nTraining Features:")

print(
X_train.shape
)

print("\nTesting Features:")

print(
X_test.shape
)

print("\nTraining Labels:")

print(
y_train.shape
)

print("\nTesting Labels:")

print(
y_test.shape
)

