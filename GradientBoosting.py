import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = r"E:\PythonProject2\placement_predict_50k Dataset (3)(in).csv"

data = pd.read_csv(FILE_PATH)

print("==========================================")
print("   GRADIENT BOOSTING - PLACEMENT STATUS")
print("==========================================")

print("\nOriginal Dataset Shape:")
print(data.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

print("\nColumns Available in Dataset:")
for column in data.columns:
    print("-", column)


# ============================================================
# 3. FIND TARGET COLUMN
# ============================================================

target_candidates = [
    "Placement Status",
    "PlacementStatus",
    "placement status",
    "placement_status",
    "Placement_Status",
    "Placement"
]

TARGET = None

for column in data.columns:
    cleaned_column = column.strip().lower().replace("_", " ")

    cleaned_candidates = [
        x.lower().replace("_", " ")
        for x in target_candidates
    ]

    if cleaned_column in cleaned_candidates:
        TARGET = column
        break


# If target is not found, stop with useful message
if TARGET is None:
    print("\nERROR: Could not find the Placement Status column.")
    print("\nPlease check the column names printed above.")
    print("\nAvailable columns:")
    print(data.columns.tolist())
    exit()


print("\nTarget Column Found:")
print(TARGET)


# ============================================================
# 4. REMOVE UNWANTED COLUMNS
# ============================================================

REMOVE_COLUMNS = [
    "Student ID",
    TARGET,
    "Salary Package",
    "IsAnomoly"
]

# Only remove columns that actually exist
columns_to_remove = [
    column for column in REMOVE_COLUMNS
    if column in data.columns
]

X = data.drop(columns=columns_to_remove)

y = data[TARGET]


# ============================================================
# 5. DISPLAY TARGET INFORMATION
# ============================================================

print("\n==========================================")
print("TARGET VARIABLE")
print("==========================================")

print("Target:", TARGET)

print("\nTarget Values:")
print(y.value_counts())

print("\nNumber of Classes:")
print(y.nunique())


# ============================================================
# 6. IDENTIFY FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\n==========================================")
print("FEATURE INFORMATION")
print("==========================================")

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 7. PREPROCESSING
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n==========================================")
print("TRAIN-TEST SPLIT")
print("==========================================")

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 9. GRADIENT BOOSTING CLASSIFIER
# ============================================================

gradient_boosting = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)


# ============================================================
# 10. CREATE PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", gradient_boosting)
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n==========================================")
print("TRAINING GRADIENT BOOSTING MODEL")
print("==========================================")

model.fit(X_train, y_train)

print("Training completed successfully!")


# ============================================================
# 12. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 13. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL ACCURACY")
print("==========================================")

print("Accuracy:", accuracy)
print("Accuracy Percentage: {:.2f}%".format(accuracy * 100))


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(classification_report(y_test, y_pred))


# ============================================================
# 16. SAMPLE PREDICTIONS
# ============================================================

print("\n==========================================")
print("SAMPLE PREDICTIONS")
print("==========================================")

results = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

print(results)


# ============================================================
# 17. COMPLETED
# ============================================================

print("\n==========================================")
print("GRADIENT BOOSTING MODEL EXECUTION COMPLETED")
print("==========================================")