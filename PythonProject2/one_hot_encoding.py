import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv(r"E:\PythonProject2\placement_predict_50k Dataset (3)(in).csv")

print("Dataset shape:", df.shape)
print("\nColums:")
print(df.columns.tolist())


nominal_cols = [
    "Gender",
    "City",
    "Stream",
    "Specialisation",
    "Hostel",
    "HistoryOfBacklogs"
]

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["PlacementStatus"])
print("\nTraining Data Shape:", train_df.shape)
print("\nTesting Data Shape:", test_df.shape)

ohe = OneHotEncoder(
    drop="first", sparse_output=False, handle_unknown="ignore"
)
train_ohe = ohe.fit_transform(train_df[nominal_cols])
test_ohe = ohe.transform(test_df[nominal_cols])
ohe_cols = ohe.get_feature_names_out()

train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_cols, index=train_df.index)
test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_cols, index=test_df.index)
print("\nNumber of original categories:",
      len(ohe_cols))

print("\nNumber of generated encoded columns:",
      len(ohe_cols))

print("\nGenerated One-Hot Columns:")
for col in ohe_cols:
    print(col)

print("\nFirst 5 rows of encoded training data:")
print(train_ohe_df.head())

print("\nFirst 5 rows of encoded testing data:")
print(test_ohe_df.head())