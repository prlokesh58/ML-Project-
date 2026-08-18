import pandas as pd
from sklearn.model_selection import OrdinalEncoder

tier_order = ['Tier3', 'Tier2', 'Tier1']
cgpa_tier_order = ['Low', 'Medium', 'High']

ord_enc = OrdinalEncoder(
    categories=[tier_order,cgpa_tier_order],
    handle_unknown='use_encoded_value',unknown_value=-1,
)

ordinal_cols = ['CollegeTier','CGPA_Tier']
train_df[['CollegeTier_enc','CGPA_Tier_enc']] = ord_enc.fit_transform(
    train_df[ordinal_cols]
)
test_df[['CollegeTier_enc','CGPA_Tier_enc']] = ord_enc.fit_transform(
    test_df[ordinal_cols]
)
