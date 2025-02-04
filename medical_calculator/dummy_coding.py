import pandas as pd
import numpy as np

def dummy_code_data(query_df):

    # concatenate current query df and original one to perform dummy coding, otherwise not all categories will be seen by the dummy coding mdoel
    original_filepath = '/Users/mirandazheng/Desktop/first/ensure/medical_calculator/pseudo_data_for_dc.xlsx'
    original_df = pd.read_excel(original_filepath)
    combined_df = pd.concat([original_df, query_df], ignore_index=True)


    for col in combined_df.columns:
        # Replace 'unknown' and 'Unknown' with np.nan
        combined_df[col] = combined_df[col].replace({'unknown': np.nan, 'Unknown': np.nan}, inplace=False)

    # Save unique categories from the original dataset for dummy coding
    categorical_vars = ['Clinical T stage', 'Clinical N stage',  'Clinical Differentiation', 'Tumour site', 'Pathologic T stage', 'Pathologic N stage', 'Differentiation', 'Treatment protocol', 'Operation type', 'Approach']

    # Applying pd.get_dummies() to these variables
    dummies = pd.get_dummies(combined_df, columns=categorical_vars, drop_first=True, dummy_na=False)

    # Reintroduce NaN where the original data was missing
    for column in categorical_vars:
        # Mask where original column was NaN
        mask = combined_df[column].isnull()
        # Columns to be adjusted (all dummy columns generated from the original column)
        dummy_cols = [col for col in dummies.columns if col.startswith(column + '_')]
        dummies[dummy_cols] = dummies[dummy_cols].astype(float)  # Convert to float to avoid warning
        # Apply NaN where the original data was NaN
        dummies.loc[mask, dummy_cols] = np.nan

    # Select only the single row (last row in the combined dataset)
    query_df = dummies.iloc[[-1]]

    return query_df

