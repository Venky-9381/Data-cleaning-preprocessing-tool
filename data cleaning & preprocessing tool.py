import pandas as pd
import numpy as np

def clean_data(file_path, output_path):
    df = pd.read_csv(file_path)

    # Drop rows with missing values
    df_cleaned = df.dropna()

    # Fill missing values using appropriate statistical methods
    df_filled = df.copy()
    if 'column_name1' in df.columns:
        df_filled['column_name1'].fillna(df_filled['column_name1'].mean(), inplace=True)
    if 'column_name2' in df.columns:
        df_filled['column_name2'].fillna(df_filled['column_name2'].median(), inplace=True)
    if 'column_name3' in df.columns:
        df_filled['column_name3'].fillna(df_filled['column_name3'].mode()[0] if not df_filled['column_name3'].mode().empty else np.nan, inplace=True)

    # Remove duplicate rows
    df_no_duplicates = df_filled.drop_duplicates()

    # Remove outliers using IQR method
    if 'column_name' in df.columns:
        Q1 = df_no_duplicates['column_name'].quantile(0.25)
        Q3 = df_no_duplicates['column_name'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_outliers_removed = df_no_duplicates[
            (df_no_duplicates['column_name'] >= lower_bound) &
            (df_no_duplicates['column_name'] <= upper_bound)
        ]
    else:
        df_outliers_removed = df_no_duplicates

    # Save the cleaned data to CSV
    df_outliers_removed.to_csv(output_path, index=False)