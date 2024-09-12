# data_preprocessing.py

import pandas as pd

def load_data():
    # Load datasets
    demographics = pd.read_csv('dataset1.csv')
    screen_time = pd.read_csv('dataset2.csv')
    well_being = pd.read_csv('dataset3.csv')
    return demographics, screen_time, well_being

def merge_data(demographics, screen_time, well_being):
    # Merge datasets on ID
    merged_df = demographics.merge(screen_time, on='ID').merge(well_being, on='ID')
    return merged_df

def preprocess_data(df):
    # Calculate total screen time for each respondent
    df['Total_Screen_Time'] = df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']].sum(axis=1)

    # Calculate average well-being score for each respondent
    df['Avg_Well_Being_Score'] = df[['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thkclr', 'Goodme', 
                                     'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']].mean(axis=1)

    # Handle missing values
    df.dropna(inplace=True)

    return df

if __name__ == "__main__":
    # Load data
    demographics, screen_time, well_being = load_data()

    # Merge datasets
    merged_df = merge_data(demographics, screen_time, well_being)

    # Preprocess data
    preprocessed_df = preprocess_data(merged_df)

    # Save the preprocessed data to a CSV file for further analysis
    preprocessed_df.to_csv('preprocessed_data.csv', index=False)
