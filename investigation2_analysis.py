# investigation2_analysis.py

import pandas as pd
from scipy.stats import ttest_ind

def perform_ttest(data):
    # Group data by gender
    group1 = data[data['gender'] == 1]['Avg_Well_Being_Score']
    group2 = data[data['gender'] == 0]['Avg_Well_Being_Score']

    # Perform t-test
    t_stat, p_val = ttest_ind(group1, group2)

    return t_stat, p_val

if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data = pd.read_csv('preprocessed_data.csv')

    # Perform T-test
    t_stat, p_val = perform_ttest(preprocessed_data)

    # Display results
    print(f'T-statistic: {t_stat}, P-value: {p_val}')
