# investigation1_analysis.py

import pandas as pd
import statsmodels.api as sm

def perform_regression_analysis(data):
    # Define dependent and independent variables
    X = data['Total_Screen_Time']
    y = data['Avg_Well_Being_Score']

    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    return model

if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data = pd.read_csv('preprocessed_data.csv')

    # Perform regression analysis
    model = perform_regression_analysis(preprocessed_data)

    # Display results
    print(model.summary())
