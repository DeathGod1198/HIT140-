
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the datasets
file_path_1 = 'dataset1.csv'
file_path_2 = 'dataset2.csv'
file_path_3 = 'dataset3.csv'

# Reading the CSV files
dataset1 = pd.read_csv(file_path_1)
dataset2 = pd.read_csv(file_path_2)
dataset3 = pd.read_csv(file_path_3)

# Merging the datasets on the 'ID' column
merged_df = pd.merge(pd.merge(dataset1, dataset2, on='ID'), dataset3, on='ID')

# Calculating the total daily screen time for weekdays and weekends
merged_df['total_screen_time_wk'] = merged_df[['C_wk', 'G_wk', 'S_wk', 'T_wk']].sum(axis=1)
merged_df['total_screen_time_we'] = merged_df[['C_we', 'G_we', 'S_we', 'T_we']].sum(axis=1)

# Creating a new column for total weekly screen time (weekdays + weekends)
merged_df['total_screen_time'] = merged_df['total_screen_time_wk'] * 5 + merged_df['total_screen_time_we'] * 2

# Defining a threshold for high and low screen time (e.g., median)
threshold = merged_df['total_screen_time'].median()

# Splitting the dataset into high and low screen time groups
high_screen_time = merged_df[merged_df['total_screen_time'] > threshold]
low_screen_time = merged_df[merged_df['total_screen_time'] <= threshold]

# Calculating the mean well-being score for each group
high_screen_wellbeing = high_screen_time.iloc[:, 15:26].mean(axis=1)
low_screen_wellbeing = low_screen_time.iloc[:, 15:26].mean(axis=1)

# Performing an independent T-test
t_stat, p_value = ttest_ind(high_screen_wellbeing, low_screen_wellbeing)

# Linear Regression: Predicting well-being scores based on screen time
# Preparing the independent variables (screen time features)
X = merged_df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']]
# Dependent variable (average well-being score)
y = merged_df.iloc[:, 15:26].mean(axis=1)

# Creating and fitting the linear regression model
model = LinearRegression().fit(X, y)
regression_coefficients = model.coef_
regression_intercept = model.intercept_
r_squared = model.score(X, y)

# Output results
print("T-test Results:")
print(f"T-test Statistic: {t_stat}")
print(f"P-value: {p_value}\n")

print("Linear Regression Results:")
print(f"Regression Coefficients: {regression_coefficients}")
print(f"Regression Intercept: {regression_intercept}")
print(f"R-squared: {r_squared}")

# Visualize Results

# Plot for T-test results - Well-being Scores Distribution
plt.figure(figsize=(10, 6))
sns.histplot(high_screen_wellbeing, color='red', label='High Screen Time', kde=True)
sns.histplot(low_screen_wellbeing, color='blue', label='Low Screen Time', kde=True)
plt.title('Distribution of Well-being Scores by Screen Time Groups')
plt.xlabel('Average Well-being Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Correlation Heatmap for Screen Time and Well-being Indicators
plt.figure(figsize=(10, 8))
correlation_matrix = merged_df[['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk'] + list(merged_df.columns[15:26])].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap: Screen Time and Well-being Indicators')
plt.show()

