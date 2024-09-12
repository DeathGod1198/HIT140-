import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Load your datasets (update the file paths accordingly)
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Investigation 1: Distribution of Minority and Deprivation Status (Dataset 1)

# Frequency distribution
minority_distribution = dataset1['minority'].value_counts()
deprived_distribution = dataset1['deprived'].value_counts()

# Cross-tabulation between minority and deprived status
minority_deprived_crosstab = pd.crosstab(dataset1['minority'], dataset1['deprived'])

# Investigation 2: Analysis of Weekend and Weekday Performance Metrics (Dataset 2)

# Calculating mean and standard deviation for weekend and weekday metrics
weekend_mean_std = dataset2[['C_we', 'G_we', 'S_we', 'T_we']].describe().loc[['mean', 'std']]
weekday_mean_std = dataset2[['C_wk', 'G_wk', 'S_wk', 'T_wk']].describe().loc[['mean', 'std']]

# Output the mean and standard deviation for weekend and weekday metrics
print("Weekend Mean and Std:\n", weekend_mean_std)
print("Weekday Mean and Std:\n", weekday_mean_std)

# Output the frequency distributions and cross-tabulation for minority and deprived status
print("Minority Distribution:\n", minority_distribution)
print("Deprived Distribution:\n", deprived_distribution)
print("Cross-tabulation between Minority and Deprived Status:\n", minority_deprived_crosstab)

# Investigation 3: Analysis of Well-Being Indicators (Dataset 3)

# Descriptive Statistics for all columns in Dataset 3
dataset3_summary_full = dataset3.describe()

# Exclude the ID column
dataset3_no_id = dataset3.drop(columns=['ID'])  # Replace 'ID' with the exact name of the ID column if needed

# Descriptive Statistics for all columns (excluding ID) in Dataset 3
dataset3_summary_full = dataset3_no_id.describe()

# Output the full descriptive statistics for all columns (excluding ID)
print("Full Descriptive Statistics for Dataset 3 (excluding ID):\n", dataset3_summary_full)

# Plotting bar charts for minority and deprived distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

minority_distribution.plot(kind='bar', ax=axes[0], color='aquamarine')
axes[0].set_title('Minority Status Distribution')

deprived_distribution.plot(kind='bar', ax=axes[1], color='khaki')
axes[1].set_title('Deprived Status Distribution')

plt.tight_layout()
plt.show()

# Plotting box plots to visualize weekend vs weekday performance metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

dataset2[['C_we', 'C_wk']].boxplot(ax=axes[0, 0])
axes[0, 0].set_title('C_we vs C_wk')

dataset2[['G_we', 'G_wk']].boxplot(ax=axes[0, 1])
axes[0, 1].set_title('G_we vs G_wk')

dataset2[['S_we', 'S_wk']].boxplot(ax=axes[1, 0])
axes[1, 0].set_title('S_we vs S_wk')

dataset2[['T_we', 'T_wk']].boxplot(ax=axes[1, 1])
axes[1, 1].set_title('T_we vs T_wk')

plt.tight_layout()
plt.show()

# Plotting histograms for all numeric columns in Dataset 3
dataset3_no_id.hist(bins=20, figsize=(16, 12), color='peru')
plt.suptitle("Histograms for all numeric variables in Dataset 3 (excluding ID)", fontsize=16)
plt.tight_layout()
plt.show()

# Box plot for all numeric variables
dataset3_no_id.plot(kind='box', figsize=(12, 8), title="Boxplot for all numeric variables in Dataset 3 (excluding ID)")
plt.tight_layout()
plt.show()
