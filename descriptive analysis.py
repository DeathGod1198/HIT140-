import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Combining all datasets by 'ID'
combined_dataset = pd.merge(pd.merge(dataset1, dataset2, on='ID'), dataset3, on='ID')

# 1. Correlation Heatmap for All Factors
plt.figure(figsize=(20, 16))
correlation_matrix = combined_dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap: Combined Socio-Demographic, Behavioral, and Academic Data')
plt.show()

# 2. Box Plot: Academic Performance (C_we) by Gender
plt.figure(figsize=(12, 6))
if 'C_we' in dataset1.columns:
    sns.boxplot(x='gender', y='C_we', data=dataset1)
else:
    sns.boxplot(x='gender', y='C_we', data=combined_dataset)
plt.title('Box Plot: Academic Performance (C_we) by Gender')
plt.xlabel('Gender')
plt.ylabel('C_we (Academic Performance)')
plt.show()

# 3. Define the column names you want to plot
columns_to_plot = ['C_we', 'G_we', ...]  # add more column names as needed

for col in columns_to_plot:
    if col in dataset1.columns:
        sns.boxplot(x='gender', y=col, data=dataset1)
    else:
        sns.boxplot(x='gender', y=col, data=combined_dataset)
    plt.title(f'Box Plot: Academic Performance ({col}) by Gender')
    plt.xlabel('Gender')
    plt.ylabel(f'{col} (Academic Performance)')
    plt.show()

# 4. Check if the column 'C_we' exists in any of the datasets
if 'C_we' in dataset1.columns:
    data_to_use = dataset1
elif 'C_we' in dataset2.columns:
    data_to_use = dataset2
elif 'C_we' in dataset3.columns:
    data_to_use = dataset3
elif 'C_we' in combined_dataset.columns:
    data_to_use = combined_dataset
else:
    print("Column 'C_we' does not exist in any of the datasets")
    # If the column doesn't exist, you can't create a box plot for it
    # You may want to add some error handling or logging here
    
# 5. Scatter Plot: Optimism vs. Academic Performance (C_we)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Optm', y='C_we', data=combined_dataset)
plt.title('Scatter Plot: Optimism vs. Academic Performance (C_we)')
plt.xlabel('Optimism')
plt.ylabel('C_we (Academic Performance)')
plt.show()

# 6. Scatter Plot: Confidence vs. Academic Performance (C_we)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Conf', y='C_we', data=combined_dataset)
plt.title('Scatter Plot: Confidence vs. Academic Performance (C_we)')
plt.xlabel('Confidence')
plt.ylabel('C_we (Academic Performance)')
plt.show()

# 7. 3D Scatter Plot: Optimism, Confidence, and Academic Performance (C_we)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(combined_dataset['Optm'], combined_dataset['Conf'], combined_dataset['C_we'], c='r', marker='o')

ax.set_xlabel('Optimism')
ax.set_ylabel('Confidence')
ax.set_zlabel('C_we (Academic Performance)')

plt.title('3D Scatter Plot: Optimism, Confidence, and Academic Performance')
plt.show()

# 8. Scatter Plot: Usage of Technology vs. Academic Performance (C_we)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Usef', y='C_we', data=combined_dataset)
plt.title('Scatter Plot: Usage of Technology vs. Academic Performance (C_we)')
plt.xlabel('Usage of Technology')
plt.ylabel('C_we (Academic Performance)')
plt.show()