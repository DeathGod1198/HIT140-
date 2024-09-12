# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_screen_time_vs_wellbeing(data):
    plt.figure(figsize=(8, 6))
    sns.regplot(x='Total_Screen_Time', y='Avg_Well_Being_Score', data=data, scatter_kws={'alpha':0.3}, line_kws={'color': 'red'})
    plt.title('Total Screen Time vs. Average Well-being Score')
    plt.xlabel('Total Screen Time (hours per day)')
    plt.ylabel('Average Well-being Score')
    plt.show()

def plot_wellbeing_by_gender(data):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='gender', y='Avg_Well_Being_Score', data=data)
    plt.title('Well-being Scores by Gender')
    plt.xlabel('Gender (1 = Male, 0 = Female)')
    plt.ylabel('Average Well-being Score')
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data = pd.read_csv('preprocessed_data.csv')

    # Plot total screen time vs well-being
    plot_screen_time_vs_wellbeing(preprocessed_data)

    # Plot well-being scores by gender
    plot_wellbeing_by_gender(preprocessed_data)
