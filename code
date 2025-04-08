import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set the style for our plots
sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Load the dataset
df = pd.read_csv('TMDB_movie_dataset_v11 new.csv')

# Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'], format='%m/%d/%Y', errors='coerce')

# Extract year from release_date
df['release_year'] = df['release_date'].dt.year

# Drop rows with NaN values
df = df.dropna()

# Create decade column
df['decade'] = (df['release_year'] // 10) * 10

# Filter to remove outliers or erroneous data (movies with budgets less than $100,000)
df = df[df['budget'] >= 100000]

# Analysis by decade
decade_budgets = df.groupby('decade')['budget'].agg(['mean', 'median', 'count']).reset_index()
decade_budgets['mean'] = decade_budgets['mean'] / 1000000  # Convert to millions
decade_budgets['median'] = decade_budgets['median'] / 1000000  # Convert to millions

# Plot 1: Budget trends by decade
plt.figure(figsize=(14, 8))
bar_width = 4
plt.bar(decade_budgets['decade'] - bar_width/2, decade_budgets['mean'], 
        width=bar_width, label='Mean Budget', color='skyblue', alpha=0.7)
plt.bar(decade_budgets['decade'] + bar_width/2, decade_budgets['median'], 
        width=bar_width, label='Median Budget', color='salmon', alpha=0.7)

plt.xlabel('Decade', fontsize=14)
plt.ylabel('Budget (Millions $)', fontsize=14)
plt.title('Movie Budget Trends by Decade', fontsize=16)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(decade_budgets['decade'])
plt.savefig('budget_trends_by_decade.png')
plt.show()

# Plot 2: Distribution of movie budgets
plt.figure(figsize=(14, 8))
sns.histplot(df['budget'] / 1000000, bins=30, kde=True)
plt.xlabel('Budget (Millions $)', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.title('Distribution of Movie Budgets', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('budget_distribution.png')
plt.show()

# Plot 3: Budget trends over time (scatterplot)
plt.figure(figsize=(14, 8))
plt.scatter(df['release_year'], df['budget'] / 1000000, alpha=0.5, color='purple')
plt.xlabel('Release Year', fontsize=14)
plt.ylabel('Budget (Millions $)', fontsize=14)
plt.title('Movie Budgets Over Time', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('budget_scatter_over_time.png')
plt.show()

# Plot 4: Monthly distribution of movie releases
df['release_month'] = df['release_date'].dt.month
monthly_counts = df.groupby('release_month').size()

plt.figure(figsize=(14, 8))
sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='viridis')
plt.xlabel('Month', fontsize=14)
plt.ylabel('Number of Movies Released', fontsize=14)
plt.title('Movie Releases by Month', fontsize=16)
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('releases_by_month.png')
plt.show()

# Plot 5: Budget comparison: Box plot by decade
plt.figure(figsize=(14, 8))
sns.boxplot(x='decade', y=df['budget'] / 1000000, data=df, palette='Set3')
plt.xlabel('Decade', fontsize=14)
plt.ylabel('Budget (Millions $)', fontsize=14)
plt.title('Distribution of Movie Budgets by Decade', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('budget_boxplot_by_decade.png')
plt.show()

# Print summary statistics
print("\nSummary Statistics of Movie Budgets:")
budget_stats = df['budget'].describe() / 1000000
print(budget_stats)

print("\nAverage Budget by Decade (Millions $):")
print(decade_budgets[['decade', 'mean', 'count']])