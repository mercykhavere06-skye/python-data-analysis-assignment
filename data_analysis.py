# =========================================================
# Task 1: Load and Explore the Dataset
# =========================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset using a try-except block for error handling
try:
    iris = load_iris(as_frame=True)
    df = iris.frame
    print("Dataset loaded successfully!")
    
    # FIX: Add a 'species' column to the DataFrame from the target data
    df['species'] = iris.target_names[df['target']]

except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows to inspect the data, including the new 'species' column
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Explore the structure and check for missing values
print("\nDataset structure and info:")
df.info()

# =========================================================
# Task 2: Basic Data Analysis
# =========================================================

# Compute basic statistics
print("\nBasic statistics of the numerical columns:")
print(df.describe())

# Group by species and find the mean of sepal length
print("\nMean of sepal length for each species:")
print(df.groupby('species')['sepal length (cm)'].mean())

# =========================================================
# Task 3: Data Visualization
# =========================================================

# Set a stylish plotting theme using seaborn
sns.set_theme(style="whitegrid")

# Create a figure to hold all plots
plt.figure(figsize=(15, 12))

# 1. Histogram (for distribution)
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='petal length (cm)', bins=20, kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# 2. Scatter Plot (for relationship between two variables)
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

# 3. Bar Chart (for comparing values across categories)
# Create a summary for the bar chart
avg_petal_width = df.groupby('species')['petal width (cm)'].mean().reset_index()
plt.subplot(2, 2, 3)
sns.barplot(data=avg_petal_width, x='species', y='petal width (cm)')
plt.title('Average Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Width (cm)')
plt.xticks(ticks=[0, 1, 2], labels=['Setosa', 'Versicolor', 'Virginica'])

# 4. Line Chart (for a trend over time, a general example)
plt.subplot(2, 2, 4)
plt.plot(df.index, df['sepal width (cm)'], color='teal')
plt.title('Sepal Width Over Index (Example of a Line Plot)')
plt.xlabel('Data Point Index')
plt.ylabel('Sepal Width (cm)')

plt.tight_layout()
plt.show()