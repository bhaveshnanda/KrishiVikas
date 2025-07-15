import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Display the first few rows of the dataset
print(df.head())

# Pair plot to visualize feature relationships
sns.pairplot(df, hue='label', diag_kind='kde', palette='husl')
plt.show()

# Heatmap to show the correlation between numerical features
plt.figure(figsize=(10, 8))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Plot distribution of each numerical feature
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

plt.figure(figsize=(14, 12))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[feature], kde=True, color='blue')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Boxplot to visualize outliers for each feature
plt.figure(figsize=(14, 12))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[feature], color='green')
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()
