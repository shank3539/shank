# pyrefly: ignore [missing-import]
from numpy.lib._function_base_impl import median
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("understanding the problem data")

file_name = "sales_data.csv" 
if not os.path.exists(file_name):
    print(f"Error: File '{file_name}' not found.")
    exit()

df = pd.read_csv(file_name)
print("successful loaded")
print(f"shape of the dataset:Row:{df.shape[0]},columns:{df.shape[1]}")  

print(df.head())
print(df.info())
print(df.describe())

print("handling misiing values")

print(df.isnull().sum())

#with using meadian 

median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)

median_spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_spending)
print(median_spending)

plt.figure(figsize=(7, 4))
df['Spending'].hist(bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Spending')
plt.xlabel('Spending Amount')
plt.ylabel('Number of Customers')

#Correlaion Matrix

correlation=df.corr(numeric_only=True)
print(correlation)

plt.show()

print('Plotting Correlation Heatmap')
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

