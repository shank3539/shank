import pandas as pd

# Load dataset
df = pd.read_csv("winequality-red.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns)

# Sort wines by quality and alcohol
top_wines = df.sort_values(
    by=["quality", "alcohol"],
    ascending=[False, False]
)

# Top 10 wines
top10 = top_wines.head(10)

print("\nTop 10 Highest Quality Wines\n")

print(top10)

# Save results
top10.to_csv("Top10_Red_Wines.csv", index=False)

print("\nSaved as Top10_Red_Wines.csv")