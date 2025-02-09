import pandas as pd

# Load the original CSV file
data = pd.read_csv("globalterrorismdb_0718dist.csv")

# Sample 50% of the data randomly
sampled_data = data.sample(frac=0.10, random_state=42)  # 30% of the data

# Save the sampled data to a new CSV file
sampled_data.to_csv("globalterrorismdb_50_percent.csv", index=False)

print("Sampled CSV file with 50% data created successfully!")
