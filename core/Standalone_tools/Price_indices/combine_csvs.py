import pandas as pd
from pathlib import Path

# Define the directory and CSV files
price_indices_dir = Path(__file__).parent
csv_files = [
    "PCU326199326199_glass.csv",
    "PCU335991335991P_carbon.csv",
    "WPU101704_steel.csv",
    "WPU10260314_copper.csv"
]

# Read all CSV files
dfs = []
for csv_file in csv_files:
    file_path = price_indices_dir / csv_file
    df = pd.read_csv(file_path)
    # Convert observation_date to datetime for proper sorting
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    dfs.append(df)

# Merge all dataframes on observation_date
combined_df = dfs[0]
for df in dfs[1:]:
    combined_df = combined_df.merge(df, on='observation_date', how='outer')

# Sort by observation_date
combined_df = combined_df.sort_values('observation_date').reset_index(drop=True)

# Save to combined CSV
output_file = price_indices_dir / "combined_price_indices.csv"
combined_df.to_csv(output_file, index=False)

print(f"Combined CSV created successfully: {output_file}")
print(f"Shape: {combined_df.shape}")
print(f"\nFirst few rows:")
print(combined_df.head())
print(f"\nDate range: {combined_df['observation_date'].min()} to {combined_df['observation_date'].max()}")
