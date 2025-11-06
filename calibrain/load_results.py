import pandas as pd

# === Load the CSV file ===
file_name = "benchmark_results_20251030_200932"

file_path = f"/Users/orabe/0.braindata/CaliBrain/results/benchmark_results/{file_name}.csv"
df = pd.read_csv(file_path)

# === Keep only key columns ===
key_cols = ["solver", "noise_type", "alpha_SNR", "gamma", "max_iter", "nnz", ]
df = df[key_cols]

# === Sort by key columns ===
df_sorted = df.sort_values(by=key_cols, ascending=True)

# === Save to new CSV ===
df_sorted.to_csv(
    f"/Users/orabe/0.braindata/CaliBrain/results/benchmark_results/benchmark_results_sorted_{file_name}.csv",
    index=False
)

# === Display first rows ===
print(df_sorted.head())