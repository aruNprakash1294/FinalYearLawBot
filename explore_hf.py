from datasets import load_dataset

# Load only a small slice so it's light on your system
ds = load_dataset("ninadn/indian-legal", split="train[:200]")

print("Number of examples loaded:", len(ds))
print("Column names:", ds.column_names)
print("\n--- Example row ---")
print(ds[0])
