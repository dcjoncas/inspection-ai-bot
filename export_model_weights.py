from safetensors.torch import load_file
import pandas as pd
import os

MODEL_PATH = "./model/checkpoint-10"
EXPORT_PATH = "./model_weights_csv"
os.makedirs(EXPORT_PATH, exist_ok=True)

print("Loading model (SafeTensors)...")
weights = load_file(os.path.join(MODEL_PATH, "model.safetensors"))

count = 0
max_tensors = 10

for key, tensor in weights.items():
    if tensor.dim() <= 2:
        df = pd.DataFrame(tensor.numpy())
        csv_file = os.path.join(EXPORT_PATH, f"{key.replace('.', '_')}.csv")
        df.to_csv(csv_file, index=False)
        print(f"Saved: {csv_file}")
        count += 1
    if count >= max_tensors:
        break

print("✅ Export complete.")
