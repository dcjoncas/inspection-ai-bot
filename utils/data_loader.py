import json
from pathlib import Path

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# Test (absolute path resolves based on script location)
if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent  # Takes you to /chat_model_training
    file_path = base_path / "data" / "training_data.jsonl"

    print(f"Loading from: {file_path}")
    sample = load_jsonl(file_path)

    for item in sample:
        print(item)
