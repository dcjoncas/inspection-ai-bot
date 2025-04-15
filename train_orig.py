# Training script goes here
from utils.data_loader import load_jsonl
from utils.tokenizer_utils import get_tokenizer, tokenize_data

# Load training data
data = load_jsonl("data/training_data.jsonl")

# Load tokenizer
tokenizer = get_tokenizer()

# Tokenize data
tokenized_dataset = tokenize_data(data, tokenizer)

# Print one example to verify
print("Sample Encoded Input IDs:")
print(tokenized_dataset[0]['input_ids'])
