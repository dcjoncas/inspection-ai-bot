# Tokenizer-related utilities
from transformers import AutoTokenizer

def get_tokenizer(model_name="distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 models have no pad_token by default
    return tokenizer

def tokenize_data(data, tokenizer, max_length=512):
    tokenized = []
    for item in data:
        combined = f"### PROMPT:\n{item['prompt']}\n\n### RESPONSE:\n{item['response']}"
        encoded = tokenizer(
            combined,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized.append(encoded)
    return tokenized
