from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from utils.data_loader import load_jsonl
from utils.tokenizer_utils import get_tokenizer, tokenize_data
import torch

# 1. Load and tokenize your training data
data = load_jsonl("data/training_data.jsonl")

# Use Falcon tokenizer directly (override get_tokenizer)
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
tokenizer.pad_token = tokenizer.eos_token  # Falcon requires explicit padding

# Tokenize with updated tokenizer
tokenized_data = tokenize_data(data, tokenizer)

# 2. Wrap in a PyTorch Dataset class
class InspectionDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'].squeeze(),
            'attention_mask': self.data[idx]['attention_mask'].squeeze(),
            'labels': self.data[idx]['input_ids'].squeeze()
        }

dataset = InspectionDataset(tokenized_data)

# 3. Load Falcon model
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
model.resize_token_embeddings(len(tokenizer))  # Align with tokenizer

# 4. Set training parameters
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="epoch",
    evaluation_strategy="no",
    warmup_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=False  # Set to True if using a GPU with mixed precision
)

# 5. Set up Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 6. Train
trainer.train()

# 7. Save model + tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
