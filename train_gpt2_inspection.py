from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="training_corpus.txt",
    block_size=128
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

# Train and save
trainer.train()
trainer.save_model("./model")
tokenizer.save_pretrained("./model")