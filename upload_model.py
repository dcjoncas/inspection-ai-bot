from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# 1. Log into Hugging Face (this will prompt for your token in browser or terminal)
login()

# 2. Load your trained model and tokenizer from the local folder
model = AutoModelForCausalLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")

# 3. Push to Hugging Face under your account
model.push_to_hub("djoncas99/inspection-bot")
tokenizer.push_to_hub("djoncas99/inspection-bot")

