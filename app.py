from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Load tokenizer from ./model and model weights from checkpoint-10
tokenizer_path = os.path.abspath("./model")
model_path = os.path.abspath("./model/checkpoint-10")

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load the local model. Make sure the directories are correctly structured.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"response": "Please enter a valid question."})

    full_prompt = f"{prompt}\n"
    inputs = tokenizer(full_prompt, return_tensors="pt")

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=.2,
            top_k=.95,
            top_p=None,
            repetition_penalty=1.1, 
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # If the model parrots the prompt, clean it up
        if prompt in decoded:
            decoded = decoded.replace(prompt, "").strip()

        print("\n--- Prompt ---")
        print(prompt)
        print("--- Decoded Output ---")
        print(decoded)

        return jsonify({"response": decoded or "The model returned an empty or unclear response."})

    except Exception as gen_error:
        print(f"Error generating response: {gen_error}")
        return jsonify({"response": "An error occurred while generating the response."})

if __name__ == "__main__":
    app.run(debug=True)
