from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = Flask(__name__)

# Load local model
tokenizer_path = os.path.abspath("./model")
model_path = os.path.abspath("./model/checkpoint-10")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
model.eval()

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"response": "Please enter a valid question."})

    # Extract generation settings from UI
    do_sample = data.get("do_sample", False)
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.95)
    repetition_penalty = data.get("repetition_penalty", 1.1)
    max_new_tokens = data.get("max_new_tokens", 100)

    full_prompt = f"{prompt}\n"
    inputs = tokenizer(full_prompt, return_tensors="pt")

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        if prompt in decoded:
            decoded = decoded.replace(prompt, "").strip()

        return jsonify({"response": decoded or "The model returned an empty or unclear response."})

    except Exception as gen_error:
        print(f"Error generating response: {gen_error}")
        return jsonify({"response": "An error occurred while generating the response."})

if __name__ == "__main__":
    app.run(debug=True)