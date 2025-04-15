from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForCausalLM.from_pretrained("model")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"response": "Please enter a valid question."})

    # Prime the model with known format and a starter word
    full_prompt = f"### PROMPT:\n{prompt}\n\n### RESPONSE:\nThe"

    inputs = tokenizer(full_prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### RESPONSE:" in decoded:
        answer = decoded.split("### RESPONSE:")[-1].strip()
        if answer.lower().startswith("the "):
            answer = answer[4:]
    else:
        answer = decoded.strip()

    print("\n--- Prompt ---")
    print(prompt)
    print("--- Full Decoded Output ---")
    print(decoded)
    print("--- Final Extracted Answer ---")
    print(answer)

    return jsonify({"response": answer or "The model did not return a meaningful answer."})

if __name__ == "__main__":
    app.run(debug=True)
