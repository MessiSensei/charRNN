from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt

from day1_PytorchRNN import CharRNN, load_data, sample

app = Flask(__name__)

# Load and preprocess text
text, encoded, stoi, itos, vocab_size = load_data("sample.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = CharRNN(vocab_size, hidden_size=128).to(device)
if os.path.exists("model.pt"):
    model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    generated_text = ""
    show_plot = os.path.exists("static/training_loss.png")
    if request.method == "POST":
        start_text = request.form.get("start", "The ")
        temperature = float(request.form.get("temperature", 1.0))
        length = int(request.form.get("length", 200))
        generated_text = sample(model, start_text, stoi, itos, device, length=length, temperature=temperature)
    return render_template("index.html", result=generated_text, show_plot=show_plot)

@app.route("/train", methods=["POST"])
def train():
    data = request.json or {}
    epochs = int(data.get("epochs", 10))
    hidden_size = int(data.get("hidden_size", 128))
    seq_length = 100
    batch_size = 64

    model = CharRNN(vocab_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    n = len(encoded) - seq_length
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, n, batch_size):
            inputs = [encoded[j:j+seq_length] for j in range(i, min(i+batch_size, n))]
            targets = [encoded[j+1:j+seq_length+1] for j in range(i, min(i+batch_size, n))]

            inputs = torch.tensor(inputs, dtype=torch.long).to(device)
            targets = torch.tensor(targets, dtype=torch.long).to(device)

            optimizer.zero_grad()
            output, _ = model(inputs, None)
            loss = criterion(output, targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        losses.append(total_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pt")

    # Plot loss curve
    plt.figure()
    plt.plot(range(1, epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.grid(True)
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/training_loss.png")
    plt.close()

    return {"status": "Training complete", "losses": losses}

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    start = data.get("start", "The ")
    temperature = float(data.get("temperature", 1.0))
    length = int(data.get("length", 200))

    model.eval()
    generated = sample(model, start, stoi, itos, device, length=length, temperature=temperature)
    return {"generated": generated}

if __name__ == "__main__":
    app.run(debug=True, port=5000)
