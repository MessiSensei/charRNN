
import torch                                                 # PyTorch core library for tensor operations
import torch.nn as nn                                        # Module for neural network layers (e.g., LSTM, Linear)
import torch.nn.functional as F                              # Functional utilities (e.g., softmax, relu)
import numpy as np                                           # NumPy for numerical operations (used for sampling)
import argparse                                              # Library for parsing command-line arguments

# Text cleaning function to remove excessive white space
def clean_text(text):
    import re
    text = re.sub(r'\n+', '\n', text)        # Collapse multiple consecutive newlines
    text = re.sub(r' +', ' ', text)            # Collapse multiple consecutive spaces
    text = text.strip()                        # Trim leading and trailing whitespace
    return text

# Load and preprocess dataset
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = clean_text(f.read())                                # Clean and read entire text
    chars = sorted(list(set(text)))                                # Extract unique characters in sorted order
    stoi = {ch: i for i, ch in enumerate(chars)}                   # String-to-index mapping
    itos = {i: ch for ch, i in stoi.items()}                       # Index-to-string mapping
    vocab_size = len(chars)                                        # Total number of unique characters
    encoded = [stoi[c] for c in text]                              # Encode full text as list of integers
    return text, encoded, stoi, itos, vocab_size

# Define RNN model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)         # Embedding layer: maps char indices to vectors
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_size, vocab_size)               # Final output layer (logits over characters)

    def forward(self, x, hidden):
        x = self.embed(x)                                          # Convert input indices to embeddings
        out, hidden = self.rnn(x, hidden)                          # Process sequence with LSTM
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))  # Flatten and pass through linear layer
        return out, hidden

# Sampling function
def sample(model, start_str, stoi, itos, device, length=200, temperature=1.0):
    start_str = ''.join([c for c in start_str if c in stoi])       # Filter out unknown chars in start string
    if not start_str:
        raise ValueError("Seed string contains no known characters from the vocabulary.")
    model.eval()                                                    # Set model to evaluation mode
    input_eval = torch.tensor([[stoi[c] for c in start_str]], dtype=torch.long).to(device)
    hidden = None
    result = list(start_str)

    with torch.no_grad():                                          # Disable gradient calculation (inference mode)
        for _ in range(length):
            output, hidden = model(input_eval, hidden)             # Forward pass
            output = output[-1, :] / temperature                   # Adjust output by temperature
            probabilities = F.softmax(output, dim=0).cpu().numpy() # Get probability distribution
            predicted_id = np.random.choice(len(probabilities), p=probabilities)  # Sample next character
            result.append(itos[predicted_id])                      # Add predicted character to result
            input_eval = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    return ''.join(result)                                         # Return the generated string

# Training function
def train_model(encoded, vocab_size, stoi, itos, epochs=10, seq_length=100, batch_size=64, hidden_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharRNN(vocab_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()
    n = len(encoded) - seq_length

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
        if (epoch+1) % 2 == 0:
            preview = sample(model, "Once upon", stoi, itos, device)
            print(f"\n[Sample after epoch {epoch+1}]:\n{preview}\n")

    torch.save(model.state_dict(), 'char_rnn.pt')                   # Save trained model
    print("✅ Model saved to char_rnn.pt")
    return model

# CLI interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Character-Level RNN Text Generator')
    parser.add_argument('--input', type=str, default='sample.txt', help='Training text file')
    parser.add_argument('--generate', action='store_true', help='Generate text using saved model')
    parser.add_argument('--start', type=str, default='Once upon', help='Seed text for generation')
    parser.add_argument('--length', type=int, default=200, help='Characters to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the RNN')
    args = parser.parse_args()

    text, encoded, stoi, itos, vocab_size = load_data(args.input)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.generate:
        model = CharRNN(vocab_size, hidden_size=args.hidden_size).to(device)
        model.load_state_dict(torch.load('char_rnn.pt', map_location=device))
        print(sample(model, args.start, stoi, itos, device, length=args.length, temperature=args.temperature))
    else:
        train_model(encoded, vocab_size, stoi, itos, epochs=args.epochs, hidden_size=args.hidden_size)
