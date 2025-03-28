
import torch                                                 # PyTorch core library for tensor operations and tensor management
import torch.nn as nn                                        # Module for building neural networks (layers like LSTM, Linear)
import torch.nn.functional as F                              # Functional API for layers and activations (e.g., softmax)
import numpy as np                                           
import argparse                                              # Library for parsing command-line arguments

# Text cleaning function to remove excessive white space
def clean_text(text):
    import re                                                # Python regex module for pattern matching
    text = re.sub(r'\n+', '\n', text)                        # Replace multiple newlines with a single newline
    text = re.sub(r' +', ' ', text)                          # Replace multiple spaces with a single space
    text = text.strip()                                      # Remove leading and trailing whitespace
    print("Text is cleaned up ... ")
    return text                                              # Return the cleaned text

# Load and preprocess dataset
def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:         # Open text file for reading
        text = clean_text(f.read())                          # Read and clean the text
    chars = sorted(list(set(text)))                          # Create sorted list of unique characters
    stoi = {ch: i for i, ch in enumerate(chars)}             # Create char to index mapping
    itos = {i: ch for ch, i in stoi.items()}                 # Create index to char mapping
    vocab_size = len(chars)                                  # Total number of unique characters
    print("Vocab size is : " , vocab_size)
    encoded = [stoi[c] for c in text]                        # Encode entire text as a list of integers
    return text, encoded, stoi, itos, vocab_size             # Return all processed data

# Define RNN model for character-level prediction
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super().__init__()                                   # Initialize parent nn.Module
        self.embed = nn.Embedding(vocab_size, hidden_size)   # Embedding layer to convert indices to dense vectors
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)  # LSTM for sequence modeling
        self.fc = nn.Linear(hidden_size, vocab_size)         # Linear layer to produce output logits

    def forward(self, x, hidden):
        x = self.embed(x)                                    # Convert indices to embeddings
        out, hidden = self.rnn(x, hidden)                    # Pass embeddings through LSTM
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))  # Reshape and pass through output layer
        return out, hidden                                   # Return output logits and new hidden state

# Function to generate text from a trained model
def sample(model, start_str, stoi, itos, device, length=200, temperature=1.0):
    start_str = ''.join([c for c in start_str if c in stoi]) # Remove unknown characters from start string
    if not start_str:
        raise ValueError("Seed string contains no known characters from the vocabulary.")
    model.eval()                                             # Set model to evaluation mode (disable dropout)
    input_eval = torch.tensor([[stoi[c] for c in start_str]], dtype=torch.long).to(device)  # Convert seed to tensor
    hidden = None                                            # Start with no hidden state
    result = list(start_str)                                 # Initialize result with seed text

    with torch.no_grad():                                    # Disable gradient tracking for faster inference
        for _ in range(length):                              # Generate 'length' number of characters
            output, hidden = model(input_eval, hidden)       # Forward pass through model
            output = output[-1, :] / temperature             # Adjust temperature to control randomness
            probabilities = F.softmax(output, dim=0).cpu().numpy()  # Apply softmax to get probabilities
            predicted_id = np.random.choice(len(probabilities), p=probabilities)  # Sample next char index
            result.append(itos[predicted_id])                # Append predicted character
            input_eval = torch.tensor([[predicted_id]], dtype=torch.long).to(device)  # Update input for next step

    return ''.join(result)                                   # Return the final generated text

# Function to train the model on the dataset
def train_model(encoded, vocab_size, stoi, itos, epochs=10, seq_length=100, batch_size=64, hidden_size=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
    model = CharRNN(vocab_size, hidden_size).to(device)     # Create model and move it to device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)  # Adam optimizer for updating weights
    criterion = nn.CrossEntropyLoss()                       # Loss function for classification
    n = len(encoded) - seq_length                           # Number of training samples

    for epoch in range(epochs):                             # Loop over epochs
        model.train()                                       # Set model to training mode
        total_loss = 0                                      # Accumulate loss for each epoch
        for i in range(0, n, batch_size):                   # Loop over mini-batches
            inputs = [encoded[j:j+seq_length] for j in range(i, min(i+batch_size, n))]  # Input sequences
            targets = [encoded[j+1:j+seq_length+1] for j in range(i, min(i+batch_size, n))]  # Target sequences
            inputs = torch.tensor(inputs, dtype=torch.long).to(device)  # Convert to tensor and move to device
            targets = torch.tensor(targets, dtype=torch.long).to(device)

            optimizer.zero_grad()                           # Clear previous gradients
            output, _ = model(inputs, None)                 # Forward pass
            loss = criterion(output, targets.reshape(-1))   # Compute loss
            loss.backward()                                 # Backpropagate gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to prevent exploding
            optimizer.step()                                # Update model weights
            total_loss += loss.item()                       # Accumulate batch loss

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")  # Log epoch loss
        if (epoch+1) % 2 == 0:                              # Sample after every 2 epochs
            preview = sample(model, "He was a ", stoi, itos, device)
            print(f"\n[Sample after epoch {epoch+1}]:\n{preview}\n")

    torch.save(model.state_dict(), 'model.pt')               # Save trained model to file
    print("Model saved to model.pt")                         # Confirm model saving
    return model                                             # Return trained model

# Main entry point when script is run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Character-Level RNN Text Generator')  # Create CLI parser
    parser.add_argument('--input', type=str, default='sample.txt', help='Training text file')  # Input file
    parser.add_argument('--generate', action='store_true', help='Generate text using saved model')  # Generation mode flag
    parser.add_argument('--start', type=str, default=' He was a ', help='Seed text for generation')  # Seed string
    parser.add_argument('--length', type=int, default=200, help='Characters to generate')  # Output length
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')  # Sampling randomness
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')  # Number of training epochs
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the RNN')  # LSTM memory size
    args = parser.parse_args()                                 # Parse all CLI arguments

    text, encoded, stoi, itos, vocab_size = load_data(args.input)  # Load and preprocess input data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device
    print("device is set to :" , device)
    if args.generate:                                          # If generation flag is used
        model = CharRNN(vocab_size, hidden_size=args.hidden_size).to(device)  # Load model
        model.load_state_dict(torch.load('model.pt', map_location=device))  # Load saved weights
        print(sample(model, args.start, stoi, itos, device, length=args.length, temperature=args.temperature))  # Generate text
    else:                                                      # Otherwise, train model
        train_model(encoded, vocab_size, stoi, itos, epochs=args.epochs, hidden_size=args.hidden_size)  # Start training
