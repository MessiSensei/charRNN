import torch                                                 # PyTorch core library for tensor operations
import torch.nn as nn                                        # Neural network layers like LSTM, Linear
import torch.nn.functional as F                              # Functional layer utilities like softmax
import numpy as np                                           # NumPy for numerical operations like sampling
import argparse                                              # Argument parser for command-line interaction

# Load and preprocess dataset
def load_data(filename):                                     # Reads text, builds vocab, returns encoded characters
    with open(filename, 'r', encoding='utf-8') as f:         # Open and read text file
        text = f.read()                                      # Read full file content
    chars = sorted(list(set(text)))                          # Get sorted list of unique characters
    stoi = {ch: i for i, ch in enumerate(chars)}             # Map characters to indices
    itos = {i: ch for ch, i in stoi.items()}                 # Reverse mapping: index to char
    vocab_size = len(chars)                                  # Total number of unique characters
    encoded = [stoi[c] for c in text]                        # Encode text into list of indices
    return text, encoded, stoi, itos, vocab_size             # Return everything for training

# Define RNN model
class CharRNN(nn.Module):                                    # Character-level RNN model using LSTM
    def __init__(self, vocab_size, hidden_size, num_layers=1): # Initialize layers
        super().__init__()                                   # Call parent constructor
        self.embed = nn.Embedding(vocab_size, hidden_size)   # Embedding: map chars to vectors
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True) # LSTM: sequence processing
        self.fc = nn.Linear(hidden_size, vocab_size)         # Linear: map LSTM output to vocab logits

    def forward(self, x, hidden):                            # Forward pass through the model
        x = self.embed(x)                                    # Convert indices to embeddings
        out, hidden = self.rnn(x, hidden)                    # Run through LSTM
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2))) # Flatten and project output
        return out, hidden                                   # Return logits and hidden state

# Sampling function
def sample(model, start_str, stoi, itos, device, length=200, temperature=1.0): # Generate text from trained model
    start_str = ''.join([c for c in start_str if c in stoi]) # Clean seed string
    if not start_str:                                        # Validate seed
        raise ValueError("Seed string contains no known characters from the vocabulary.")
    model.eval()                                             # Set model to eval mode
    input_eval = torch.tensor([[stoi[c] for c in start_str]], dtype=torch.long).to(device) # Convert seed to tensor
    hidden = None                                            # No initial hidden state
    result = list(start_str)                                 # Store result characters
    with torch.no_grad():                                    # Disable gradient tracking
        for _ in range(length):                              # Loop to generate characters
            output, hidden = model(input_eval, hidden)       # Get next output from model
            output = output[-1, :] / temperature             # Apply temperature
            probabilities = F.softmax(output, dim=0).cpu().numpy() # Softmax to probabilities
            predicted_id = np.random.choice(len(probabilities), p=probabilities) # Sample next char
            result.append(itos[predicted_id])                # Add predicted character to result
            input_eval = torch.tensor([[predicted_id]], dtype=torch.long).to(device) # Prepare next input
    return ''.join(result)                                   # Return final generated string

# Training function
def train_model(encoded, vocab_size, stoi, itos, epochs=10, seq_length=100, batch_size=64, hidden_size=128): # Train the RNN model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available
    model = CharRNN(vocab_size, hidden_size).to(device)      # Create and move model to device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003) # Adam optimizer
    criterion = nn.CrossEntropyLoss()                        # Classification loss function
    n = len(encoded) - seq_length                            # Total training chunks
    for epoch in range(epochs):                              # Epoch loop
        model.train()                                        # Set model to train mode
        total_loss = 0                                       # Track loss
        for i in range(0, n, batch_size):                    # Mini-batch loop
            inputs = [encoded[j:j+seq_length] for j in range(i, min(i+batch_size, n))] # Build input sequences
            targets = [encoded[j+1:j+seq_length+1] for j in range(i, min(i+batch_size, n))] # Build target sequences
            inputs = torch.tensor(inputs, dtype=torch.long).to(device) # Convert to tensor
            targets = torch.tensor(targets, dtype=torch.long).to(device)
            optimizer.zero_grad()                            # Clear gradients
            output, _ = model(inputs, None)                  # Forward pass
            loss = criterion(output, targets.reshape(-1))    # Compute loss
            loss.backward()                                  # Backpropagate
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
            optimizer.step()                                 # Update weights
            total_loss += loss.item()                        # Accumulate loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}") # Show loss
        if (epoch+1) % 2 == 0:                               # Generate sample every 2 epochs
            preview = sample(model, "Once upon", stoi, itos, device)
            print(f"\n[Sample after epoch {epoch+1}]:\n{preview}\n")
    torch.save(model.state_dict(), 'model.pt')            # Save trained model
    print("Model saved to model.pt")
    return model                                             # Return model object

# CLI interface
if __name__ == '__main__':                                   # Main entry point
    parser = argparse.ArgumentParser(description='Character-Level RNN Text Generator') # Setup argument parser
    parser.add_argument('--input', type=str, default='sample.txt', help='Training text file') # Input text file
    parser.add_argument('--generate', action='store_true', help='Generate text using saved model') # Flag for generation
    parser.add_argument('--start', type=str, default='Once upon', help='Seed text for generation') # Seed string
    parser.add_argument('--length', type=int, default=200, help='Characters to generate') # Output length
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature') # Sampling randomness
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs') # Number of epochs
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the RNN') # LSTM size
    args = parser.parse_args()                               # Parse CLI arguments
    text, encoded, stoi, itos, vocab_size = load_data(args.input) # Load data from file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device
    if args.generate:                                        # If generation mode
        model = CharRNN(vocab_size, hidden_size=args.hidden_size).to(device) # Load model
        model.load_state_dict(torch.load('model.pt', map_location=device)) # Load weights
        print(sample(model, args.start, stoi, itos, device, length=args.length, temperature=args.temperature)) # Generate text
    else:                                                    # If training mode
        train_model(encoded, vocab_size, stoi, itos, epochs=args.epochs, hidden_size=args.hidden_size) # Start training
