
#  Character-Level RNN Text Generator (PyTorch)

using PyTorch to generate text one character at a time.

---

##  Requirements

- PyTorch
- NumPy

Install dependencies:

```bash
pip install torch numpy
```

---

##  Training the Model

Train on `sample.txt` with default settings:
upload your sample.txt in the same folder as .py file 
```bash
python day1_PytorchRNN.py
```

Train on custom text with more epochs:

```bash
python day1_PytorchRNN.py --input custom.txt --epochs 50
```

Change the hidden layer size (e.g., 256 units):

```bash
python day1_PytorchRNN.py --hidden_size 256
```

---

## Generate Text

Generate 200 characters using the trained model:

```bash
python day1_PytorchRNN.py --generate
```

Start with a custom seed and generate 300 characters:

```bash
python day1_PytorchRNN.py --generate --start "He was a " --length 300
```

---

## Temperature Explained

**Temperature** controls the randomness of the output:

- `--temperature 1.0`: Balanced (default)
- `--temperature 0.5`: More predictable and repetitive
- `--temperature 1.2`: More creative and diverse

Example:

```bash
python day1_PytorchRNN.py --generate --temperature 1.2
```

---

##  Example Output

```
[Sample after epoch 10]:
He was a shadow of the throne,
The moonlight sang through windows faint,
And night was born in stone.
```

---

##  Files

- `charGenRNN.py`: Main script for training and generation
- `sample.txt`: Example training data
- `model.pt`: Saved model after training (auto-generated)

---

