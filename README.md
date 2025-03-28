
# 🧠 Character-Level RNN Text Generator (PyTorch)

This project trains a character-level Recurrent Neural Network (RNN) using PyTorch to generate text one character at a time.

---

## 📦 Requirements

- Python 3.7+
- PyTorch
- NumPy

Install dependencies:

```bash
pip install torch numpy
```

---

## 🚀 Training the Model

Train on `sample.txt` with default settings:

```bash
python your_script.py
```

Train on custom text with more epochs:

```bash
python your_script.py --input custom.txt --epochs 50
```

Change the hidden layer size (e.g., 256 units):

```bash
python your_script.py --hidden_size 256
```

---

## ✍️ Generate Text

Generate 200 characters using the trained model:

```bash
python your_script.py --generate
```

Start with a custom seed and generate 300 characters:

```bash
python your_script.py --generate --start "Once upon a time" --length 300
```

---

## 🔥 Temperature Explained

**Temperature** controls the randomness of the output:

- `--temperature 1.0`: Balanced (default)
- `--temperature 0.5`: More predictable and repetitive
- `--temperature 1.2`: More creative and diverse

Example:

```bash
python your_script.py --generate --temperature 1.2
```

---

## 🧪 Example Output

```
[Sample after epoch 10]:
Once upon the morning, and the shadow of the throne,
The moonlight sang through windows faint,
And night was born in stone.
```

---

## 📁 Files

- `charGenRNN.py`: Main script for training and generation
- `sample.txt`: Example training data
- `model.pt`: Saved model after training (auto-generated)

---

