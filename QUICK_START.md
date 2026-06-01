# Quick Start Guide

## For University Submission

### Step 1: Clone the Repository
```bash
git clone https://github.com/malikahmed034351-byte/ml-english-urdu-translator.git
cd ml-english-urdu-translator
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Model (5-10 minutes)
```bash
python train.py
```

Expected Output:
```
============================================================
English to Urdu Translator - Training Script
============================================================

[1/5] Loading data...
Total samples: 1000
Training samples: 800
Test samples: 200

[2/5] Preprocessing data...
English sequences shape: (800, 50)
Urdu sequences shape: (800, 50)

[3/5] Building Seq2Seq model...
...
[Model summary displayed]

[4/5] Training model...
Epoch 1/20
...
Training completed!

[5/5] Saving model...
Model saved to models/translator_model.h5

============================================================
Training completed successfully!
You can now test translations using: python translate.py
============================================================
```

### Step 5: Test Translations
```bash
# Test 1
python translate.py "hello how are you"

# Test 2
python translate.py "good morning"

# Test 3
python translate.py "thank you very much"
```

Expected Output:
```
============================================================
English to Urdu Translation
============================================================

[1/3] Loading trained model...
✓ Model loaded successfully!

[2/3] Preparing preprocessor...
✓ Preprocessor ready!

[3/3] Translating text...

============================================================
English: hello how are you
Urdu: ہیلو آپ کیسے ہیں
============================================================
```

## Project Structure
```
ml-english-urdu-translator/
├── src/
│   ├── preprocessing.py      # Text cleaning and tokenization
│   ├── model.py             # Seq2Seq neural network
│   ├── data_loader.py       # Dataset handling
│   └── translator.py        # Translation interface
├── models/                  # Saved trained models
├── config.py                # Configuration parameters
├── train.py                 # Training script
├── translate.py             # Testing script
├── requirements.txt         # Dependencies
└── README.md               # Full documentation
```

## What This Project Does

This is a **Machine Learning project** that:
1. **Preprocesses** English and Urdu text data
2. **Builds** a Sequence-to-Sequence neural network (Encoder-Decoder architecture)
3. **Trains** the model on English-Urdu sentence pairs
4. **Translates** new English sentences to Urdu

## Key Features

✅ **Bidirectional LSTM Encoder** - Captures context from both directions
✅ **LSTM Decoder** - Generates translated sequences
✅ **Text Preprocessing** - Cleaning, tokenization, padding
✅ **Configurable** - Easy hyperparameter tuning
✅ **Well-Documented** - Clear code with docstrings
✅ **University-Ready** - Perfect for academic projects

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Make sure you've installed requirements:
```bash
pip install -r requirements.txt
```

### Issue: "Model not found at models/translator_model.h5"
**Solution:** Train the model first:
```bash
python train.py
```

### Issue: Takes too long to train
**Solution:** Reduce EPOCHS in config.py (default is 20)
```python
EPOCHS = 10  # Change from 20 to 10
```

## Submission Link

📍 **https://github.com/malikahmed034351-byte/ml-english-urdu-translator**

Share this link with your university for evaluation!

## Project Description for Submission

"This project implements a Sequence-to-Sequence (Seq2Seq) neural network using TensorFlow/Keras to translate English sentences to Urdu. The model consists of a bidirectional LSTM encoder that processes English input and an LSTM decoder that generates Urdu translations. The project includes data preprocessing, model training with early stopping and learning rate reduction, and a translation interface for testing."

---

Need help? Check the README.md for detailed documentation!
