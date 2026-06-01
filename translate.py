"""Script to test translation"""

import sys
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import TextPreprocessor
from src.data_loader import DataLoader
from config import *

def main():
    if len(sys.argv) < 2:
        text = "hello how are you"
    else:
        text = sys.argv[1]
    
    print("=" * 60)
    print("English to Urdu Translation")
    print("=" * 60)
    
    # Check if model exists
    model_path = os.path.join(MODEL_DIR, 'translator_model.h5')
    if not os.path.exists(model_path):
        print(f"\nError: Model not found at {model_path}")
        print("Please run 'python train.py' first to train the model.")
        return
    
    # Load model
    print("\n[1/3] Loading trained model...")
    try:
        model = load_model(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare preprocessor
    print("\n[2/3] Preparing preprocessor...")
    data_loader = DataLoader()
    df = data_loader.create_synthetic_dataset(num_samples=100)
    preprocessor = TextPreprocessor(max_vocab_size=VOCAB_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH)
    preprocessor.preprocess_dataset(df)
    print("✓ Preprocessor ready!")
    
    # Translate
    print("\n[3/3] Translating text...")
    english_seq = preprocessor.english_tokenizer.texts_to_sequences([text.lower()])
    english_seq = pad_sequences(english_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    decoder_input = np.zeros((1, MAX_SEQUENCE_LENGTH))
    prediction = model.predict([english_seq, decoder_input], verbose=0)
    
    predicted_ids = np.argmax(prediction[0], axis=-1)
    urdu_translation = ' '.join([
        preprocessor.urdu_tokenizer.index_word.get(int(idx), '<UNK>') 
        for idx in predicted_ids if idx != 0
    ])
    
    print("\n" + "=" * 60)
    print(f"English: {text}")
    print(f"Urdu: {urdu_translation}")
    print("=" * 60)

if __name__ == "__main__":
    main()
