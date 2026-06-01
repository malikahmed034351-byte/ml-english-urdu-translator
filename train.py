"""Script to train the English-Urdu translator model"""

from src.preprocessing import TextPreprocessor
from src.model import Seq2SeqTranslator
from src.data_loader import DataLoader
from config import *
import numpy as np
import os

def main():
    print("=" * 60)
    print("English to Urdu Translator - Training Script")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    data_loader = DataLoader()
    df = data_loader.create_synthetic_dataset(num_samples=1000)
    train_df, test_df = data_loader.split_dataset(df)
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Step 2: Preprocess data
    print("\n[2/5] Preprocessing data...")
    preprocessor = TextPreprocessor(max_vocab_size=VOCAB_SIZE, max_sequence_length=MAX_SEQUENCE_LENGTH)
    english_train, urdu_train = preprocessor.preprocess_dataset(train_df)
    print(f"English sequences shape: {english_train.shape}")
    print(f"Urdu sequences shape: {urdu_train.shape}")
    
    # Step 3: Build model
    print("\n[3/5] Building Seq2Seq model...")
    translator_model = Seq2SeqTranslator(
        english_vocab_size=VOCAB_SIZE,
        urdu_vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_units=HIDDEN_UNITS,
        learning_rate=LEARNING_RATE
    )
    model = translator_model.build_model(sequence_length=MAX_SEQUENCE_LENGTH)
    print("Model architecture:")
    model.summary()
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    history = translator_model.train(
        english_train, 
        english_train, 
        np.expand_dims(urdu_train, -1),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    print("Training completed!")
    
    # Step 5: Save model
    print("\n[5/5] Saving model...")
    model_path = os.path.join(MODEL_DIR, 'translator_model.h5')
    translator_model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("You can now test translations using: python translate.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
