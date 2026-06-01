"""Sequence-to-Sequence model for English-Urdu translation"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Bidirectional, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class Seq2SeqTranslator:
    """Sequence-to-Sequence model for translation"""
    
    def __init__(self, english_vocab_size, urdu_vocab_size, embedding_dim=128, 
                 hidden_units=256, learning_rate=0.001):
        self.english_vocab_size = english_vocab_size
        self.urdu_vocab_size = urdu_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self, sequence_length):
        """Build Seq2Seq model with attention-like architecture"""
        # Encoder
        encoder_inputs = Input(shape=(sequence_length,))
        encoder_embedding = Embedding(self.english_vocab_size, self.embedding_dim)(encoder_inputs)
        encoder_lstm = Bidirectional(LSTM(self.hidden_units, return_state=True))
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
        
        # State initialization
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        
        # Decoder
        decoder_inputs = Input(shape=(sequence_length,))
        decoder_embedding = Embedding(self.urdu_vocab_size, self.embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(self.hidden_units * 2, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
        
        # Output layer
        decoder_dense = Dense(self.urdu_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, encoder_input_data, decoder_input_data, decoder_target_data, 
              epochs=20, batch_size=32, validation_split=0.2):
        """Train the model"""
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
        
        history = self.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Load a trained model"""
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
