"""Main translator class"""

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class EnglishUrduTranslator:
    """English to Urdu translator using trained model"""
    
    def __init__(self, model, english_tokenizer, urdu_tokenizer, 
                 urdu_reverse_vocab, max_sequence_length=50):
        self.model = model
        self.english_tokenizer = english_tokenizer
        self.urdu_tokenizer = urdu_tokenizer
        self.urdu_reverse_vocab = urdu_reverse_vocab
        self.max_sequence_length = max_sequence_length
        
    def translate(self, english_text):
        """Translate English text to Urdu"""
        # Preprocess input
        english_text = english_text.lower()
        english_seq = self.english_tokenizer.texts_to_sequences([english_text])
        english_seq = pad_sequences(english_seq, maxlen=self.max_sequence_length, padding='post')
        
        # Create decoder input (start token)
        decoder_input = np.zeros((1, self.max_sequence_length))
        
        # Predict
        prediction = self.model.predict([english_seq, decoder_input], verbose=0)
        
        # Convert predictions to words
        predicted_ids = np.argmax(prediction[0], axis=-1)
        urdu_translation = ' '.join([
            self.urdu_reverse_vocab.get(idx, '<UNK>') 
            for idx in predicted_ids if idx != 0
        ])
        
        return urdu_translation
