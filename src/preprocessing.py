"""Data preprocessing module for English-Urdu translation"""

import re
import string
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextPreprocessor:
    """Handles text preprocessing for translation task"""
    
    def __init__(self, max_vocab_size=10000, max_sequence_length=50):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.english_tokenizer = None
        self.urdu_tokenizer = None
        
    def clean_text(self, text, language='english'):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_pad(self, texts, language='english', fit=False):
        """Tokenize and pad sequences"""
        if language == 'english':
            if fit:
                self.english_tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
                self.english_tokenizer.fit_on_texts(texts)
            sequences = self.english_tokenizer.texts_to_sequences(texts)
        else:
            if fit:
                self.urdu_tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
                self.urdu_tokenizer.fit_on_texts(texts)
            sequences = self.urdu_tokenizer.texts_to_sequences(texts)
        
        padded = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        return padded
    
    def preprocess_dataset(self, df, english_col='english', urdu_col='urdu'):
        """Preprocess entire dataset"""
        df[english_col] = df[english_col].apply(lambda x: self.clean_text(x, 'english'))
        df[urdu_col] = df[urdu_col].apply(lambda x: self.clean_text(x, 'urdu'))
        
        # Tokenize and pad
        english_sequences = self.tokenize_and_pad(df[english_col].tolist(), language='english', fit=True)
        urdu_sequences = self.tokenize_and_pad(df[urdu_col].tolist(), language='urdu', fit=True)
        
        return english_sequences, urdu_sequences
