"""Data loading and preparation"""

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and prepare training data"""
    
    @staticmethod
    def load_csv(filepath):
        """Load CSV file"""
        return pd.read_csv(filepath)
    
    @staticmethod
    def create_synthetic_dataset(num_samples=1000):
        """Create synthetic English-Urdu dataset for testing"""
        # Sample phrase pairs
        english_phrases = [
            "hello how are you",
            "what is your name",
            "good morning",
            "thank you very much",
            "where is the bathroom",
            "i love this place",
            "can you help me",
            "see you tomorrow",
            "have a nice day",
            "welcome to our home",
            "how old are you",
            "nice to meet you",
            "where do you live",
            "what do you do",
            "do you speak english"
        ]
        
        urdu_phrases = [
            "ہیلو آپ کیسے ہیں",
            "آپ کا نام کیا ہے",
            "صبح بخیر",
            "بہت شکریہ",
            "باتھ روم کہاں ہے",
            "مجھے یہ جگہ پسند ہے",
            "کیا آپ مجھے مدد کر سکتے ہیں",
            "کل ملیں گے",
            "خوش رہو",
            "ہمارے گھر میں خوش آمدید",
            "آپ کی عمر کتنی ہے",
            "آپ سے ملکر خوشی ہوئی",
            "آپ کہاں رہتے ہیں",
            "آپ کیا کام کرتے ہیں",
            "کیا آپ انگریزی بولتے ہیں"
        ]
        
        data = []
        for i in range(num_samples):
            idx = i % len(english_phrases)
            data.append({
                'english': english_phrases[idx],
                'urdu': urdu_phrases[idx]
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def split_dataset(df, test_size=0.2, random_state=42):
        """Split dataset into train and test sets"""
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_df, test_df
