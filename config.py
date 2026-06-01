"""Configuration file for the English-Urdu Translator project"""

# Model Configuration
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_UNITS = 256
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Data Configuration
TRAIN_TEST_SPLIT = 0.8
MAX_SEQUENCE_LENGTH = 50
MIN_WORD_FREQUENCY = 2

# Paths
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
NOTEBOOK_DIR = 'notebooks/'

# Preprocessing
REMOVE_PUNCTUATION = True
LOWERCASE = True
REMOVE_STOPWORDS = False
