# config.py
# Configuration settings for the Telugu NER project

# Model training parameters
N_ITERATIONS = 50
DROPOUT = 0.3
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# Dataset parameters - START WITH SMALL VALUES AND INCREASE GRADUALLY
TRAIN_SUBSET_SIZE = 1000  # Start with a small subset (100-500), then increase
VALIDATION_SIZE = 200

# Paths - UPDATED FOR LOCAL FILES
MODEL_DIR = "telugu_ner_model_merged"

WIKI_DIR = r"C:\Users\dell\Downloads\Telugu\Telugu"

DATA_DIR = r"C:\Users\dell\Downloads\Telugu\Telugu"

BEST_CHECKPOINT_DIR = r"C:\Users\dell\Downloads\Telugu\Telugu\telugu_ner_complete_fixed_best"

TRAIN_FILE = "te_train.json"
VAL_FILE = "te_val.json"
TEST_FILE = "test.spacy"

PSEUDO_LABELED_DATA = r"C:\Users\dell\Downloads\Telugu\Telugu\pseudo_labels.json"  # ← update this to your actual path

# Your local dataset file names - ADD THESE
TRAIN_FILE = "te_train.json"
VAL_FILE = "te_val.json" 
TEST_FILE = "te_test.json"
PSEUDO_LABELED_DATA = r'C:\Users\dell\Desktop\Telugu\pseudo_labels.json'

# Updated entity labels (based on your dataset)
LABELS = ["PERSON", "ORG", "LOC", "GPE"]

# ID to label mapping - UPDATE THIS BASED ON YOUR ACTUAL TAGS
ID_TO_LABEL = {
    "O": "O",
    "B-PER": "PERSON",
    "I-PER": "PERSON", 
    "B-ORG": "ORG",
    "I-ORG": "ORG",
    "B-LOC": "LOC",
    "I-LOC": "LOC",
    "B-GPE": "GPE",
    "I-GPE": "GPE",
}

# Label to ID mapping (reverse of above)
LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items() if k != 0}
LABEL_TO_ID["O"] = 0

# Fine-grained entity types (extend this based on your research)
FINE_GRAINED_TYPES = {
    "PERSON": ["POLITICIAN", "ARTIST", "SCIENTIST", "ATHLETE"],
    "ORG": ["GOVERNMENT", "EDUCATIONAL", "COMMERCIAL", "NON_PROFIT"],
    "GPE": ["CITY", "STATE", "COUNTRY", "DISTRICT"],
    "LOC": ["RIVER", "MOUNTAIN", "TEMPLE", "HISTORICAL_SITE"]
}

# Memory management settings - ADD THESE NEW SETTINGS
MAX_DOC_LENGTH = 200  # Maximum number of characters per document to process
SKIP_LONG_DOCS = True  # Skip documents longer than MAX_DOC_LENGTH to save memory
CHECKPOINT_EVERY = 5  # Save model checkpoint every N iterations
