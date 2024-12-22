import os
import yaml
import random
import re
import json
import numpy as np
from transformers import Wav2Vec2CTCTokenizer

# ****************** CONFIGURATION ******************

# Load configuration
config = yaml.load(open("utils/seq2seq/config.yaml", "r"), Loader=yaml.FullLoader)

# Paths
preprocessed_path = config["path"]["root_path"]
nam_features_path = os.path.join(preprocessed_path, "nam_features")
text_path = os.path.join(preprocessed_path, "text")

# Following code creates this many files
output_train_file = os.path.join(preprocessed_path, "train.txt")
output_val_file = os.path.join(preprocessed_path, "val.txt")
store_vocab_json = os.path.join(preprocessed_path, "vocab_character.json")
asr_tokens_save_path = os.path.join(preprocessed_path, "ASR_tokens_character")

# Regex for special characters to ignore
chars_to_ignore_regex = r'[\,%=\?\.\!\-\;\:\"\'\(\)\[\]\’\“\”]'

# Character replacements
character_replacements = {
    'é': 'e', 'à': 'a', 'â': 'a', 'è': 'e', 'ê': 'e', 'ü': 'u'
}

# ****************** UTILITY FUNCTIONS ******************

def remove_special_characters(texts):
    """Remove special characters from text."""
    return [re.sub(chars_to_ignore_regex, '', text).lower() for text in texts]

def replace_characters(texts, replacements):
    """Replace specific characters in text."""
    return [
        "".join(replacements.get(char, char) for char in text)
        for text in texts
    ]

def extract_vocab(texts):
    """Extract unique characters and vocabulary from text."""
    all_text = " ".join(texts)
    unique_chars = list(set(all_text))
    return {"vocab": unique_chars, "all_text": all_text}

def create_ctc_labels(texts, tokenizer):
    """Generate CTC labels using a tokenizer."""
    return [tokenizer.encode(text) for text in texts]

# ****************** CREATE TRAIN AND VAL FILES ******************

# Load all feature IDs
all_ids = [os.path.splitext(file)[0] for file in os.listdir(nam_features_path)]

# Shuffle and split data
random.shuffle(all_ids)
split_ratio = config.get("split_ratio", 0.95)
split_index = int(len(all_ids) * split_ratio)
train_ids, val_ids = all_ids[:split_index], all_ids[split_index:]

# Write train and validation IDs to files
with open(output_train_file, "w") as train_file:
    train_file.write("\n".join(train_ids))
with open(output_val_file, "w") as val_file:
    val_file.write("\n".join(val_ids))

print(f"Train data saved to: {output_train_file}")
print(f"Validation data saved to: {output_val_file}")

# ****************** CREATE FILES FOR CTC TRAINING ******************

# Load text files
text_files = [f for f in os.listdir(text_path) if f.endswith('.txt')]
texts_list = []
names_list = []

for text_file in text_files:
    with open(os.path.join(text_path, text_file), "r") as f:
        lines = f.readlines()
        if lines:
            texts_list.append(lines[0].strip())  # Only the first line is considered
        else:
            print(f"Skipping empty file: {text_file}")
        names_list.append(os.path.splitext(text_file)[0])

if len(texts_list) != len(names_list):
    print("Mismatch detected! Investigate empty or malformed files.")

# Process text: remove special characters and replace specified characters
texts_list = remove_special_characters(texts_list)
texts_list = replace_characters(texts_list, character_replacements)

# Extract vocabulary
vocab_data = extract_vocab(texts_list)
vocab_list = sorted(vocab_data["vocab"])
vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

# Save vocabulary to JSON
with open(store_vocab_json, "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

# Initialize tokenizer
tokenizer = Wav2Vec2CTCTokenizer(
    store_vocab_json,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

# Generate CTC labels and save them
os.makedirs(asr_tokens_save_path, exist_ok=True)
ctc_labels = create_ctc_labels(texts_list, tokenizer)

for name, labels in zip(names_list, ctc_labels):
    with open(os.path.join(asr_tokens_save_path, f"{name}.npy"), "wb") as f:
        np.save(f, labels)

print(f"Vocabulary size: {len(vocab_dict)}")
print(f"ASR tokens saved to: {asr_tokens_save_path}")