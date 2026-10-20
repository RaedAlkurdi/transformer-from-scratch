"""Text loading, character vocabulary, encoding, decoding, and batching."""
from pathlib import Path
import torch

def read_text(file_path):
    return Path(file_path).read_text(encoding="utf-8")

def build_vocabulary(text):
    chars = sorted(set(text))
    return chars

def build_mappings(chars):
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    return stoi, itos

def encode(text, stoi):
    encoded = []

    for character in text:
        token_id = stoi[character]
        encoded.append(token_id)
    return encoded

def decode(token_ids, itos):
    decoded = []

    for token_id in token_ids:
        character = (itos[token_id])
        decoded.append(character)

    return ''.join(decoded)

def split_data(token_ids, train_fraction=0.9): 
    n = int(train_fraction*len(token_ids))
    train_data = token_ids[:n]
    validation_data = token_ids[n:]
    return train_data, validation_data

def get_batch(data, batch_size, context_length):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    input_ids = torch.stack([data[pos: pos + context_length] for pos in ix])
    target_ids = torch.stack([data[pos+1: pos + context_length + 1] for pos in ix])
    return input_ids, target_ids
