"""Attention heads and the character-level Transformer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    """A causal self-attention head that attends to current and earlier positions."""

    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.query_layer = nn.Linear(embedding_size, head_size)
        self.key_layer = nn.Linear(embedding_size, head_size)
        self.value_layer = nn.Linear(embedding_size, head_size)

    def forward(self, x):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        key = torch.transpose(key, 1, 2)
        T = x.shape[1]
        mask = torch.tril(torch.ones(T, T, device=x.device))
        head_size = query.shape[-1]
        attention = (query @ key) / head_size ** 0.5
        attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        return attention @ value


class TransformerLanguageModel(nn.Module):

    def __init__(self, vocab_size, embedding_size, context_length, head_size):
        super().__init__()
        self.feed_forward = nn.Sequential(nn.Linear(embedding_size,embedding_size*4),
                                           nn.ReLU(), 
                                           nn.Linear(embedding_size*4, embedding_size))
        self.attention_projection = nn.Linear(2*head_size,embedding_size)
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_embedding = nn.Embedding(context_length, embedding_size)
        self.head1 = AttentionHead(embedding_size, head_size)
        self.head2 = AttentionHead(embedding_size, head_size)
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, token_ids):
        token_embeddings = self.token_embedding(token_ids)
        B, T = token_ids.shape
        position_ids = torch.arange(T, device=token_ids.device)
        position_embeddings = self.pos_embedding(position_ids)
        x = token_embeddings +  position_embeddings 
        normalized_x = self.layer_norm1(x)
        
        head_cat = torch.cat((self.head1(normalized_x), self.head2(normalized_x)), dim=-1)
        head_cat = self.attention_projection(head_cat)
        sum_ = x + head_cat 
        normalized_sum = self.layer_norm2(sum_)
        output = self.feed_forward(normalized_sum) 
        sum_ = output + sum_
        logits = self.output_layer(sum_)
        
        return logits