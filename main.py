import tokenmonster
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from numpy import sin, cos, pi

vocab_size = 32000
vocab = tokenmonster.load(f"english-{vocab_size}-consistent-v1")

print("Hello world")

s = t.tensor([1, 2, 3, 4])
s = t.vstack([s, s])
print(s)
tokens = vocab.tokenize("This is a test.")


class Transformer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # hyperparameters
        self.batchsize = 32  # how many independent sequences will we process in parallel?
        self.blocksize = 8  # what is the maximum context length for predictions?
        self.maximum_sequence_length = self.blocksize
        self.max_iters = 3000
        self.eval_interval = 300
        self.learning_rate = 1e-2

        self.embedding_dimension = 1024
        self.embedding_matrix = t.tensor(np.random.randn(self.embedding_dimension, vocab_size))
        # self.de_embedding_matrix = t.tensor(np.random.randn(vocab_size, self.embedding_dimension))

        self.d_key = 64
        self.d_value = 63
        self.num_attention_heads = 8

        # Initialize N groups of self.num_attention_heads multihead attentions
        self.N = 10
        self.left_attention = [MultiheadAttention(self.d_key, self.d_value, self.num_attention_heads, self.blocksize, self.embedding_dimension) for _ in range(self.N)]

        # Initialize N feedforward layers
        self.left_feedforward_layers = [nn.Linear(self.embedding_dimension, self.embedding_dimension) for _ in range(self.N)]

        # Initialization using Xavier Uniform method
        nn.init.xavier_uniform_(self.multi_head_linear)

        self.norm = nn.LayerNorm(self.embedding_dimension)

    def text_to_tokens(self, input):
        tokens = vocab.tokenize(input)
        return tokens

    def embed_tokens(self, tokens):
        """
        :param tokens: a [blocksize] tensor of tokens
        :return: a [blocksize, embedding_dimension] tensor of positionally encoded embeddings
        """

        # One hot embedding: takes in a [batchsize, blocksize] tensor of tokens and returns a [batchsize, blocksize, vocab_size] tensor of one-hot embeddings
        one_hot = F.one_hot(t.tensor(tokens.astype(np.int64)).to(t.int64), num_classes=vocab_size).to(t.float64)

        embedded = self.embedding_matrix @ one_hot.T

        # Positional encoding
        # Create a row of sin(w1 * 0), sin(w1 * 1), sin(w1 * 2), ..., sin(w1 * blocksize)
        # Followed by a row of cos(w1 * 0), cos(w1 * 1), cos(w1 * 2), ..., cos(w1 * blocksize)
        num_tokens = len(tokens)
        pos_enc = t.tensor([sin(i) for i in range(num_tokens)] + [cos(i) for i in range(num_tokens)]).reshape(2,
                                                                                                              num_tokens)
        for k in range(1, self.embedding_dimension // 2):
            w_k = 1 / (10000 ** (2 * k / self.embedding_dimension))
            sine_row = t.tensor([sin(w_k * i) for i in range(num_tokens)])
            cosine_row = t.tensor([cos(w_k * i) for i in range(num_tokens)])
            pos_enc = t.vstack([pos_enc, sine_row, cosine_row])

        # Add the positional encoding to the embedding
        embedded = embedded + pos_enc

        # Create a [embedding_dimension, blocksize] matrix of zeros, and paste the embedded tokens into it
        final_embedding = t.zeros((self.blocksize, self.embedding_dimension)).T
        final_embedding[:embedded.shape[0], :embedded.shape[1]] = embedded

        return final_embedding.T


    def forward(self, text):
        tokens = self.text_to_tokens(text)
        embeddings = self.embed_tokens(tokens)

        # Pass the embeddings through the N multihead attention layers
        for i in range(self.N):
            attended = self.left_attention[i](embeddings)
            added = embeddings + attended
            normed = self.norm(added)
            feedforwarded = self.left_feedforward_layers[i](normed)
            nextadded = normed + feedforwarded
            nextnormed = self.norm(nextadded)

            embeddings = nextnormed



        pass

class MultiheadAttention(nn.Module):
    def __init__(self, d_key, d_value, num_attention_heads, blocksize, embedding_dimension):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.num_attention_heads = num_attention_heads
        self.blocksize = blocksize
        self.embedding_dimension = embedding_dimension

        self.attention_heads = [AttentionHead(self.d_key, self.d_value, self.blocksize, self.embedding_dimension) for _ in range(self.num_attention_heads)]

        self.multi_head_linear = nn.Parameter(t.empty(self.num_attention_heads * self.d_value, self.embedding_dimension))

        # Initialization using Xavier Uniform method
        nn.init.xavier_uniform_(self.multi_head_linear)

    def forward(self, embedding_matrix):
        # Concatenate the outputs of each attention head
        concatenated = t.cat([head.attention(embedding_matrix) for head in self.attention_heads], dim=1)

        # Multiply by the linear layer
        assert (concatenated @ self.multi_head_linear).shape == (self.blocksize, self.embedding_dimension)
        return concatenated @ self.multi_head_linear

class AttentionHead(nn.Module):
    def __init__(self, d_key, d_value, block_size, embedding_dimension):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.block_size = block_size
        self.embedding_dimension = embedding_dimension

        self.Q = nn.Parameter(t.empty(embedding_dimension, d_key))
        self.K = nn.Parameter(t.empty(embedding_dimension, d_key))
        self.V = nn.Parameter(t.empty(embedding_dimension, d_value))

        # Initialization using Xavier Uniform method
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

    def attention(self, embedding_matrix):
        q = embedding_matrix @ self.Q  # [block_size, d_key]
        k = embedding_matrix @ self.K  # [block_size, d_key]
        v = embedding_matrix @ self.V  # [block_size, d_value]

        product = q @ k.T  # [block_size, block_size]
        scaled = product / (self.d_key ** 0.5)  # [block_size, block_size]
        # TODO: add a mask here
        softmaxed = F.softmax(scaled, dim=1)  # [block_size, block_size]
        return softmaxed @ v  # [block_size, d_value]


transformer = Transformer()

tokens = transformer.text_to_tokens("This is a test.")
print(tokens)

embedded = transformer.embed_tokens(tokens)
print(embedded)

attended = transformer.multi_head_attention(embedded)
print(attended)
