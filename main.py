import pickle

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numpy import sin, cos
from torch import optim
from torch._dynamo.utils import tabulate
from torch.nn import init

from tokenizer import PreambleTokenizer

vocab_size = 40
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

def visualize_attention_maps(model, prompt):
    tokens = model.text_to_tokens(prompt)
    _ = model.forward(tokens)
    for i, layer in enumerate(model.attention):
        for j, head in enumerate(layer.attention_heads):
            plt.figure(figsize=(10, 10))
            plt.imshow(head.attention_scores)
            plt.title(f'Layer {i + 1}, Head {j + 1}')
            plt.colorbar()
            plt.show()

def visualize_attention_outputs(model, prompt):
    tokens = model.text_to_tokens(prompt)
    _ = model.forward(tokens)
    for i, layer in enumerate(model.attention):
        plt.figure(figsize=(10, 10))
        plt.imshow(layer.attention_output)
        plt.title(f'Layer {i + 1} Attention Output')
        plt.colorbar()
        plt.show()

def visualize_layer_outputs(model, prompt):
    tokens = model.text_to_tokens(prompt)
    _ = model.forward(tokens)
    for i, output in enumerate(model.layer_outputs):
        plt.figure(figsize=(10, 10))
        plt.imshow(output)
        plt.title(f'Layer {i + 1} Output')
        plt.colorbar()
        plt.show()


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

        self.embedding_dimension = 20
        self.embedding_matrix = nn.Parameter(t.empty(self.embedding_dimension, vocab_size, dtype=t.float32).to(DEVICE))
        init.xavier_uniform_(self.embedding_matrix)

        self.de_embedding_matrix = nn.Parameter(t.empty(vocab_size, self.embedding_dimension, dtype=t.float32).to(DEVICE))
        init.xavier_uniform_(self.de_embedding_matrix)

        self.d_key = 8
        self.d_value = 7
        self.num_attention_heads = 3

        # Initialize N groups of self.num_attention_heads multihead attentions
        self.N = 1
        self.attention = nn.ModuleList([MaskedMultiheadAttention(self.d_key, self.d_value, self.num_attention_heads, self.blocksize, self.embedding_dimension) for _ in range(self.N)])

        # Initialize N feedforward layers
        self.feedforward_layers = nn.ModuleList([nn.Linear(self.embedding_dimension, self.embedding_dimension) for _ in range(self.N)])

        self.norm = nn.LayerNorm(self.embedding_dimension)


        # Visualizations
        self.layer_outputs = []

    def text_to_tokens(self, input):
        tokens = PreambleTokenizer.encode_text(input)
        return t.tensor(np.asarray(tokens)).to(DEVICE)

    def embed_tokens(self, tokens):
        """
        :param tokens: a [blocksize] tensor of tokens
        :return: a [blocksize, embedding_dimension] tensor of positionally encoded embeddings
        """

        # One hot embedding: takes in a [batchsize, blocksize] tensor of tokens and returns a [batchsize, blocksize, vocab_size] tensor of one-hot embeddings
        one_hot = F.one_hot(t.tensor(tokens).to(DEVICE), num_classes=vocab_size).float()

        embedded = self.embedding_matrix @ one_hot.T

        # Positional encoding
        # Create a row of sin(w1 * 0), sin(w1 * 1), sin(w1 * 2), ..., sin(w1 * blocksize)
        # Followed by a row of cos(w1 * 0), cos(w1 * 1), cos(w1 * 2), ..., cos(w1 * blocksize)
        num_tokens = len(tokens)
        pos_enc = t.tensor([sin(i) for i in range(num_tokens)] + [cos(i) for i in range(num_tokens)]).reshape(2,
                                                                                                              num_tokens).to(DEVICE)
        for k in range(1, self.embedding_dimension // 2):
            w_k = 1 / (10000 ** (2 * k / self.embedding_dimension))
            sine_row = t.tensor([sin(w_k * i) for i in range(num_tokens)]).to(DEVICE)
            cosine_row = t.tensor([cos(w_k * i) for i in range(num_tokens)]).to(DEVICE)
            pos_enc = t.vstack([pos_enc, sine_row, cosine_row])

        # Add the positional encoding to the embedding
        embedded = embedded + pos_enc

        # Create a [embedding_dimension, blocksize] matrix of zeros, and paste the embedded tokens into it
        final_embedding = t.zeros((self.blocksize, self.embedding_dimension)).to(DEVICE).T
        final_embedding[:embedded.shape[0], :embedded.shape[1]] = embedded

        return final_embedding.T

    def forward(self, tokens):
        embeddings = self.embed_tokens(tokens)

        # Pass the embeddings through the N multihead attention layers
        for i in range(self.N):
            attended = self.attention[i](embeddings)
            added = embeddings + attended
            normed = self.norm(added)
            feedforwarded = self.feedforward_layers[i](normed)
            nextadded = normed + feedforwarded
            nextnormed = self.norm(nextadded)

            self.layer_outputs.append(embeddings.detach().cpu().numpy())  # save for visualization

            embeddings = nextnormed

        # Apply the final linear layer
        output = embeddings @ self.de_embedding_matrix.T

        # Softmax
        output = F.softmax(output, dim=1)

        return output

    def generate_text(self, prompt):
        tokens = self.text_to_tokens(prompt)


        for i in range(100):
            output = self.forward(tokens[-self.blocksize:])
            next_token = t.argmax(output[-1])

            # Append the next token to the tokens tensor array
            tokens = t.cat([tokens, next_token.unsqueeze(0)])

        return PreambleTokenizer.decode_tokens(tokens.detach().cpu().numpy().tolist())

    def view_probs(self, prompt):
        tokens = self.text_to_tokens(prompt)

        output = self.forward(tokens)

        # Get the top ten token probabilities
        top_ten = t.topk(output[-1], 10)

        # Display the top ten decoded tokens, with their probabilities in a nice ascii table
        print(tabulate([[PreambleTokenizer.decode_tokens([i.item()]), j.item()] for i, j in zip(top_ten.indices, top_ten.values)], headers=['Token', 'Probability']))

class MaskedMultiheadAttention(nn.Module):
    def __init__(self, d_key, d_value, num_attention_heads, blocksize, embedding_dimension):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.num_attention_heads = num_attention_heads
        self.blocksize = blocksize
        self.embedding_dimension = embedding_dimension

        self.attention_heads = nn.ModuleList([AttentionHead(self.d_key, self.d_value, self.blocksize, self.embedding_dimension) for _ in range(self.num_attention_heads)])

        self.multi_head_linear = nn.Linear(self.num_attention_heads * self.d_value, self.embedding_dimension)

    def forward(self, embedding_matrix):
        # Concatenate the outputs of each attention head
        concatenated = t.cat([head.attention(embedding_matrix) for head in self.attention_heads], dim=1)

        self.attention_output = concatenated.detach().cpu().numpy()  # save for visualization

        # Multiply by the linear layer
        # assert (concatenated @ self.multi_head_linear).shape == (self.blocksize, self.embedding_dimension)
        return self.multi_head_linear(concatenated)

class AttentionHead(nn.Module):
    def __init__(self, d_key, d_value, block_size, embedding_dimension):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.block_size = block_size
        self.embedding_dimension = embedding_dimension

        self.Q = nn.Parameter(t.empty(embedding_dimension, d_key).to(DEVICE))
        self.K = nn.Parameter(t.empty(embedding_dimension, d_key).to(DEVICE))
        self.V = nn.Parameter(t.empty(embedding_dimension, d_value).to(DEVICE))

        # Initialization using Xavier Uniform method
        nn.init.xavier_uniform_(self.Q)
        nn.init.xavier_uniform_(self.K)
        nn.init.xavier_uniform_(self.V)

        self.mask = t.triu(t.ones(self.block_size, self.block_size) * -1e9, diagonal=1)
        self.mask = self.mask.to(t.float32)
        self.mask = self.mask.to(DEVICE)

    def attention(self, embedding_matrix):
        q = embedding_matrix @ self.Q  # [block_size, d_key]
        k = embedding_matrix @ self.K  # [block_size, d_key]
        v = embedding_matrix @ self.V  # [block_size, d_value]

        product = q @ k.T  # [block_size, block_size]
        scaled = product / (self.d_key ** 0.5)  # [block_size, block_size]
        # Add a mask here
        masked = scaled + self.mask

        softmaxed = F.softmax(masked, dim=-1)  # [block_size, block_size]
        self.attention_scores = softmaxed.detach().cpu().numpy()  # save for visualization
        return softmaxed @ v  # [block_size, d_value]






# Training loop for the transformer

# 1. Load the data. It's an array in tokenized_samples_targets.pkl, where each row contains the tokens of a sample and the targets.
# E.g., pairs[0] is (array([ 9396,    36,  7859, 15442,  3969,  5464, 19135,  7450,  5803,
#         6206,  9975,    15,  8715,  3770, 11497, 10963,    36,  7859,
#        15442,  1724], dtype=uint16), array([   36,  7859, 15442,  3969,  5464, 19135,  7450,  5803,  6206,
#         9975,    15,  8715,  3770, 11497, 10963,    36,  7859, 15442,
#         1724, 10467], dtype=uint16))

sample_data = pickle.load(open("tokenized_samples_targets.pkl", "rb"))

# 2. Create a transformer
transformer = Transformer()
transformer = transformer.to(DEVICE)

for name, param in transformer.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

# 4. Create an optimizer
optimizer = optim.Adam(transformer.parameters(), lr=0.001)

# 5. Train the transformer
for epoch in range(1002):
    for i, (sample, target) in enumerate(sample_data):
        sample, target = t.tensor(sample).to(DEVICE), t.tensor(target).to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = transformer(sample)

        # Calculate loss
        # This takes the [target]th element of each row of output, and sums the logs of 'em
        loss = t.sum(-t.log(output[t.arange(output.size(0)), t.tensor(target.to(t.int32))]))

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        if i % 1000 == 0:
            print(f"Epoch {epoch}, sample {i}, loss {loss.detach().cpu().numpy().item()}")

            # Generate some text
            print(transformer.generate_text("We the People of the United States, in"))

            if epoch % 1000 == 0 and i % 1000 == 0 and epoch > 2:
                visualize_attention_maps(transformer, "We the People of the ")
                visualize_attention_outputs(transformer, "We the")
                visualize_layer_outputs(transformer, "We the")

