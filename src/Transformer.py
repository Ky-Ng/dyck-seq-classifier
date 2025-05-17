import torch
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


@dataclass
class TransformerConfig:
    block_size: int = 24  # Max Context length
    vocab_size: int = 2
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 36  # Hidden Size
    n_classes: int = 1  # Grammatical Function returns [0, 1]
    # d_head = 3


class CausalSelfAttention(nn.Module):
    """
    Multi Headed Attention Mechanism
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # QKV Matrices
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output project (Mixing of all of the heads)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_hs = config.n_embd // config.n_head

    def forward(self, x: torch.tensor):
        # B = Batch Size, T = sequence length, C = n_embd = "_" for now
        B, T, C = x.size()

        # Step 1) Generate the K,Q,V values (matrix format)
        """
        x = (B,T,C)
        c_attn = (C, 3*C)
        qkv = (B,T,C) x (C, 3*C) = (B,T,3*C)
        """
        qkv = self.c_attn(x)

        # in the T,3*C, we'll split the massive matrix into three separate TxC matrices
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Split each QKV matrix into its respective heads
        # First, split the TxC with vertical lines in the matrix denoting each of the heads: shape = (B, T, self.n_head, self.n_hs)
        # Then create shape (B, self.n_head, T, self.n_hs);
        # allow processing for each head in parallel with (T x self.n_hs); imagine self.n_head as a 3rd "depth" dimension
        q = q.view(B, T, self.n_head, self.n_hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.n_hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_hs).transpose(1, 2)

        # Step 2) Scaled Dot Product Attention
        """
        q = (B, self.n_head, T, self.n_hs)
        k.transpose(-2, -1) = (B, self.n_head, self.n_hs, T)
        attn = (B, self.n_head, T, T)
        - Note: each row, attn[i] in the attn matrix is how much token[i] should attend to all other token[j]
        - We'll apply this in the next part to each component of all v[j] in the self.n_embd
        """
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.n_hs))

        # Apply softmax row-wise
        attn = F.softmax(attn, dim=-1)

        # Step 3) Reduce/Take weighted sum of the value vectors
        """
        attn = (B, self.n_head, T, T)
        v = (B, self.n_head, T, self.n_hs) 
        y = (B, self.n_head, T, self.n_hs) where each row y[i] corresponds to new rep of token[i]

        applies the weighting of the vectors to each component of the vectors
        """
        y = attn @ v

        # Step 4) Concatentate the vectors tip to tip
        # y = (B, T, self.n_head, self.n_hs) => (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 5) Final mixing
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """
    Feed Forward Network projecting to 4 * model dimension (config.n_embd)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        # project to 4 * the model dimension in FFN

        # c stand for component (part of a nn.Module component rather than a block in the diagram)
        # fc = fully connected
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        # proj = project back down
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Normalization 1
        self.ln_1 = nn.LayerNorm(config.n_embd)

        # MHA Attention
        self.attn = CausalSelfAttention(config)

        # Normalization 2
        self.ln_2 = nn.LayerNorm(config.n_embd)

        # Feed Forward Network
        self.mlp = MLP(config)

    def forward(self, x):
        # Residually apply the layer normalization to the input, then pass to attention
        x = x + self.attn(self.ln_1(x))

        # Residually apply the layer normalization to output of MHA, then pass to FFN
        x = x + self.mlp(self.ln_2(x))
        return x


class Pooling(nn.Module):
    """
    Takes input of (B,T,d) => (B,d) by averaging over the Time dimension

    (B,T,d) input
    [
     ----h_1----
     ----h_2----
         ...
     ----h_T----
    ]

    (B,d) output where each vector is a pooled average of all h_i
    [
     ----Pooled----
    ]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=1)


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # Embedding = wrapper for a tensor that you can index into the rows
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),

                # Index each layer from [0, config.n_layer); gray image in AIAYN
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),

                # special for GPT2
                ln_f=nn.LayerNorm(config.n_embd),

            )
        )

        # Average Pooling across final hidden representations (B,T,d) => (B,d)
        self.pool = Pooling()

        # Softmax regression (apply sigmoid elementwise on (B,d) shape --> (B,1))
        self.classification_head = nn.Linear(config.n_embd, config.n_classes)

    def forward(self, idx, targets=None):

        # (B, T) batch size and token size
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence request of {T} larger than {self.config.block_size}"

        # Generate positional and token embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        x = tok_emb + pos_emb  # Implicit broadcasting of pos_emb to every batch

        # Propogate input through each transformer block
        for block in self.transformer.h:
            x = block(x)

        # Run the final encoder's hidden representation through the layer norm and Language Head
        x = self.transformer.ln_f(x)
        x = self.pool(x)
        logits = self.classification_head(x)

        if targets is not None:
            # Flatten by (1) fixing the logits into (B*T, vocab_size) and (B*T)
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1, logits.size(-1)), targets.view(-1).float())
            return logits, loss
        return logits
