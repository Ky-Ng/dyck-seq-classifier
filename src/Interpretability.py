import matplotlib.pyplot as plt
import torch

from Tokenizer import Tokenizer
from Transformer import Transformer


class Interpretability():
    def __init__(self, tokenizer: Tokenizer, model: Transformer):
        self.tokenizer = tokenizer
        self.model = model

    def interpret(self, seq: str):
        # Tokenize Sequence
        tokenized = self.tokenizer.encode(seq)
        tokenized = torch.tensor(tokenized)

        # Make shape (T) => (B=1,T)
        if tokenized.dim() == 1:
            tokenized = tokenized.unsqueeze(0)

        # Run forward pass and extract attention matrix
        pred, attn_matrix = self.model.get_attentions(tokenized)

        # Output Prediction and plots
        print(
            f"Interpreting sequence {seq} as grammatical: {pred.item():.03f}")
        self.plot_attention_grid(attn_matrix, seq)

    def plot_attention_grid(self, attn_matrix: torch.Tensor, tokens: list[str]):
        """
        Note: Written by GPT4o
        Plot a grid of attention matrices:
        - Columns = Layers (left = early)
        - Rows = Heads (top = Head 0)
        - Each heatmap is T x T with dynamic contrast scaling
        """

        n_layer, n_head, T, _ = attn_matrix.shape
        config = self.model.get_config()
        n_embd = config.n_embd
        
        fig, axes = plt.subplots(
            n_head, n_layer, figsize=(n_layer * 2.2, n_head * 2.2))

        fig.suptitle(
            f"Attention Matrix Heatmaps: {n_head} Heads | {n_layer} Layers | {n_embd} Hidden Size",
            fontsize=12,
            y=0.98
        )

        for i in range(n_layer):
            for j in range(n_head):
                ax = axes[j, i] if n_head > 1 else axes[i]
                attn = attn_matrix[i, j].numpy()

                vmin = attn.min()
                vmax = attn.max()
                im = ax.imshow(attn, cmap="plasma",
                               aspect="equal", vmin=vmin, vmax=vmax)

                ax.set_xticks(range(T))
                ax.set_yticks(range(T))

                ax.set_xticklabels(tokens, fontsize=6)
                ax.set_yticklabels(tokens, fontsize=6)

                ax.tick_params(length=0)
                ax.set_title(f"H{j} | L{i}", fontsize=6)

        plt.tight_layout()
        plt.show()
