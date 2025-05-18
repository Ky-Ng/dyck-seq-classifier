import os
import matplotlib.pyplot as plt
import torch

from Tokenizer import Tokenizer
from Transformer import Transformer
from dyck_lib import is_valid_dyck_word


class Interpretability():
    def __init__(self, tokenizer: Tokenizer, model: Transformer, save_dir: str = None):
        self.tokenizer = tokenizer
        self.model = model
        self.save_dir = save_dir

    def interpret(self, seq: str):
        # Tokenize Sequence
        tokenized = self.tokenizer.encode(seq)
        tokenized = torch.tensor(tokenized)

        # Make shape (T) => (B=1,T)
        if tokenized.dim() == 1:
            tokenized = tokenized.unsqueeze(0)

        # Run forward pass and extract attention matrix
        pred, attn_matrix = self.model.get_attentions(tokenized)
        pred_prob = pred.item()

        # Output Prediction and plots
        print(
            f"Interpreting sequence {seq} with grammatical score: {pred_prob:.03f}")
        self.plot_attention_grid(attn_matrix, seq, pred_prob)

    def plot_attention_grid(self, attn_matrix: torch.Tensor, tokens: list[str], grammatical_score: float = None):
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

        # Generate Title
        grammatical_label = "Grammatical" if is_valid_dyck_word(
            "".join(tokens)) else "Ungrammatical"
        
        grammatical_score_text = f" | Grammatical Score: {grammatical_score:.03f}" if grammatical_score else ""

        title = f"Attention Matrix Heatmaps: {n_layer} Layers | {n_head} Heads | {n_embd} Hidden Size | {grammatical_label} Seq = {tokens}"
        title += grammatical_score_text

        fig.suptitle(
            title,
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

        # Save Image if specified
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_fname = f"AttnMat_{n_layer}_Layers_{n_head}_Heads_{n_embd}_HiddenSize_{grammatical_label}_{tokens}"
            plt.savefig(os.path.join(self.save_dir, save_fname), bbox_inches="tight")
        
        plt.show()
