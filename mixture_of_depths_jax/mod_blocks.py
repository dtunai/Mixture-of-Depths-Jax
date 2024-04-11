import flax.linen as nn
import jax.numpy as jnp


class MoDBlock(nn.Module):
    layer_id: int
    n_heads: int
    dim: int
    head_dim: int
    router: nn.Module
    aux_router: nn.Module = None
    norm_eps: float = 1e-6
    router_skip_blocks: int = 2
    top_k: int = 256

    def setup(self):
        self.attention = Attention(
            n_heads=self.n_heads, head_dim=self.head_dim
        )  # Implement your own Attn layer
        self.feed_forward = FeedForward(dim=self.dim)  # Implement your own FFN layer
        self.attention_norm = RMSNorm(self.dim, eps=self.norm_eps)  # Implement RMSNorm layer
        self.ffn_norm = RMSNorm(self.dim, eps=self.norm_eps)  # Implement RMSNorm layer

    def __call__(self, x, start_pos, freqs_cis, mask=None):
        batch_size, seq_len, _ = x.shape

        if self.layer_id % self.router_skip_blocks == 0 and self.router is not None:
            token_weights = self.router(x)
            topk_values, topk_indices = jnp.topk(
                token_weights.squeeze(-1), self.top_k, sorted=False
            )
            mask = jnp.zeros((batch_size, seq_len))
            mask = mask.at[jnp.arange(batch_size)[:, None], topk_indices].set(1)
        else:
            mask = jnp.ones((batch_size, seq_len))

        masked_input = x * mask[:, :, None]
        h = masked_input + self.attention(
            self.attention_norm(masked_input), start_pos, freqs_cis, mask
        )
        out = self.feed_forward(self.ffn_norm(h))

        return (
            out,
            token_weights if self.router is not None else None,
            None,
            topk_indices if self.router is not None else None,
        )
