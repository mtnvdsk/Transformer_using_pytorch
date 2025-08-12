## Transformers from Scratch (PyTorch)

A minimal, educational implementation of a Transformer encoder-decoder in pure PyTorch, built end-to-end inside a single notebook: `Transformers_from_scratch.ipynb`.

### What’s inside
- **Embedding**: token embedding with `nn.Embedding`
- **PositionalEncoding**: sinusoidal positional encodings (non-trainable buffer)
- **MultiHeadAttention**: scaled dot-product attention with multiple heads
- **TransformerBlock (Encoder block)**: MHA + residual + LayerNorm + MLP
- **TransformerEncoder**: stack of encoder blocks
- **DecoderBlock**: masked self-attention + cross-attention via encoder block
- **TransformerDecoder**: stack of decoder blocks + output projection
- **Transformer (full model)**: encoder-decoder wrapper with `forward` and `decode`

### Requirements
- Python 3.11+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- Jupyter

Install:
```bash
pip install "torch>=2.0" numpy matplotlib pandas jupyter
```

### Run the notebook
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `Transformers_from_scratch.ipynb`
3. Run cells top-to-bottom.

### Quick usage
Create toy inputs and run a forward pass:
```python
import torch
from math import sqrt

# Example hyperparameters
src_vocab_size = 11
tgt_vocab_size = 11
seq_len = 12
embed_dim = 512
num_layers = 6
n_heads = 8

# Example inputs (batch=2, length=12)
src = torch.tensor([[0,2,5,6,4,3,9,5,2,9,10,1],
                    [0,2,8,7,3,4,5,6,7,2,10,1]])
tgt = torch.tensor([[0,1,7,4,3,5,9,2,8,10,9,1],
                    [0,1,5,6,2,4,7,6,2,8,10,1]])

# Build the model (same API as in the notebook)
model = Transformer(
    embed_dim=embed_dim,
    src_vocab_size=src_vocab_size,
    target_vocab_size=tgt_vocab_size,
    seq_length=seq_len,
    num_layers=num_layers,
    expansion_factor=4,
    n_heads=n_heads,
)

out = model(src, tgt)            # shape: [batch, seq_len, tgt_vocab_size]
print(out.shape)                 # e.g., torch.Size([2, 12, 11])
```

Greedy decode demo (untrained model; outputs are not meaningful yet):
```python
src = torch.tensor([[0,2,5,6,4,3,9,5,2,9,10,1]])
trg_start = torch.tensor([[0]])  # BOS token
pred_ids = model.decode(src, trg_start)  # list of token ids
print(pred_ids)
```

### Notes and caveats
- The decoder `softmax` call should specify `dim=-1` to avoid a PyTorch warning:
  - In `TransformerDecoder.forward`: replace
    ```python
    torch.nn.functional.softmax(self.fc_out(x))
    ```
    with
    ```python
    torch.nn.functional.softmax(self.fc_out(x), dim=-1)
    ```
- The notebook shows architecture behavior on toy inputs; it does not include a training loop.
- `decode` is a simple greedy loop; results from an untrained model will be mostly constant/zeros.

### Suggested next steps
- Add a training pipeline:
  - Loss: cross-entropy over target tokens with padding mask
  - Optimizer: Adam with warmup or cosine LR schedule
  - Teacher forcing on the decoder inputs
  - Proper source/target padding and attention masks
- Improve efficiency:
  - Vectorize positional encoding creation
  - Use fused projections for Q/K/V
- Quality features:
  - Label smoothing
  - Weight tying (decoder embed and output projection)
  - Beam search decoding

### References
- Vaswani et al., “Attention Is All You Need” (2017)
- “The Annotated Transformer” (Harvard NLP)

### License
MIT
