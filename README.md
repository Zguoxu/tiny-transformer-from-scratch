# Tiny Transformer from Scratch

**English** | [简体中文](README.zh-CN.md)

**Understand how a Transformer translates English into Chinese—with a single Python file.**

A small Transformer implemented from scratch in PyTorch, covering token embeddings, positional encoding, multi-head attention, the encoder and decoder, training, and autoregressive generation.

No `nn.Transformer` or `nn.MultiheadAttention`. Designed for learners who know the basics of PyTorch and want to follow the computations inside a Transformer.

## Features

- **One Python file**: data processing, the model, training, and translation all live in `transformer.py`.
- **Core components implemented by hand**: scaled dot-product attention, multi-head attention, causal masking, and cross-attention.
- **No dataset downloads**: 8 built-in English–Chinese sentence pairs, a model dimension of 32, 2 attention heads, 2 encoder layers, and 2 decoder layers.
- **A complete workflow**: build vocabularies → train the model → save the loss curve → generate a Chinese translation.
- **Chinese code comments**: explanations of key components and training steps for Chinese-speaking learners.

## Architecture

![Transformer encoder–decoder architecture](assets/architecture.svg)

The encoder and decoder each stack two layers. In decoder cross-attention, queries come from the decoder, while keys and values come from the encoder output. Each attention and feed-forward sublayer uses a residual connection followed by layer normalization, matching the implementation.

[Download PNG](assets/architecture.png)

## Example

Input:

```text
there is a black dog under the chair
```

Output from an actual run:

```text
椅子下面有一只黑色狗
```

Training log:

```text
Epoch   1 | loss=2.1297
Epoch  20 | loss=0.0163
Epoch  40 | loss=0.0050
Epoch  60 | loss=0.0025
Epoch  80 | loss=0.0015
Epoch 100 | loss=0.0010
```

### Training loss

![Transformer training loss over 100 epochs on the 8 built-in sentence pairs](assets/loss_curve.png)

The horizontal axis shows training epochs. The vertical axis shows the average of the per-sentence cross-entropy losses within each epoch. Each run saves the curve to `assets/loss_curve.png`.

This is a small learning demo with only 8 training pairs and fixed vocabularies. Input containing words outside the source vocabulary raises an error. The example input combines familiar training words in a new way; the sample output and low training loss do not demonstrate general translation ability. Results may vary slightly across environments.

## Quick start

Verified with Python 3.12. Requires PyTorch and Matplotlib.

```bash
git clone https://github.com/Zguoxu/tiny-transformer-from-scratch.git
cd tiny-transformer-from-scratch
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python transformer.py
```

The program trains for 100 epochs, then saves and displays the loss curve. **Close the plot window** to continue and see the translation in your terminal.

To save the plot and print the translation on a server without a graphical display, run (macOS / Linux):

```bash
MPLBACKEND=Agg python transformer.py
```

## Suggested reading order

| Step | Code | What to look for |
| --- | --- | --- |
| 1. Data and vocabularies | `en_cn_pairs`, `source_ids`, `target_ids` | Converting text to token IDs; the roles of BOS and EOS |
| 2. Input representations | `TokenEmbedding`, `PositionalEncoding` | Adding token embeddings and sinusoidal positional encodings |
| 3. Attention | `attention`, `AttentionHead`, `MultiHeadAttention` | Q, K, V, scaled dot products, and concatenating heads |
| 4. Encoder | `EncoderLayer`, `Encoder` | Self-attention, feed-forward networks, residual connections, and layer normalization |
| 5. Decoder | `DecoderLayer`, `Decoder` | Causal masking and cross-attention to encoder outputs |
| 6. Complete model | `Transformer` | Encoding, decoding, and projecting to the target vocabulary |
| 7. Training and generation | Training loop, `translate` | Shifted targets, cross-entropy, and greedy token-by-token generation |

## Try it yourself

- Change `HEADS` and compare training behavior. Make sure `D` is divisible by `HEADS`.
- Change `NUM_LAYERS` to explore how model depth affects training.
- Add sentence pairs to `en_cn_pairs`, then rerun to rebuild the vocabularies and train.
- Edit the English input on the last line to try new combinations of words from the existing vocabulary.
- Read the causal mask in `attention` to understand how the decoder avoids attending to future tokens.

## Project structure

```text
tiny-transformer-from-scratch/
├── README.md
├── README.zh-CN.md
├── LICENSE
├── transformer.py
├── requirements.txt
├── .gitignore
└── assets/
    ├── architecture.svg
    ├── architecture.png
    └── loss_curve.png
```

If this project helps you understand Transformers, give it a ⭐ Star so you can revisit it later. Questions and learning discussions are welcome in Issues.

## License

Released under the [MIT License](LICENSE). You may use, modify, and redistribute this project, including commercially, provided you retain the copyright and license notices. The software is provided without warranty.
