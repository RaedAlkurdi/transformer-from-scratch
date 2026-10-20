# Character-level Transformer from scratch

A small decoder-only Transformer in PyTorch, trained to predict the next character in *Alice's Adventures in Wonderland*.

I started with a bigram model and name generation, then built up the attention model step by step. This is my first Transformer project. 

The model learns some spelling, dialogue, and punctuation patterns. It still invents words, repeats itself, and loses track of what a sentence is saying.

## Model

The architecture uses ordinary PyTorch layers and explicit attention calculations:

```text
Character IDs
  -> token embeddings + learned position embeddings
  -> LayerNorm -> two causal attention heads -> projection -> residual addition
  -> LayerNorm -> feed-forward network -> residual addition
  -> vocabulary logits
```

Each head computes queries, keys, and values. A triangular mask prevents a position from attending to later characters. The two head outputs are concatenated and projected back to the embedding size. The feed-forward network expands that size by four, applies ReLU, then projects back.

The model returns logits with shape `(batch_size, sequence_length, vocab_size)`. Training targets are the input shifted forward by one character. Cross-entropy compares the logits at each position with the next character.

The current model has one block and two heads. It uses no dropout, pretrained weights, or `nn.Transformer`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Download the dataset

The text comes from [Project Gutenberg eBook 11](https://www.gutenberg.org/ebooks/11), by Lewis Carroll. Dataset files and checkpoints are excluded from Git. Download and prepare the text locally:

```powershell
@'
from pathlib import Path
from urllib.request import urlopen

url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

raw_bytes = urlopen(url, timeout=60).read()
(data_dir / "alice_original.txt").write_bytes(raw_bytes)
raw_text = raw_bytes.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
body = raw_text.split("*** START OF THE PROJECT GUTENBERG EBOOK", 1)[1]
body = body.split("\n", 1)[1]
body = body.split("*** END OF THE PROJECT GUTENBERG EBOOK", 1)[0]
text = body.strip() + "\n"
(data_dir / "alice.txt").write_text(text, encoding="utf-8", newline="\n")
print(f"{len(text)} characters, {len(set(text))} unique characters")
'@ | python -
```

The edition used here produces 144,600 characters and a vocabulary of 75 characters. Front matter inside the Gutenberg markers stays in the training text. The first 90% of tokens are training data and the last 10% are validation data: 130,140 and 14,460 tokens.

The vocabulary is built from the complete text, including the held-out portion. Validation tokens do not receive gradient updates. This is one sequential split from one book; it does not measure performance on other books.

Keep the original download with its notices. Gutenberg lists this edition as public domain in the USA; its page and the downloaded file contain the distribution terms.

## Train

For a quick check that training and saving work:

```powershell
python -m src.train --data data/alice.txt --checkpoint checkpoints/smoke.pt --steps 1 --context-length 64
```

For a longer run:

```powershell
python -m src.train --data data/alice.txt --checkpoint checkpoints/alice.pt --steps 5000 --context-length 64 --embedding-size 128 --head-size 32 --learning-rate 0.001 --device cpu
```

Training uses batches of four and AdamW. It prints mean training and validation losses after the final update, evaluated over 50 sampled batches per split. The checkpoint stores the weights, model configuration, and character vocabulary.

The CLI defaults to a context length of 9; these examples explicitly use 64. The embedding size defaults to 128 and each head size to 32. `--steps` counts optimizer updates, not full passes over the dataset.

Each training command creates a new model. There is no resume option, training seed flag, or optimizer state in the saved checkpoint. Using an existing checkpoint path overwrites that file.

## Generate

After training:

```powershell
python -m src.generate --checkpoint checkpoints/alice.pt --prompt "Alice" --max-new-tokens 220 --temperature 0.8
```

Generation keeps the most recent context window, predicts the next character, samples it, and appends it to the prompt. The printed text includes the prompt.

Use a nonempty prompt containing characters from the saved vocabulary. An unseen character causes a lookup error. Temperature must be positive; lower values favor more likely characters. Sampling is random, so output changes between runs.

## Measured results

I evaluated the local `alice_directml.pt` checkpoint on CPU on October 21, 2026.

| Setting | Value |
| --- | ---: |
| Transformer blocks | 1 |
| Attention heads | 2 |
| Embedding size | 128 |
| Size of each head | 32 |
| Context length | 64 |
| Vocabulary size | 75 |
| Trainable parameters | 192,779 |
| Mean training loss | 1.5874 |
| Mean validation loss | 1.6870 |

These are character cross-entropy losses in natural-log units, averaged over 200 random batches of four per split. Evaluation reset `torch.manual_seed(1337)` immediately before sampling the training evaluation batches, followed by the validation evaluation batches. This uses more batches than the CLI's default evaluation.

The checkpoint does not record its training seed or update count. These numbers describe an evaluation of that saved checkpoint; the training command above does not reproduce its exact weights. The checkpoint is not distributed with the repository. Details are in [results/experiments.csv](results/experiments.csv).

A sample from that checkpoint, with prompt `Alice`, temperature 0.8, and sampling seed 1337:

```text
Alice; but what of the was them had matter.
This
first you know dance, _that quitell a me the Engled
little
games, and be mistupid of it,
```

This is an excerpt. The full output and settings are in [results/generated_samples.txt](results/generated_samples.txt). Recognizable words and dialogue patterns appear, but the passage is not coherent.

## Files

| Path | Contents |
| --- | --- |
| `src/data.py` | Text loading, vocabulary, encoding, decoding, splitting, and batching |
| `src/model.py` | Attention heads and the one-block Transformer |
| `src/train.py` | Training CLI, loss evaluation, and checkpoint saving |
| `src/generate.py` | Checkpoint loading and text generation CLI |
| `data/README.md` | Dataset source and preparation notes |
| `results/` | Recorded evaluation and generated sample |
| `notebooks/` | Earlier learning experiments; the original notebook contains local paths |

The Python modules are the entry points for running the project. Raw datasets and trained checkpoints stay local.

## Checks and limitations

Checks covered encoding/decoding round trips, shifted batch targets, output shapes, causal prefix consistency, parameter updates, checkpoint loading, and generation length. CPU and DirectML each passed a one-update training command followed by checkpoint loading and generation in a separate process.

These were manual checks; there is no committed automated test suite. A fresh virtual environment installation was not tested.

The model's 64-character context and single block limit what it can learn. Training uses one small book, and the output has weak grammar and long-range coherence. There is no controlled baseline comparison in the recorded results. Invalid dataset lengths and empty prompts also lack friendly error messages.

