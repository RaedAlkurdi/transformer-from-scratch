# Dataset

The main text experiment uses Lewis Carroll's *Alice's Adventures in Wonderland*, from [Project Gutenberg eBook 11](https://www.gutenberg.org/ebooks/11).

Follow the download and preprocessing commands in the [project README](../README.md#download-the-dataset). They save the original download, including its notices, as `data/alice_original.txt`, normalize line endings to LF, and save the text between the Gutenberg START and END markers as `data/alice.txt`. Front matter inside the markers is retained.

The local edition has 144,600 characters and 75 unique characters. The pipeline builds the vocabulary from the complete text, then splits character IDs in order: the first 90% for training (130,140 tokens) and the last 10% for validation (14,460 tokens).

