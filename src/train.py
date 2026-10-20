"""Model training, evaluation, and checkpoint saving."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from .data import (
    build_mappings,
    build_vocabulary,
    encode,
    get_batch,
    read_text,
    split_data,
)
from .model import TransformerLanguageModel


def evaluate_loss(model, data, batch_size, context_length, num_batches=50):
    device = next(model.parameters()).device
    losses = []

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, batch_size, context_length)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )
            losses.append(loss.item())
    return sum(losses) / len(losses)


def train_model(model, data, batch_size, context_length, num_steps, learning_rate):
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []

    for _ in range(num_steps):
        x, y = get_batch(data, batch_size, context_length)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def save_checkpoint(model, model_config, chars, checkpoint_path):
    checkpoint = {
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "model_config": model_config,
        "chars": chars,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train a character-level Transformer.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--context-length", type=int, default=9)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--head-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "directml"], default="cpu")
    args = parser.parse_args()

    if args.device == "directml":
        try:
            import torch_directml
        except ImportError:
            parser.error("DirectML is not installed; use your DirectML Python interpreter.")
        device = torch_directml.device()
    else:
        device = torch.device("cpu")
    print(f"Training device: {device}")

    text = read_text(args.data)
    chars = build_vocabulary(text)
    stoi, _ = build_mappings(chars)
    token_ids = torch.tensor(encode(text, stoi), dtype=torch.long)
    training_data, validation_data = split_data(token_ids)

    print(f"Vocabulary size: {len(chars)}")
    print(f"Training tokens: {len(training_data)}")
    print(f"Validation tokens: {len(validation_data)}")

    model_config = {
        "vocab_size": len(chars),
        "embedding_size": args.embedding_size,
        "context_length": args.context_length,
        "head_size": args.head_size,
    }
    model = TransformerLanguageModel(**model_config).to(device)
    training_losses = train_model(
        model=model,
        data=training_data,
        batch_size=4,
        context_length=model_config["context_length"],
        num_steps=args.steps,
        learning_rate=args.learning_rate,
    )
    print(f"Training updates completed: {len(training_losses)}")

    model.eval()
    training_loss = evaluate_loss(
        model, training_data, batch_size=4,
        context_length=model_config["context_length"],
    )
    validation_loss = evaluate_loss(
        model, validation_data, batch_size=4,
        context_length=model_config["context_length"],
    )
    print(f"Average training loss: {training_loss:.4f}")
    print(f"Average validation loss: {validation_loss:.4f}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, model_config, chars, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
