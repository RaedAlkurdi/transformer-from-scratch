"""Load a trained checkpoint and generate text one character at a time."""
import math

import torch
import torch.nn.functional as F
from .model import TransformerLanguageModel
import argparse
from .data import build_mappings, encode, decode

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    config = checkpoint["model_config"]
    chars = checkpoint["chars"]

    model = TransformerLanguageModel(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, chars

def generate(model, token_ids, context_length, max_new_tokens, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite number greater than zero")
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(token_ids[:, -context_length:])
            probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)

    return token_ids

def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained Transformer."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="emma")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Positive sampling temperature; lower values favor likely characters.")

    args = parser.parse_args()
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be a finite number greater than zero")
    model, config, chars = load_checkpoint(args.checkpoint)
    stoi, itos = build_mappings(chars)

    prompt_ids = torch.tensor(
        encode(args.prompt, stoi),
        dtype=torch.long,
    ).unsqueeze(0)

    generated_ids = generate(
        model,
        prompt_ids,
        context_length=config["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    generated_text = decode(generated_ids[0].tolist(), itos)
    print(generated_text)
    
if __name__ == "__main__":
    main()


    