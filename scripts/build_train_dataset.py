import json
import argparse
import os
from datasets import load_dataset


def stream_code_dataset(output_path: str, num_samples: int = 50000, skip_samples: int = 0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Streaming CodeSearchNet and saving to {output_path}...")

    dataset = load_dataset("sentence-transformers/codesearchnet", split="train", streaming=True)

    if skip_samples > 0:
        dataset = dataset.skip(skip_samples)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break

            docstring = item.get('comment', '').strip()
            code = item.get('code', '').strip()

            if not docstring or not code:
                continue

            triplet = {
                "query": docstring,
                "positive_doc": code,
                "negative_doc": ""
            }
            f.write(json.dumps(triplet, ensure_ascii=False) + '\n')

    print(f"Successfully saved {num_samples} JSONL records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/processed/code_triplets.jsonl")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    stream_code_dataset(args.out, args.samples, args.skip)