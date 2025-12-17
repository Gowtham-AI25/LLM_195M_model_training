"""
LLM Tokenization Pipeline (Correct Version)
-------------------------------------------
- YAML-driven
- CLI runnable
- Multi-file
- Fixed-length sequences ONLY
- Partial batches SAVED
- Token remainders DISCARDED (correct for LLM pretraining)
"""

import re
import os
import yaml
import argparse
import duckdb
import torch
from typing import List, Generator
from transformers import AutoTokenizer


# --------------------------------------------------
# Normalization (UNCHANGED)
# --------------------------------------------------
def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# --------------------------------------------------
# Token stream (FIXED SEQ, remainder discarded)
# --------------------------------------------------
def token_stream(
    texts: List[str],
    tokenizer,
    max_seq: int,
    eos_token: str,
) -> Generator[List[int], None, None]:
    buffer: List[int] = []

    for doc in texts:
        token_ids = tokenizer.encode(
            doc + eos_token, add_special_tokens=False
        )
        buffer.extend(token_ids)

        while len(buffer) >= max_seq:
            yield buffer[:max_seq]
            buffer = buffer[max_seq:]

    #  remainder tokens are intentionally discarded


# --------------------------------------------------
# Batchify (partial batch SAVED)
# --------------------------------------------------
def batchify(
    token_chunks: Generator[List[int], None, None],
    batch_size: int,
) -> Generator[torch.Tensor, None, None]:
    batch: List[List[int]] = []

    for chunk in token_chunks:
        batch.append(chunk)
        if len(batch) == batch_size:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []

    if batch:
        yield torch.tensor(batch, dtype=torch.long)


# --------------------------------------------------
# Process a single file (CORE LOGIC)
# --------------------------------------------------
def process_file(
    file_url: str,
    tokenizer,
    text_column: str,
    max_seq: int,
    batch_size: int,
    eos_token: str,
    output_dir: str,
    filename_prefix: str,
    start_index: int,
) -> int:
    print(f"\n[INFO] Processing file: {file_url}")

    query = f"SELECT {text_column} FROM '{file_url}'"
    df = duckdb.sql(query).df()
    print(f"[INFO] Loaded {len(df)} rows")

    texts = [normalize_text(t) for t in df[text_column].dropna()]

    chunks = token_stream(
        texts=texts,
        tokenizer=tokenizer,
        max_seq=max_seq,
        eos_token=eos_token,
    )

    batch_idx = start_index

    for batch in batchify(chunks, batch_size):
        save_path = os.path.join(
            output_dir,
            f"{filename_prefix}_{batch_idx:05d}.pt"
        )
        torch.save(batch, save_path)
        print(f"[SAVE] {batch.shape} -> {save_path}")
        batch_idx += 1

    return batch_idx


# --------------------------------------------------
# Main
# --------------------------------------------------
def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["processing"]["output_dir"], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer"]["name"],
        token=cfg["tokenizer"]["hf_token"],
        use_fast=cfg["tokenizer"]["use_fast"],
    )

    global_index = cfg["processing"]["start_index"]

    for file_url in cfg["files"]:
        global_index = process_file(
            file_url=file_url,
            tokenizer=tokenizer,
            text_column=cfg["processing"]["text_column"],
            max_seq=cfg["processing"]["max_sequence_length"],
            batch_size=cfg["processing"]["batch_size"],
            eos_token=cfg["tokenizer"]["eos_token"],
            output_dir=cfg["processing"]["output_dir"],
            filename_prefix=cfg["processing"]["filename_prefix"],
            start_index=global_index,
        )

    print(f"\n[DONE] Tokenization complete. Last batch index: {global_index}")


# --------------------------------------------------
# CLI Entry
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tokenizer_config.yaml",
    )
    args = parser.parse_args()

    main(args.config)
