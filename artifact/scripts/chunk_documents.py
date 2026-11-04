#!/usr/bin/env python3
"""
RAG Pipeline Document Chunker
=============================
Chunks text documents from `document-store/` using a sliding window of tokens
with configurable chunk size (default: 500 tokens) and overlap (default: 50 tokens).
Outputs chunk files and a metadata manifest into `chunks/`.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def tokenize(text: str) -> List[str]:
    """
    Lossless tokenization that splits text into word tokens while preserving whitespace.
    Concatenating all tokens (''.join(tokens)) reconstructs the exact original text.
    """
    tokens = re.findall(r'\S+\s*', text)
    return tokens


def chunk_tokens(
    tokens: List[str], chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Tuple[int, int, str]]:
    """
    Splits tokens into overlapping chunks using a sliding window.
    Returns a list of tuples: (start_token_idx, end_token_idx, chunk_text)
    """
    if not tokens:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(f"Overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size}).")

    stride = chunk_size - chunk_overlap
    n = len(tokens)
    chunks = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = "".join(tokens[start:end])
        chunks.append((start, end, chunk_text))
        if end == n:
            break
        start += stride

    return chunks


def resolve_default_path(target_name: str) -> Path:
    """
    Resolves directory path checking current working directory,
    script parent directories, and artifact/ subdirectories.
    """
    candidates = [
        Path(target_name),
        Path("artifact") / target_name,
        Path(__file__).resolve().parent / target_name,
        Path(__file__).resolve().parent.parent / target_name,
        Path(__file__).resolve().parent.parent / "artifact" / target_name,
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return Path(target_name)


def process_documents(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Reads all text files in input_dir, chunks each document, writes chunk files,
    and returns a manifest list of all generated chunks.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect and sort text files
    doc_files = sorted(
        [f for f in input_dir.iterdir() if f.is_file() and f.suffix in ('.txt', '.md')]
    )

    if not doc_files:
        print(f"Warning: No .txt or .md files found in {input_dir}")
        return []

    print(f"Processing {len(doc_files)} documents from: {input_dir.resolve()}")
    print(f"Chunk settings: size={chunk_size} tokens, overlap={chunk_overlap} tokens (stride={chunk_size - chunk_overlap})\n")

    manifest = []
    total_tokens_all = 0
    total_chunks_all = 0

    for doc_idx, doc_path in enumerate(doc_files, start=1):
        content = doc_path.read_text(encoding="utf-8")
        tokens = tokenize(content)
        total_tokens_all += len(tokens)

        # Extract document title if available from first non-empty line
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        doc_title = lines[0] if lines else doc_path.stem

        chunks = chunk_tokens(tokens, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total_chunks_all += len(chunks)

        stem = doc_path.stem

        for chunk_idx, (start_tok, end_tok, chunk_text) in enumerate(chunks, start=1):
            chunk_filename = f"{stem}_chunk_{chunk_idx:03d}.txt"
            chunk_file_path = output_dir / chunk_filename
            chunk_file_path.write_text(chunk_text, encoding="utf-8")

            token_count = end_tok - start_tok
            char_count = len(chunk_text)

            chunk_meta = {
                "chunk_id": f"{stem}_{chunk_idx:03d}",
                "source_file": doc_path.name,
                "document_title": doc_title,
                "document_index": doc_idx,
                "chunk_index": chunk_idx,
                "total_chunks_in_doc": len(chunks),
                "token_count": token_count,
                "char_count": char_count,
                "start_token": start_tok,
                "end_token": end_tok,
                "overlap_previous": chunk_overlap if chunk_idx > 1 else 0,
                "overlap_next": chunk_overlap if chunk_idx < len(chunks) else 0,
                "filename": chunk_filename,
            }
            manifest.append(chunk_meta)

        print(
            f"  [{doc_idx:02d}/{len(doc_files):02d}] {doc_path.name:16s} : "
            f"{len(tokens):5d} tokens -> {len(chunks):2d} chunks"
        )

    # Save manifest.json in output_dir
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSuccessfully generated {total_chunks_all} chunks across {len(doc_files)} documents.")
    print(f"Total tokens processed: {total_tokens_all}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Metadata manifest saved to: {manifest_path.resolve()}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Chunk text files from document-store/ for RAG vector embedding."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=None,
        help="Path to directory containing input documents (default: document-store/)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Path to directory to write chunk files (default: chunks/)",
    )
    parser.add_argument(
        "--chunk-size",
        "-s",
        type=int,
        default=500,
        help="Token count per chunk (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        "-v",
        type=int,
        default=50,
        help="Token overlap between consecutive chunks at both ends (default: 50)",
    )

    args = parser.parse_args()

    # Resolve default paths if not explicitly specified
    input_dir = args.input_dir or resolve_default_path("document-store")
    output_dir = args.output_dir or resolve_default_path("chunks")

    try:
        process_documents(
            input_dir=input_dir,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except Exception as e:
        print(f"Error during document chunking: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

