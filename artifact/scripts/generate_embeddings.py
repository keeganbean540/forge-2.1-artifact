#!/usr/bin/env python3
"""
RAG Pipeline Embedding Generator
================================
Generates dense vector embeddings for text chunks in `artifact/chunks/`
and saves individual embedding files and a master manifest into `artifact/embeddings/`.

Supports:
- OpenAI API (e.g., text-embedding-3-small, 1536 dims)
- Ollama Local API (e.g., nomic-embed-text, all-minilm)
- Google Gemini API (e.g., text-embedding-004)
- Local deterministic feature projection (zero external dependencies, 1536 dims)
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def resolve_default_paths() -> Tuple[Path, Path]:
    """
    Intelligently resolves chunk input directory and embedding output directory
    whether the script is executed from the repository root, artifact/, or scripts/.
    """
    cwd = Path.cwd()

    # Determine chunks input dir
    chunks_candidates = [
        cwd / "artifact" / "chunks",
        cwd / "chunks",
        Path(__file__).resolve().parent.parent / "chunks",
        Path(__file__).resolve().parent.parent / "artifact" / "chunks",
    ]
    input_dir = None
    for cand in chunks_candidates:
        if cand.exists() and cand.is_dir() and any(cand.glob("*.txt")):
            input_dir = cand
            break
    if input_dir is None:
        input_dir = cwd / "artifact" / "chunks"

    # Determine embeddings output dir (must be artifact/embeddings/)
    if (cwd / "artifact").is_dir():
        output_dir = (cwd / "artifact" / "embeddings").resolve()
    elif cwd.name == "artifact":
        output_dir = (cwd / "embeddings").resolve()
    else:
        output_dir = (Path(__file__).resolve().parent.parent / "artifact" / "embeddings").resolve()

    return input_dir, output_dir


def generate_local_embedding(text: str, dim: int = 1536) -> List[float]:
    """
    Generates a deterministic, L2-normalized dense vector using n-gram feature hashing.
    Requires no external packages or network access.
    """
    words = text.lower().split()
    vec = [0.0] * dim
    if not words:
        return vec

    for i in range(len(words)):
        # Unigram hash
        w = words[i]
        h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 16) & 1) else -1.0
        vec[idx] += sign * 1.5

        # Bigram hash for semantic adjacency
        if i + 1 < len(words):
            bg = f"{w} {words[i + 1]}"
            h_bg = int(hashlib.sha256(bg.encode("utf-8")).hexdigest(), 16)
            idx_bg = h_bg % dim
            sign_bg = 1.0 if ((h_bg >> 16) & 1) else -1.0
            vec[idx_bg] += sign_bg * 2.0

    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        return [round(x / norm, 7) for x in vec]
    return vec


def request_openai_embeddings(
    texts: List[str], model: str = "text-embedding-3-small", api_key: Optional[str] = None
) -> List[List[float]]:
    """Generates embeddings using the OpenAI API."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    payload = json.dumps({"input": texts, "model": model}).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Sort embeddings by return index
    data["data"].sort(key=lambda x: x["index"])
    return [item["embedding"] for item in data["data"]]


def request_ollama_embeddings(
    text: str, model: str = "nomic-embed-text", host: str = "http://localhost:11434"
) -> List[float]:
    """Generates an embedding for a single text using a local Ollama instance."""
    url = f"{host.rstrip('/')}/api/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embedding"]


def request_gemini_embeddings(
    text: str, model: str = "text-embedding-004", api_key: Optional[str] = None
) -> List[float]:
    """Generates an embedding using the Google Gemini API."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={key}"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]}
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["embedding"]["values"]


def detect_provider(requested_provider: str) -> Tuple[str, str, int]:
    """
    Selects provider, default model, and vector dimension.
    """
    if requested_provider != "auto":
        provider = requested_provider.lower()
    else:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("GEMINI_API_KEY"):
            provider = "gemini"
        else:
            # Check if local Ollama is running
            try:
                req = urllib.request.Request("http://localhost:11434/api/tags")
                with urllib.request.urlopen(req, timeout=1):
                    provider = "ollama"
            except Exception:
                provider = "local"

    if provider == "openai":
        return "openai", "text-embedding-3-small", 1536
    elif provider == "ollama":
        return "ollama", "nomic-embed-text", 768
    elif provider == "gemini":
        return "gemini", "text-embedding-004", 768
    else:
        return "local", "deterministic-hash-1536", 1536


def generate_embeddings_pipeline(
    input_dir: Path,
    output_dir: Path,
    provider: str = "auto",
    model: Optional[str] = None,
    dim: Optional[int] = None,
    batch_size: int = 50,
):
    """
    Main embedding ingestion loop.
    Reads chunk files, generates dense vector representations, and saves results.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    provider_name, default_model, default_dim = detect_provider(provider)
    model_name = model or default_model
    embedding_dim = dim or default_dim

    print(f"=== RAG Embedding Generation ===")
    print(f"Source chunks dir   : {input_dir.resolve()}")
    print(f"Output embeddings dir: {output_dir.resolve()}")
    print(f"Provider             : {provider_name}")
    print(f"Model                : {model_name}")
    print(f"Embedding dimension  : {embedding_dim}")

    if provider_name == "local":
        print(
            "Note: Generating local deterministic 1536-dim unit embeddings.\n"
            "      (Set OPENAI_API_KEY or run Ollama for neural embeddings)\n"
        )

    # Load chunk metadata if manifest.json exists
    manifest_file = input_dir / "manifest.json"
    chunk_meta_map = {}
    if manifest_file.exists():
        try:
            items = json.loads(manifest_file.read_text(encoding="utf-8"))
            for it in items:
                chunk_meta_map[it.get("filename")] = it
        except Exception as e:
            print(f"Warning: Failed to parse manifest.json: {e}")

    chunk_files = sorted(input_dir.glob("*.txt"))
    if not chunk_files:
        print(f"No .txt chunk files found in {input_dir}")
        return

    print(f"Processing {len(chunk_files)} text chunks...")

    total_chunks = len(chunk_files)
    all_embeddings_manifest = []
    start_time = time.time()

    if provider_name == "openai":
        # Process in batches for efficient API throughput
        for i in range(0, total_chunks, batch_size):
            batch_files = chunk_files[i : i + batch_size]
            batch_texts = [f.read_text(encoding="utf-8") for f in batch_files]

            print(f"  Embedding batch [{i + 1} - {min(i + batch_size, total_chunks)} / {total_chunks}] via OpenAI...")
            vectors = request_openai_embeddings(batch_texts, model=model_name)

            for f_path, text, vec in zip(batch_files, batch_texts, vectors):
                save_chunk_embedding(f_path, text, vec, model_name, embedding_dim, output_dir, chunk_meta_map, all_embeddings_manifest)
    else:
        # Item-by-item processing
        for idx, f_path in enumerate(chunk_files, start=1):
            text = f_path.read_text(encoding="utf-8")

            if provider_name == "ollama":
                vec = request_ollama_embeddings(text, model=model_name)
            elif provider_name == "gemini":
                vec = request_gemini_embeddings(text, model=model_name)
            else:
                vec = generate_local_embedding(text, dim=embedding_dim)

            save_chunk_embedding(f_path, text, vec, model_name, embedding_dim, output_dir, chunk_meta_map, all_embeddings_manifest)

            if idx % 25 == 0 or idx == total_chunks:
                print(f"  Progress: [{idx:3d}/{total_chunks:3d}] chunks embedded ({idx / total_chunks * 100:.1f}%)")

    # Save comprehensive manifest
    manifest_out_path = output_dir / "embeddings_manifest.json"
    manifest_out_path.write_text(
        json.dumps(all_embeddings_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    elapsed = time.time() - start_time
    print(f"\nEmbedding generation complete in {elapsed:.2f}s!")
    print(f"Total embeddings generated : {len(all_embeddings_manifest)}")
    print(f"Embedding vector dimension : {embedding_dim}")
    print(f"Directory                  : {output_dir.resolve()}")
    print(f"Master manifest saved to   : {manifest_out_path.resolve()}")


def save_chunk_embedding(
    chunk_path: Path,
    text: str,
    vector: List[float],
    model_name: str,
    dimension: int,
    output_dir: Path,
    meta_map: Dict[str, Any],
    manifest_collector: List[Dict],
):
    """Saves per-chunk embedding JSON and registers entry in manifest."""
    stem = chunk_path.stem
    out_file = output_dir / f"{stem}.json"

    meta = meta_map.get(chunk_path.name, {})

    record = {
        "chunk_id": meta.get("chunk_id", stem),
        "source_file": meta.get("source_file", chunk_path.name.split("_chunk_")[0] + ".txt"),
        "document_title": meta.get("document_title", ""),
        "chunk_index": meta.get("chunk_index", 1),
        "chunk_filename": chunk_path.name,
        "token_count": meta.get("token_count", len(text.split())),
        "char_count": len(text),
        "model": model_name,
        "dimension": dimension or len(vector),
        "embedding": vector,
    }

    out_file.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    manifest_collector.append(record)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dense vector embeddings for RAG pipeline chunks."
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        default=None,
        help="Directory containing text chunks (default: auto-detected, e.g. artifact/chunks/)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory to save embeddings (default: artifact/embeddings/)",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["auto", "openai", "ollama", "gemini", "local"],
        default="auto",
        help="Embedding provider (default: auto)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Embedding model name (e.g., text-embedding-3-small, nomic-embed-text)",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=int,
        default=1536,
        help="Vector dimension (default: 1536)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Batch size for API providers (default: 50)",
    )

    args = parser.parse_args()

    default_in, default_out = resolve_default_paths()
    input_dir = args.input_dir or default_in
    output_dir = args.output_dir or default_out

    try:
        generate_embeddings_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            provider=args.provider,
            model=args.model,
            dim=args.dim,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"Error during embedding generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

