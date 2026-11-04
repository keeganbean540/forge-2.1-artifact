#!/bin/bash

# Generate chunks
cd artifact
python scripts/chunk_documents.py
python scripts/generate_embeddings.py


