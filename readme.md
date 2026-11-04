
```
python3 scripts/chunk_documents.py

python3 scripts/chunk_documents.py --input-dir document-store --output-dir chunks --chunk-size 500 --chunk-overlap 50

docker compose up -d

docker compose down -v
```

In this artifact demo, the small document corpus is the 24 books of the Odyssey. Each book is a separate document.

On initial startup, the 24 documents are chunked. Each chunk is 500 tokens in size, with 50 tokens overlap at both ends. WHy 500 and 50?


