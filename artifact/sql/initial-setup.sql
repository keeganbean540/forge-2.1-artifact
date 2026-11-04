-- =============================================================================
-- RAG Pipeline Vector Store Setup for PostgreSQL + pgvector
-- Target Database: ragdb
-- =============================================================================

-- 1. Enable Required Extensions
-- -----------------------------------------------------------------------------
-- 'vector' enables high-performance vector types, similarity metrics, and indexes
CREATE EXTENSION IF NOT EXISTS vector;

-- 'uuid-ossp' provides standard UUID generation for documents and chunks
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 'pg_trgm' enables trigram fuzzy matching for text search / keyword filtering
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- 2. Documents Table (Parent Corpus)
-- -----------------------------------------------------------------------------
-- Stores top-level document metadata (e.g., chapters, articles, source files)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    source_uri TEXT,
    doc_type TEXT DEFAULT 'text',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);


-- 3. Document Chunks Table (Vector Store Core)
-- -----------------------------------------------------------------------------
-- Stores chunked passages, chunk embeddings, and rich JSON metadata.
-- Default dimension: 1536 (OpenAI text-embedding-3-small / Ada-002).
-- Note: For GCP Vertex AI text-embedding-004 / Nomic Embed, change dimension to 768.
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_chunk UNIQUE (document_id, chunk_index)
);


-- 4. High-Performance Indexes for RAG Retrieval
-- -----------------------------------------------------------------------------
-- HNSW Index for Approximate Nearest Neighbor (ANN) search using Cosine distance (<=>)
-- m = 16 (bidirectional links per node), ef_construction = 64 (build-time search depth)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- B-Tree index for fast sequential chunk retrieval by document
CREATE INDEX IF NOT EXISTS idx_chunks_doc_chunk
    ON document_chunks (document_id, chunk_index);

-- GIN index on metadata for fast JSONB metadata filtering (e.g., {"chapter": "1"})
CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON document_chunks USING gin (metadata);

-- GIN full-text index on chunk content to enable hybrid search (dense vector + sparse keyword)
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
    ON document_chunks USING gin (to_tsvector('english', content));


-- 5. Trigger for Updating Document Timestamps
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();


-- 6. Stored Procedures / Helper Functions for RAG Queries
-- -----------------------------------------------------------------------------

-- Vector Similarity Search (Cosine Distance) with optional metadata filtering
CREATE OR REPLACE FUNCTION match_chunks (
    query_embedding vector(1536),
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 5,
    filter_metadata jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT,
    metadata JSONB,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (1 - (c.embedding <=> query_embedding))::float AS similarity
    FROM document_chunks c
    WHERE (filter_metadata = '{}'::jsonb OR c.metadata @> filter_metadata)
      AND (1 - (c.embedding <=> query_embedding)) >= match_threshold
    ORDER BY c.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$;


-- Hybrid Search Function (Dense Vector + Full-Text BM25/ts_rank)
CREATE OR REPLACE FUNCTION hybrid_search_chunks (
    query_text text,
    query_embedding vector(1536),
    match_count int DEFAULT 5,
    full_text_weight float DEFAULT 0.3,
    semantic_weight float DEFAULT 0.7
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT,
    metadata JSONB,
    combined_score float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.chunk_index,
        c.content,
        c.metadata,
        (
            semantic_weight * (1 - (c.embedding <=> query_embedding)) +
            full_text_weight * ts_rank_cd(to_tsvector('english', c.content), plainto_tsquery('english', query_text))
        )::float AS combined_score
    FROM document_chunks c
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;

