# Knowledge Base Tools

Various tools for loading and querying the vector store using the documents in the knowledge base [data](../data) folder

# Chunking and loading the vector store

Use **dry-run again** if you want to (re)fit the BM25 encoder **before** upserting, which is the cleanest path.

## Recommended sequence (blue/green to a new namespace)

1. **Start fresh corpus file** (avoid stale lines):

   ```bash
   rm -f bm25_corpus_schh.jsonl
   ```
2. **Generate JSONL without upserting** (dry-run):

   ```bash
   export INDEX_NAME=schh
   export NAMESPACE=schh_v2
   export BM25_CORPUS_PATH=./bm25_corpus_schh.jsonl
   python chunker.py --index_name "$INDEX_NAME" --namespace "$NAMESPACE" --dryrun data/
   ```
3. **Fit the BM25 encoder** from that JSONL:

   ```bash
   export BM25_ENCODER_PATH=./bm25_encoder_schh.joblib
   python tools/build_bm25_encoder.py
   ```
4. **Real ingest** (dense + sparse in one pass):

   ```bash
   python chunker.py --index_name "$INDEX_NAME" --namespace "$NAMESPACE" data/
   ```
5. **Point the server** at the new namespace:

   ```bash
   export DEFAULT_NAMESPACE=schh_v2
   uvicorn serve:app --port 8080
   ```

## When dry-run is *not* needed

* If you don’t mind doing two passes: upsert dense-only first, then re-upsert after you train the encoder. (Works, but slower and more Pinecone writes.)
* If you already have a current `.joblib` that reflects **exactly** the same corpus you’re about to ingest. In that case, skip dry-run and just upsert—sparse will be included.

## Small gotchas

* **Truncate the JSONL** before regenerating; the builder doesn’t dedupe old lines.
* Keep `BM25_ENCODER_PATH` and `BM25_CORPUS_PATH` consistent between build and serve.
* If you change embedding **dimension** (e.g., move to `text-embedding-3-large`), you need a **new Pinecone index**; otherwise reuse the existing one.

TL;DR: For your updated corpus, do **dry-run → build encoder → real ingest** to get hybrid (dense + sparse) in a single clean pass.
