#!/usr/bin/env python
import os, json, joblib
from pinecone_text.sparse import BM25Encoder

CORPUS_JSONL = os.getenv("BM25_CORPUS_PATH", "./bm25_corpus_schh.jsonl")
ENCODER_PATH = os.getenv("BM25_ENCODER_PATH", "./bm25_encoder_schh.joblib")

def main():
    texts = []
    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            txt = (rec.get("text") or "").strip()
            if txt:
                texts.append(txt)
    enc = BM25Encoder()
    enc.fit(texts)
    joblib.dump(enc, ENCODER_PATH)
    print(f"Saved encoder to {ENCODER_PATH} (docs={len(texts)})")

if __name__ == "__main__":
    main()