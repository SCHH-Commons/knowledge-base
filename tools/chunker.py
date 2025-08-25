#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SCHH Knowledge Base Chunker & Loader (Hybrid-ready)

Features
- Embeddings: OpenAI text-embedding-3-(small|large)
- Markdown-first: split by headers (H1/H2/H3) then sub-chunk (~400/64)
- PDFs: prefer pymupdf4llm -> Markdown -> same pipeline; PyPDFLoader fallback (700/100)
- Deterministic IDs: path + header_path + seq; parent_id / sib_ids
- Optional section summaries (improves recall)
- Dedupe: hash-based; skip tiny/footer chunks
- Batched embeddings & Pinecone upserts with retries
- Rich metadata: header_path, section, doc_id, updated_at, is_summary/is_parent
- HYBRID: Writes JSONL corpus and, if BM25 encoder is present, upserts sparse_values for Pinecone hybrid queries.

Prep for hybrid
- Generate a BM25 encoder once per corpus with tools/build_bm25_encoder.py
- Set BM25_ENCODER_PATH to that joblib so this script can load it at ingest time
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import logging
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple

import yaml
from yaml import CLoader as Loader

# LangChain + splitters
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# PDF → Markdown
import pymupdf4llm

# Pinecone + embeddings
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# Hybrid sparse encoder (optional)
try:
    import joblib
    from pinecone_text.sparse import BM25Encoder
except Exception:
    joblib = None
    BM25Encoder = None  # type: ignore


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("chunker")


# ------------------------------------------------------------------------------
# Env / Defaults
# ------------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

BASEDIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATADIR = os.path.join(BASEDIR, "data")

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # or text-embedding-3-large

# Hybrid encoder/corpus paths (used if present)
BM25_ENCODER_PATH = os.getenv("BM25_ENCODER_PATH", "./bm25_encoder_schh.joblib")
BM25_CORPUS_PATH  = os.getenv("BM25_CORPUS_PATH",  "./bm25_corpus_schh.jsonl")

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def batched(iterable, n: int):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def now_ms() -> int:
    return int(time.time() * 1000)


# ------------------------------------------------------------------------------
# Markdown / PDF processing
# ------------------------------------------------------------------------------
FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n", flags=re.S)
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])', flags=re.M)

SECTION_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # ~400 chars proxy
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)

PDF_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
    length_function=len,
)


def pdf_to_md(path: str) -> str:
    return pymupdf4llm.to_markdown(path)


def parse_frontmatter(markdown: str) -> Tuple[str, Dict[str, Any]]:
    m = FRONTMATTER_RE.match(markdown)
    if not m:
        return markdown, {}
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except Exception:
        meta = {}
    return markdown[m.end():], meta


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts, out, in_code = text.split("\n"), [], False
    buf = []
    for line in parts:
        if line.strip().startswith("```"):
            in_code = not in_code
        if in_code:
            buf.append(line)
        else:
            if buf:
                out.append("\n".join(buf))
                buf = []
            sents = SENT_RE.split(line.strip())
            for s in sents:
                if s:
                    out.append(s.strip())
    if buf:
        out.append("\n".join(buf))
    return [s for s in out if s.strip()]


def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()


def should_skip_chunk(text: str, min_len: int = 40) -> bool:
    t = normalize_text(text)
    if len(t) < min_len:
        return True
    if re.search(r"^(page \d+ of \d+|table of contents)$", t, re.I):
        return True
    return False


def stable_base_id(path: str, header_path: List[str]) -> str:
    rel = re.sub(r"^(\.\./)?data/", "", path)
    prefix = rel.replace("/", ":")
    hp = ":".join(re.sub(r"[^A-Za-z0-9]+", "", h.title()) for h in header_path)
    return f"{prefix}:{hp}" if hp else prefix


def attach_relations_and_ids(docs: List[Document], path: str) -> List[Document]:
    # group by header_path
    groups: Dict[Tuple[str, ...], List[Document]] = {}
    for d in docs:
        hp = tuple(d.metadata.get("header_path") or [])
        groups.setdefault(hp, []).append(d)

    # parent id per header_path
    parent_id_by_hp: Dict[Tuple[str, ...], str] = {}
    for hp, lst in groups.items():
        base = stable_base_id(path, list(hp))
        parent_id_by_hp[hp] = f"{base}:P"
        for i, d in enumerate(lst, 1):
            d.metadata["header_path"] = list(hp)
            d.metadata["seq"] = i
            d.id = f"{base}:{i}"
            d.metadata["id"] = d.id

    # set parent + siblings
    for hp, lst in groups.items():
        if len(hp) > 0:
            parent_hp = hp[:-1]
            if parent_hp in parent_id_by_hp:
                for d in lst:
                    d.metadata["parent_id"] = parent_id_by_hp[parent_hp]
        ids = [d.id for d in lst]
        for d in lst:
            d.metadata["sib_ids"] = [x for x in ids if x != d.id]

    # synthetic parent docs (optional; lightweight headings)
    parents: List[Document] = []
    for hp, pid in parent_id_by_hp.items():
        if len(hp) == 0:
            continue
        pd = Document(page_content=" ".join(hp))
        pd.metadata = {"id": pid, "is_parent": True, "header_path": list(hp)}
        pd.id = pid
        parents.append(pd)

    # enrich metadata
    mtime = int(os.path.getmtime(path)) if os.path.exists(path) else None
    for d in docs:
        hp = d.metadata.get("header_path") or []
        d.metadata["section"] = " > ".join(hp)
        base = d.id.rsplit(":", 1)[0]
        d.metadata["doc_id"] = base
        d.metadata["updated_at"] = mtime
        d.metadata["size"] = len(d.page_content or "")
        d.metadata.setdefault("title", os.path.basename(path))
        # print(json.dumps(d.metadata, indent=2))  # Debug print

    return docs + parents


def dedupe_by_hash(docs: List[Document]) -> List[Document]:
    seen = set()
    out: List[Document] = []
    for d in docs:
        t = normalize_text(d.page_content)
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        if h in seen or should_skip_chunk(t):
            continue
        seen.add(h)
        d.metadata["text_norm_hash"] = h
        out.append(d)
    return out


def add_section_summaries(docs: List[Document], path: str, max_chars: int = 600) -> List[Document]:
    # group by first two headers
    groups: Dict[Tuple[str, ...], List[Document]] = {}
    for d in docs:
        hp = tuple((d.metadata.get("header_path") or [])[:2])
        groups.setdefault(hp, []).append(d)

    summaries: List[Document] = []
    for hp, lst in groups.items():
        if not hp:
            continue
        text = " ".join((d.page_content or "")[:800] for d in lst[:8])
        text = normalize_text(text)[:8000]
        if not text:
            continue
        summ = f"Section summary: {' > '.join(hp)} — key points extracted.\n\n{text[:max_chars]}"
        sd = Document(page_content=summ)
        sd.metadata = {**lst[0].metadata, "is_summary": True}
        base = stable_base_id(path, list(hp))
        sd.id = f"{base}:S"
        summaries.append(sd)

    return docs + summaries


# ------------------------------------------------------------------------------
# Chunkers
# ------------------------------------------------------------------------------
def chunk_markdown(markdown: str, path: str) -> List[Document]:
    markdown, fm = parse_frontmatter(markdown)
    source = fm.get("source")

    headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,
    ).split_text(markdown)

    docs: List[Document] = []
    for sec in sections:
        hp = [sec.metadata[k] for k in sorted(sec.metadata) if k.startswith("Header")]
        sec.metadata["header_path"] = hp
        if source:
            sec.metadata["source"] = source
        subs = SECTION_SPLITTER.split_documents([sec])
        docs.extend(subs)

    docs = attach_relations_and_ids(docs, path)
    return docs


def chunk_pdf(path: str, prefer_md: bool = True) -> List[Document]:
    if prefer_md:
        try:
            md = pdf_to_md(path)
            return chunk_markdown(md, path)
        except Exception as e:
            log.warning(f"pymupdf4llm failed on {path}: {e}; falling back to PyPDFLoader")

    loader = PyPDFLoader(path)
    pages = loader.load()
    docs = PDF_SPLITTER.split_documents(pages)
    page_counts: Dict[int, int] = {}
    for d in docs:
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1
        page_counts[d.metadata["page"]] = page_counts.get(d.metadata["page"], 0) + 1
        seq = page_counts[d.metadata["page"]]
        rel = re.sub(r"^(\.\./)?data/", "", path).replace("/", ":")
        d.id = f"{rel}:{d.metadata['page']}:{seq}"
        d.metadata["id"] = d.id
        d.metadata["header_path"] = [f"Page {d.metadata['page']}"]

    docs = attach_relations_and_ids(docs, path)
    return docs


# ------------------------------------------------------------------------------
# Embeddings + Pinecone + Hybrid helpers
# ------------------------------------------------------------------------------
def build_embeddings_client(model: str) -> OpenAIEmbeddings:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAIEmbeddings(model=model, openai_api_key=OPENAI_API_KEY)


def embed_texts(emb_client: OpenAIEmbeddings, texts: List[str], batch_size: int = 128, max_retries: int = 3) -> List[List[float]]:
    out: List[List[float]] = []
    for batch in batched(texts, batch_size):
        for attempt in range(max_retries):
            try:
                out.extend(emb_client.embed_documents(batch))
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
    return out


def upsert_pinecone(index_name: str, items: List[Dict[str, Any]], batch_size: int = 100, namespace: Optional[str] = None) -> None:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    total = 0
    for batch in batched(items, batch_size):
        for attempt in range(3):
            try:
                index.upsert(vectors=batch, namespace=namespace)
                total += len(batch)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
    log.info(f"Upserted {total} vectors into '{index_name}' namespace='{namespace or ''}'")


def delete_records(index_name: str, path: str, namespace: Optional[str] = None) -> None:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(index_name)
    prefix = re.sub(r"^(\.\./)?data/", "", path).replace("/", ":")
    for ids in index.list(prefix=prefix, namespace=namespace):
        log.warning(f"Deleting {len(ids)} records in '{index_name}' (ns='{namespace or ''}') for '{path}'")
        index.delete(ids=ids, namespace=namespace)


# Hybrid encoder loader
_bm25: Optional[BM25Encoder] = None  # type: ignore
def get_bm25_encoder() -> Optional["BM25Encoder"]:
    global _bm25
    if _bm25 is not None:
        return _bm25
    if joblib is None or BM25Encoder is None:
        return None
    if not os.path.exists(BM25_ENCODER_PATH):
        return None
    try:
        _bm25 = joblib.load(BM25_ENCODER_PATH)
        log.info(f"[hybrid] BM25 encoder loaded from {BM25_ENCODER_PATH}")
        return _bm25
    except Exception as e:
        log.warning(f"[hybrid] failed to load BM25 encoder: {e}")
        return None


def append_to_corpus_jsonl(docs: List[Document]):
    if not BM25_CORPUS_PATH:
        return
    try:
        os.makedirs(os.path.dirname(BM25_CORPUS_PATH) or ".", exist_ok=True)
        with open(BM25_CORPUS_PATH, "a", encoding="utf-8") as wf:
            for d in docs:
                wf.write(json.dumps({
                    "id": d.id,
                    "text": normalize_text(d.page_content)
                }) + "\n")
    except Exception as e:
        log.warning(f"Failed to append to corpus JSONL: {e}")


# ------------------------------------------------------------------------------
# Source map
# ------------------------------------------------------------------------------
_sources_cache = None
def get_sources() -> Dict[str, Dict[str, Any]]:
    global _sources_cache
    if _sources_cache is None:
        sources_path = os.path.join(DATADIR, "sources.yml")
        if os.path.exists(sources_path):
            _sources_cache = yaml.load(open(sources_path, "r"), Loader) or {}
        else:
            _sources_cache = {}
    return _sources_cache


# ------------------------------------------------------------------------------
# Load pipeline
# ------------------------------------------------------------------------------
def load_one(path: str,
             index_name: str,
             embed_model: str,
             namespace: Optional[str] = None,
             prefer_pdf_md: bool = True,
             add_summaries: bool = True,
             dryrun: bool = False,
             verbose: bool = False) -> None:

    fname = os.path.basename(path)
    name, extension = os.path.splitext(fname)
    extension = extension.lower().lstrip(".")

    # 1) Chunk
    if extension == "md":
        markdown = open(path, "r", encoding="utf-8").read()
        docs = chunk_markdown(markdown, path)
    elif extension == "pdf":
        docs = chunk_pdf(path, prefer_md=prefer_pdf_md)
    else:
        log.info(f"Skipping unsupported file type: {path}")
        return

    if add_summaries:
        docs = add_section_summaries(docs, path)

    docs = dedupe_by_hash(docs)
    if not docs:
        log.warning(f"No chunks produced for {path}")
        return

    # 2) Source/title enrichment
    sources = get_sources()
    meta_src = sources.get(name.lower()) or {}
    default_title = meta_src.get("title") or fname
    default_url = meta_src.get("url")
    if not default_url and meta_src.get("docid"):
        default_url = f"https://suncityhiltonhead.org/ResourceCenter/Download/46134/{name.lower()}?doc_id={meta_src['docid']}&print=1&view=1"
    gh_url = f"https://github.com/SCHH-Commons/knowledge-base/blob/main/{path}"

    for d in docs:
        d.metadata.setdefault("title", default_title)
        d.metadata.setdefault("source", default_url or gh_url)
        
    # 3) (Optional but recommended) append to corpus JSONL (for BM25 fitting)
    append_to_corpus_jsonl(docs)

    # 4) Embeddings
    emb_client = build_embeddings_client(embed_model)
    texts = [d.page_content for d in docs]
    t0 = now_ms()
    embs = embed_texts(emb_client, texts, batch_size=128)
    t1 = now_ms()

    # 5) Hybrid sparse (if encoder present)
    bm25 = get_bm25_encoder()

    '''
    for doc in docs:
        print(json.dumps(doc.metadata, indent=2))
        print(doc.page_content + '\n\n---\n')  # Debug print
    '''
        
    # 6) Build Pinecone items
    items = []
    for d, e in zip(docs, embs):
        meta = dict(d.metadata)
        meta["text"] = d.page_content
        item: Dict[str, Any] = {"id": d.id, "values": e, "metadata": meta}
        if bm25:
            try:
                sv = bm25.encode_documents([meta["text"]])[0]  # {"indices":[...], "values":[...]}
                item["sparse_values"] = sv
            except Exception as ex:
                log.warning(f"[hybrid] encode_documents failed for {d.id}: {ex}")
        items.append(item)
        # print(json.dumps(item['metadata'], indent=2)[:600] + "\n---\n")  # Debug print

    if verbose:
        print(f"\nDocs to upsert: {len(items)} | embed_model={embed_model} | namespace={namespace or ''}")
        print(f"Embedding latency: {t1 - t0} ms\n---\n")
        for it in items[:3]:
            print(it["metadata"]["text"][:600], "\n")
            print(json.dumps({k: v for k, v in it["metadata"].items() if k != "text"}, indent=2))
            print("\n---\n")
    else:
        log.info(f"{path}: docs={len(items)} load={not dryrun} model={embed_model} ns={namespace or ''} hybrid={'on' if bm25 else 'off'}")

    # 7) Upsert
    if not dryrun:
        upsert_pinecone(index_name=index_name, items=items, batch_size=100, namespace=namespace)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SCHH Knowledge Base Loader (Hybrid-ready)")
    parser.add_argument("--dryrun", action="store_true", default=False, help="Do not load data into Pinecone")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print verbose output")
    parser.add_argument("--index_name", default="schh", help="Pinecone index name")
    parser.add_argument("--namespace", default=None, help="Pinecone namespace (optional)")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="Embedding model (text-embedding-3-small/large)")
    parser.add_argument("--no_pdf_md", action="store_true", default=False, help="Disable PDF→Markdown (use PyPDFLoader fallback)")
    parser.add_argument("--no_summaries", action="store_true", default=False, help="Disable section summary chunks")
    parser.add_argument("--delete", action="store_true", default=False, help="Delete all records in the index for the path prefix")
    parser.add_argument("path", nargs="?", default=DATADIR, help="Path to a file or directory to load")
    args = parser.parse_args()

    if args.delete:
        delete_records(index_name=args.index_name, path=args.path, namespace=args.namespace)
        return

    prefer_pdf_md = not args.no_pdf_md
    add_summaries = not args.no_summaries

    if os.path.isdir(args.path):
        for root, dirs, files in os.walk(args.path):
            files = [f for f in files if not f.startswith(".") and f not in ["README.md", "LICENSE", "requirements.txt"]]
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for file in files:
                name, ext = os.path.splitext(file)
                # Skip PDF if same-named .md exists in the same dir
                if ext.lower() == ".pdf" and (f"{name}.md" in files) and prefer_pdf_md:
                    continue
                fpath = os.path.join(root, file)
                try:
                    load_one(
                        path=fpath,
                        index_name=args.index_name,
                        embed_model=args.embed_model,
                        namespace=args.namespace,
                        prefer_pdf_md=prefer_pdf_md,
                        add_summaries=add_summaries,
                        dryrun=args.dryrun,
                        verbose=args.verbose,
                    )
                except Exception as e:
                    log.exception(f"Failed to load {fpath}: {e}")
    elif os.path.isfile(args.path):
        path = args.path
        name, ext = os.path.splitext(path)
        if ext.lower() == ".pdf" and prefer_pdf_md:
            md_path = f"{name}.md"
            if os.path.exists(md_path):
                path = md_path
        load_one(
            path=path,
            index_name=args.index_name,
            embed_model=args.embed_model,
            namespace=args.namespace,
            prefer_pdf_md=prefer_pdf_md,
            add_summaries=add_summaries,
            dryrun=args.dryrun,
            verbose=args.verbose,
        )
    else:
        log.error(f"Path not found: {args.path}")
        sys.exit(1)


if __name__ == "__main__":
    main()