#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import argparse, json, os, re, sys
import hashlib
from typing import List
from slugify import slugify

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm

from pinecone import Pinecone

from langchain_openai import OpenAIEmbeddings
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDINGS = OpenAIEmbeddings( model='text-embedding-ada-002', openai_api_key=os.getenv('OPENAPI_API_KEY') )

# from langchain_community.embeddings import HuggingFaceEmbeddings
# EMBEDDINGS = HuggingFaceEmbeddings(model_name='intfloat/e5-large-v2')
# EMBEDDINGS = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

CHUNK_SIZE = 8191 * 4 # Maximum size of text (in characters) that can be processed by OpenAI text-embedding-ada-002 embedding model
CHUNK_OVERLAP = 200

def pdf_to_md(path):
  return pymupdf4llm.to_markdown(path)

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]], index_name, **kwargs) -> None:
  # print (f'Upsert data to Pinecone: {len(data_with_metadata)} index_name={index_name}')
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(index_name)
  index.upsert(vectors=data_with_metadata)

def chunk_markdown(markdown, source):
  headerLists = [
    [ ('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3') ],
    [ ('#', 'Header 1'), ('##', 'Header 2') ]
  ]
  docs = []
  for headers_to_split_on in headerLists:
    docs += MarkdownHeaderTextSplitter(
      headers_to_split_on = headers_to_split_on, 
      strip_headers=False
    ).split_text(markdown)
  
  # Char-level splits
  docs = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
  ).split_documents(docs)

  heading_1_ids = {}
  heading_2_ids = {}
  
  for idx in range(len(docs)):
    doc = docs[idx]
    doc.metadata['source'] = source
    headings_ids = []
    for heading in [doc.metadata[key] for key in sorted(doc.metadata) if key.startswith('Header')]:
      headings_ids.append(re.sub(r'[^a-zA-Z0-9]', '',''.join([token.title() for token in heading.split()])))
    doc.id = f'{source.replace("/",":")}:{":".join(headings_ids)}'
    doc.metadata['id'] = f'{source.replace("/",":")}:{":".join(headings_ids)}'

    if 'Header 1' in doc.metadata and 'Header 2' not in doc.metadata:
      heading_1_ids[doc.metadata['Header 1']] = doc.metadata['id']
    if 'Header 2' in doc.metadata and 'Header 3' not in doc.metadata:
      heading_2_ids[doc.metadata['Header 2']] = doc.metadata['id']
      if 'Header 1' in doc.metadata and doc.metadata['Header 1'] in heading_1_ids:
        doc.metadata['parent_id'] = heading_1_ids[doc.metadata['Header 1']]
    if 'Header 3' in doc.metadata:
      if 'Header 2' in doc.metadata and doc.metadata['Header 2'] in heading_2_ids:
        doc.metadata['parent_id'] = heading_2_ids[doc.metadata['Header 2']]
  return docs
        
def delete_records(index_name, path, **kwargs):
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(index_name)
  fname = path.split('/')[-1]
  for ids in index.list(prefix=fname):
    print(f'Deleting {len(ids)} records in "{index_name}" associated with "{path}"')
    index.delete(ids=ids)

def load(path, dryrun=False, verbose=False, **kwargs):
  fname = path.split('/')[-1]
  _ , extension = fname.split('.')
  docs = []
  
  if 'md' == extension:
    markdown = open(path).read()
  elif 'pdf' == extension:
    markdown = pdf_to_md(path)
  else:
    return
  
  docs = chunk_markdown(markdown, path)
  doc_embeddings = EMBEDDINGS.embed_documents([doc.page_content for doc in docs])
  
  data_with_metadata = []
  for doc, embedding in zip(docs, doc_embeddings):
    anchor = doc.metadata.get('Header 3', doc.metadata.get('Header 2', doc.metadata.get('Header 1')))
    url = f'https://www.schh-commons.org/knowledge-base/{path.replace("/index.md", "").replace(".md","")}' + (f'#{slugify(anchor)}' if anchor else '')
    data_item = {
        'id': doc.id,
        'values': embedding,
        'metadata': doc.metadata | {'text': doc.page_content, 'source': url},  # add text as metadata
    }
    data_with_metadata.append(data_item)  # Append the data item to the list
    
  if verbose:
    print(f'\nDocs: {len(data_with_metadata)}' + '\n\n---\n')
    for doc in data_with_metadata:
      print(doc['metadata']['text'] + '\n')
      print(json.dumps(dict([k,v] for k, v in doc['metadata'].items() if k not in ('text',))) + '\n\n---\n')
  else:
    print(f'{path}: docs={len(data_with_metadata)} load={not dryrun}')  
  if not dryrun:
    upsert_data_to_pinecone(data_with_metadata=data_with_metadata, **kwargs)

if __name__ == '__main__':
  BASEDIR = os.path.abspath(os.path.dirname(__file__))

  parser = argparse.ArgumentParser(description='SCHH Knowledge Base Loader')  
  parser.add_argument('--dryrun', default=False, action='store_true', help='Don\'t load data into Pinecone')
  parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose output')
  parser.add_argument('--content', default=BASEDIR, help='Knowledge base root directory')
  parser.add_argument('--index_name', default='schh', help='Pinecone index name')
  parser.add_argument('--delete', action='store_true', default=False, help='Delete all records in the index associated with the source path')
  parser.add_argument('path', help='Path to a file or directory to load')

  args = vars(parser.parse_args())
  
  path = sys.argv[1]
  if args['delete']:
    delete_records(**args) # Delete all records in the index associated with the source path
  else: # Load data into Pinecone
    if os.path.isdir(args['path']):
      for root, dirs, files in os.walk(args['path']):
        files = [f for f in files if not f[0] == '.' and not f.endswith('.py') and not f.endswith('.ipynb') and not f in ['README.md', 'LICENSE', 'requirements.txt']]
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in files:
          name, extension = os.path.splitext(file)
          if (extension == '.pdf' and f'{name}.md' in files):
            continue # Skip PDF files that have corresponding markdown files
          args['path'] = os.path.join(root, file)
          load(**args)
    elif os.path.isfile(args['path']):
      load(**args)