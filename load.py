#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import argparse, json, os, re, sys
import hashlib
from typing import List

from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDINGS = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
  
def generate_short_id(content: str) -> str:
  """Generate a short ID based on the content using SHA-256 hash."""
  hash_obj = hashlib.sha256()
  hash_obj.update(content.encode('utf-8'))
  return hash_obj.hexdigest()

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]], index_name, **kwargs) -> None:
  # print (f'Upsert data to Pinecone: {len(data_with_metadata)} index_name={index_name}')
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(index_name)
  index.upsert(vectors=data_with_metadata)

def chunk_markdown(path):
  markdown = open(path).read()
  docs = MarkdownHeaderTextSplitter(
    headers_to_split_on = [ ('#', 'Header 1'), ('##', 'Header 2') ], 
    strip_headers=False
  ).split_text(markdown)
  
  # Char-level splits
  '''
  docs = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP
  ).split_documents(docs)
  '''
  return docs

def chunk_pdf(path):
  loader = PyPDFLoader(path)
  pages = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
  docs = text_splitter.split_documents(pages)
  return docs

ids = {}
def get_id(doc, fname):
  extension = fname.split('.')[-1]
  if extension == 'md':
    # Generate a unique ID based on the headings in the metadata
    headings_ids = []
    for heading in [doc.metadata[key] for key in sorted(doc.metadata) if key.startswith('Header')]:
      headings_ids.append(re.sub(r'[^a-zA-Z0-9]', '',''.join([token.title() for token in heading.split()])))
    id = f'{fname}:{":".join(headings_ids)}'
  elif extension == 'pdf':
    id = f'{fname}:{doc.metadata["page"]}'
  if id in ids:
    ids[id] += 1
    id = f'{id}-{ids[id]}'
  else:
    ids[id] = 1
  return id
        
def delete_records(index_name, path, **kwargs):
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(index_name)
  fname = path.split('/')[-1]
  for ids in index.list(prefix=fname):
    print(f'Deleting {len(ids)} records in "{index_name}" associated with "{path}"')
    index.delete(ids=ids)

def load(path, dryrun=False, verbose=False, **kwargs):
  logger.info(f'Loading {path} into Pinecone, dryrun={dryrun}')
  fname = path.split('/')[-1]
  _ , extension = fname.split('.')
  docs = []
  
  if 'md' == extension:
    docs = chunk_markdown(path)
  elif 'pdf' == extension:
    docs = chunk_pdf(path)

  if docs:
    if verbose:
      print(f'\nDocs: {len(docs)}' + '\n\n---\n')
      for doc in docs:
        print(doc.page_content + '\n\n---\n')
    else:
      print(f'{path}: docs={len(docs)}')
    
    doc_embeddings = EMBEDDINGS.embed_documents([doc.page_content for doc in docs])
    
    data_with_metadata = []
    for doc, embedding in zip(docs, doc_embeddings):
      data_item = {
          'id': get_id(doc, fname),
          'values': embedding,
          'metadata': doc.metadata | {'text': doc.page_content, 'source': path},  # add text as metadata
      }
      data_with_metadata.append(data_item)  # Append the data item to the list
    
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
  print(json.dumps(args))
  
  path = sys.argv[1]
  if args['delete']:
    delete_records(**args) # Delete all records in the index associated with the source path
  else: # Load data into Pinecone
    if os.path.isdir(args['path']):
      for root, dirs, files in os.walk(args['path']):
        for file in files:
          name, extension = os.path.splitext(file)
          if (extension == '.pdf' and f'{name}.md' in files):
            continue # Skip PDF files that have corresponding markdown files
          args['path'] = os.path.join(root, file)
          load(**args)
    elif os.path.isfile(args['path']):
      load(**args)