#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(format='%(asctime)s : %(filename)s : %(levelname)s : %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

import argparse, json, os, re, sys
import yaml
from yaml import CLoader as Loader

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
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

BASEDIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DATADIR = os.path.join(BASEDIR, 'data')
  
def pdf_to_md(path):
  md = pymupdf4llm.to_markdown(path)
  out_path = path.rsplit('.', 1)[0] + '.md'
  with open(out_path, 'wb') as fp:
    fp.write(md.encode('utf-8'))
  return md

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]], index_name, **kwargs) -> None:
  # print (f'Upsert data to Pinecone: {len(data_with_metadata)} index_name={index_name}')
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(index_name)
  index.upsert(vectors=data_with_metadata)

def chunk_markdown(markdown, path):
  source = None
  m = re.match(r'\s*source:\s*(.+)', markdown)
  if m:
    source = m.group(1)
    markdown = re.sub(r'\s*source:\s*(.+)', '', markdown)
    
  headerLists = [
    [ ('#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3') ],
    # [ ('#', 'Header 1'), ('##', 'Header 2') ]
  ]
  docs = []
  for headers_to_split_on in headerLists:
    docs += MarkdownHeaderTextSplitter(
      headers_to_split_on = headers_to_split_on, 
      strip_headers=False
    ).split_text(markdown)
  
  # Char-level splits
  docs = RecursiveCharacterTextSplitter(chunk_size=8192, chunk_overlap=0).split_documents(docs)

  heading_1_ids = {}
  heading_2_ids = {}
  
  _docids = {}
  def docid (doc):
    headings_ids = []
    for heading in [doc.metadata[key] for key in sorted(doc.metadata) if key.startswith('Header')]:
      headings_ids.append(re.sub(r'[^a-zA-Z0-9]', '',''.join([token.title() for token in heading.split()])))
    base = re.sub(r"^(\.\.\/)?data\/", "", path).replace("/",":") + ':' + ':'.join(headings_ids)
    if base in _docids:
      _docids[base] += 1
    else:
      _docids[base] = 1
    return f'{base}:{_docids[base]}'
  
  for idx in range(len(docs)):
    doc = docs[idx]
    if source:
      doc.metadata['source'] = source
    doc.id = docid(doc)
    doc.metadata['id'] = doc.id
    doc.metadata['size'] = len(doc.page_content)

    if 'Header 1' in doc.metadata and 'Header 2' not in doc.metadata:
      heading_1_ids[doc.metadata['Header 1']] = doc.id
    if 'Header 2' in doc.metadata and 'Header 3' not in doc.metadata:
      heading_2_ids[doc.metadata['Header 2']] = doc.id
      if 'Header 1' in doc.metadata and doc.metadata['Header 1'] in heading_1_ids:
        doc.metadata['parent_id'] = heading_1_ids[doc.metadata['Header 1']]
    if 'Header 3' in doc.metadata:
      if 'Header 2' in doc.metadata and doc.metadata['Header 2'] in heading_2_ids:
        doc.metadata['parent_id'] = heading_2_ids[doc.metadata['Header 2']]
        
  for doc in docs:
    base, seq = doc.id.rsplit(':', 1)
    sibs = [f'{base}:{idx+1}' for idx in range(_docids[base]) if (idx + 1) != int(seq)] if _docids[base] > 1 else []
    if len(sibs) > 0:
      doc.metadata['sib_ids'] = sibs
  return docs
        
def delete_records(index_name, path, **kwargs):
  pc = Pinecone(api_key=PINECONE_API_KEY)
  index = pc.Index(index_name)
  # prefix = f'{path.replace("../data/", "").replace("/",":")}'
  prefix = re.sub(r"^(\.\.\/)?data\/", "", path).replace("/",":")
  print(f'Delete: prefix={prefix}')
  for ids in index.list(prefix=prefix):
    print(f'Deleting {len(ids)} records in "{index_name}" associated with "{path}"')
    index.delete(ids=ids)

_sources = None
def get_sources():
  global _sources
  if not _sources:
    _sources = yaml.load(open(f'{DATADIR}/sources.yml', 'r'), Loader)
  return _sources

def chunk_pdf(path):
  loader = PyPDFLoader(path)
  pages = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  docs = text_splitter.split_documents(pages)
  
  page_chunks = {}
  for doc in docs:
    doc.metadata['page'] = doc.metadata['page'] + 1
    if doc.metadata['page'] not in page_chunks: page_chunks[doc.metadata['page']] = 0
    page_chunks[doc.metadata['page']] += 1
    # doc.id = f'{path.replace("../data/", "").replace("/",":")}:{doc.metadata["page"]}:{page_chunks[doc.metadata["page"]]}'
    doc.id = re.sub(r"^(\.\.\/)?data\/", "", path).replace("/",":") + f':{doc.metadata["page"]}:{page_chunks[doc.metadata["page"]]}'
    doc.metadata['id'] = doc.id
  return docs

def load(path, dryrun=False, verbose=False, **kwargs):
  sources = get_sources()
  fname = path.split('/')[-1]
  name , extension = fname.split('.')
  docs = []
  
  if 'md' == extension:
    markdown = open(path).read()
    docs = chunk_markdown(markdown, path)
  elif 'pdf' == extension:
    # docs = chunk_markdown(pdf_to_md(path), path)
    docs = chunk_pdf(path)
  else:
    return
  
  doc_embeddings = EMBEDDINGS.embed_documents([doc.page_content for doc in docs])
  
  data_with_metadata = []
  for doc, embedding in zip(docs, doc_embeddings):
    custom_metadata = {'text': doc.page_content}
    source = sources.get(name.lower())
    if source:
      custom_metadata['title'] = source['title']
      if 'url' in source:
        url = source['url']
      elif 'docid' in source:
        url = f'https://suncityhiltonhead.org/ResourceCenter/Download/46134/{name.lower()}?doc_id={source["docid"]}&print=1&view=1'
      custom_metadata['source'] = url
    else:
      custom_metadata['title'] = fname
      custom_metadata['source'] = f'https://github.com/SCHH-Commons/knowledge-base/blob/main/{path}'
    data_item = {
        'id': doc.id,
        'values': embedding,
        'metadata': doc.metadata | custom_metadata,
    }
    data_with_metadata.append(data_item)  # Append the data item to the list
    
  if verbose:
    print(f'\nDocs: {len(data_with_metadata)}' + '\n\n---\n')
    for doc in data_with_metadata:
      # print(doc['metadata']['text'] + '\n')
      # print(json.dumps(dict([k,v] for k, v in doc['metadata'].items() if k not in ('text',))) + '\n\n---\n')
      print(json.dumps(dict([k,v] for k, v in doc['metadata'].items() if k not in ('text',))))
  else:
    print(f'{path}: docs={len(data_with_metadata)} load={not dryrun}')

  if not dryrun:
    upsert_data_to_pinecone(data_with_metadata=data_with_metadata, **kwargs)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='SCHH Knowledge Base Loader')  
  parser.add_argument('--dryrun', default=False, action='store_true', help='Don\'t load data into Pinecone')
  parser.add_argument('--verbose', action='store_true', default=False, help='Print verbose output')
  parser.add_argument('--index_name', default='schh', help='Pinecone index name')
  parser.add_argument('--delete', action='store_true', default=False, help='Delete all records in the index associated with the source path')
  parser.add_argument('path', default=DATADIR, help='Path to a file or directory to load')

  args = vars(parser.parse_args())
  
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
      name, extension = os.path.splitext(args['path'])
      if extension == '.pdf':
        md_path = f'{name}.md'
        if os.path.exists(md_path):
          args['path'] = md_path
      load(**args)