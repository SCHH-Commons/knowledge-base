#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib, os, sys
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
EMBEDDINGS = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index('schh')

chunk_size = 1000
chunk_overlap = 200
  
def generate_embeddings(documents: list[any]) -> list[list[float]]:
  print (f'generate_embeddings: {len(documents)}')
  
  embeddings = []
  for i, doc in enumerate(documents):
    if i and i % 20 == 0: print (i)
    embeddings.append(EMBEDDINGS.embed_documents(doc))
  return embeddings

def generate_short_id(content: str) -> str:
  hash_obj = hashlib.sha256()
  hash_obj.update(content.encode('utf-8'))
  return hash_obj.hexdigest()

def combine_vector_and_text(documents: list[any], doc_embeddings: list[list[float]]) -> list[dict[str, any]]:
  print (f'combine_vector_and_text: {len(documents)} {len(doc_embeddings)}')
  
  data_with_metadata = []

  for doc_text, embedding in zip(documents, doc_embeddings):
    # Convert doc_text to string if it's not already a string
    if not isinstance(doc_text, str):
      doc_text = str(doc_text)

    doc_id = generate_short_id(doc_text) # Generate a unique ID based on the text content

    # Create a data item dictionary
    data_item = {
      'id': doc_id,
      'values': embedding[0],
      'metadata': {'text': doc_text},  # Include the text as metadata
    }
    data_with_metadata.append(data_item)  # Append the data item to the list

  return data_with_metadata

def upsert_data_to_pinecone(data_with_metadata: list[dict[str, any]]) -> None:
  print (f'upsert_data_to_pinecone: {len(data_with_metadata)}')
  index.upsert(vectors=data_with_metadata)

def chunk_text_for_list(docs: list[str], max_chunk_size: int = 1000) -> list[list[str]]:
  """
  Break down each text in a list of texts into chunks of a maximum size, attempting to preserve whole paragraphs.
  """

  def chunk_text(text: str, max_chunk_size: int) -> List[str]:
    # Ensure each text ends with a double newline to correctly split paragraphs
    if not text.endswith("\n\n"):
        text += "\n\n"
    # Split text into paragraphs
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    # Iterate over paragraphs and assemble chunks
    for paragraph in paragraphs:
      # Check if adding the current paragraph exceeds the maximum chunk size
      if (
        len(current_chunk) + len(paragraph) + 2 > max_chunk_size
        and current_chunk
      ):
        # If so, add the current chunk to the list and start a new chunk
        chunks.append(current_chunk.strip())
        current_chunk = ""
      # Add the current paragraph to the current chunk
      current_chunk += paragraph.strip() + "\n\n"
    # Add any remaining text as the last chunk
    if current_chunk:
      chunks.append(current_chunk.strip())
    return chunks

  # Apply the chunk_text function to each document in the list
  return [chunk_text(doc, max_chunk_size) for doc in docs]

def chunk_markdown_document(markdown_path):
  markdown_document = open(markdown_path).read()
  headers_to_split_on = [ ('#', 'Header 1'), ('##', 'Header 2') ]

  # MD splits
  md_header_splits = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, 
    strip_headers=False
  ).split_text(markdown_document)

  # Char-level splits
  splits = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap
  ).split_documents(md_header_splits)
  
  return chunk_text_for_list([doc.page_content for doc in splits])
  # return [split.page_content for split in splits]

def load(markdown_path):
  chunks = chunk_markdown_document(markdown_path)
  doc_embeddings = generate_embeddings(documents=chunks)
  data_with_metadata = combine_vector_and_text(documents=chunks, doc_embeddings=doc_embeddings)
  upsert_data_to_pinecone(data_with_metadata=data_with_metadata)

if '__main__' == __name__:
  path = sys.argv[1]
  load(path)
