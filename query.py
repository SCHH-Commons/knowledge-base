#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json, os, sys
import argparse

from langchain_openai import OpenAIEmbeddings  
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA 
from langchain_pinecone import PineconeVectorStore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from pinecone import Pinecone

openai_api_key = os.environ.get('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

model_name = 'text-embedding-ada-002'  
text_field = 'text'

llm = ChatOpenAI(
  openai_api_key=openai_api_key,
  model_name='gpt-4o',
  temperature=0.0,
  streaming=True,
  callbacks=[StreamingStdOutCallbackHandler()]
)

def do_query(query: str, index: str, docs: bool , **kwargs):
  pc = Pinecone(api_key=pinecone_api_key)
  index = pc.Index(index)
  embeddings = OpenAIEmbeddings( model=model_name, openai_api_key=openai_api_key )
  vectorstore = PineconeVectorStore( index, embeddings, text_field )  
  results = vectorstore.similarity_search(query=query) # returns a list of Document objects with the most similar documents to the query (k=4 by default)
  if docs:
    for i, doc in enumerate(results):
        print(f'\n{doc.page_content}\n\nmetadata: [{doc.metadata}]\n')
        if i < len(results) - 1:
            print('-------')
  else:
    return RetrievalQA.from_chain_type( llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever() ).invoke(query)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='SCHH Knowledge Base Query Tool')  
  parser.add_argument('--index', default='schh', help='Pinecone index name')
  parser.add_argument('query', help='Chatbot query')
  parser.add_argument('--docs', action='store_true', default=False, help='Print raw vector text')
  args = vars(parser.parse_args())

  do_query(**args)
