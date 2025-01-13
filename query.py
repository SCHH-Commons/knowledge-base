#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys

from langchain_openai import OpenAIEmbeddings  
from langchain_openai import ChatOpenAI  
from langchain.chains import RetrievalQA 
from langchain_pinecone import PineconeVectorStore
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from pinecone import Pinecone

openai_api_key = os.environ.get('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')

model_name = 'text-embedding-ada-002'  
index_name = 'schh'
text_field = 'text'

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings( model=model_name, openai_api_key=openai_api_key )
vectorstore = PineconeVectorStore( index, embeddings, text_field )  

llm = ChatOpenAI(
  openai_api_key=openai_api_key,
  model_name='gpt-4o-mini',
  temperature=0.0,
  streaming=True,
  callbacks=[StreamingStdOutCallbackHandler()]
)

def do_query(query: str):
  print (f'Query: {query}')
  return RetrievalQA.from_chain_type( llm=llm, chain_type='stuff', retriever=vectorstore.as_retriever() ).invoke(query)

if __name__ == '__main__':
  query = ' '.join(sys.argv[1:])
  do_query(query)
