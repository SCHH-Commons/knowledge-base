#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://artifex.com/news/introducing-pymupdf4llm-a-breakthrough-in-pdf-to-markdown-conversion-for-python-developers

import os
import argparse
import pathlib
import pymupdf4llm

def convert(path, **kwargs):
  md_text = pymupdf4llm.to_markdown(path)
  output_path = pathlib.Path(path).with_suffix('.md')
  pathlib.Path(output_path).write_bytes(md_text.encode())

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Converts a PDF file to Markdown')  
  parser.add_argument('path', help='Path to a PDF file or directory containing PDF files')
  parser.add_argument('--force', type=bool, default=False, help='Force conversion even if a Markdown file already exists')

  args = vars(parser.parse_args())
  path = args['path']
  force = args['force'] if args['force'] else False
  
  if os.path.isdir(path):
    for root, dirs, files in os.walk(path):
      for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() == '.pdf':
          md_path = os.path.join(root, f"{name}.md")
          if not os.path.exists(md_path) or force:
            convert(os.path.join(root, f))
  elif os.path.isfile(path) and path.lower().endswith('.pdf'):
    name, ext = os.path.splitext(path)
    md_path = f"{name}.md"
    if not os.path.exists(md_path) or force:
      convert(path)