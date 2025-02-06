#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://artifex.com/news/introducing-pymupdf4llm-a-breakthrough-in-pdf-to-markdown-conversion-for-python-developers

import argparse
import pathlib
import pymupdf4llm

def convert(path, **kwargs):
  md_text = pymupdf4llm.to_markdown(path)
  output_path = pathlib.Path(path).with_suffix('.md')
  pathlib.Path(output_path).write_bytes(md_text.encode())

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Converts a PDF file to Markdown')  
  parser.add_argument('path', help='Path to a PDF file or convert')

  args = vars(parser.parse_args())

  convert(**args)
