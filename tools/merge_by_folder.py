#!/usr/bin/env python3

import re

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BASEDIR = os.path.dirname(SCRIPT_DIR)

# Directory containing your Markdown files
SOURCE_DIR = os.path.join(BASEDIR, 'docs')
OUTPUT_DIR = os.path.join(BASEDIR, 'merged')

# Regex to detect YAML front matter (--- ... --- at start of file)
FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

def getCommentMetadata(text: str) -> dict:
    """Extract metadata from HTML comments at the start of the file."""
    metadata = {}
    comment_re = re.compile(r"^\s*<!--\s*(.*?)\s*-->\s*\n", re.DOTALL)
    match = comment_re.match(text)
    if match:
        content = match.group(1)
        for line in content.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
    return metadata

def strip_front_matter(text: str) -> str:
    """Remove YAML front matter if present."""
    return FRONT_MATTER_RE.sub("", text, count=1)

def merge_by_folder():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    index_path = os.path.join(BASEDIR, 'index.md')
    with open(index_path, 'w', encoding='utf-8') as index:
        for root, dir, files in os.walk(SOURCE_DIR):
            if root == SOURCE_DIR: continue
            merged_path = os.path.join(OUTPUT_DIR, root.split(os.sep)[-1] + '.md')
            merged = []
            for file in files:
                if file.endswith('.md'):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        print(root, file)
                        content = f.read()
                        metadata = getCommentMetadata(content)
                        if metadata:
                            title = metadata.get('title', file)
                            url = metadata.get('url', f'https://www.SCHH-commons.org/knowledge-base/{os.path.relpath(os.path.join(root, file.replace('.md', '')))}')
                            index.write(f"- [{title}]({url})\n")
                        else:
                            print(f"Warning: No metadata found in {file}")
                        content = strip_front_matter(content)
                        if (file == 'index.md'):
                            merged.insert(0, content.strip())
                        else:
                            merged.append(content.strip())
            with open(merged_path, 'w', encoding='utf-8') as out:
                out.write("\n\n---\n\n".join(merged))
        
if __name__ == "__main__":
    merge_by_folder()