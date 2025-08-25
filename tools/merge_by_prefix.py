#!/usr/bin/env python3
import pathlib
import re

# Directory containing your Markdown files
SOURCE_DIR = pathlib.Path("./docs")
# Output directory for merged files
OUTPUT_DIR = pathlib.Path("./merged")
OUTPUT_DIR.mkdir(exist_ok=True)

# Regex to detect YAML front matter (--- ... --- at start of file)
FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)

def strip_front_matter(text: str) -> str:
    """Remove YAML front matter if present."""
    return FRONT_MATTER_RE.sub("", text, count=1)

def merge_by_prefix():
    groups = {}
    for md_file in SOURCE_DIR.glob("*.md"):
        name = md_file.stem  # filename without extension
        if "-" not in name:
            continue  # skip files without prefix
        prefix, rest = name.split("-", 1)
        groups.setdefault(prefix, []).append((rest, md_file))

    for prefix, files in groups.items():
        output_path = OUTPUT_DIR / f"{prefix}.md"
        with open(output_path, "w", encoding="utf-8") as out:
            # out.write(f"# {prefix} Documents\n\n")
            for rest, md_file in sorted(files):
                with open(md_file, encoding="utf-8") as f:
                    content = strip_front_matter(f.read())
                # Use filename (minus prefix) as heading
                title = rest.replace("_", " ")
                # out.write(f"## {title}\n\n")
                out.write(content.strip())
                out.write("\n\n---\n\n")
        print(f"Wrote {output_path}")

if __name__ == "__main__":
    merge_by_prefix()