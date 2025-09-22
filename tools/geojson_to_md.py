#!/usr/bin/env python3
"""
geojson_to_markdown.py

Convert a GeoJSON FeatureCollection (with features like the example below)
into a nicely formatted, RAG-friendly Markdown file.

Example feature:
{
  "type": "Feature",
  "properties": {
    "name": "Town Square Tennis Courts",
    "description": "",
    "address": "124 Del Webb Blvd, Okatie, SC 29909",
    "phone": "",
    "category": "Tennis",
    "marker-symbol": "table-tennis-paddle-ball",
    "marker-color": "#43A047"
  },
  "geometry": {
    "coordinates": [-80.95366953812885, 32.29380760543188],
    "type": "Point"
  },
  "id": 36
}

Usage:
  python geojson_to_markdown.py input.geojson output.md --title "SCHH Amenities"
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def load_geojson(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize to FeatureCollection
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        return data
    elif isinstance(data, dict) and data.get("type") == "Feature":
        return {"type": "FeatureCollection", "features": [data]}
    elif isinstance(data, list):
        # list of features
        return {"type": "FeatureCollection", "features": data}
    else:
        raise ValueError("Input must be a GeoJSON FeatureCollection, a Feature, or a list of Features.")

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return f"{x}"
    return str(x)

def md_escape(text: str) -> str:
    """Lightweight markdown escape for headings/inline text."""
    if not text:
        return ""
    for ch in ["#", "*", "_", "`", "<", ">", "|"]:
        text = text.replace(ch, f"\\{ch}")
    return text

def get_props(feat: Dict[str, Any]) -> Dict[str, Any]:
    props = feat.get("properties") or {}
    # Normalize expected fields
    return {
        "id": feat.get("id"),
        "name": props.get("name"),
        "description": props.get("description"),
        "address": props.get("address"),
        "phone": props.get("phone"),
        "category": props.get("category"),
        "marker-symbol": props.get("marker-symbol"),
        "marker-color": props.get("marker-color"),
    }

def get_coords(feat: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    geom = feat.get("geometry") or {}
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    if gtype == "Point" and isinstance(coords, (list, tuple)) and len(coords) >= 2:
        # GeoJSON order: [lon, lat]
        lon, lat = float(coords[0]), float(coords[1])
        return lat, lon, "Point"
    return None, None, gtype

def maps_links(lat: Optional[float], lon: Optional[float]) -> str:
    if lat is None or lon is None:
        return ""
    g = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
    osm = f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f}"
    return f"[Google Maps]({g}) · [OpenStreetMap]({osm})"

def clean(s: Optional[str]) -> Optional[str]:
    if s is None: 
        return None
    s = s.strip()
    return s if s else None

def feature_key(feat: Dict[str, Any]) -> Tuple[str, str]:
    p = get_props(feat)
    cat = (p.get("category") or "~").lower()
    name = (p.get("name") or "~").lower()
    return (cat, name)

def write_header(out, title: str, count: int):
    now = dt.datetime.now().isoformat(timespec="seconds")
    out.write("---\n")
    out.write(f'title: "{title}"\n')
    out.write(f"generated_at: {now}\n")
    out.write(f"feature_count: {count}\n")
    out.write("schema: v1\n")
    out.write("---\n\n")
    out.write(f"# {md_escape(title)}\n\n")
    out.write("_This file is auto-generated from a GeoJSON source. Each record begins with a YAML block for reliable parsing, followed by a human-readable section._\n\n")

def write_feature(out, feat: Dict[str, Any], index: int):
    p = get_props(feat)
    lat, lon, gtype = get_coords(feat)

    # Normalized fields
    rec = {
        "id": p["id"],
        "name": clean(p["name"]),
        "category": clean(p["category"]),
        "address": clean(p["address"]),
        "phone": clean(p["phone"]),
        "description": clean(p["description"]),
        "coordinates": {"lat": lat, "lon": lon} if lat is not None and lon is not None else None,
        "geometry_type": gtype,
        "marker": {
            "symbol": clean(p["marker-symbol"]),
            "color": clean(p["marker-color"]),
        },
        "source_index": index
    }

    # --- Machine-friendly YAML block (delimits each record) ---
    out.write("---\n")
    # Minimal YAML emitter to keep script dependency-free
    def yaml_write(k, v, indent=0):
        sp = "  " * indent
        if isinstance(v, dict):
            out.write(f"{sp}{k}:\n")
            for kk, vv in v.items():
                yaml_write(kk, vv, indent + 1)
        elif isinstance(v, list):
            out.write(f"{sp}{k}:\n")
            for item in v:
                if isinstance(item, (dict, list)):
                    out.write(f"{sp}-\n")
                    # write nested with an extra indent
                    if isinstance(item, dict):
                        for kk, vv in item.items():
                            yaml_write(kk, vv, indent + 2)
                    else:
                        for i in item:
                            yaml_write("-", i, indent + 2)
                else:
                    out.write(f"{sp}- {repr(item)}\n")
        else:
            # Quote strings; leave numbers/None unquoted
            if isinstance(v, str):
                out.write(f"{sp}{k}: {json.dumps(v)}\n")
            elif v is None:
                out.write(f"{sp}{k}: null\n")
            else:
                out.write(f"{sp}{k}: {v}\n")

    for key in ["id","name","category","address","phone","description","geometry_type","coordinates","marker","source_index"]:
        yaml_write(key, rec[key])
    out.write("---\n\n")

    # --- Human-readable section ---
    title = rec["name"] or f"Feature {index}"
    out.write(f"## {md_escape(title)}\n\n")

    bullets: List[str] = []
    if rec["category"]:
        bullets.append(f"**Category:** {md_escape(rec['category'])}")
    if rec["address"]:
        bullets.append(f"**Address:** {md_escape(rec['address'])}")
    if rec["phone"]:
        bullets.append(f"**Phone:** {md_escape(rec['phone'])}")
    if lat is not None and lon is not None:
        bullets.append(f"**Coordinates:** {lat:.6f}, {lon:.6f}")
        link_str = maps_links(lat, lon)
        if link_str:
            bullets.append(f"**Map:** {link_str}")
    m = rec["marker"]
    if m and (m.get("symbol") or m.get("color")):
        sym = m.get("symbol") or "—"
        col = m.get("color") or "—"
        bullets.append(f"**Marker:** symbol `{sym}`, color `{col}`")

    if bullets:
        out.write("- " + "\n- ".join(bullets) + "\n\n")

    if rec["description"]:
        out.write(md_escape(rec["description"]) + "\n\n")

def convert(input_path: Path, output_path: Path, title: Optional[str] = None):
    fc = load_geojson(input_path)
    feats = list(fc.get("features") or [])
    feats.sort(key=feature_key)

    if title is None:
        title = f"{input_path.stem} (GeoJSON Export)"

    with output_path.open("w", encoding="utf-8") as out:
        write_header(out, title, len(feats))
        for i, f in enumerate(feats, start=1):
            write_feature(out, f, i)

def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert GeoJSON to RAG-friendly Markdown.")
    parser.add_argument("input", type=Path, help="Path to input GeoJSON/JSON file")
    parser.add_argument("output", type=Path, help="Path to output Markdown file")
    parser.add_argument("--title", type=str, default=None, help="Title for the Markdown document")
    args = parser.parse_args(argv)

    try:
        convert(args.input, args.output, args.title)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())