#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
schh_md_to_calendar.py — Parse SCHH Markdown events into CSV and ICS.

Input shape (as in your sample):
### Event Title
  - When: 2025-09-03, 4:00 PM → 6:00 PM
  - Where: Hidden Cypress and Lakehouse Outdoor Pools
  - Details: (free text; may be long and wrap)

Or:
  - When: 2025-09-04 (All Day)

Usage:
  python schh_md_to_calendar.py input.md --csv events.csv --ics events.ics --tz America/New_York
"""

import os
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
BASEDIR = os.path.dirname(SCRIPT_DIR)

# Directory containing source Markdown files
SOURCE_DIR = os.path.join(BASEDIR, 'docs', 'Events')

import argparse
import csv
import re
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import unicodedata

# ---- Config ----
CSV_HEADERS = ["Subject","Start Date","Start Time","End Date","End Time","All Day Event","Description","Location","Private"]
DEFAULT_DURATION_MIN = 60

# Regexes for structure
RE_H3      = re.compile(r'^\s{0,3}#{3}\s+(.+?)\s*$')                 # "### Title"
RE_BULLET  = re.compile(r'^\s*[-*]\s*(\w+)\s*:\s*(.+?)\s*$')         # "- Key: Value"
RE_WHEN    = re.compile(r'^\s*(\d{4}-\d{2}-\d{2})(?:\s*,\s*(.+))?\s*$')
RE_ALLDAY  = re.compile(r'\(\s*All\s*Day\s*\)', re.IGNORECASE)
RE_RANGE   = re.compile(r'\s*(.+?)\s*(?:→|->)\s*(.+?)\s*$')          # "4:00 PM → 6:00 PM" or "->"
RE_TIME    = re.compile(r'^\s*(\d{1,2}:\d{2}\s*[AP]M)\s*$', re.IGNORECASE)
RE_TIME_LABEL = re.compile(r'\b(\d{1,2}:\d{2}\s*[AP]M)\b', re.IGNORECASE)

def _norm_text(s: str) -> str:
    """Lowercase, strip, collapse whitespace, and remove diacritics for stable hashing."""
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s

def stable_uid(ev: dict, namespace: str = "schh", include_time: bool = True) -> str:
    """
    Create a stable UID for the event. By default includes start/end times in the key.
    If you want time edits to *update* the same logical event on re-import, set include_time=False.
    """
    title = _norm_text(ev.get("title", ""))
    loc   = _norm_text(ev.get("location", ""))
    details = _norm_text(ev.get("details", ""))  # optional in the key; often omitted

    if ev.get("all_day"):
        date_key = ev["start"].strftime("%Y-%m-%d")
        time_key = "ALLDAY"
    else:
        # Use local times exactly as parsed (no TZ conversion) for stability
        date_key = ev["start"].strftime("%Y-%m-%d")
        if include_time:
            time_key = f"{ev['start'].strftime('%H:%M')}..{(ev.get('end') or ev['start']).strftime('%H:%M')}"
        else:
            time_key = "NO-TIME"

    key = f"{namespace}|{title}|{date_key}|{time_key}|{loc}"
    # Hash to keep UID compact and RFC-safe
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:32]
    return f"{namespace}-{digest}@md2cal"

def parse_time_label(s: str) -> datetime.time:
    """Parse a 'h:mm AM/PM' string into a time object."""
    return datetime.strptime(s.strip().upper().replace(' ', ''), "%I:%M%p").time()

def clean_location(s: str) -> str:
    return s.rstrip(' ,').strip()

def ics_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace(",", "\\,").replace(";", "\\;")

def fold_ics_line(line: str, limit: int = 75) -> str:
    # Basic UTF-8 fold per RFC 5545 (safe enough for typical content)
    b = line.encode("utf-8")
    out = []
    while len(b) > limit:
        out.append(b[:limit].decode("utf-8", errors="ignore"))
        b = b[limit:]
        b = b"\r\n " + b
    out.append(b.decode("utf-8", errors="ignore"))
    return "".join(out)

def dt_ics_local(dt: datetime, tzid: str) -> str:
    # Local time with TZID (no Z).
    return f";TZID={tzid}:{dt.strftime('%Y%m%dT%H%M%S')}"

def parse_markdown(md_text: str):
    """
    Returns a list of dict events with:
      title, start (datetime or date-only sentinel), end (datetime or None), all_day(bool), location, details
    """
    lines = md_text.splitlines()
    events = []
    i = 0
    current = None  # temp buffer for accumulating multi-line Details

    def push_current():
        nonlocal current
        if current and current.get("title") and current.get("when_seen"):
            # finalize any accumulated details
            if "details_buf" in current:
                current["details"] = " ".join(current["details_buf"]).strip()
                del current["details_buf"]
            norm = normalize_event(current)
            events.extend(norm if isinstance(norm, list) else [norm])
        current = None

    while i < len(lines):
        line = lines[i]

        # Event title line
        m_h3 = RE_H3.match(line)
        if m_h3:
            # Push previous event (if any)
            push_current()
            current = {"title": m_h3.group(1).strip(), "details_buf": []}
            i += 1
            continue

        # If we're within an event, parse bullets
        if current is not None:
            m_b = RE_BULLET.match(line)
            if m_b:
                key = m_b.group(1).strip().lower()
                val = m_b.group(2).strip()
                if key == "when":
                    current["when_seen"] = True
                    current["when_raw"] = val
                elif key == "where":
                    current["location"] = clean_location(val)
                elif key in ("details", "description", "info"):
                    current.setdefault("details_buf", []).append(val)
                else:
                    # unknown key: stash into details
                    current.setdefault("details_buf", []).append(f"{key.title()}: {val}")
            else:
                # Continuation line for Details (indented or blank lines)
                if line.strip():
                    current.setdefault("details_buf", []).append(line.strip())

        i += 1

    # push the last one
    push_current()
    return events

def normalize_event(ev):
    """
    Return a *list* of normalized events to support multiple showtimes on the same date.

    Input ev fields:
      title, when_raw, location?, details?
    Output list items:
      { title, all_day(bool), start(dt), end(dt or None), location, details }
    """
    title = ev["title"].strip()
    details = ev.get("details", "").strip()
    location = ev.get("location", "").strip()
    when_raw = ev.get("when_raw", "").strip()

    # All-day detection
    if RE_ALLDAY.search(when_raw):
        m = RE_WHEN.match(when_raw.replace("(All Day)", "").strip())
        if not m:
            raise ValueError(f"Unrecognized All Day format: {when_raw}")
        date_obj = datetime.strptime(m.group(1), "%Y-%m-%d")
        return [{
            "title": title,
            "all_day": True,
            "start": date_obj,
            "end": None,
            "location": location,
            "details": details
        }]

    # Timed event(s)
    m = RE_WHEN.match(when_raw)
    if not m:
        raise ValueError(f"Unrecognized 'When' format: {when_raw}")
    date_str, time_part = m.group(1), (m.group(2) or "").strip()
    base_date = datetime.strptime(date_str, "%Y-%m-%d")

    # Case 1: explicit range "X → Y" or "X -> Y" (single event)
    mr = RE_RANGE.match(time_part) if time_part else None
    if mr:
        t1, t2 = mr.group(1).strip(), mr.group(2).strip()
        start_dt = datetime.combine(base_date.date(), parse_time_label(t1))
        end_dt   = datetime.combine(base_date.date(), parse_time_label(t2))
        return [{
            "title": title,
            "all_day": False,
            "start": start_dt,
            "end": end_dt,
            "location": location,
            "details": details
        }]

    # Case 2: multiple showtimes on the same day, e.g. "5:30 PM and 8:00 PM"
    # Extract *all* time labels; if we get 2+ times, make one event per time.
    times = RE_TIME_LABEL.findall(time_part) if time_part else []
    if len(times) >= 2:
        out = []
        for t in times:
            st = datetime.combine(base_date.date(), parse_time_label(t))
            out.append({
                "title": title,
                "all_day": False,
                "start": st,
                "end": st + timedelta(minutes=DEFAULT_DURATION_MIN),
                "location": location,
                "details": details
            })
        return out

    # Case 3: single time (or date-only fallback)
    if time_part:
        # single time like "5:30 PM"
        mt = RE_TIME_LABEL.search(time_part)
        if not mt:
            raise ValueError(f"Unrecognized time segment: {time_part}")
        st = datetime.combine(base_date.date(), parse_time_label(mt.group(1)))
        en = st + timedelta(minutes=DEFAULT_DURATION_MIN)
        return [{
            "title": title,
            "all_day": False,
            "start": st,
            "end": en,
            "location": location,
            "details": details
        }]
    else:
        # Date provided without time; treat as default-duration from 00:00
        st = base_date
        en = st + timedelta(minutes=DEFAULT_DURATION_MIN)
        return [{
            "title": title,
            "all_day": False,
            "start": st,
            "end": en,
            "location": location,
            "details": details
        }]

def write_csv(events, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADERS)
        for ev in events:
            if ev["all_day"]:
                sd = ev["start"].strftime("%m/%d/%Y")
                w.writerow([ev["title"], sd, "", sd, "", "True", ev["details"], ev["location"], "False"])
            else:
                sd = ev["start"].strftime("%m/%d/%Y")
                st = ev["start"].strftime("%H:%M")
                ed = (ev["end"] or (ev["start"] + timedelta(minutes=DEFAULT_DURATION_MIN)))
                w.writerow([ev["title"], sd, st, ed.strftime("%m/%d/%Y"), ed.strftime("%H:%M"), "False", ev["details"], ev["location"], "False"])

def write_ics(events, ics_path: Path, tzid: str, include_time_in_uid: bool = True):
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//SCHH md2cal//EN"
    ]
    for ev in events:
        uid = stable_uid(ev, namespace="schh", include_time=include_time_in_uid)

        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{uid}")
        lines.append(f"DTSTAMP:{now}")

        title = ics_escape(ev["title"])
        desc  = ics_escape(ev.get("details", ""))
        loc   = ics_escape(ev.get("location", ""))

        if ev["all_day"]:
            d = ev["start"]
            lines.append(f"DTSTART;VALUE=DATE:{d.strftime('%Y%m%d')}")
            lines.append(f"DTEND;VALUE=DATE:{(d + timedelta(days=1)).strftime('%Y%m%d')}")  # exclusive
        else:
            st = ev["start"]
            en = ev["end"] or (st + timedelta(minutes=DEFAULT_DURATION_MIN))
            lines.append("DTSTART" + dt_ics_local(st, tzid))
            lines.append("DTEND"   + dt_ics_local(en, tzid))

        lines.append(fold_ics_line(f"SUMMARY:{title}"))
        if desc:
            lines.append(fold_ics_line(f"DESCRIPTION:{desc}"))
        if loc:
            lines.append(fold_ics_line(f"LOCATION:{loc}"))
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")

    ics_path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Markdown file to parse")
    ap.add_argument("--csv", default=f"{SOURCE_DIR}/events.csv", help="Output CSV path")
    ap.add_argument("--ics", default=f"{SOURCE_DIR}/events.ics", help="Output ICS path")
    ap.add_argument("--tz",  default="America/New_York", help="TZID for ICS local times")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    events = parse_markdown(text)

    if not events:
        print("No events parsed.")
        return

    write_csv(events, Path(args.csv))
    write_ics(events, Path(args.ics), args.tz)
    print(f"Wrote {args.csv} and {args.ics} with {len(events)} events.")

if __name__ == "__main__":
    main()