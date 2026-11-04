#!/usr/bin/env python3
"""
Odyssey Text Extraction Script
Extracts Homer's Odyssey (translated by Robert Fitzgerald) from PDF
into individual chapter files (one file per Book, 24 books in total).
"""

import os
import re
import sys
import zlib
from pathlib import Path


def parse_pdf_string(s: bytes) -> str:
    """Decodes PDF string literal bytes into MacRoman / Unicode string."""
    ba = bytearray()
    i = 0
    while i < len(s):
        if s[i:i + 1] == b'\\':
            i += 1
            if i >= len(s):
                break
            c = s[i:i + 1]
            if c == b'n':
                ba.append(10)
            elif c == b'r':
                ba.append(13)
            elif c == b't':
                ba.append(9)
            elif c in (b'(', b')', b'\\'):
                ba.append(c[0])
            elif c.isdigit():
                m_oct = re.match(rb'^[0-7]{1,3}', s[i:i + 3])
                if m_oct:
                    ba.append(int(m_oct.group(0), 8))
                    i += len(m_oct.group(0)) - 1
                else:
                    ba.append(c[0])
            else:
                ba.append(c[0])
        else:
            ba.append(s[i])
        i += 1
    return ba.decode('mac_roman', errors='replace')


def decode_tj(operator_block: bytes, is_heading: bool = False) -> str:
    """Parses text from TJ or Tj operator, adding spacing for large kerns in titles."""
    out = []
    # Match strings inside parentheses or numeric kerning adjustments
    for m in re.finditer(rb'\(((?:\\.|[^\\()])*)\)|(-?\d+(?:\.\d+)?)', operator_block):
        if m.group(1) is not None:
            out.append(parse_pdf_string(m.group(1)))
        elif m.group(2) is not None and is_heading:
            val = float(m.group(2))
            if val < -200:
                out.append(' ')
    return ''.join(out)


def merge_horizontal_segments(line_elements: list) -> str:
    """
    Merges text segments sharing the same horizontal baseline (y-coordinate).
    Handles composite accents (e.g. acute accent followed by 'y'), quotes, and punctuation.
    """
    line_elements.sort(key=lambda item: item[1])  # sort by x-coordinate
    merged = ''
    for j, (y, x, t) in enumerate(line_elements):
        if j == 0:
            merged = t
        else:
            prev = merged
            # Handle acute accent over y / Y (typeset as ´ + y)
            if prev.endswith('´'):
                accent_char = 'ý' if t.startswith('y') else ('Ý' if t.startswith('Y') else '´' + t[0])
                merged = prev[:-1] + accent_char + t[1:]
            # Handle open quotes / brackets
            elif prev.endswith(' ‘') or prev.endswith('“') or prev.endswith('—') or prev.endswith('-') or prev.endswith('('):
                merged = prev + t
            # Handle closing punctuation and apostrophes
            elif t.startswith('’') or t.startswith(',') or t.startswith('.') or t.startswith(';') or \
                 t.startswith(':') or t.startswith('?') or t.startswith('!') or t.startswith('”') or \
                 t.startswith('—') or t.startswith('-') or t.startswith(')'):
                merged = prev.rstrip() + t
            elif prev.endswith(' ') or t.startswith(' '):
                merged = prev + t
            else:
                merged = prev + ' ' + t
    return merged.strip()


def extract_chapters(pdf_path: str, output_dir: str):
    """
    Parses the Odyssey PDF, extracts the 24 Books, and writes individual files.
    """
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"Reading PDF from {pdf_path}...")
    with open(pdf_file, 'rb') as f:
        data = f.read()

    # Index all PDF indirect objects
    print("Indexing PDF objects...")
    objs = {}
    for m in re.finditer(rb'(\d+)\s+0\s+obj\b(.*?)\bendobj\b', data, re.DOTALL):
        objs[int(m.group(1))] = m.group(2)

    # Traverse Page Tree to ensure strictly sequential page ordering
    def get_pages(oid: int) -> list:
        body = objs[oid]
        if re.search(rb'/Type\s*/Pages\b', body):
            km = re.search(rb'/Kids\s*\[(.*?)\]', body, re.DOTALL)
            res = []
            for k in [int(x) for x in re.findall(rb'(\d+)\s+0\s+R', km.group(1))]:
                res.extend(get_pages(k))
            return res
        elif re.search(rb'/Type\s*/Page\b', body):
            return [oid]
        return []

    root_m = re.search(rb'/Root\s+(\d+)\s+0\s+R', data)
    catalog = objs[int(root_m.group(1))]
    pages_root = int(re.search(rb'/Pages\s+(\d+)\s+0\s+R', catalog).group(1))
    page_oids = get_pages(pages_root)

    print(f"Discovered {len(page_oids)} pages in PDF document.")
    # The Odyssey text runs on pages 1 to 322 (pages 323-339 contain Aesop's Fables)
    odyssey_pages = min(322, len(page_oids))

    books = []
    current_book = None
    pending_title = None

    print("Extracting verse lines and segmenting chapters...")
    for p_num in range(1, odyssey_pages + 1):
        poid = page_oids[p_num - 1]
        cm = re.search(rb'/Contents\s+(\d+)\s+0\s+R', objs[poid])
        if not cm:
            continue
        raw_stream = objs[int(cm.group(1))]
        sm = re.search(rb'stream\r?\n(.*)\r?\nendstream', raw_stream, re.DOTALL)
        if not sm:
            continue
        dec = zlib.decompress(sm.group(1))

        elements = []
        for bt in re.finditer(rb'\bBT\b(?:(?:\((?:\\.|[^\\()])*\))|[^()])*?\bET\b', dec, re.DOTALL):
            b = bt.group(0)
            tm = re.search(rb'([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+Tm', b)
            if not tm:
                continue

            d = round(float(tm.group(4)), 2)
            x = float(tm.group(5))
            y = float(tm.group(6))

            # Filter running headers and margin marks at top and bottom of page
            if y >= 638 or y < 45:
                continue
            # Filter non-body font sizes (footnotes: 8pt/6pt; line numbers: 6pt; superscripts: 7.5pt)
            if d < 9.0:
                continue
            # Filter introductory headnote on page 1 (size 9pt above y=300)
            if p_num == 1 and y > 300:
                continue
            # 9pt text elsewhere is only the Sirens' lyric songs on pages 153-154
            if d == 9.0 and p_num not in (153, 154):
                continue

            tj_m = re.search(rb'\[(.*?)\]\s*TJ|\((?:\\.|[^\\()])*\)\s*Tj', b, re.DOTALL)
            is_hd = (d >= 11.0)
            txt = decode_tj(tj_m.group(0), is_heading=is_hd) if tj_m else ''
            if txt.strip():
                elements.append((y, x, d, txt))

        # Group elements on the same horizontal line (within 1.5 pt tolerance)
        grouped = []
        for elem in sorted(elements, key=lambda e: -e[0]):
            if grouped and abs(grouped[-1][0][0] - elem[0]) < 1.5:
                grouped[-1].append(elem)
            else:
                grouped.append([elem])

        for grp in grouped:
            max_d = max(e[2] for e in grp)
            # Chapter heading detection (12pt font)
            if max_d >= 11.0:
                title_part = ' '.join(e[3].strip() for e in sorted(grp, key=lambda e: e[1]))
                title_part = re.sub(r'\s+', ' ', title_part).strip()
                if 'BOOK' in title_part:
                    if pending_title:
                        current_book = {'title': pending_title, 'pages': [p_num], 'lines': []}
                        books.append(current_book)
                        pending_title = None
                    pending_title = title_part
                elif pending_title:
                    # Multi-line title wrap
                    pending_title = f"{pending_title} {title_part}"
                    current_book = {'title': pending_title, 'pages': [p_num], 'lines': []}
                    books.append(current_book)
                    pending_title = None
            else:
                if pending_title:
                    current_book = {'title': pending_title, 'pages': [p_num], 'lines': []}
                    books.append(current_book)
                    pending_title = None
                if current_book is not None:
                    line_text = merge_horizontal_segments([(e[0], e[1], e[3]) for e in grp])
                    if line_text:
                        if p_num not in current_book['pages']:
                            current_book['pages'].append(p_num)
                        current_book['lines'].append(line_text)

    # Save chapters to output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(books)} chapters to {out_path.resolve()}:")
    for idx, b in enumerate(books, 1):
        filename = f"chapter_{idx:02d}.txt"
        file_path = out_path / filename

        content = [b['title'], "=" * len(b['title']), ""]
        content.extend(b['lines'])
        content.append("")  # trailing newline

        file_path.write_text("\n".join(content), encoding="utf-8")
        print(f"  [{idx:02d}/24] {filename:16s} : {b['title']} ({len(b['lines'])} lines, pp. {b['pages'][0]}-{b['pages'][-1]})")

    print("\nExtraction successfully completed.")


if __name__ == '__main__':
    default_pdf = "docs/odyssey-fitzgerald.pdf"
    if not os.path.exists(default_pdf) and os.path.exists("docs/odyssey-firtzgerald.pdf"):
        default_pdf = "docs/odyssey-firtzgerald.pdf"

    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else default_pdf
    out_arg = sys.argv[2] if len(sys.argv) > 2 else "artifact/document-store"

    extract_chapters(pdf_arg, out_arg)

