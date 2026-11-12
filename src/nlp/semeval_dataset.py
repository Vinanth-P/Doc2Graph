"""Strict SemEval-2010 Task 8 parser; offsets refer to untagged text."""
import re
from pathlib import Path

FAMILIES = ('Cause-Effect', 'Instrument-Agency', 'Product-Producer',
            'Content-Container', 'Entity-Origin', 'Entity-Destination',
            'Component-Whole', 'Member-Collection', 'Message-Topic')
LABELS = tuple(f'{r}({d})' for r in FAMILIES for d in ('e1,e2', 'e2,e1')) + ('Other',)


def parse_marked(text):
    parts, spans, opened, pos, cursor = [], {}, None, 0, 0
    for match in re.finditer(r'</?e[12]>', text):
        segment = text[cursor:match.start()]
        parts.append(segment)
        pos += len(segment)
        tag = match.group()
        key = tag.strip('</>')
        if tag.startswith('</'):
            if opened != key or pos <= spans[key][0]:
                raise ValueError('Malformed or empty entity span')
            spans[key].append(pos)
            opened = None
        else:
            if opened or key in spans:
                raise ValueError('Nested or repeated entity marker')
            spans[key] = [pos]
            opened = key
        cursor = match.end()
    parts.append(text[cursor:])
    if opened or set(spans) != {'e1', 'e2'} or any(len(v) != 2 for v in spans.values()):
        raise ValueError('Exactly two complete entity spans required')
    return {'text': ''.join(parts), 'e1': spans['e1'], 'e2': spans['e2']}


def read_dataset(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    records, seen, i = [], set(), 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        match = re.fullmatch(r'(\d+)\t"(.*)"', lines[i])
        if not match or i + 1 >= len(lines):
            raise ValueError(f'Malformed record at line {i+1}')
        eid, marked = match.groups()
        label = lines[i+1].strip()
        if eid in seen or label not in LABELS:
            raise ValueError(f'Duplicate ID or invalid label: {eid}, {label}')
        seen.add(eid)
        records.append(dict(parse_marked(marked), example_id=eid, label=label))
        i += 2
        while i < len(lines) and lines[i].strip():
            if not lines[i].startswith('Comment:'):
                raise ValueError(f'Unexpected metadata at line {i+1}')
            i += 1
    return records


def family(label):
    return label.split('(')[0]
