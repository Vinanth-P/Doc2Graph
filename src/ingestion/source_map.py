"""Conservative covering-span maps through lossy NFKC/whitespace normalization."""
from __future__ import annotations
import bisect
import unicodedata


class SourceMap:
    def __init__(self, raw_text, normalized_text, source_spans):
        if len(normalized_text) != len(source_spans):
            raise ValueError('One source span is required per normalized character')
        self.raw_text, self.normalized_text = raw_text, normalized_text
        self._starts = [s for s, _ in source_spans]
        self._ends = [e for _, e in source_spans]
        if any(not 0 <= s < e <= len(raw_text) for s, e in source_spans):
            raise ValueError('Invalid source span')
        if self._starts != sorted(self._starts) or self._ends != sorted(self._ends):
            raise ValueError('Source spans must be monotonic')

    @classmethod
    def create(cls, raw_text, normalize_unicode=True, normalize_whitespace=False):
        segments = []
        start, segment = 0, ''
        for index, char in enumerate(raw_text):
            if not normalize_unicode:
                segments.append((char, index, index + 1))
                continue
            separate = unicodedata.normalize('NFKC', segment) + unicodedata.normalize('NFKC', char)
            combined = unicodedata.normalize('NFKC', segment + char)
            if segment and not unicodedata.combining(char) and separate == combined:
                segments.append((unicodedata.normalize('NFKC', segment), start, index))
                start, segment = index, char
            else:
                segment += char
        if normalize_unicode and segment:
            segments.append((unicodedata.normalize('NFKC', segment), start, len(raw_text)))
        chars, spans = [], []
        for text, start, end in segments:
            for char in text:
                if normalize_whitespace and char.isspace():
                    char = ' '
                    if chars and chars[-1] == ' ':
                        spans[-1] = (spans[-1][0], end)
                        continue
                chars.append(char)
                spans.append((start, end))
        normalized = ''.join(chars)
        return normalized, cls(raw_text, normalized, spans)

    @staticmethod
    def _check(start, end, length):
        if not 0 <= start <= end <= length:
            raise IndexError(f'Span [{start}, {end}) outside length {length}')

    def normalized_to_source(self, start, end):
        """Cover all source characters contributing to a nonempty normalized span."""
        self._check(start, end, len(self.normalized_text))
        if start == end:
            boundary = self._starts[start] if start < len(self._starts) else len(self.raw_text)
            return boundary, boundary
        return self._starts[start], self._ends[end - 1]

    def source_to_normalized(self, raw_start, raw_end):
        """Cover normalized characters overlapping a source span; not a bijection."""
        self._check(raw_start, raw_end, len(self.raw_text))
        start = bisect.bisect_right(self._ends, raw_start)
        if raw_start == raw_end:
            return start, start
        return start, bisect.bisect_left(self._starts, raw_end)
