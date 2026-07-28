#!/usr/bin/env python3
"""
a2_entropy.py -- characterizes what the anonymization map destroys and preserves.

This needs no trained model and no new dataset: the anonymization is a
deterministic per-character map, so its effect on the pixel alphabet can be
measured directly on the source corpus. It answers "how much information is
suppressed" independently of "how much accuracy is lost", which is what makes
the pair of results interesting.

Usage:
    python a2_entropy.py --src /path/to/repos
    python a2_entropy.py --src ... --extensions .java,.js,.json,.svg
"""
import argparse
import math
import os
from collections import Counter

DEFAULT_EXTENSIONS = ('.java', '.js', '.json', '.svg', '.jsp', '.xml', '.sql',
                      '.html', '.css', '.py', '.properties', '.sh', '.yml', '.yaml')


def encrypt_chars(code: int) -> int:
    if code < 33:   return 32
    if code < 48:   return code
    if code < 58:   return 53
    if code < 65:   return code
    if code < 91:   return 77
    if code < 97:   return code
    if code < 123:  return 109
    if code < 127:  return code
    return 130


def entropy(counter):
    n = sum(counter.values())
    return -sum((v / n) * math.log2(v / n) for v in counter.values() if v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True)
    ap.add_argument('--extensions', default=','.join(DEFAULT_EXTENSIONS))
    ap.add_argument('--max-files', type=int, default=20000)
    ap.add_argument('--per-class', action='store_true',
                    help='also break the measurement down by file extension')
    args = ap.parse_args()

    exts = tuple(e if e.startswith('.') else '.' + e
                 for e in args.extensions.split(','))

    files = []
    for root, dirs, names in os.walk(args.src):
        dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '__pycache__')]
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
                if len(files) >= args.max_files:
                    break
        if len(files) >= args.max_files:
            break

    print(f'files analyzed: {len(files)}')
    if not files:
        return

    raw, anon = Counter(), Counter()
    by_ext = {}
    for path in files:
        try:
            text = open(path, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        ext = os.path.splitext(path)[1].lower()
        r_local, a_local = Counter(), Counter()
        for ch in text:
            code = ord(ch)
            if code > 255:
                code = 130
            enc = encrypt_chars(code)
            raw[code] += 1
            anon[enc] += 1
            if args.per_class:
                r_local[code] += 1
                a_local[enc] += 1
        if args.per_class:
            acc = by_ext.setdefault(ext, [Counter(), Counter()])
            acc[0].update(r_local)
            acc[1].update(a_local)

    h_raw, h_anon = entropy(raw), entropy(anon)
    total = sum(raw.values())
    preserved = sum(v for k, v in raw.items() if encrypt_chars(k) == k)

    print(f'\nalphabet size      : {len(raw)} -> {len(anon)} distinct intensities')
    print(f'entropy per pixel  : {h_raw:.3f} -> {h_anon:.3f} bits '
          f'({100 * (1 - h_anon / h_raw):.1f}% reduction)')
    print(f'characters passing through unchanged: {100 * preserved / total:.1f}% '
          f'(the symbol classes)')

    if args.per_class:
        print(f'\n{"extension":<14}{"H raw":>9}{"H anon":>9}{"reduction":>11}')
        for ext, (r, a) in sorted(by_ext.items()):
            hr, ha = entropy(r), entropy(a)
            print(f'{ext:<14}{hr:>9.3f}{ha:>9.3f}{100 * (1 - ha / hr):>10.1f}%')


if __name__ == '__main__':
    main()
