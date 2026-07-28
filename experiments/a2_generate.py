#!/usr/bin/env python3
"""
a2_generate.py -- paired minimap generation for the anonymization ablation (A2).

Renders every discovered source file TWICE in a single pass: once with the
character-level anonymization applied and once without it. Both renderings share
the output filename, the image geometry and the label, so the two datasets are
paired at the instance level. That pairing is what allows the paired statistical
tests in a2_experiment.py; it would be lost if the two datasets were generated
by two independent runs.

The anonymization map and the 8-pixel border reproduce minimaps.py from
munifgebara/codeminimap.

Output layout:
    <out>/anon/<label>/<name>.png
    <out>/plain/<label>/<name>.png
    <out>/catalog.csv

Usage:
    python a2_generate.py --src /path/to/repos --out /path/to/dataset_a2
    python a2_generate.py --src ... --out ... --variable-size --max-per-class 1000
"""
import argparse
import csv
import hashlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter

from PIL import Image

BORDER = 8
FIXED_SIZE = (128, 128)

# Extension -> label. Mirrors the EXTENSOES table of minimaps.py.
EXTENSIONS = {
    '.adoc': 'asciidoc', '.asciidoc': 'asciidoc', '.asm': 'asm', '.bat': 'batch',
    '.c': 'c', '.cc': 'cpp', '.cfg': 'config', '.clj': 'clojure', '.cmake': 'cmake',
    '.coffee': 'coffeescript', '.conf': 'config', '.cpp': 'cpp', '.cs': 'csharp',
    '.css': 'css', '.cxx': 'cpp', '.dart': 'dart', '.dockerfile': 'dockerfile',
    '.editorconfig': 'editorconfig', '.ejs': 'ejs', '.env': 'env',
    '.gitattributes': 'gitattributes', '.gitignore': 'gitignore', '.go': 'go',
    '.gradle': 'gradle', '.graphql': 'graphql', '.groovy': 'groovy', '.h': 'c-header',
    '.hh': 'cpp-header', '.hpp': 'cpp-header', '.htm': 'html', '.html': 'html',
    '.hxx': 'cpp-header', '.ini': 'ini', '.java': 'java', '.jl': 'julia', '.js': 'js',
    '.json': 'json', '.json5': 'json', '.jsonc': 'json', '.jsp': 'javajsp',
    '.jsx': 'javascript-jsx', '.kt': 'kotlin', '.kts': 'kotlin-script', '.less': 'less',
    '.lua': 'lua', '.m': 'objc', '.makefile': 'makefile', '.md': 'markdown',
    '.mk': 'makefile', '.mm': 'objc-cpp', '.php': 'php', '.pl': 'perl', '.pm': 'perl',
    '.properties': 'properties', '.ps1': 'powershell', '.psql': 'sql', '.py': 'python',
    '.r': 'r', '.rb': 'ruby', '.rs': 'rust', '.rst': 'rst', '.s': 'asm', '.sass': 'sass',
    '.scala': 'scala', '.scss': 'scss', '.sh': 'sh', '.sql': 'sql', '.svg': 'svg',
    '.swift': 'swift', '.toml': 'toml', '.ts': 'typescript', '.tsx': 'typescript-jsx',
    '.twig': 'twig', '.txt': 'text', '.vue': 'vue', '.xml': 'xml', '.xsl': 'xsl',
    '.yaml': 'yaml', '.yml': 'yaml',
}


# --- Java stereotype labelling -------------------------------------------
# Reconstructed from Java/JEE naming conventions: the stereotype is inferred
# from the class name suffix first and from the package path second. The
# original definitions used in the IWSSIP study are not available in the
# published artifact, so this rule set must be validated before the numbers it
# produces are reported as comparable to that study.

CLASS_SUFFIX_STEREOTYPES = [
    ('Impl',      'javaimplementation'),
    ('DAO',       'javadao'),
    ('DTO',       'javadto'),
    ('Service',   'javaservice'),
    ('Action',    'javaaction'),
    ('Form',      'javaform'),
    ('Converter', 'javaconverter'),
    ('Exception', 'javaexception'),
    ('Factory',   'javafactory'),
    ('Handler',   'javahandler'),
    ('Tag',       'javatag'),
]

PACKAGE_STEREOTYPES = [
    ('entity',      'javaentity'),
    ('persistence', 'javapersistence'),
    ('service',     'javaservice'),
    ('converter',   'javaconverter'),
    ('form',        'javaform'),
    ('dto',         'javadto'),
]

NON_JAVA_STEREOTYPES = {'.jsp': 'javajsp', '.jrxml': 'javajasper'}

# Directories that hold build output rather than sources.
SKIP_DIR_TOKENS = ('build', 'target', 'classes', 'generated', 'out')


def java_stereotype(path, filename):
    stem = os.path.splitext(filename)[0]
    lowered = path.replace('\\', '/').lower()

    if '/tst/' in lowered or '/test/' in lowered or stem.endswith(('Test', 'Tests', 'IT')):
        return 'javaintegrationtest' if 'integration' in lowered else 'javaunittest'

    for suffix, label in CLASS_SUFFIX_STEREOTYPES:
        if stem.endswith(suffix):
            return label

    segments = set(lowered.rsplit('/', 1)[0].split('/'))
    for segment, label in PACKAGE_STEREOTYPES:
        if segment in segments:
            return label
    return 'java'


def label_for(path, mode):
    filename = os.path.basename(path)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in EXTENSIONS:
        return None
    if mode == 'extension':
        return EXTENSIONS[ext]
    if ext == '.java':
        return java_stereotype(path, filename)
    return NON_JAVA_STEREOTYPES.get(ext, EXTENSIONS[ext])


def encrypt_chars(code: int) -> int:
    """Character-level anonymization. Letters and digits collapse onto a single
    intensity each; symbols pass through untouched."""
    if code < 33:   return 32    # non-printable -> space
    if code < 48:   return code  # symbols
    if code < 58:   return 53    # digits
    if code < 65:   return code  # symbols
    if code < 91:   return 77    # uppercase
    if code < 97:   return code  # symbols
    if code < 123:  return 109   # lowercase
    if code < 127:  return code  # symbols
    return 130


def read_lines(path):
    try:
        with open(path, encoding='utf-8', errors='ignore') as f:
            return f.read().split('\n')
    except (OSError, UnicodeError):
        return None


def render_pair(lines, fixed):
    """Returns (anonymized, plain) images with identical geometry."""
    if fixed:
        width, height = FIXED_SIZE
    else:
        width = max((len(l) for l in lines), default=1) + 2 * BORDER
        height = len(lines) + 2 * BORDER
        width = max(width, 2 * BORDER + 1)
        height = max(height, 2 * BORDER + 1)

    img_anon = Image.new('RGB', (width, height), 'black')
    img_plain = Image.new('RGB', (width, height), 'black')
    px_anon, px_plain = img_anon.load(), img_plain.load()

    y = 0
    for line in lines:
        limit = min(len(line), width - 2 * BORDER)
        for x in range(limit):
            raw = ord(line[x])
            if raw > 255:
                raw = 130
            enc = encrypt_chars(raw)
            px_plain[x + BORDER, y + BORDER] = (raw, raw, raw)
            px_anon[x + BORDER, y + BORDER] = (enc, enc, enc)
        y += 1
        if y + 2 * BORDER >= height:
            break
    return img_anon, img_plain


def job(args):
    path, label, out, fixed = args
    name = hashlib.sha1(path.encode('utf-8', 'ignore')).hexdigest()[:16] + '.png'
    done = all(os.path.exists(os.path.join(out, v, label, name))
               for v in ('anon', 'plain'))
    if done:
        return name, label, path
    lines = read_lines(path)
    if lines is None or not any(line.strip() for line in lines):
        return None
    try:
        img_anon, img_plain = render_pair(lines, fixed)
    except Exception:
        return None

    name = hashlib.sha1(path.encode('utf-8', 'ignore')).hexdigest()[:16] + '.png'
    for variant, img in (('anon', img_anon), ('plain', img_plain)):
        d = os.path.join(out, variant, label)
        os.makedirs(d, exist_ok=True)
        img.save(os.path.join(d, name))
    return name, label, path


def discover(src, min_bytes, mode):
    """Only files whose extension is in EXTENSIONS are ever opened; binaries
    such as .class, .jar or images are never read."""
    found = []
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs
                   if d not in ('.git', 'node_modules', '__pycache__')
                   and d.lower() not in SKIP_DIR_TOKENS]
        for fn in files:
            p = os.path.join(root, fn)
            label = label_for(p, mode)
            if not label:
                continue
            try:
                if os.path.getsize(p) < min_bytes:
                    continue
            except OSError:
                continue
            found.append((p, label))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='root directory containing the repositories')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--variable-size', action='store_true',
                    help='render at the natural file size instead of 128x128')
    ap.add_argument('--min-per-class', type=int, default=200)
    ap.add_argument('--max-per-class', type=int, default=1000)
    ap.add_argument('--min-bytes', type=int, default=200)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--seed', type=int, default=18081975)
    ap.add_argument('--file-list', default=None,
                    help='text file with one absolute path per line; skips the '
                         'directory walk, which is slow on network mounts')
    ap.add_argument('--no-source-paths', action='store_true',
                    help='omit original file paths from catalog.csv (use for '
                         'confidential corpora, where the paths themselves '
                         'reveal package and module structure)')
    ap.add_argument('--label-mode', choices=('extension', 'java-stereotype'),
                    default='extension',
                    help='how to derive the class label (default: extension)')
    args = ap.parse_args()

    fixed = not args.variable_size

    if args.file_list:
        print(f'reading file list {args.file_list} ...', flush=True)
        found = []
        with open(args.file_list, encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                candidate = line.rstrip('\n')
                if not candidate:
                    continue
                label = label_for(candidate, args.label_mode)
                if label:
                    found.append((candidate, label))
    else:
        print(f'scanning {args.src} ...', flush=True)
        found = discover(args.src, args.min_bytes, args.label_mode)
    counts = Counter(label for _, label in found)
    keep = {c for c, n in counts.items() if n >= args.min_per_class}
    if not keep:
        sys.exit(f'no class reached --min-per-class={args.min_per_class}; '
                 f'largest was {counts.most_common(1)}')

    import random
    rng = random.Random(args.seed)
    selected = []
    for label in sorted(keep):
        pool = [p for p, l in found if l == label]
        rng.shuffle(pool)
        selected += [(p, label) for p in pool[:args.max_per_class]]

    print(f'{len(keep)} classes kept, {len(selected)} files selected')
    for label in sorted(keep):
        print(f'  {label:22s} {min(counts[label], args.max_per_class):6d} '
              f'(available {counts[label]})')

    os.makedirs(args.out, exist_ok=True)
    rows = []
    tasks = [(p, l, args.out, fixed) for p, l in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(job, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r:
                rows.append(r)
            if i % 2000 == 0:
                print(f'  rendered {i}/{len(tasks)}')

    with open(os.path.join(args.out, 'catalog.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if args.no_source_paths:
            w.writerow(['image', 'label'])
            w.writerows(sorted((n, l) for n, l, _ in rows))
        else:
            w.writerow(['image', 'label', 'source_path'])
            w.writerows(sorted(rows))

    print(f'\ndone: {len(rows)} paired minimaps under {args.out}')
    print('  anon/   character-level anonymization applied')
    print('  plain/  identical geometry, original intensities')


if __name__ == '__main__':
    main()
