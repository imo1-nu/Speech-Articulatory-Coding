#!/usr/bin/env python3
"""Create a plain HTML listening page from the saved VQ/RVQ comparison."""

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


LABELS = {
    'original': '元音声',
    'original_ema': '元の調音軌道から合成',
    'vq128': '一段VQ・128コード',
    'vq512': '一段VQ・512コード',
    'rvq1': 'RVQ・128コード×1段',
    'rvq2': 'RVQ・128コード×2段',
    'rvq4': 'RVQ・128コード×4段',
    'rvq8': 'RVQ・128コード×8段',
    'continuous_ae': '連続AE（量子化なし・参考）',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('outputs/jsut_rvq/comparison_verified'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/jsut_vq_rvq_listening'))
    args = parser.parse_args()
    if args.source_dir.resolve() == args.output_dir.resolve():
        parser.error('Choose an output directory different from the source')
    manifest = json.loads((args.source_dir/'manifest.json').read_text())
    with (args.source_dir/'audio_metrics.csv').open() as stream:
        hashes = {(r['utterance_id'], r['condition']): r['audio_sha256'] for r in csv.DictReader(stream)}
    sections = []
    for record in manifest['audio_records']:
        players = []
        for condition, label in LABELS.items():
            path = args.source_dir/record['directory']/f'{condition}.wav'
            if not path.resolve().is_relative_to(args.source_dir.resolve()):
                raise ValueError('Audio path escapes source directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[record['utterance_id'], condition]:
                raise ValueError(f'Audio checksum mismatch: {path}')
            url = html.escape(quote(os.path.relpath(path, args.output_dir)), quote=True)
            players.append(f'<div><h3>{html.escape(label)}</h3><audio controls preload="none" src="{url}"></audio></div>')
        sections.append(f'<section><h2>{html.escape(record["utterance_id"])}</h2>'
                        f'<p>{html.escape(record["text"])}</p><div class="players">'+''.join(players)+'</div></section>')
    page = '''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>一段VQとRVQの音声聴き比べ</title>
<style>body{font-family:system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:16px;line-height:1.6}
section{border:1px solid #ccc;border-radius:6px;padding:16px;margin:24px 0}.players{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px}
h3{font-size:1rem;margin:0 0 8px}audio{width:100%}</style></head><body>
<h1>一段VQとRVQの音声聴き比べ</h1>
<p>同じ5発話の元音声・元軌道合成・一段VQ・RVQを並べています。編集・FBなしの再構成音声です。
40 ms窓・各段50 Hzで、F0・音量・話者・合成ゲインは合成条件間で共通です。</p>'''
    page += ''.join(sections)+'</body></html>'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir/'index.html'
    destination.write_text(page, encoding='utf-8')
    print(f'{destination}: {len(sections)} utterances, {len(sections)*len(LABELS)} audio players')


if __name__ == '__main__':
    main()
