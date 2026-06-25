"""
KCL → CYMO CSV (Split by Task)
===============================
Reads Whisper JSON transcriptions from the KCL dataset and produces
two separate CYMO-compatible CSVs:
  1. cymo_kcl_readtext.csv      — ReadText recordings only
  2. cymo_kcl_dialogue.csv      — SpontaneousDialogue recordings only

Usage:
    python kcl_to_cymo_split.py --root ../KCL

Requirements:
    pip install pandas
"""

import os
import sys
import json
import argparse
import pandas as pd


def parse_kcl_filename(filename):
    """Parse ID02_pd_1_2_1.wav into components."""
    name = os.path.splitext(filename)[0]
    parts = name.split('_')
    return {
        'subject_id': parts[0] if len(parts) >= 1 else name,
        'group_code': parts[1] if len(parts) >= 2 else 'unknown',
        'hy_score': parts[2] if len(parts) >= 3 else None,
        'updrs_ii_5': parts[3] if len(parts) >= 4 else None,
        'updrs_iii_18': parts[4] if len(parts) >= 5 else None,
    }


def collect_transcriptions(root, task_name):
    """
    Read all Whisper JSON files for one task (ReadText or SpontaneousDialogue).
    Returns (cymo_rows, metadata_rows, issues).
    """
    cymo_rows = []
    meta_rows = []
    issues = []

    for group in ['HC', 'PD']:
        folder = os.path.join(root, task_name, group)
        if not os.path.isdir(folder):
            issues.append(f'{task_name}/{group}/ not found')
            continue

        json_files = sorted([f for f in os.listdir(folder) if f.endswith('.json')])

        for jf in json_files:
            json_path = os.path.join(folder, jf)
            wav_name = jf.replace('.json', '.wav')
            meta = parse_kcl_filename(wav_name)

            group_label = 'PD' if group == 'PD' else 'CN'

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                issues.append(f'{task_name}/{group}/{jf}: corrupt JSON — {e}')
                continue

            segments = data.get('segments', [])
            if not segments:
                full_text = data.get('text', '').strip()
                if full_text:
                    segments = [{'text': full_text, 'start': 0, 'end': 0}]
                else:
                    issues.append(f'{task_name}/{group}/{jf}: no text')
                    continue

            for seg_idx, seg in enumerate(segments):
                text = seg.get('text', '').strip()
                if not text:
                    continue

                tid = f'{group_label}_{meta["subject_id"]}_seg{seg_idx:04d}'

                cymo_rows.append({
                    'TID': tid,
                    'text': text,
                })

                meta_rows.append({
                    'TID': tid,
                    'text': text,
                    'group': group_label,
                    'subject_id': meta['subject_id'],
                    'task': task_name,
                    'segment_idx': seg_idx,
                    'start_s': seg.get('start'),
                    'end_s': seg.get('end'),
                    'hy_score': meta.get('hy_score'),
                    'updrs_ii_5': meta.get('updrs_ii_5'),
                    'updrs_iii_18': meta.get('updrs_iii_18'),
                    'source_file': wav_name,
                })

    return cymo_rows, meta_rows, issues


def main():
    parser = argparse.ArgumentParser(
        description='Convert KCL Whisper transcriptions to two CYMO CSVs (ReadText + Dialogue).'
    )
    parser.add_argument('--root', default='../KCL', help='Path to KCL dataset')
    args = parser.parse_args()

    root = args.root

    tasks = [
        ('ReadText', 'cymo_kcl_readtext.csv'),
        ('SpontaneousDialogue', 'cymo_kcl_dialogue.csv'),
    ]

    for task_name, output_file in tasks:
        print(f'\n{"=" * 60}')
        print(f'  Processing: {task_name}')
        print(f'{"=" * 60}')

        cymo_rows, meta_rows, issues = collect_transcriptions(root, task_name)

        if not cymo_rows:
            print(f'  No transcriptions found for {task_name}.')
            print(f'  Make sure Whisper JSON files exist in {root}/{task_name}/HC/ and PD/')
            if issues:
                for iss in issues:
                    print(f'    — {iss}')
            continue

        cymo_df = pd.DataFrame(cymo_rows)
        meta_df = pd.DataFrame(meta_rows)

        # Stats
        cn = meta_df[meta_df['group'] == 'CN']
        pd_g = meta_df[meta_df['group'] == 'PD']
        words = cymo_df['text'].str.split().str.len()

        print(f'  Segments   : {len(cymo_df)}')
        print(f'  CN         : {len(cn)} segments from {cn["subject_id"].nunique()} subjects')
        print(f'  PD         : {len(pd_g)} segments from {pd_g["subject_id"].nunique()} subjects')
        print(f'  Total words: {words.sum():,}')
        print(f'  Avg words  : {words.mean():.1f} per segment')

        # Preview
        print(f'\n  Preview:')
        for _, row in cymo_df.head(5).iterrows():
            t = row['text'][:65] + '...' if len(row['text']) > 65 else row['text']
            print(f'    {row["TID"]:<30s} {t}')

        # Save CYMO CSV
        cymo_df.to_csv(output_file, index=False)
        print(f'\n  Saved CYMO CSV    → {output_file} ({len(cymo_df)} rows)')

        # Save metadata
        meta_file = output_file.replace('.csv', '_metadata.csv')
        meta_df.to_csv(meta_file, index=False)
        print(f'  Saved metadata    → {meta_file} ({len(meta_df)} rows)')

        if issues:
            print(f'\n  Issues ({len(issues)}):')
            for iss in issues:
                print(f'    — {iss}')

    print(f'\n{"=" * 60}')
    print(f'  Done. Upload these to CYMO:')
    print(f'    1. cymo_kcl_readtext.csv')
    print(f'    2. cymo_kcl_dialogue.csv')
    print(f'  Then run the evaluation notebook with the CYMO results.')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
