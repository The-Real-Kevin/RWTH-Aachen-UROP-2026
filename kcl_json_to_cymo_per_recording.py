"""
KCL Gold-Standard JSON → CYMO CSV (One Row Per Recording)
==========================================================
Each .wav file becomes exactly ONE row in the CSV.
All segments from that recording are concatenated into a single text entry.

Usage:
    python kcl_json_to_cymo_per_recording.py --root ../KCL

Output:
    cymo_kcl_readtext_full.csv          ← upload to CYMO
    cymo_kcl_readtext_full_metadata.csv
    cymo_kcl_dialogue_full.csv          ← upload to CYMO
    cymo_kcl_dialogue_full_metadata.csv
"""

import os
import sys
import json
import argparse
import pandas as pd


def parse_filename(filename):
    """Parse ID02_pd_1_2_1.wav → subject metadata."""
    name = os.path.splitext(filename)[0].replace('_goldstandard', '')
    parts = name.split('_')
    return {
        'subject_id': parts[0] if len(parts) >= 1 else name,
        'group_code': parts[1] if len(parts) >= 2 else 'unknown',
        'hy_score': parts[2] if len(parts) >= 3 else None,
        'updrs_ii_5': parts[3] if len(parts) >= 4 else None,
        'updrs_iii_18': parts[4] if len(parts) >= 5 else None,
    }


def build_csv_for_task(root, task_name, output_name):
    """Build a CYMO CSV where each recording = one row."""
    cymo_rows = []
    meta_rows = []
    issues = []

    for group in ['HC', 'PD']:
        folder = os.path.join(root, task_name, group)
        if not os.path.isdir(folder):
            issues.append(f'{task_name}/{group}/ not found')
            continue

        json_files = sorted([f for f in os.listdir(folder) if f.endswith('_goldstandard.json')])

        for jf in json_files:
            json_path = os.path.join(folder, jf)

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                issues.append(f'{jf}: corrupt JSON — {e}')
                continue

            wav_name = jf.replace('_goldstandard.json', '.wav')
            meta = parse_filename(wav_name)
            group_label = 'PD' if group == 'PD' else 'CN'

            # Concatenate ALL segments into one text
            all_text_parts = []
            total_hes = 0
            total_unk = 0
            n_segments = 0
            duration = 0

            for seg in data.get('segments', []):
                text = seg.get('gold_text', '').strip()
                if text:
                    all_text_parts.append(text)
                    total_hes += seg.get('n_HES', 0)
                    total_unk += seg.get('n_UNK', 0)
                    n_segments += 1
                    end = seg.get('end', 0)
                    if end and end > duration:
                        duration = end

            full_text = ' '.join(all_text_parts)

            if not full_text:
                issues.append(f'{jf}: no text after concatenation')
                continue

            # TID: one per recording
            tid = f'{group_label}_{meta["subject_id"]}'

            word_count = len(full_text.split())

            cymo_rows.append({
                'TID': tid,
                'text': full_text,
            })

            meta_rows.append({
                'TID': tid,
                'text': full_text,
                'group': group_label,
                'subject_id': meta['subject_id'],
                'task': task_name,
                'source_file': wav_name,
                'hy_score': meta.get('hy_score'),
                'updrs_ii_5': meta.get('updrs_ii_5'),
                'updrs_iii_18': meta.get('updrs_iii_18'),
                'n_segments': n_segments,
                'word_count': word_count,
                'n_HES': total_hes,
                'n_UNK': total_unk,
                'duration_s': round(duration, 2),
            })

    if not cymo_rows:
        print(f'  {task_name}: no gold-standard JSONs found')
        if issues:
            for iss in issues:
                print(f'    — {iss}')
        return

    cymo_df = pd.DataFrame(cymo_rows)
    meta_df = pd.DataFrame(meta_rows)

    cn = meta_df[meta_df['group'] == 'CN']
    pd_g = meta_df[meta_df['group'] == 'PD']

    print(f'\n  {task_name}')
    print(f'  {"─" * 50}')
    print(f'  Recordings (rows)  : {len(cymo_df)}')
    print(f'    CN               : {len(cn)} subjects')
    print(f'    PD               : {len(pd_g)} subjects')
    print(f'  Total words        : {meta_df["word_count"].sum():,}')
    print(f'  Avg words/recording: {meta_df["word_count"].mean():.0f}')
    print(f'  Total HES markers  : {meta_df["n_HES"].sum()}')
    print(f'  Total <UNK>        : {meta_df["n_UNK"].sum()}')

    print(f'\n  Preview:')
    print(f'  {"TID":<20s} {"Words":>6s} {"HES":>5s} {"Text (first 60 chars)"}')
    print(f'  {"─"*20} {"─"*6} {"─"*5} {"─"*60}')
    for _, row in meta_df.head(6).iterrows():
        t = row['text'][:60] + '...' if len(row['text']) > 60 else row['text']
        print(f'  {row["TID"]:<20s} {row["word_count"]:>6d} {row["n_HES"]:>5d} {t}')

    # Save
    cymo_df.to_csv(output_name, index=False)
    print(f'\n  Saved CYMO CSV → {output_name} ({len(cymo_df)} rows)')

    meta_path = output_name.replace('.csv', '_metadata.csv')
    meta_df.to_csv(meta_path, index=False)
    print(f'  Saved metadata → {meta_path}')

    if issues:
        print(f'\n  Issues ({len(issues)}):')
        for iss in issues:
            print(f'    — {iss}')


def main():
    parser = argparse.ArgumentParser(
        description='Convert KCL gold-standard JSONs to CYMO CSV (one row per recording).'
    )
    parser.add_argument('--root', default='../KCL', help='KCL dataset path')
    args = parser.parse_args()

    print('KCL → CYMO CSV (one row per recording)')
    print('=' * 55)

    build_csv_for_task(args.root, 'ReadText', 'cymo_kcl_readtext_full.csv')
    build_csv_for_task(args.root, 'SpontaneousDialogue', 'cymo_kcl_dialogue_full.csv')

    print(f'\n{"=" * 55}')
    print(f'Done. Upload these to CYMO:')
    print(f'  1. cymo_kcl_readtext_full.csv')
    print(f'  2. cymo_kcl_dialogue_full.csv')
    print(f'\nEach row = one complete recording from one subject.')


if __name__ == '__main__':
    main()
