"""
ParkCeleb → CYMO CSV Converter
===============================
Iterates through every CN and PD subject folder, reads each recording's
speakers_info.csv to identify the Target speaker, then extracts every
utterance from the Target speaker in the JSON transcription and writes
it as a separate row in the final CYMO CSV.

Output: a CSV with two columns (TID, text) ready to upload to CYMO.

Usage:
    cd ../ParkCeleb
    python parkceleb_to_cymo.py --root .
    python parkceleb_to_cymo.py --root . -o my_cymo_upload.csv

Requirements:
    pip install pandas
"""

import os
import sys
import json
import argparse
import pandas as pd


def find_target_speaker(speakers_info_path):
    """
    Read speakers_info.csv and return the speaker ID marked as 'target'.

    The CSV looks like:
        video_id    speakers     status    years_from_diagnosis  ...
        4zHrnCPJTFM SPEAKER_00   non-target
                    SPEAKER_01   target    34                    ...
                    SPEAKER_02   non-target

    Returns:
        (target_speaker_id, metadata_dict) or (None, {}) if no target found.
    """
    try:
        df = pd.read_csv(speakers_info_path)
    except Exception as e:
        return None, {'error': f'cannot read CSV: {e}'}

    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Find the status column
    status_col = None
    for candidate in ['status', 'speaker_status']:
        if candidate in df.columns:
            status_col = candidate
            break

    if status_col is None:
        return None, {'error': f'no status column found, columns: {list(df.columns)}'}

    # Find the speakers column
    speaker_col = None
    for candidate in ['speakers', 'speaker', 'speaker_id']:
        if candidate in df.columns:
            speaker_col = candidate
            break

    if speaker_col is None:
        return None, {'error': f'no speakers column found, columns: {list(df.columns)}'}

    # Find the row where status == 'target' (case-insensitive)
    target_mask = df[status_col].astype(str).str.strip().str.lower() == 'target'
    target_rows = df[target_mask]

    if len(target_rows) == 0:
        return None, {'error': 'no row with status=target'}

    target_row = target_rows.iloc[0]
    target_speaker = str(target_row[speaker_col]).strip()

    # Collect metadata
    metadata = {}
    for col in ['years_from_diagnosis', 'before_after_diagnosis', 'use_as_control']:
        if col in df.columns:
            val = target_row.get(col)
            if pd.notna(val):
                metadata[col] = val

    return target_speaker, metadata


def extract_target_utterances(json_path, target_speaker):
    """
    Read a ParkCeleb JSON transcription and return a list of text
    segments spoken by the target speaker.

    Each entry: {'start': float, 'end': float, 'text': str}
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None  # signals corrupt/empty file (distinct from empty list)

    utterances = []
    for seg in data.get('segments', []):
        speaker = seg.get('speaker', '')
        text = seg.get('text', '').strip()

        if speaker == target_speaker and text:
            utterances.append({
                'start': seg.get('start'),
                'end': seg.get('end'),
                'text': text,
            })

    return utterances


def main():
    parser = argparse.ArgumentParser(
        description='Convert ParkCeleb transcriptions to CYMO CSV format.'
    )
    parser.add_argument('--root', default='.', help='Path to ParkCeleb directory')
    parser.add_argument('-o', '--output', default='cymo_parkceleb.csv',
                        help='Output CSV filename (default: cymo_parkceleb.csv)')
    args = parser.parse_args()

    root = args.root

    # Verify directory structure
    if not os.path.isdir(os.path.join(root, 'CN')) and not os.path.isdir(os.path.join(root, 'PD')):
        print(f'ERROR: Neither CN/ nor PD/ found under {os.path.abspath(root)}')
        sys.exit(1)

    # ---- Counters ----
    total_subjects = 0
    total_recordings = 0
    total_utterances = 0
    skipped_no_json = 0
    skipped_no_speakers_info = 0
    skipped_no_target = 0
    skipped_no_utterances = 0
    issues = []

    # ---- Collect all rows ----
    cymo_rows = []       # for CYMO upload (TID, text only)
    metadata_rows = []   # full version with extra columns

    for group in ['CN', 'PD']:
        group_dir = os.path.join(root, group)
        if not os.path.isdir(group_dir):
            print(f'WARNING: {group_dir} not found, skipping.')
            continue

        subjects = sorted([
            d for d in os.listdir(group_dir)
            if os.path.isdir(os.path.join(group_dir, d))
        ])

        for subj_name in subjects:
            subj_dir = os.path.join(group_dir, subj_name)
            total_subjects += 1

            # Each subfolder in the subject dir is a recording (video ID)
            recording_folders = sorted([
                d for d in os.listdir(subj_dir)
                if os.path.isdir(os.path.join(subj_dir, d))
            ])

            for vid_id in recording_folders:
                vid_dir = os.path.join(subj_dir, vid_id)

                # ---- Find the JSON transcription ----
                json_path = os.path.join(vid_dir, f'{vid_id}.json')
                if not os.path.isfile(json_path):
                    # Try any .json in the folder
                    json_files = [f for f in os.listdir(vid_dir) if f.endswith('.json')]
                    if json_files:
                        json_path = os.path.join(vid_dir, json_files[0])
                    else:
                        skipped_no_json += 1
                        continue

                # ---- Find speakers_info.csv ----
                speakers_path = os.path.join(vid_dir, 'speakers_info.csv')
                if not os.path.isfile(speakers_path):
                    skipped_no_speakers_info += 1
                    issues.append(f'{subj_name}/{vid_id}: no speakers_info.csv')
                    continue

                # ---- Identify target speaker ----
                target_speaker, meta = find_target_speaker(speakers_path)
                if target_speaker is None:
                    skipped_no_target += 1
                    issues.append(f'{subj_name}/{vid_id}: {meta.get("error", "no target")}')
                    continue

                # ---- Extract target utterances ----
                utterances = extract_target_utterances(json_path, target_speaker)
                if utterances is None:
                    skipped_no_json += 1
                    issues.append(f'{subj_name}/{vid_id}: corrupt or empty JSON file')
                    continue
                if not utterances:
                    skipped_no_utterances += 1
                    issues.append(f'{subj_name}/{vid_id}: target={target_speaker} but 0 utterances')
                    continue

                total_recordings += 1

                # ---- Create rows ----
                for i, utt in enumerate(utterances):
                    total_utterances += 1

                    # TID format: GROUP_SUBJECT_VIDEOID_SEGnnnn
                    tid = f'{group}_{subj_name}_{vid_id}_seg{i:04d}'

                    cymo_rows.append({
                        'TID': tid,
                        'text': utt['text'],
                    })

                    metadata_rows.append({
                        'TID': tid,
                        'text': utt['text'],
                        'group': group,
                        'subject': subj_name,
                        'video_id': vid_id,
                        'segment_idx': i,
                        'start_s': utt['start'],
                        'end_s': utt['end'],
                        'target_speaker': target_speaker,
                        'years_from_diagnosis': meta.get('years_from_diagnosis'),
                        'before_after': meta.get('before_after_diagnosis'),
                    })

            # Progress
            if total_subjects % 10 == 0:
                print(f'  Processed {total_subjects} subjects, {total_utterances} utterances so far...')

    # ---- Summary ----
    print()
    print('=' * 60)
    print('  CONVERSION COMPLETE')
    print('=' * 60)
    print(f'  Subjects processed            : {total_subjects}')
    print(f'  Recordings with target speech  : {total_recordings}')
    print(f'  Total target utterances        : {total_utterances}')
    print()
    print(f'  Skipped (no JSON file)         : {skipped_no_json}')
    print(f'  Skipped (no speakers_info.csv) : {skipped_no_speakers_info}')
    print(f'  Skipped (no target speaker)    : {skipped_no_target}')
    print(f'  Skipped (0 target utterances)  : {skipped_no_utterances}')

    if not cymo_rows:
        print('\n  ERROR: No utterances extracted. Nothing to save.')
        if issues:
            print(f'\n  Issues ({len(issues)}):')
            for iss in issues[:15]:
                print(f'    — {iss}')
        return

    # ---- Group breakdown ----
    meta_df = pd.DataFrame(metadata_rows)
    cn_count = len(meta_df[meta_df['group'] == 'CN'])
    pd_count = len(meta_df[meta_df['group'] == 'PD'])
    cn_subj = meta_df[meta_df['group'] == 'CN']['subject'].nunique()
    pd_subj = meta_df[meta_df['group'] == 'PD']['subject'].nunique()
    print(f'\n  CN: {cn_count} utterances from {cn_subj} subjects')
    print(f'  PD: {pd_count} utterances from {pd_subj} subjects')

    word_counts = meta_df['text'].str.split().str.len()
    print(f'\n  Avg words per utterance : {word_counts.mean():.1f}')
    print(f'  Min words               : {word_counts.min()}')
    print(f'  Max words               : {word_counts.max()}')
    print(f'  Total words             : {word_counts.sum():,}')

    # ---- Preview ----
    print(f'\n  Preview (first 8 rows):')
    print(f'  {"TID":<50s} {"text (first 70 chars)"}')
    print(f'  {"─"*50} {"─"*70}')
    for row in cymo_rows[:8]:
        text_short = row['text'][:70] + '...' if len(row['text']) > 70 else row['text']
        print(f'  {row["TID"]:<50s} {text_short}')

    # ---- Save CYMO CSV (TID + text only) ----
    cymo_df = pd.DataFrame(cymo_rows)
    cymo_df.to_csv(args.output, index=False)
    print(f'\n  Saved CYMO CSV → {args.output}')
    print(f'    {len(cymo_df)} rows, 2 columns (TID, text)')
    print(f'    Upload this file to CYMO.')

    # ---- Save metadata CSV ----
    meta_path = args.output.replace('.csv', '_metadata.csv')
    meta_df.to_csv(meta_path, index=False)
    print(f'\n  Saved metadata CSV → {meta_path}')
    print(f'    {len(meta_df)} rows, {len(meta_df.columns)} columns')
    print(f'    Use this later to merge CYMO features back with PD/CN labels.')

    # ---- Print issues ----
    if issues:
        print(f'\n  Issues encountered ({len(issues)}):')
        for iss in issues[:20]:
            print(f'    — {iss}')
        if len(issues) > 20:
            print(f'    ... and {len(issues) - 20} more')

    print()
    print('=' * 60)


if __name__ == '__main__':
    main()
