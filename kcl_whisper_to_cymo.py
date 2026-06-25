"""
KCL Dataset: Whisper Transcription → CYMO CSV
==============================================
Step 1: Transcribe all KCL .wav files using OpenAI Whisper
Step 2: Convert transcriptions to CYMO-compatible CSV

Usage:
    # Install whisper first:
    pip install openai-whisper

    # Transcribe all files (uses 'small' model by default)
    python kcl_whisper_to_cymo.py --root ../KCL --step transcribe

    # Convert transcriptions to CYMO CSV
    python kcl_whisper_to_cymo.py --root ../KCL --step cymo

    # Do both in one go
    python kcl_whisper_to_cymo.py --root ../KCL --step all

    # Use a different Whisper model (tiny/base/small/medium/large)
    python kcl_whisper_to_cymo.py --root ../KCL --step all --model medium

Requirements:
    pip install openai-whisper pandas
    ffmpeg must be installed (brew install ffmpeg)

KCL directory structure expected:
    KCL/
    ├── ReadText/
    │   ├── HC/
    │   │   ├── ID00_hc_0_0_0.wav
    │   │   └── ...
    │   └── PD/
    │       ├── ID02_pd_1_2_1.wav
    │       └── ...
    └── SpontaneousDialogue/
        ├── HC/
        └── PD/
"""

import os
import sys
import json
import glob
import re
import argparse
import time

import pandas as pd


# ============================================================
# Step 1: Transcription
# ============================================================

def parse_kcl_filename(filename):
    """
    Parse KCL filename like 'ID02_pd_1_2_1.wav' into components.
    Returns dict with: subject_id, group (hc/pd), hy_score, updrs_ii_5, updrs_iii_18
    """
    name = os.path.splitext(filename)[0]  # remove .wav
    parts = name.split('_')

    if len(parts) >= 5:
        return {
            'subject_id': parts[0],        # e.g. ID02
            'group_code': parts[1],        # hc or pd
            'hy_score': parts[2],          # H&Y scale
            'updrs_ii_5': parts[3],        # UPDRS II part 5
            'updrs_iii_18': parts[4],      # UPDRS III part 18
        }
    else:
        return {
            'subject_id': parts[0] if parts else name,
            'group_code': parts[1] if len(parts) > 1 else 'unknown',
            'hy_score': None,
            'updrs_ii_5': None,
            'updrs_iii_18': None,
        }


def transcribe_all(root, model_name='small', language='en'):
    """
    Transcribe all .wav files in the KCL dataset using Whisper.
    Saves JSON transcription next to each .wav file.
    """
    try:
        import whisper
    except ImportError:
        print('ERROR: openai-whisper not installed.')
        print('  Run: pip install openai-whisper')
        sys.exit(1)

    # Find all wav files
    wav_files = []
    for task in ['ReadText', 'SpontaneousDialogue']:
        for group in ['HC', 'PD']:
            folder = os.path.join(root, task, group)
            if not os.path.isdir(folder):
                continue
            for f in sorted(os.listdir(folder)):
                if f.endswith('.wav'):
                    wav_files.append({
                        'path': os.path.join(folder, f),
                        'filename': f,
                        'task': task,
                        'group': group,
                    })

    print(f'Found {len(wav_files)} .wav files to transcribe')
    print(f'Loading Whisper model: {model_name}')
    print(f'  (first run will download the model — this takes a few minutes)\n')

    model = whisper.load_model(model_name)
    print(f'Model loaded.\n')

    transcribed = 0
    skipped = 0
    errors = []

    for i, info in enumerate(wav_files):
        fpath = info['path']
        fname = info['filename']

        # Output JSON path (same directory, same name, .json extension)
        json_path = os.path.splitext(fpath)[0] + '.json'

        # Skip if already transcribed
        if os.path.isfile(json_path):
            skipped += 1
            if (i + 1) % 20 == 0:
                print(f'  [{i+1}/{len(wav_files)}] Skipping (already done): {fname}')
            continue

        print(f'  [{i+1}/{len(wav_files)}] Transcribing: {fname}', end='', flush=True)
        start_time = time.time()

        try:
            result = model.transcribe(
                fpath,
                language=language,
                word_timestamps=True,     # get word-level timing
                verbose=False,
            )

            # Save full result as JSON
            # Convert to serialisable format
            output = {
                'file': fname,
                'task': info['task'],
                'group': info['group'],
                'language': result.get('language', language),
                'text': result.get('text', ''),
                'segments': [],
            }

            for seg in result.get('segments', []):
                seg_data = {
                    'id': seg.get('id'),
                    'start': seg.get('start'),
                    'end': seg.get('end'),
                    'text': seg.get('text', '').strip(),
                }
                # Include word-level data if available
                if 'words' in seg:
                    seg_data['words'] = [
                        {
                            'word': w.get('word', w) if isinstance(w, dict) else str(w),
                            'start': w.get('start') if isinstance(w, dict) else None,
                            'end': w.get('end') if isinstance(w, dict) else None,
                        }
                        for w in seg['words']
                    ]
                output['segments'].append(seg_data)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - start_time
            n_words = len(result.get('text', '').split())
            print(f'  — {elapsed:.1f}s, {n_words} words, {len(output["segments"])} segments')
            transcribed += 1

        except Exception as e:
            print(f'  — ERROR: {e}')
            errors.append({'file': fname, 'error': str(e)})

    # Summary
    print(f'\nTranscription complete:')
    print(f'  Transcribed : {transcribed}')
    print(f'  Skipped     : {skipped} (already had .json)')
    print(f'  Errors      : {len(errors)}')

    if errors:
        print(f'\n  Failed files:')
        for e in errors:
            print(f'    {e["file"]}: {e["error"]}')

    return transcribed, skipped, errors


# ============================================================
# Step 2: Convert to CYMO CSV
# ============================================================

def build_cymo_csv(root, output_path='cymo_kcl.csv'):
    """
    Read all Whisper JSON transcriptions and build a CYMO-compatible CSV.
    Each segment becomes one row with a unique TID.
    """
    rows = []
    metadata_rows = []
    issues = []

    for task in ['ReadText', 'SpontaneousDialogue']:
        for group in ['HC', 'PD']:
            folder = os.path.join(root, task, group)
            if not os.path.isdir(folder):
                continue

            for f in sorted(os.listdir(folder)):
                if not f.endswith('.json'):
                    continue

                json_path = os.path.join(folder, f)
                wav_name = f.replace('.json', '.wav')

                # Parse filename for metadata
                meta = parse_kcl_filename(wav_name)
                subject_id = meta['subject_id']
                group_label = 'PD' if group == 'PD' else 'CN'

                # Read transcription
                try:
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                except (json.JSONDecodeError, ValueError):
                    issues.append(f'{task}/{group}/{f}: corrupt JSON')
                    continue

                segments = data.get('segments', [])
                if not segments:
                    # Use full text as single segment if no segments
                    full_text = data.get('text', '').strip()
                    if full_text:
                        segments = [{'text': full_text, 'start': 0, 'end': 0}]
                    else:
                        issues.append(f'{task}/{group}/{f}: no text found')
                        continue

                for seg_idx, seg in enumerate(segments):
                    text = seg.get('text', '').strip()
                    if not text:
                        continue

                    # TID format: GROUP_SUBJECTID_TASK_SEGnnnn
                    task_short = 'RT' if task == 'ReadText' else 'SD'
                    tid = f'{group_label}_{subject_id}_{task_short}_seg{seg_idx:04d}'

                    rows.append({
                        'TID': tid,
                        'text': text,
                    })

                    metadata_rows.append({
                        'TID': tid,
                        'text': text,
                        'group': group_label,
                        'subject_id': subject_id,
                        'task': task,
                        'task_short': task_short,
                        'segment_idx': seg_idx,
                        'start_s': seg.get('start'),
                        'end_s': seg.get('end'),
                        'hy_score': meta.get('hy_score'),
                        'updrs_ii_5': meta.get('updrs_ii_5'),
                        'updrs_iii_18': meta.get('updrs_iii_18'),
                        'source_file': wav_name,
                    })

    if not rows:
        print('ERROR: No transcriptions found. Run --step transcribe first.')
        if issues:
            for iss in issues:
                print(f'  {iss}')
        return None

    cymo_df = pd.DataFrame(rows)
    meta_df = pd.DataFrame(metadata_rows)

    # Summary
    print(f'CYMO CSV Summary')
    print(f'================')
    print(f'  Total segments : {len(cymo_df)}')

    cn_count = meta_df[meta_df['group'] == 'CN']
    pd_count = meta_df[meta_df['group'] == 'PD']
    print(f'  CN segments    : {len(cn_count)} from {cn_count["subject_id"].nunique()} subjects')
    print(f'  PD segments    : {len(pd_count)} from {pd_count["subject_id"].nunique()} subjects')

    rt_count = meta_df[meta_df['task'] == 'ReadText']
    sd_count = meta_df[meta_df['task'] == 'SpontaneousDialogue']
    print(f'  ReadText       : {len(rt_count)} segments')
    print(f'  Dialogue       : {len(sd_count)} segments')

    word_counts = cymo_df['text'].str.split().str.len()
    print(f'  Total words    : {word_counts.sum():,}')
    print(f'  Avg words/seg  : {word_counts.mean():.1f}')

    # Preview
    print(f'\n  Preview:')
    print(f'  {"TID":<35s} {"text (first 60 chars)"}')
    print(f'  {"─"*35} {"─"*60}')
    for _, row in cymo_df.head(8).iterrows():
        t = row['text'][:60] + '...' if len(row['text']) > 60 else row['text']
        print(f'  {row["TID"]:<35s} {t}')

    # Save CYMO CSV (TID + text only)
    cymo_df.to_csv(output_path, index=False)
    print(f'\n  Saved CYMO CSV → {output_path}')
    print(f'    {len(cymo_df)} rows, 2 columns (TID, text)')
    print(f'    Upload this to CYMO.')

    # Save metadata CSV
    meta_path = output_path.replace('.csv', '_metadata.csv')
    meta_df.to_csv(meta_path, index=False)
    print(f'\n  Saved metadata → {meta_path}')
    print(f'    {len(meta_df)} rows, {len(meta_df.columns)} columns')
    print(f'    Use this to merge CYMO features back with PD/CN labels.')

    if issues:
        print(f'\n  Issues ({len(issues)}):')
        for iss in issues[:10]:
            print(f'    — {iss}')

    return cymo_df


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe KCL audio with Whisper and convert to CYMO CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline
    python kcl_whisper_to_cymo.py --root ../KCL --step all

    # Just transcribe
    python kcl_whisper_to_cymo.py --root ../KCL --step transcribe --model small

    # Just build CYMO CSV (after transcription is done)
    python kcl_whisper_to_cymo.py --root ../KCL --step cymo
        """
    )
    parser.add_argument('--root', default='../KCL',
                        help='Path to KCL dataset directory')
    parser.add_argument('--step', default='all', choices=['transcribe', 'cymo', 'all'],
                        help='Which step to run')
    parser.add_argument('--model', default='small',
                        choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Whisper model size (default: small)')
    parser.add_argument('-o', '--output', default='cymo_kcl.csv',
                        help='Output CYMO CSV path')
    args = parser.parse_args()

    root = args.root

    # Verify directory
    found_task = False
    for task in ['ReadText', 'SpontaneousDialogue']:
        if os.path.isdir(os.path.join(root, task)):
            found_task = True
    if not found_task:
        print(f'ERROR: Neither ReadText/ nor SpontaneousDialogue/ found under {root}')
        print(f'  Expected: {root}/ReadText/HC/, {root}/ReadText/PD/, etc.')
        sys.exit(1)

    if args.step in ('transcribe', 'all'):
        print('STEP 1: Whisper Transcription')
        print('=' * 50)
        print(f'  Model: {args.model}')
        print(f'  Note: first run downloads the model (~1-2 GB for "small")')
        print(f'  On CPU this will take ~1-5 minutes per file depending on length.')
        print(f'  Already-transcribed files (with .json) will be skipped.\n')
        transcribe_all(root, model_name=args.model)
        print()

    if args.step in ('cymo', 'all'):
        print('STEP 2: Build CYMO CSV')
        print('=' * 50)
        build_cymo_csv(root, output_path=args.output)
