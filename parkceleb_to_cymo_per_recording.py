"""
ParkCeleb → CYMO CSV (One Row Per Recording)
=============================================
Each video recording becomes exactly one row.
All target speaker segments concatenated into a single text entry.

Usage:
    cd ../ParkCeleb
    python parkceleb_to_cymo_per_recording.py --root .
"""

import os, sys, json, argparse
import pandas as pd
from urllib.parse import urlparse, parse_qs


def find_target_speaker(speakers_csv_path):
    """Return target speaker ID from speakers_info.csv."""
    try:
        df = pd.read_csv(speakers_csv_path)
    except Exception:
        return None
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    status_col = next((c for c in df.columns if c in ('status', 'speaker_status')), None)
    speaker_col = next((c for c in df.columns if c in ('speakers', 'speaker', 'speaker_id')), None)
    if not status_col or not speaker_col:
        return None
    
    target = df[df[status_col].astype(str).str.strip().str.lower() == 'target']
    return str(target.iloc[0][speaker_col]).strip() if len(target) > 0 else None


def process_video(vid_dir, group, subject, vid_id):
    """Extract concatenated target speaker text from one video folder."""
    # Find JSON
    json_path = os.path.join(vid_dir, f'{vid_id}.json')
    if not os.path.isfile(json_path):
        json_files = [f for f in os.listdir(vid_dir) if f.endswith('.json')]
        if not json_files:
            return None, 'no JSON'
        json_path = os.path.join(vid_dir, json_files[0])

    # Find target speaker
    speakers_csv = os.path.join(vid_dir, 'speakers_info.csv')
    target = find_target_speaker(speakers_csv) if os.path.isfile(speakers_csv) else None

    # Read JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        return None, 'corrupt JSON'

    # Collect target speaker text
    parts = []
    for seg in data.get('segments', []):
        speaker = seg.get('speaker', '')
        text = seg.get('text', '').strip()
        if text and (target is None or speaker == target):
            parts.append(text)

    if not parts:
        return None, 'no target text'

    return ' '.join(parts), None


def main():
    parser = argparse.ArgumentParser(description='ParkCeleb transcripts → CYMO CSV (one row per recording)')
    parser.add_argument('--root', default='.', help='ParkCeleb directory')
    parser.add_argument('-o', '--output', default='cymo_parkceleb_per_recording.csv')
    args = parser.parse_args()

    cymo_rows = []
    meta_rows = []
    issues = []
    
    for group in ['CN', 'PD']:
        group_dir = os.path.join(args.root, group)
        if not os.path.isdir(group_dir):
            continue
        for subj in sorted(os.listdir(group_dir)):
            subj_dir = os.path.join(group_dir, subj)
            if not os.path.isdir(subj_dir):
                continue
            for vid_id in sorted(os.listdir(subj_dir)):
                vid_dir = os.path.join(subj_dir, vid_id)
                if not os.path.isdir(vid_dir):
                    continue
                if not any(f.endswith('.json') for f in os.listdir(vid_dir)):
                    continue

                text, err = process_video(vid_dir, group, subj, vid_id)
                if err:
                    issues.append(f'{subj}/{vid_id}: {err}')
                    continue

                tid = f'{group}_{subj}_{vid_id}'
                cymo_rows.append({'TID': tid, 'text': text})
                meta_rows.append({
                    'TID': tid, 'text': text,
                    'group': group, 'subject': subj, 'video_id': vid_id,
                    'word_count': len(text.split()),
                })

    if not cymo_rows:
        print('ERROR: No transcriptions found.')
        sys.exit(1)

    cymo_df = pd.DataFrame(cymo_rows)
    meta_df = pd.DataFrame(meta_rows)

    cn = meta_df[meta_df['group'] == 'CN']
    pd_g = meta_df[meta_df['group'] == 'PD']

    print(f'Recordings : {len(cymo_df)}')
    print(f'  CN       : {len(cn)} recordings from {cn["subject"].nunique()} subjects')
    print(f'  PD       : {len(pd_g)} recordings from {pd_g["subject"].nunique()} subjects')
    print(f'  Words    : {meta_df["word_count"].sum():,} total, {meta_df["word_count"].mean():.0f} avg/recording')

    cymo_df.to_csv(args.output, index=False)
    meta_df.to_csv(args.output.replace('.csv', '_metadata.csv'), index=False)
    print(f'\nSaved: {args.output} ({len(cymo_df)} rows)')
    print(f'Saved: {args.output.replace(".csv", "_metadata.csv")}')

    if issues:
        print(f'\nIssues ({len(issues)}):')
        for iss in issues[:10]:
            print(f'  — {iss}')


if __name__ == '__main__':
    main()
