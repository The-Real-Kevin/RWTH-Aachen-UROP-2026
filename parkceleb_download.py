"""
ParkCeleb YouTube Audio Downloader — Batch Script
===================================================
Downloads all YouTube videos referenced in metadata.xlsx files
across the ParkCeleb dataset and saves them as .wav audio files.

Usage:
    cd /path/to/ParkCeleb
    python parkceleb_download.py

    Or with options:
    python parkceleb_download.py --root ./ParkCeleb --format wav --dry-run

Requirements:
    pip install yt-dlp openpyxl pandas
    brew install ffmpeg  (or apt install ffmpeg on Linux)

Directory structure expected:
    ParkCeleb/
    ├── CN/
    │   ├── cn_01/
    │   │   ├── metadata.xlsx       ← contains "link" column
    │   │   ├── [video_id_1]/       ← audio saved here
    │   │   └── [video_id_2]/
    │   ├── cn_02/
    │   └── ...
    └── PD/
        ├── pd_01/
        └── ...
"""

import os
import sys
import subprocess
import argparse
import time
import logging
from urllib.parse import urlparse, parse_qs

import pandas as pd


# ============================================================
# Configuration
# ============================================================
DEFAULT_ROOT = '../ParkCeleb'       # run from inside ParkCeleb directory
AUDIO_FORMAT = 'wav'     # wav, mp3, flac, etc.
MAX_RETRIES  = 2         # retry failed downloads
RETRY_DELAY  = 5         # seconds between retries
RATE_LIMIT   = 2         # seconds between downloads (be polite to YouTube)


# ============================================================
# Utility functions
# ============================================================
def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from a URL."""
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    parsed = urlparse(url)

    if parsed.hostname in ('youtu.be',):
        vid = parsed.path.lstrip('/')
        return vid.split('/')[0] if vid else None

    if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            qs = parse_qs(parsed.query)
            v = qs.get('v')
            return v[0] if v else None
        if parsed.path.startswith(('/embed/', '/v/')):
            parts = parsed.path.split('/')
            return parts[2] if len(parts) > 2 else None

    return None


def read_metadata_links(xlsx_path: str) -> list[dict]:
    """Read metadata.xlsx and extract YouTube links."""
    if not os.path.isfile(xlsx_path):
        return []

    df = pd.read_excel(xlsx_path)

    # Find link column (case-insensitive search)
    link_col = None
    for col in df.columns:
        if col.strip().lower() in ('link', 'links', 'url', 'youtube',
                                    'youtube_link', 'video_link'):
            link_col = col
            break
    if link_col is None:
        for col in df.columns:
            sample = df[col].dropna().astype(str)
            if sample.str.contains('youtube.com|youtu.be', case=False).any():
                link_col = col
                break
    if link_col is None:
        return []

    entries = []
    for raw_url in df[link_col].dropna():
        url = str(raw_url).strip()
        vid = extract_video_id(url)
        if vid:
            entries.append({'url': url, 'video_id': vid})
    return entries


def download_audio(url: str, output_dir: str, video_id: str,
                   audio_format: str = 'wav') -> bool:
    """
    Download a YouTube video as an audio file using yt-dlp.
    Returns True on success, False on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f'{video_id}.{audio_format}')

    # Skip if already downloaded
    if os.path.isfile(output_path) and os.path.getsize(output_path) > 1000:
        logging.info(f'  SKIP (exists): {video_id}')
        return True

    cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', audio_format,
        '--audio-quality', '0',         # best quality
        '--no-playlist',
        '--output', os.path.join(output_dir, f'{video_id}.%(ext)s'),
        '--quiet',
        '--no-warnings',
        '--socket-timeout', '30',
        '--retries', '3',
        url,
    ]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                # Verify file was created
                if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                    return True
                # Sometimes yt-dlp creates with a different extension before converting
                # Check if any file with the video_id exists
                for f in os.listdir(output_dir):
                    if f.startswith(video_id):
                        return True
                logging.warning(f'  yt-dlp succeeded but no output file found')
                return False
            else:
                stderr = result.stderr.strip()
                if 'Private video' in stderr or 'Video unavailable' in stderr:
                    logging.warning(f'  UNAVAILABLE: {video_id} — video is private or removed')
                    return False
                if 'Sign in to confirm your age' in stderr:
                    logging.warning(f'  AGE-RESTRICTED: {video_id}')
                    return False
                logging.warning(f'  Attempt {attempt}/{MAX_RETRIES} failed: {stderr[:150]}')
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        except subprocess.TimeoutExpired:
            logging.warning(f'  Attempt {attempt}/{MAX_RETRIES} timed out (300s)')
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            logging.error(f'  Unexpected error: {e}')
            return False

    return False


def discover_subjects(root: str) -> list[dict]:
    """Find all subject folders with metadata.xlsx."""
    subjects = []
    for group in ['CN', 'PD']:
        group_dir = os.path.join(root, group)
        if not os.path.isdir(group_dir):
            continue
        for subj in sorted(os.listdir(group_dir)):
            subj_dir = os.path.join(group_dir, subj)
            if not os.path.isdir(subj_dir):
                continue
            xlsx = os.path.join(subj_dir, 'metadata.xlsx')
            subjects.append({
                'group': group,
                'subject': subj,
                'dir': subj_dir,
                'xlsx': xlsx if os.path.isfile(xlsx) else None,
            })
    return subjects


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Download ParkCeleb YouTube videos as audio files.'
    )
    parser.add_argument('--root', default=DEFAULT_ROOT,
                        help='Path to ParkCeleb directory (default: current dir)')
    parser.add_argument('--format', default=AUDIO_FORMAT,
                        help=f'Audio format: wav, mp3, flac (default: {AUDIO_FORMAT})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be downloaded without actually downloading')
    parser.add_argument('--subject', default=None,
                        help='Only process a specific subject, e.g. cn_01 or pd_05')
    parser.add_argument('--group', default=None, choices=['CN', 'PD'],
                        help='Only process CN or PD group')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show verbose output')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    root = args.root
    audio_fmt = args.format

    # Check yt-dlp and ffmpeg
    if not args.dry_run:
        import shutil
        if not shutil.which('yt-dlp'):
            logging.error('yt-dlp not found. Install: pip install yt-dlp')
            sys.exit(1)
        if not shutil.which('ffmpeg'):
            logging.error('ffmpeg not found. Install: brew install ffmpeg')
            sys.exit(1)

    # Discover subjects
    subjects = discover_subjects(root)
    if not subjects:
        logging.error(f'No subject folders found under {root}/')
        logging.error(f'Expected structure: {root}/CN/cn_01/, {root}/PD/pd_01/, etc.')
        sys.exit(1)

    # Filter by group/subject if specified
    if args.group:
        subjects = [s for s in subjects if s['group'] == args.group]
    if args.subject:
        subjects = [s for s in subjects if s['subject'] == args.subject]

    logging.info(f'ParkCeleb Audio Downloader')
    logging.info(f'Root directory : {os.path.abspath(root)}')
    logging.info(f'Audio format   : {audio_fmt}')
    logging.info(f'Subjects found : {len(subjects)}')
    if args.dry_run:
        logging.info(f'MODE: DRY RUN (no downloads)')
    logging.info('')

    # Process each subject
    total_videos = 0
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    failed_list = []

    for subj in subjects:
        if subj['xlsx'] is None:
            logging.warning(f"[{subj['subject']}] No metadata.xlsx found — skipping")
            continue

        entries = read_metadata_links(subj['xlsx'])
        if not entries:
            logging.warning(f"[{subj['subject']}] No YouTube links found in metadata.xlsx")
            continue

        logging.info(f"[{subj['subject']}] {len(entries)} videos to process")

        for entry in entries:
            total_videos += 1
            vid = entry['video_id']
            url = entry['url']

            # Output directory is the video_id subfolder under the subject
            output_dir = os.path.join(subj['dir'], vid)

            if args.dry_run:
                exists = os.path.isdir(output_dir) and any(
                    f.endswith(f'.{audio_fmt}') for f in os.listdir(output_dir)
                ) if os.path.isdir(output_dir) else False
                status = 'EXISTS' if exists else 'WOULD DOWNLOAD'
                logging.info(f'  {status}: {vid}  ←  {url[:60]}')
                continue

            logging.info(f'  Downloading: {vid}')
            success = download_audio(url, output_dir, vid, audio_fmt)

            if success:
                # Check if it was a skip (already existed) vs new download
                total_downloaded += 1
            else:
                total_failed += 1
                failed_list.append({
                    'subject': subj['subject'],
                    'video_id': vid,
                    'url': url,
                })
                logging.error(f'  FAILED: {vid}')

            # Rate limiting — be polite
            time.sleep(RATE_LIMIT)

    # Summary
    logging.info('')
    logging.info('=' * 50)
    logging.info('DOWNLOAD SUMMARY')
    logging.info('=' * 50)
    logging.info(f'  Total videos     : {total_videos}')
    logging.info(f'  Downloaded / OK  : {total_downloaded}')
    logging.info(f'  Failed           : {total_failed}')

    if failed_list:
        logging.info('')
        logging.info('Failed downloads:')
        for f in failed_list:
            logging.info(f"  {f['subject']:>10s}  {f['video_id']:>15s}  {f['url'][:50]}")

        # Save failed list to CSV for re-running later
        failed_csv = os.path.join(root, 'failed_downloads.csv')
        pd.DataFrame(failed_list).to_csv(failed_csv, index=False)
        logging.info(f'\n  Failed list saved to: {failed_csv}')


if __name__ == '__main__':
    main()
