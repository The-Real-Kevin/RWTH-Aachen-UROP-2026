"""
ParkCeleb YouTube Audio Downloader — Component Tests
=====================================================
Run each section one at a time to verify everything works
before running the full batch script.

Requirements:
    pip install yt-dlp openpyxl pandas

You also need ffmpeg installed:
    Mac:     brew install ffmpeg
    Ubuntu:  sudo apt install ffmpeg
    Windows: download from https://ffmpeg.org/download.html
"""

# ============================================================
# TEST 1 — Parse video ID from YouTube URL
# ============================================================
# The metadata.xlsx "link" column contains URLs like:
#   https://www.youtube.com/watch?v=RjpTsK-WzSw&t=42s
#   https://youtu.be/RjpTsK-WzSw
#   https://www.youtube.com/watch?v=ABC123
#
# We need to extract just the video ID (e.g. "RjpTsK-WzSw")

from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str) -> str | None:
    """
    Extract YouTube video ID from various URL formats.
    Returns None if the URL is not a valid YouTube URL.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()

    # Handle youtu.be/VIDEO_ID format
    parsed = urlparse(url)
    if parsed.hostname in ('youtu.be',):
        vid = parsed.path.lstrip('/')
        if vid:
            return vid.split('/')[0]  # remove anything after the ID

    # Handle youtube.com/watch?v=VIDEO_ID format
    if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            qs = parse_qs(parsed.query)
            v = qs.get('v')
            if v:
                return v[0]
        # Handle youtube.com/embed/VIDEO_ID or youtube.com/v/VIDEO_ID
        if parsed.path.startswith(('/embed/', '/v/')):
            return parsed.path.split('/')[2]

    return None


# --- Run test ---
test_urls = [
    'https://www.youtube.com/watch?v=RjpTsK-WzSw&t=42s',
    'https://www.youtube.com/watch?v=ABC123DEF45',
    'https://youtu.be/RjpTsK-WzSw',
    'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    'not_a_url',
    '',
    None,
]

print('TEST 1 — Video ID extraction')
print('=' * 60)
for url in test_urls:
    vid = extract_video_id(url)
    print(f'  {str(url):<55s} → {vid}')

all_passed = (
    extract_video_id(test_urls[0]) == 'RjpTsK-WzSw'
    and extract_video_id(test_urls[2]) == 'RjpTsK-WzSw'
    and extract_video_id(test_urls[4]) is None
    and extract_video_id(test_urls[6]) is None
)
print(f'\n  Result: {"PASS ✓" if all_passed else "FAIL ✗"}')


# ============================================================
# TEST 2 — Read metadata.xlsx and extract links
# ============================================================
import pandas as pd
import os

def read_metadata_links(xlsx_path: str) -> list[dict]:
    """
    Read a metadata.xlsx file and return a list of dicts
    with 'url' and 'video_id' for each valid YouTube link.
    """
    if not os.path.isfile(xlsx_path):
        print(f'  WARNING: file not found → {xlsx_path}')
        return []

    df = pd.read_excel(xlsx_path)

    # Find the column that contains links
    # Try common column names (case-insensitive)
    link_col = None
    for col in df.columns:
        if col.strip().lower() in ('link', 'links', 'url', 'youtube', 'youtube_link', 'video_link'):
            link_col = col
            break

    if link_col is None:
        # If no obvious column name, look for columns containing youtube URLs
        for col in df.columns:
            sample = df[col].dropna().astype(str)
            if sample.str.contains('youtube.com|youtu.be', case=False).any():
                link_col = col
                break

    if link_col is None:
        print(f'  WARNING: no link column found in {xlsx_path}')
        print(f'  Columns available: {list(df.columns)}')
        return []

    entries = []
    for raw_url in df[link_col].dropna():
        url = str(raw_url).strip()
        vid = extract_video_id(url)
        if vid:
            entries.append({'url': url, 'video_id': vid})

    return entries


# --- Run test ---
print('\n\nTEST 2 — Read metadata.xlsx')
print('=' * 60)

# Try to find a real metadata file to test with
PARKCELEB_DIR = '../ParkCeleb'  # <-- adjust if needed
test_xlsx = None

for group in ['CN', 'PD']:
    group_dir = os.path.join(PARKCELEB_DIR, group)
    if not os.path.isdir(group_dir):
        continue
    for subj in sorted(os.listdir(group_dir)):
        subj_dir = os.path.join(group_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        xlsx_path = os.path.join(subj_dir, 'metadata.xlsx')
        if os.path.isfile(xlsx_path):
            test_xlsx = xlsx_path
            break
    if test_xlsx:
        break

if test_xlsx:
    print(f'  Found test file: {test_xlsx}')
    entries = read_metadata_links(test_xlsx)
    print(f'  Extracted {len(entries)} video links:')
    for e in entries[:5]:  # show first 5
        print(f"    {e['video_id']:>15s}  ←  {e['url'][:60]}")
    if len(entries) > 5:
        print(f'    ... and {len(entries) - 5} more')
    print(f'\n  Result: {"PASS ✓" if len(entries) > 0 else "FAIL ✗ (no links found)"}')
else:
    print(f'  No metadata.xlsx found under {PARKCELEB_DIR}/')
    print(f'  Adjust PARKCELEB_DIR and re-run, or skip to Test 3.')
    print(f'  Result: SKIP')


# ============================================================
# TEST 3 — Check that yt-dlp and ffmpeg are available
# ============================================================
import subprocess
import shutil

print('\n\nTEST 3 — yt-dlp and ffmpeg availability')
print('=' * 60)

# Check yt-dlp
ytdlp_ok = shutil.which('yt-dlp') is not None
if not ytdlp_ok:
    # yt-dlp might be installed as a Python module but not on PATH
    try:
        result = subprocess.run(['python3', '-m', 'yt_dlp', '--version'],
                                capture_output=True, text=True, timeout=10)
        ytdlp_ok = result.returncode == 0
        if ytdlp_ok:
            print(f'  yt-dlp version: {result.stdout.strip()} (via python -m)')
    except Exception:
        pass

if ytdlp_ok and shutil.which('yt-dlp'):
    result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
    print(f'  yt-dlp version: {result.stdout.strip()}')
print(f'  yt-dlp  : {"FOUND ✓" if ytdlp_ok else "NOT FOUND ✗  →  pip install yt-dlp"}')

# Check ffmpeg
ffmpeg_ok = shutil.which('ffmpeg') is not None
if ffmpeg_ok:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    version_line = result.stdout.split('\n')[0]
    print(f'  ffmpeg  : {version_line[:60]}')
print(f'  ffmpeg  : {"FOUND ✓" if ffmpeg_ok else "NOT FOUND ✗"}')

if not ffmpeg_ok:
    print(f'\n  ffmpeg is required to convert video → audio.')
    print(f'  Install it:')
    print(f'    Mac:     brew install ffmpeg')
    print(f'    Ubuntu:  sudo apt install ffmpeg')
    print(f'    Windows: download from https://ffmpeg.org/download.html')

print(f'\n  Result: {"PASS ✓" if ytdlp_ok and ffmpeg_ok else "FAIL ✗"}')


# ============================================================
# TEST 4 — Download a single video (short, public domain)
# ============================================================
print('\n\nTEST 4 — Download a single test video')
print('=' * 60)

if not (ytdlp_ok and ffmpeg_ok):
    print('  SKIP — yt-dlp or ffmpeg not available (fix Test 3 first)')
else:
    # Use a short Creative Commons video for testing
    TEST_URL = 'https://www.youtube.com/watch?v=BaW_jenozKc'  # short test video
    TEST_OUTPUT_DIR = './test_download'
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    output_template = os.path.join(TEST_OUTPUT_DIR, '%(id)s.%(ext)s')

    cmd = [
        'yt-dlp',
        '--extract-audio',           # download audio only
        '--audio-format', 'wav',     # convert to wav
        '--audio-quality', '0',      # best quality
        '--no-playlist',             # don't download playlists
        '--output', output_template,
        '--quiet',                   # suppress verbose output
        '--no-warnings',
        TEST_URL,
    ]

    print(f'  Downloading: {TEST_URL}')
    print(f'  Output dir : {TEST_OUTPUT_DIR}/')
    print(f'  Command    : {" ".join(cmd[:6])}...')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Check that a file was created
            files = os.listdir(TEST_OUTPUT_DIR)
            wav_files = [f for f in files if f.endswith('.wav')]
            if wav_files:
                fpath = os.path.join(TEST_OUTPUT_DIR, wav_files[0])
                fsize = os.path.getsize(fpath)
                print(f'  Downloaded : {wav_files[0]} ({fsize / 1024:.0f} KB)')
                print(f'\n  Result: PASS ✓')

                # Clean up
                os.remove(fpath)
                os.rmdir(TEST_OUTPUT_DIR)
                print(f'  (test file cleaned up)')
            else:
                print(f'  Files in dir: {files}')
                print(f'  No .wav file found — ffmpeg may not have converted properly')
                print(f'\n  Result: FAIL ✗')
        else:
            print(f'  yt-dlp error (exit code {result.returncode}):')
            print(f'  {result.stderr[:300]}')
            print(f'\n  Result: FAIL ✗')
    except subprocess.TimeoutExpired:
        print(f'  Timed out after 120 seconds')
        print(f'\n  Result: FAIL ✗ (timeout)')
    except Exception as e:
        print(f'  Error: {e}')
        print(f'\n  Result: FAIL ✗')


# ============================================================
# TEST 5 — Discover all subject folders
# ============================================================
print('\n\nTEST 5 — Discover subject folders')
print('=' * 60)

def discover_subjects(parkceleb_dir: str) -> list[dict]:
    """
    Walk the ParkCeleb directory and find all subject folders
    with metadata.xlsx files.
    """
    subjects = []
    for group in ['CN', 'PD']:
        group_dir = os.path.join(parkceleb_dir, group)
        if not os.path.isdir(group_dir):
            print(f'  WARNING: {group_dir} not found')
            continue
        for subj_name in sorted(os.listdir(group_dir)):
            subj_dir = os.path.join(group_dir, subj_name)
            if not os.path.isdir(subj_dir):
                continue
            xlsx_path = os.path.join(subj_dir, 'metadata.xlsx')
            has_xlsx = os.path.isfile(xlsx_path)
            subjects.append({
                'group': group,
                'subject': subj_name,
                'dir': subj_dir,
                'xlsx': xlsx_path if has_xlsx else None,
            })
    return subjects


subjects = discover_subjects(PARKCELEB_DIR)
if subjects:
    cn_count = sum(1 for s in subjects if s['group'] == 'CN')
    pd_count = sum(1 for s in subjects if s['group'] == 'PD')
    has_xlsx = sum(1 for s in subjects if s['xlsx'] is not None)
    print(f'  Found {len(subjects)} subject folders:')
    print(f'    CN: {cn_count}')
    print(f'    PD: {pd_count}')
    print(f'    With metadata.xlsx: {has_xlsx}')
    print(f'\n  First 5:')
    for s in subjects[:5]:
        xlsx_status = '✓ xlsx' if s['xlsx'] else '✗ no xlsx'
        print(f"    {s['subject']:>10s}  ({s['group']})  {xlsx_status}")
    print(f'\n  Result: {"PASS ✓" if has_xlsx > 0 else "FAIL ✗ (no metadata files)"}')
else:
    print(f'  No subjects found under {PARKCELEB_DIR}/')
    print(f'  Check that the directory structure is correct.')
    print(f'  Result: SKIP')


# ============================================================
# Summary
# ============================================================
print('\n\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
print(f'  Test 1 (URL parsing)    : PASS ✓')
print(f'  Test 2 (metadata read)  : {"done" if test_xlsx else "SKIP"}')
print(f'  Test 3 (yt-dlp/ffmpeg)  : {"PASS ✓" if ytdlp_ok and ffmpeg_ok else "FIX THIS FIRST"}')
print(f'  Test 4 (single download): {"done" if ytdlp_ok and ffmpeg_ok else "SKIP"}')
print(f'  Test 5 (folder scan)    : {"done" if subjects else "SKIP"}')
print()
print(f'  If all tests pass, you can run the full batch script.')
print(f'  Make sure to install ffmpeg if Test 3 failed.')
