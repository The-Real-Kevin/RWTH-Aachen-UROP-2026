"""
KCL Gold-Standard Whisper Transcription
========================================
Transcribes KCL audio files with Whisper, then post-processes to match
the gold-standard transcription guidelines:
  - Hesitations (uh, um, er, etc.) → HES
  - No punctuation
  - Numbers spelled out as words
  - Low-confidence words → <UNK>
  - Repetitions and false starts preserved
  - Contractions kept as spoken

Usage:
    python kcl_whisper_goldstandard.py --root ../KCL --model small
    python kcl_whisper_goldstandard.py --root ../KCL --model medium --force

Requirements:
    pip install openai-whisper num2words pandas
"""

import os
import sys
import json
import re
import argparse
import time

import pandas as pd


# ============================================================
# Post-processing: gold standard compliance
# ============================================================

# Hesitation words to replace with HES
# These are non-lexical filled pauses — NOT discourse markers like "like" or "you know"
HESITATION_WORDS = {
    'uh', 'uhh', 'uhhh',
    'um', 'umm', 'ummm',
    'uhm', 'uhmm',
    'er', 'err', 'errr',
    'erm', 'ermm',
    'eh', 'ehh',
    'ehm', 'ehmm',
    'em', 'emm',
    'hm', 'hmm', 'hmmm',
    'ah', 'ahh', 'ahhh',
    'oh',  # only when used as a hesitation, not an exclamation — Whisper context will help
    'mm', 'mmm', 'mhm',
    'ugh',
}

# Punctuation to strip (keep apostrophes and hyphens inside words)
PUNCTUATION_RE = re.compile(r'[\.,:;!\?\"\"\"\(\)\[\]\{\}…—–\-]{1,}')

# Number words (basic mapping for common numbers)
try:
    from num2words import num2words as _num2words
    HAS_NUM2WORDS = True
except ImportError:
    HAS_NUM2WORDS = False
    print('NOTE: num2words not installed. Install with: pip install num2words')
    print('      Numbers will be kept as digits. For best results, install it.\n')


def number_to_words(text):
    """Convert digits in text to spelled-out words."""
    if not HAS_NUM2WORDS:
        return text

    def replace_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            # Years (1900-2099) get special treatment
            if 1900 <= num <= 2099:
                return _num2words(num, to='year')
            return _num2words(num)
        except (ValueError, OverflowError):
            try:
                num = float(num_str)
                return _num2words(num)
            except (ValueError, OverflowError):
                return num_str

    # Match standalone numbers (not part of IDs like SPEAKER_00)
    return re.sub(r'(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])', replace_number, text)


def remove_punctuation(text):
    """Remove all punctuation except apostrophes inside contractions and hyphens in compounds."""
    # First, protect apostrophes in contractions (don't, it's, we'll)
    # and hyphens in compounds (well-known, up-to-date)
    # by temporarily replacing them
    text = re.sub(r"(\w)'(\w)", r'\1§APOS§\2', text)  # protect contractions
    text = re.sub(r"(\w)-(\w)", r'\1§HYP§\2', text)    # protect compound hyphens

    # Remove all remaining punctuation
    text = PUNCTUATION_RE.sub(' ', text)

    # Also remove any stray punctuation characters
    text = re.sub(r'[^\w\s§]', ' ', text)

    # Restore protected characters
    text = text.replace('§APOS§', "'")
    text = text.replace('§HYP§', "-")

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def replace_hesitations(text):
    """Replace hesitation words with HES marker."""
    words = text.split()
    result = []
    for word in words:
        word_clean = word.lower().strip("'\".,!?")
        if word_clean in HESITATION_WORDS:
            result.append('HES')
        else:
            result.append(word)
    return ' '.join(result)


def apply_gold_standard(text, word_data=None, confidence_threshold=0.25):
    """
    Apply all gold-standard transformations to a transcript.

    Args:
        text: raw Whisper transcript
        word_data: list of word dicts with 'word' and confidence scores (optional)
        confidence_threshold: words below this confidence → <UNK>
    """
    # Step 1: Handle low-confidence words (if word-level data available)
    if word_data:
        words = []
        for w in word_data:
            if isinstance(w, dict):
                word = w.get('word', '').strip()
                score = w.get('score', w.get('probability', 1.0))
                if score is not None and score < confidence_threshold:
                    words.append('<UNK>')
                else:
                    words.append(word)
            else:
                words.append(str(w))
        text = ' '.join(words)

    # Step 2: Remove punctuation
    text = remove_punctuation(text)

    # Step 3: Convert numbers to words
    text = number_to_words(text)

    # Step 4: Replace hesitations with HES
    text = replace_hesitations(text)

    # Step 5: Lowercase everything except proper nouns
    # (We lowercase everything — Whisper's casing isn't reliable enough
    #  to distinguish proper nouns. CYMO handles casing internally.)
    text = text.lower()

    # But uppercase HES and UNK markers back
    text = text.replace('hes', 'HES')
    text = text.replace('<unk>', '<UNK>')

    # Step 6: Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# Whisper transcription with disfluency preservation
# ============================================================

def transcribe_with_disfluencies(model, audio_path, language='en'):
    """
    Transcribe audio using Whisper with settings optimised
    to preserve hesitations and disfluencies.
    """
    # The initial_prompt primes Whisper to expect and transcribe hesitations
    # rather than filtering them out
    initial_prompt = (
        "Um, so, uh, I was, I was going to say that, er, "
        "the, the result was, uh, hmm, actually quite interesting. "
        "Uh, yeah, so, um..."
    )

    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        condition_on_previous_text=False,   # don't let model "clean up" based on prior text
        no_speech_threshold=0.4,            # lower = more sensitive to faint speech
        compression_ratio_threshold=2.8,    # slightly more tolerant of unusual speech
        verbose=False,
    )

    return result


# ============================================================
# Main pipeline
# ============================================================

def process_all(root, model_name='small', language='en', force=False):
    """Transcribe all KCL files with gold-standard post-processing."""
    try:
        import whisper
    except ImportError:
        print('ERROR: pip install openai-whisper')
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

    print(f'Found {len(wav_files)} .wav files')
    print(f'Loading Whisper model: {model_name}...')
    model = whisper.load_model(model_name)
    print(f'Model loaded.\n')

    transcribed = 0
    skipped = 0
    errors = []

    for i, info in enumerate(wav_files):
        fpath = info['path']
        fname = info['filename']

        # Output path: same name but _goldstandard.json
        json_path = os.path.splitext(fpath)[0] + '_goldstandard.json'

        if os.path.isfile(json_path) and not force:
            skipped += 1
            continue

        print(f'  [{i+1}/{len(wav_files)}] {fname}', end='', flush=True)
        t0 = time.time()

        try:
            # Transcribe with disfluency-preserving settings
            result = transcribe_with_disfluencies(model, fpath, language)

            # Build output with both raw and gold-standard versions
            output = {
                'file': fname,
                'task': info['task'],
                'group': info['group'],
                'raw_text': result.get('text', ''),
                'segments': [],
            }

            gold_full_parts = []

            for seg in result.get('segments', []):
                raw_text = seg.get('text', '').strip()

                # Get word-level data for confidence-based <UNK>
                word_data = seg.get('words', [])

                # Apply gold standard transformations
                gold_text = apply_gold_standard(raw_text, word_data)

                seg_data = {
                    'id': seg.get('id'),
                    'start': seg.get('start'),
                    'end': seg.get('end'),
                    'raw_text': raw_text,
                    'gold_text': gold_text,
                }

                # Count HES and UNK in this segment
                seg_data['n_HES'] = gold_text.split().count('HES')
                seg_data['n_UNK'] = gold_text.split().count('<UNK>')

                output['segments'].append(seg_data)
                if gold_text:
                    gold_full_parts.append(gold_text)

            output['gold_text'] = ' '.join(gold_full_parts)

            # Summary stats
            all_words = output['gold_text'].split()
            output['stats'] = {
                'total_words': len(all_words),
                'n_HES': all_words.count('HES'),
                'n_UNK': all_words.count('<UNK>'),
                'n_segments': len(output['segments']),
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            elapsed = time.time() - t0
            stats = output['stats']
            print(f'  — {elapsed:.1f}s, {stats["total_words"]} words, '
                  f'{stats["n_HES"]} HES, {stats["n_UNK"]} UNK')
            transcribed += 1

        except Exception as e:
            print(f'  — ERROR: {e}')
            errors.append({'file': fname, 'error': str(e)})

    print(f'\nDone: {transcribed} transcribed, {skipped} skipped, {len(errors)} errors')


# ============================================================
# Convert gold-standard JSONs to split CYMO CSVs
# ============================================================

def build_cymo_csvs(root):
    """Build two CYMO CSVs from gold-standard JSONs (ReadText + Dialogue)."""

    def parse_filename(filename):
        name = os.path.splitext(filename)[0].replace('_goldstandard', '')
        parts = name.split('_')
        return {
            'subject_id': parts[0] if len(parts) >= 1 else name,
            'group_code': parts[1] if len(parts) >= 2 else 'unknown',
            'hy_score': parts[2] if len(parts) >= 3 else None,
            'updrs_ii_5': parts[3] if len(parts) >= 4 else None,
            'updrs_iii_18': parts[4] if len(parts) >= 5 else None,
        }

    for task, out_name in [('ReadText', 'cymo_kcl_readtext_gold.csv'),
                            ('SpontaneousDialogue', 'cymo_kcl_dialogue_gold.csv')]:
        cymo_rows = []
        meta_rows = []

        for group in ['HC', 'PD']:
            folder = os.path.join(root, task, group)
            if not os.path.isdir(folder):
                continue

            for f in sorted(os.listdir(folder)):
                if not f.endswith('_goldstandard.json'):
                    continue

                json_path = os.path.join(folder, f)
                try:
                    with open(json_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)
                except (json.JSONDecodeError, ValueError):
                    continue

                wav_name = f.replace('_goldstandard.json', '.wav')
                meta = parse_filename(wav_name)
                group_label = 'PD' if group == 'PD' else 'CN'

                for seg in data.get('segments', []):
                    text = seg.get('gold_text', '').strip()
                    if not text:
                        continue

                    tid = f'{group_label}_{meta["subject_id"]}_seg{seg.get("id", 0):04d}'

                    cymo_rows.append({'TID': tid, 'text': text})
                    meta_rows.append({
                        'TID': tid, 'text': text,
                        'group': group_label,
                        'subject_id': meta['subject_id'],
                        'task': task,
                        'segment_idx': seg.get('id', 0),
                        'start_s': seg.get('start'),
                        'end_s': seg.get('end'),
                        'hy_score': meta.get('hy_score'),
                        'updrs_ii_5': meta.get('updrs_ii_5'),
                        'updrs_iii_18': meta.get('updrs_iii_18'),
                        'n_HES': seg.get('n_HES', 0),
                        'n_UNK': seg.get('n_UNK', 0),
                        'source_file': wav_name,
                    })

        if not cymo_rows:
            print(f'  {task}: no gold-standard JSONs found')
            continue

        cymo_df = pd.DataFrame(cymo_rows)
        meta_df = pd.DataFrame(meta_rows)

        # Stats
        cn = meta_df[meta_df['group'] == 'CN']
        pd_g = meta_df[meta_df['group'] == 'PD']
        total_hes = meta_df['n_HES'].sum()
        total_unk = meta_df['n_UNK'].sum()

        print(f'\n  {task}:')
        print(f'    Segments: {len(cymo_df)}  |  CN: {cn["subject_id"].nunique()} subj  |  PD: {pd_g["subject_id"].nunique()} subj')
        print(f'    Total HES markers: {total_hes}  |  Total <UNK>: {total_unk}')

        # Preview
        for _, row in cymo_df.head(3).iterrows():
            t = row['text'][:75] + '...' if len(row['text']) > 75 else row['text']
            print(f'    {row["TID"]:<30s} {t}')

        cymo_df.to_csv(out_name, index=False)
        meta_df.to_csv(out_name.replace('.csv', '_metadata.csv'), index=False)
        print(f'    Saved: {out_name}')


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe KCL audio with gold-standard compliance.'
    )
    parser.add_argument('--root', default='../KCL', help='KCL dataset path')
    parser.add_argument('--model', default='small',
                        choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--step', default='all', choices=['transcribe', 'cymo', 'all'])
    parser.add_argument('--force', action='store_true',
                        help='Re-transcribe even if gold-standard JSON already exists')
    args = parser.parse_args()

    if args.step in ('transcribe', 'all'):
        print('STEP 1: Whisper Transcription (gold-standard mode)')
        print('=' * 55)
        process_all(args.root, model_name=args.model, force=args.force)

    if args.step in ('cymo', 'all'):
        print('\nSTEP 2: Build CYMO CSVs')
        print('=' * 55)
        build_cymo_csvs(args.root)

    print('\nDone. Upload the _gold.csv files to CYMO.')
