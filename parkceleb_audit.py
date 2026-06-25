"""
ParkCeleb Download Audit
========================
Scans every subject folder, reads metadata.xlsx, cross-references
against actually downloaded audio files, and produces a comprehensive
report of what's missing and why.

Usage:
    python parkceleb_audit.py
    python parkceleb_audit.py --root /path/to/ParkCeleb
    python parkceleb_audit.py --root ./ParkCeleb --csv   # also save CSV reports

Requirements:
    pip install pandas openpyxl
"""

import os
import sys
import argparse
import numpy as np
from urllib.parse import urlparse, parse_qs
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend so it works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ============================================================
# Helpers
# ============================================================

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.webm', '.aac')


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    parsed = urlparse(url)

    if parsed.hostname in ('youtu.be',):
        vid = parsed.path.lstrip('/')
        return vid.split('/')[0] if vid else None

    if parsed.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            v = parse_qs(parsed.query).get('v')
            return v[0] if v else None
        if parsed.path.startswith(('/embed/', '/v/', '/shorts/')):
            parts = parsed.path.split('/')
            return parts[2] if len(parts) > 2 else None

    return None


def find_xlsx(subj_dir):
    """Find the metadata xlsx in a subject folder."""
    # Try metadata.xlsx first
    exact = os.path.join(subj_dir, 'metadata.xlsx')
    if os.path.isfile(exact):
        return exact
    # Fall back to any xlsx directly in the folder
    for f in sorted(os.listdir(subj_dir)):
        if f.endswith('.xlsx') and os.path.isfile(os.path.join(subj_dir, f)):
            return os.path.join(subj_dir, f)
    return None


def find_link_column(df):
    """Find the column containing YouTube links."""
    # Exact name match
    for col in df.columns:
        if col.strip().lower() in ('link', 'links', 'url', 'youtube',
                                    'youtube_link', 'video_link', 'video_url'):
            return col
    # Content match
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if len(sample) > 0 and sample.str.contains(r'youtube\.com|youtu\.be', case=False, regex=True).any():
            return col
    return None


def check_audio_exists(subj_dir, video_id):
    """
    Check whether audio for a given video_id exists.
    Returns (found: bool, details: str).
    """
    vid_dir = os.path.join(subj_dir, video_id)

    if not os.path.isdir(vid_dir):
        return False, 'subfolder missing'

    contents = os.listdir(vid_dir)
    if not contents:
        return False, 'subfolder empty'

    audio_files = [f for f in contents if f.lower().endswith(AUDIO_EXTENSIONS)]
    if not audio_files:
        other = ', '.join(contents[:5])
        return False, f'no audio files (found: {other})'

    # Check file size — files < 1KB are likely corrupt
    for af in audio_files:
        fpath = os.path.join(vid_dir, af)
        size_kb = os.path.getsize(fpath) / 1024
        if size_kb < 1:
            return False, f'audio file too small ({size_kb:.1f} KB)'

    return True, f'{len(audio_files)} audio file(s)'


def classify_failure(video_id, subj_dir):
    """Attempt to classify why a download might have failed."""
    vid_dir = os.path.join(subj_dir, video_id)

    if not os.path.isdir(vid_dir):
        return 'never_attempted'

    contents = os.listdir(vid_dir)
    if not contents:
        return 'empty_folder'

    # Check for partial downloads
    partials = [f for f in contents if f.endswith(('.part', '.ytdl', '.temp'))]
    if partials:
        return 'partial_download'

    # Check for non-audio files (maybe video was downloaded but not converted)
    video_files = [f for f in contents if f.endswith(('.mp4', '.mkv', '.webm'))]
    if video_files:
        return 'video_not_converted'

    # Has files but no audio
    audio = [f for f in contents if f.lower().endswith(AUDIO_EXTENSIONS)]
    if not audio:
        return 'no_audio_in_folder'

    # Audio exists but too small
    for af in audio:
        if os.path.getsize(os.path.join(vid_dir, af)) < 1024:
            return 'corrupt_file'

    return 'unknown'


# ============================================================
# Main audit
# ============================================================

def run_audit(root, save_csv=False, plot=False):
    print('=' * 70)
    print('  ParkCeleb Download Audit')
    print(f'  Root: {os.path.abspath(root)}')
    print(f'  Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)
    print()

    all_entries = []       # every video from every metadata.xlsx
    subject_summary = []   # per-subject stats
    issues = []            # problems encountered during scanning

    for group in ['CN', 'PD']:
        group_dir = os.path.join(root, group)
        if not os.path.isdir(group_dir):
            issues.append({
                'subject': f'{group}/*',
                'issue': f'Group directory not found: {group_dir}',
            })
            continue

        for subj_name in sorted(os.listdir(group_dir)):
            subj_dir = os.path.join(group_dir, subj_name)
            if not os.path.isdir(subj_dir):
                continue

            # --- Find xlsx ---
            xlsx_path = find_xlsx(subj_dir)
            if xlsx_path is None:
                issues.append({
                    'subject': subj_name,
                    'issue': 'No .xlsx file found in subject folder',
                })
                subject_summary.append({
                    'group': group, 'subject': subj_name,
                    'xlsx_file': None, 'link_column': None,
                    'total_expected': 0, 'downloaded': 0, 'missing': 0,
                    'pct_complete': 0, 'status': 'NO_XLSX',
                })
                continue

            # --- Read xlsx ---
            try:
                meta_df = pd.read_excel(xlsx_path)
            except Exception as e:
                issues.append({
                    'subject': subj_name,
                    'issue': f'Failed to read {os.path.basename(xlsx_path)}: {e}',
                })
                continue

            # --- Find link column ---
            link_col = find_link_column(meta_df)
            if link_col is None:
                issues.append({
                    'subject': subj_name,
                    'issue': f'No link column found. Columns: {list(meta_df.columns)}',
                })
                subject_summary.append({
                    'group': group, 'subject': subj_name,
                    'xlsx_file': os.path.basename(xlsx_path),
                    'link_column': None,
                    'total_expected': 0, 'downloaded': 0, 'missing': 0,
                    'pct_complete': 0, 'status': 'NO_LINK_COL',
                })
                continue

            # --- Check each video ---
            n_expected = 0
            n_downloaded = 0
            n_missing = 0

            for row_idx, url in enumerate(meta_df[link_col].dropna()):
                url = str(url).strip()
                vid = extract_video_id(url)

                if vid is None:
                    issues.append({
                        'subject': subj_name,
                        'issue': f'Could not parse video ID from: {url[:80]}',
                    })
                    continue

                n_expected += 1
                found, detail = check_audio_exists(subj_dir, vid)
                failure_type = None

                if found:
                    n_downloaded += 1
                    status = 'OK'
                else:
                    n_missing += 1
                    failure_type = classify_failure(vid, subj_dir)
                    status = 'MISSING'

                all_entries.append({
                    'group': group,
                    'subject': subj_name,
                    'video_id': vid,
                    'url': url,
                    'status': status,
                    'detail': detail,
                    'failure_type': failure_type,
                })

            pct = (n_downloaded / n_expected * 100) if n_expected > 0 else 0
            if n_missing == 0:
                subj_status = 'COMPLETE'
            elif n_downloaded == 0:
                subj_status = 'EMPTY'
            else:
                subj_status = 'PARTIAL'

            subject_summary.append({
                'group': group,
                'subject': subj_name,
                'xlsx_file': os.path.basename(xlsx_path),
                'link_column': link_col,
                'total_expected': n_expected,
                'downloaded': n_downloaded,
                'missing': n_missing,
                'pct_complete': round(pct, 1),
                'status': subj_status,
            })

    # ========================================================
    # Build DataFrames
    # ========================================================
    entries_df = pd.DataFrame(all_entries)
    summary_df = pd.DataFrame(subject_summary)
    issues_df = pd.DataFrame(issues)

    missing_df = entries_df[entries_df['status'] == 'MISSING'].copy() if len(entries_df) > 0 else pd.DataFrame()

    # ========================================================
    # Print report
    # ========================================================

    # --- Overall stats ---
    total_subjects = len(summary_df)
    total_expected = summary_df['total_expected'].sum()
    total_downloaded = summary_df['downloaded'].sum()
    total_missing = summary_df['missing'].sum()
    overall_pct = (total_downloaded / total_expected * 100) if total_expected > 0 else 0

    print('OVERALL SUMMARY')
    print('-' * 50)
    print(f'  Subjects scanned     : {total_subjects}')
    print(f'  Videos expected      : {total_expected}')
    print(f'  Videos downloaded    : {total_downloaded}')
    print(f'  Videos missing       : {total_missing}')
    print(f'  Completeness         : {overall_pct:.1f}%')

    # --- Per-group ---
    print()
    print('PER-GROUP BREAKDOWN')
    print('-' * 50)
    for g in ['CN', 'PD']:
        g_df = summary_df[summary_df['group'] == g]
        g_exp = g_df['total_expected'].sum()
        g_dl = g_df['downloaded'].sum()
        g_miss = g_df['missing'].sum()
        pct = f'{g_dl/g_exp:.1f}%' if g_exp > 0 else 'N/A'
        n_complete = len(g_df[g_df['status'] == 'COMPLETE'])
        n_partial = len(g_df[g_df['status'] == 'PARTIAL'])
        n_empty = len(g_df[g_df['status'] == 'EMPTY'])
        n_noxlsx = len(g_df[g_df['status'].isin(['NO_XLSX', 'NO_LINK_COL'])])
        print(f'  {g}:')
        print(f'    Subjects    : {len(g_df)} total  |  {n_complete} complete, {n_partial} partial, {n_empty} empty, {n_noxlsx} no metadata')
        print(f'    Videos      : {g_dl}/{g_exp} downloaded ({pct})  |  {g_miss} missing')

    # --- Failure type breakdown ---
    if len(missing_df) > 0:
        print()
        print('FAILURE TYPE BREAKDOWN')
        print('-' * 50)
        failure_counts = missing_df['failure_type'].value_counts()
        for ftype, count in failure_counts.items():
            explanations = {
                'never_attempted': 'Download was never started (no subfolder created)',
                'empty_folder': 'Subfolder exists but is empty',
                'partial_download': 'Download started but did not finish (.part file found)',
                'video_not_converted': 'Video downloaded but not converted to audio (ffmpeg issue?)',
                'no_audio_in_folder': 'Subfolder has files but none are audio',
                'corrupt_file': 'Audio file exists but is < 1 KB (likely corrupt)',
                'unknown': 'Could not determine reason',
            }
            desc = explanations.get(ftype, '')
            print(f'  {ftype:<25s} : {count:>4d}  — {desc}')

    # --- Subjects with missing downloads (sorted worst first) ---
    incomplete = summary_df[summary_df['missing'] > 0].sort_values('missing', ascending=False)
    if len(incomplete) > 0:
        print()
        print(f'SUBJECTS WITH MISSING DOWNLOADS ({len(incomplete)})')
        print('-' * 70)
        print(f'  {"Subject":<12s} {"Group":<6s} {"Downloaded":<12s} {"Missing":<10s} {"Complete%":<10s} {"Status":<10s}')
        print(f'  {"─"*12} {"─"*6} {"─"*12} {"─"*10} {"─"*10} {"─"*10}')
        for _, row in incomplete.iterrows():
            print(f'  {row["subject"]:<12s} {row["group"]:<6s} '
                  f'{row["downloaded"]:>4d}/{row["total_expected"]:<7d} '
                  f'{row["missing"]:<10d} {row["pct_complete"]:>8.1f}%  '
                  f'{row["status"]:<10s}')

    # --- Full list of missing videos ---
    if len(missing_df) > 0:
        print()
        print(f'ALL MISSING VIDEOS ({len(missing_df)})')
        print('-' * 90)
        print(f'  {"Subject":<12s} {"Video ID":<15s} {"Failure":<25s} {"URL":<40s}')
        print(f'  {"─"*12} {"─"*15} {"─"*25} {"─"*40}')
        for _, row in missing_df.iterrows():
            url_short = row['url'][:40] + '...' if len(row['url']) > 40 else row['url']
            print(f'  {row["subject"]:<12s} {row["video_id"]:<15s} '
                  f'{row["failure_type"]:<25s} {url_short}')
    else:
        print()
        print('No missing videos — all downloads are complete.')

    # --- Issues encountered ---
    if len(issues_df) > 0:
        print()
        print(f'ISSUES ENCOUNTERED ({len(issues_df)})')
        print('-' * 70)
        for _, row in issues_df.iterrows():
            print(f'  {row["subject"]:<12s}: {row["issue"]}')

    # ========================================================
    # Save CSVs
    # ========================================================
    if save_csv:
        print()
        print('SAVING CSV REPORTS')
        print('-' * 50)

        out_dir = os.path.join(root, 'audit_reports')
        os.makedirs(out_dir, exist_ok=True)

        if len(summary_df) > 0:
            path = os.path.join(out_dir, 'subject_summary.csv')
            summary_df.to_csv(path, index=False)
            print(f'  {path}  ({len(summary_df)} rows)')

        if len(missing_df) > 0:
            path = os.path.join(out_dir, 'missing_videos.csv')
            missing_df.to_csv(path, index=False)
            print(f'  {path}  ({len(missing_df)} rows)')

        if len(entries_df) > 0:
            path = os.path.join(out_dir, 'all_videos.csv')
            entries_df.to_csv(path, index=False)
            print(f'  {path}  ({len(entries_df)} rows)')

        if len(issues_df) > 0:
            path = os.path.join(out_dir, 'issues.csv')
            issues_df.to_csv(path, index=False)
            print(f'  {path}  ({len(issues_df)} rows)')

        # Save a re-download script
        if len(missing_df) > 0:
            path = os.path.join(out_dir, 'retry_missing.sh')
            with open(path, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('# Auto-generated script to retry failed downloads\n')
                f.write(f'# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                f.write(f'# Missing videos: {len(missing_df)}\n\n')
                for _, row in missing_df.iterrows():
                    out_dir_vid = os.path.join(root, row['group'], row['subject'], row['video_id'])
                    f.write(f'echo "Downloading {row["video_id"]} for {row["subject"]}..."\n')
                    f.write(f'mkdir -p "{out_dir_vid}"\n')
                    f.write(f'yt-dlp --extract-audio --audio-format wav --audio-quality 0 '
                            f'--no-playlist --output "{out_dir_vid}/{row["video_id"]}.%(ext)s" '
                            f'--socket-timeout 30 --retries 3 '
                            f'"{row["url"]}"\n')
                    f.write(f'sleep 2\n\n')
            os.chmod(path, 0o755)
            print(f'  {path}  (run this to retry {len(missing_df)} failed downloads)')

    # ========================================================
    # Generate visuals
    # ========================================================
    if plot and len(summary_df) > 0:
        out_dir = os.path.join(root, 'audit_reports')
        os.makedirs(out_dir, exist_ok=True)

        plt.rcParams.update({
            'figure.dpi': 150,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
        })

        C_CN = '#3498DB'
        C_PD = '#E74C3C'
        C_OK = '#27AE60'
        C_MISS = '#E74C3C'
        C_PARTIAL = '#F39C12'

        # ---- Figure 1: Overview dashboard (2×2) ----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Overall completeness donut
        ax = axes[0, 0]
        sizes = [total_downloaded, total_missing]
        labels = [f'Downloaded\n{total_downloaded}', f'Missing\n{total_missing}']
        colors_pie = [C_OK, C_MISS]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
        )
        for t in autotexts:
            t.set_fontsize(11)
            t.set_fontweight('bold')
        ax.set_title(f'Overall Download Completeness\n({total_expected} videos total)')

        # (b) Per-group completeness stacked bar
        ax = axes[0, 1]
        groups = ['CN', 'PD']
        dl_counts = []
        miss_counts = []
        for g in groups:
            g_df = summary_df[summary_df['group'] == g]
            dl_counts.append(g_df['downloaded'].sum())
            miss_counts.append(g_df['missing'].sum())
        x = range(len(groups))
        ax.bar(x, dl_counts, color=C_OK, alpha=0.8, label='Downloaded')
        ax.bar(x, miss_counts, bottom=dl_counts, color=C_MISS, alpha=0.7, label='Missing')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=12)
        ax.set_ylabel('Number of videos')
        ax.set_title('Completeness by Group')
        ax.legend()
        for i, g in enumerate(groups):
            total_g = dl_counts[i] + miss_counts[i]
            if total_g > 0:
                pct = dl_counts[i] / total_g * 100
                ax.text(i, total_g + 2, f'{pct:.0f}%', ha='center', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

        # (c) Subject status distribution
        ax = axes[1, 0]
        status_order = ['COMPLETE', 'PARTIAL', 'EMPTY', 'NO_XLSX', 'NO_LINK_COL']
        status_colors = {
            'COMPLETE': C_OK, 'PARTIAL': C_PARTIAL, 'EMPTY': C_MISS,
            'NO_XLSX': '#95A5A6', 'NO_LINK_COL': '#BDC3C7',
        }
        status_labels = {
            'COMPLETE': 'Complete', 'PARTIAL': 'Partial', 'EMPTY': 'No downloads',
            'NO_XLSX': 'No metadata', 'NO_LINK_COL': 'No link col',
        }
        for g_idx, g in enumerate(['CN', 'PD']):
            g_df = summary_df[summary_df['group'] == g]
            bottom = 0
            for s in status_order:
                count = len(g_df[g_df['status'] == s])
                if count > 0:
                    ax.bar(g_idx, count, bottom=bottom,
                           color=status_colors.get(s, '#999'),
                           alpha=0.8, label=status_labels.get(s, s) if g_idx == 0 else '')
                    if count >= 2:
                        ax.text(g_idx, bottom + count/2, str(count),
                                ha='center', va='center', fontsize=9, fontweight='bold')
                    bottom += count
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['CN', 'PD'], fontsize=12)
        ax.set_ylabel('Number of subjects')
        ax.set_title('Subject Download Status')
        handles, lbls = ax.get_legend_handles_labels()
        # Remove duplicate labels
        by_label = dict(zip(lbls, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.grid(axis='y', alpha=0.2)

        # (d) Failure type breakdown
        ax = axes[1, 1]
        if len(missing_df) > 0 and 'failure_type' in missing_df.columns:
            fc = missing_df['failure_type'].value_counts()
            colors_fail = ['#E74C3C', '#E67E22', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E', '#95A5A6']
            bars = ax.barh(range(len(fc)), fc.values, color=colors_fail[:len(fc)], alpha=0.8)
            ax.set_yticks(range(len(fc)))
            ax.set_yticklabels([t.replace('_', ' ') for t in fc.index], fontsize=9)
            ax.set_xlabel('Number of videos')
            ax.set_title('Failure Type Breakdown')
            ax.invert_yaxis()
            for i, v in enumerate(fc.values):
                ax.text(v + 0.5, i, str(v), va='center', fontsize=10, fontweight='bold')
            ax.grid(axis='x', alpha=0.2)
        else:
            ax.text(0.5, 0.5, 'No missing videos', ha='center', va='center',
                    fontsize=14, transform=ax.transAxes)
            ax.set_title('Failure Types')

        plt.suptitle('ParkCeleb Download Audit — Overview',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = os.path.join(out_dir, 'audit_overview.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'\n  Saved: {path}')

        # ---- Figure 2: Per-subject completeness (sorted) ----
        subj_with_data = summary_df[summary_df['total_expected'] > 0].copy()
        if len(subj_with_data) > 0:
            subj_with_data = subj_with_data.sort_values('pct_complete', ascending=True)

            fig, ax = plt.subplots(figsize=(14, max(6, len(subj_with_data) * 0.25)))

            y_pos = range(len(subj_with_data))
            bar_colors = []
            for _, row in subj_with_data.iterrows():
                if row['group'] == 'CN':
                    bar_colors.append(C_CN)
                else:
                    bar_colors.append(C_PD)

            bars = ax.barh(y_pos, subj_with_data['pct_complete'],
                          color=bar_colors, alpha=0.7, edgecolor='white', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                [f"{row['subject']} ({row['group']})" for _, row in subj_with_data.iterrows()],
                fontsize=7
            )
            ax.set_xlabel('Download Completeness (%)')
            ax.set_title('Per-Subject Download Completeness (sorted lowest → highest)')
            ax.set_xlim(0, 105)
            ax.axvline(100, color='green', ls='--', lw=0.8, alpha=0.5)
            ax.axvline(50, color='orange', ls=':', lw=0.8, alpha=0.5)

            # Annotate bars with count
            for i, (_, row) in enumerate(subj_with_data.iterrows()):
                ax.text(row['pct_complete'] + 1, i,
                        f"{row['downloaded']}/{row['total_expected']}",
                        va='center', fontsize=7)

            cn_patch = plt.Line2D([0], [0], color=C_CN, lw=8, alpha=0.7, label='CN')
            pd_patch = plt.Line2D([0], [0], color=C_PD, lw=8, alpha=0.7, label='PD')
            ax.legend(handles=[cn_patch, pd_patch], loc='lower right')
            ax.grid(axis='x', alpha=0.2)

            plt.tight_layout()
            path = os.path.join(out_dir, 'audit_per_subject.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'  Saved: {path}')

        # ---- Figure 3: Distribution of missing counts ----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: histogram of missing counts per subject
        ax = axes[0]
        for g, c in [('CN', C_CN), ('PD', C_PD)]:
            vals = summary_df[summary_df['group'] == g]['missing']
            ax.hist(vals, bins=range(0, int(vals.max()) + 3),
                    alpha=0.6, color=c, label=g, edgecolor='white')
        ax.set_xlabel('Number of missing videos per subject')
        ax.set_ylabel('Number of subjects')
        ax.set_title('Distribution of Missing Videos')
        ax.legend()
        ax.grid(axis='y', alpha=0.2)

        # Right: histogram of completeness % per subject
        ax = axes[1]
        for g, c in [('CN', C_CN), ('PD', C_PD)]:
            vals = subj_with_data[subj_with_data['group'] == g]['pct_complete']
            if len(vals) > 0:
                ax.hist(vals, bins=np.arange(0, 110, 10),
                        alpha=0.6, color=c, label=g, edgecolor='white')
        ax.set_xlabel('Completeness (%)')
        ax.set_ylabel('Number of subjects')
        ax.set_title('Distribution of Completeness')
        ax.axvline(100, color='green', ls='--', lw=1, alpha=0.5, label='100%')
        ax.legend()
        ax.grid(axis='y', alpha=0.2)

        plt.suptitle('ParkCeleb — Missing Video Distributions',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = os.path.join(out_dir, 'audit_distributions.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  Saved: {path}')

        # ---- Figure 4: Top 15 worst subjects ----
        worst = summary_df[summary_df['missing'] > 0].nlargest(15, 'missing')
        if len(worst) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = range(len(worst))
            bar_colors = [C_CN if r['group'] == 'CN' else C_PD for _, r in worst.iterrows()]

            ax.bar(x, worst['downloaded'], color=[c + '99' for c in bar_colors],
                   label='Downloaded', edgecolor='white')
            ax.bar(x, worst['missing'], bottom=worst['downloaded'],
                   color=bar_colors, alpha=0.4, hatch='///',
                   label='Missing', edgecolor='white')

            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{r['subject']}\n({r['group']})" for _, r in worst.iterrows()],
                fontsize=8, rotation=45, ha='right'
            )
            ax.set_ylabel('Number of videos')
            ax.set_title(f'Top {len(worst)} Subjects with Most Missing Downloads')

            for i, (_, r) in enumerate(worst.iterrows()):
                total = r['downloaded'] + r['missing']
                ax.text(i, total + 0.5, f"{r['pct_complete']:.0f}%",
                        ha='center', fontsize=9, fontweight='bold')

            ax.legend()
            ax.grid(axis='y', alpha=0.2)

            plt.tight_layout()
            path = os.path.join(out_dir, 'audit_worst_subjects.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'  Saved: {path}')

        print(f'\n  All visuals saved to {out_dir}/')

    print()
    print('=' * 70)
    print('  Audit complete.')
    print('=' * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit ParkCeleb download completeness.')
    parser.add_argument('--root', default='.', help='Path to ParkCeleb directory (default: current dir)')
    parser.add_argument('--csv', action='store_true', help='Save CSV reports and retry script to audit_reports/')
    parser.add_argument('--plot', action='store_true', help='Generate visual charts to audit_reports/')
    args = parser.parse_args()

    # Verify the root looks right
    root = args.root
    has_cn = os.path.isdir(os.path.join(root, 'CN'))
    has_pd = os.path.isdir(os.path.join(root, 'PD'))

    if not has_cn and not has_pd:
        print(f'ERROR: Neither CN/ nor PD/ found under {os.path.abspath(root)}')
        print(f'  Make sure --root points to the ParkCeleb directory.')
        print(f'  Expected: {root}/CN/cn_01/, {root}/PD/pd_01/, etc.')
        sys.exit(1)

    run_audit(root, save_csv=args.csv, plot=args.plot)
