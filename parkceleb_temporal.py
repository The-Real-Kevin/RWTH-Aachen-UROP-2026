"""
ParkCeleb Temporal Analysis — speakers_info.csv
================================================
Scans all subject folders, reads speakers_info.csv from each video
subfolder, filters for Target speakers, and produces statistics and
visuals about the temporal distribution of recordings relative to
diagnosis year.

Usage:
    cd /path/to/ParkCeleb
    python parkceleb_temporal.py
    python parkceleb_temporal.py --root ./ParkCeleb --csv

Requirements:
    pip install pandas matplotlib numpy
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# Configuration
# ============================================================
SPEAKERS_CSV = 'speakers_info.csv'
OUTPUT_DIR_NAME = 'temporal_analysis'


# ============================================================
# Scan and collect
# ============================================================

def scan_all_speakers_info(root):
    """
    Walk every subject folder and every video subfolder,
    read speakers_info.csv, and return a combined DataFrame.
    """
    all_rows = []
    issues = []

    for group in ['CN', 'PD']:
        group_dir = os.path.join(root, group)
        if not os.path.isdir(group_dir):
            issues.append(f'{group}/ directory not found')
            continue

        for subj_name in sorted(os.listdir(group_dir)):
            subj_dir = os.path.join(group_dir, subj_name)
            if not os.path.isdir(subj_dir):
                continue

            # Iterate over video-id subfolders
            for vid_folder in sorted(os.listdir(subj_dir)):
                vid_dir = os.path.join(subj_dir, vid_folder)
                if not os.path.isdir(vid_dir):
                    continue

                csv_path = os.path.join(vid_dir, SPEAKERS_CSV)
                if not os.path.isfile(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    issues.append(f'{subj_name}/{vid_folder}: failed to read CSV — {e}')
                    continue

                # Normalize column names (strip whitespace, lowercase)
                df.columns = [c.strip() for c in df.columns]

                # Add context columns
                df['group'] = group
                df['subject'] = subj_name
                df['video_folder'] = vid_folder
                df['csv_path'] = csv_path

                all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined, issues


def find_column(df, candidates):
    """Find a column by trying multiple name variants (case-insensitive)."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


# ============================================================
# Main analysis
# ============================================================

def run_analysis(root, save_csv=False):
    out_dir = os.path.join(root, OUTPUT_DIR_NAME)

    print('=' * 70)
    print('  ParkCeleb Temporal Analysis')
    print(f'  Root: {os.path.abspath(root)}')
    print(f'  Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)
    print()

    # --- Scan ---
    raw_df, issues = scan_all_speakers_info(root)

    if len(raw_df) == 0:
        print('ERROR: No speakers_info.csv files found.')
        if issues:
            for iss in issues:
                print(f'  {iss}')
        return

    print(f'Scanned:')
    print(f'  Total speakers_info.csv files read : {raw_df["csv_path"].nunique()}')
    print(f'  Total rows across all CSVs         : {len(raw_df)}')
    print(f'  Columns found                      : {list(raw_df.columns)}')
    if issues:
        print(f'  Issues                             : {len(issues)}')
        for iss in issues[:5]:
            print(f'    — {iss}')
        if len(issues) > 5:
            print(f'    ... and {len(issues) - 5} more')
    print()

    # --- Identify key columns ---
    status_col = find_column(raw_df, ['status', 'speaker_status', 'speakers'])
    years_col = find_column(raw_df, ['years_from_diagnosis', 'years from diagnosis',
                                      'year_from_diagnosis', 'years_from_diag'])
    before_after_col = find_column(raw_df, ['before_after_diagnosis', 'before after diagnosis',
                                             'before_after', 'diagnosis_phase'])
    video_id_col = find_column(raw_df, ['video_id', 'videoid', 'video'])

    print(f'Column mapping:')
    print(f'  Status column             : {status_col}')
    print(f'  Years from diagnosis col  : {years_col}')
    print(f'  Before/after diagnosis col: {before_after_col}')
    print(f'  Video ID column           : {video_id_col}')
    print()

    if status_col is None:
        print('ERROR: Could not find a status column.')
        print(f'  Available columns: {list(raw_df.columns)}')
        return

    # --- Show unique status values ---
    print(f'Unique values in "{status_col}" column:')
    status_counts = raw_df[status_col].value_counts()
    for val, count in status_counts.items():
        print(f'  {str(val):<25s} : {count:>5d} rows')
    print()

    # --- Filter for Target speakers ---
    target_mask = raw_df[status_col].astype(str).str.strip().str.lower() == 'target'
    target_df = raw_df[target_mask].copy()

    print(f'After filtering for Target speakers:')
    print(f'  Target rows    : {len(target_df)}')
    print(f'  Non-target rows: {len(raw_df) - len(target_df)}')
    print(f'  CN targets     : {len(target_df[target_df["group"] == "CN"])}')
    print(f'  PD targets     : {len(target_df[target_df["group"] == "PD"])}')
    print()

    if len(target_df) == 0:
        print('No Target rows found. Check the status column values above.')
        return

    # --- Parse years_from_diagnosis as numeric ---
    if years_col:
        target_df['years_num'] = pd.to_numeric(target_df[years_col], errors='coerce')
        valid_years = target_df['years_num'].notna()
        print(f'Years from diagnosis:')
        print(f'  Parseable as numeric: {valid_years.sum()}/{len(target_df)}')
        if not valid_years.all():
            unparseable = target_df[~valid_years][years_col].unique()[:5]
            print(f'  Unparseable values : {list(unparseable)}')
        print()

    # --- Show unique before/after values ---
    if before_after_col:
        print(f'Unique values in "{before_after_col}" column (Target only):')
        ba_counts = target_df[before_after_col].value_counts(dropna=False)
        for val, count in ba_counts.items():
            print(f'  {str(val):<25s} : {count:>5d} rows')
        print()

    # ========================================================
    # Per-subject statistics
    # ========================================================
    print('PER-SUBJECT STATISTICS (Target speakers only)')
    print('=' * 70)

    subj_stats = []
    for (group, subject), sdf in target_df.groupby(['group', 'subject']):
        row = {
            'group': group,
            'subject': subject,
            'n_recordings': len(sdf),
            'n_videos': sdf['video_folder'].nunique(),
        }

        if years_col and 'years_num' in sdf.columns:
            yrs = sdf['years_num'].dropna()
            if len(yrs) > 0:
                row['years_min'] = yrs.min()
                row['years_max'] = yrs.max()
                row['years_mean'] = yrs.mean()
                row['years_median'] = yrs.median()
                row['years_range'] = yrs.max() - yrs.min()

        if before_after_col:
            ba = sdf[before_after_col].astype(str).str.strip().str.lower()
            row['n_before'] = (ba == 'before').sum()
            row['n_after'] = (ba == 'after').sum()
            row['n_other_phase'] = len(ba) - row['n_before'] - row['n_after']

        subj_stats.append(row)

    subj_df = pd.DataFrame(subj_stats)

    # Print table
    print(f'\n{"Subject":<12s} {"Grp":<5s} {"Recs":<6s} {"Vids":<6s}', end='')
    if years_col:
        print(f' {"Yrs Min":<9s} {"Yrs Max":<9s} {"Yrs Mean":<9s} {"Range":<7s}', end='')
    if before_after_col:
        print(f' {"Before":<8s} {"After":<8s} {"Other":<8s}', end='')
    print()
    print('─' * 100)

    for _, r in subj_df.iterrows():
        print(f'{r["subject"]:<12s} {r["group"]:<5s} {r["n_recordings"]:<6d} {r["n_videos"]:<6d}', end='')
        if years_col and 'years_min' in r and pd.notna(r.get('years_min')):
            print(f' {r["years_min"]:<9.1f} {r["years_max"]:<9.1f} {r["years_mean"]:<9.1f} {r["years_range"]:<7.1f}', end='')
        elif years_col:
            print(f' {"N/A":<9s} {"N/A":<9s} {"N/A":<9s} {"N/A":<7s}', end='')
        if before_after_col and 'n_before' in r:
            print(f' {r["n_before"]:<8.0f} {r["n_after"]:<8.0f} {r["n_other_phase"]:<8.0f}', end='')
        print()

    # ========================================================
    # Group-level summary
    # ========================================================
    print()
    print('GROUP-LEVEL SUMMARY')
    print('=' * 70)

    for g in ['CN', 'PD']:
        g_df = target_df[target_df['group'] == g]
        g_subj = subj_df[subj_df['group'] == g]
        print(f'\n  {g} group:')
        print(f'    Subjects with Target data : {len(g_subj)}')
        print(f'    Total Target recordings   : {len(g_df)}')

        if years_col and 'years_num' in g_df.columns:
            yrs = g_df['years_num'].dropna()
            if len(yrs) > 0:
                print(f'    Years from diagnosis:')
                print(f'      Mean   : {yrs.mean():.2f}')
                print(f'      Median : {yrs.median():.1f}')
                print(f'      Std    : {yrs.std():.2f}')
                print(f'      Range  : [{yrs.min():.1f}, {yrs.max():.1f}]')

        if before_after_col:
            ba = g_df[before_after_col].astype(str).str.strip().str.lower()
            ba_counts = ba.value_counts()
            print(f'    Before/After breakdown:')
            for val, count in ba_counts.items():
                pct = count / len(ba) * 100
                print(f'      {val:<20s}: {count:>5d} ({pct:.1f}%)')

    # ========================================================
    # Visuals
    # ========================================================
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        'figure.dpi': 150,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
    })

    C_CN = '#3498DB'
    C_PD = '#E74C3C'
    C_BEFORE = '#F39C12'
    C_AFTER = '#8E44AD'

    has_years = years_col and 'years_num' in target_df.columns
    has_ba = before_after_col is not None

    # ---- Figure 1: Overview dashboard ----
    n_panels = 2 + int(has_years) + int(has_ba)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) Recording count per group
    ax = axes[0, 0]
    for i, g in enumerate(['CN', 'PD']):
        count = len(target_df[target_df['group'] == g])
        c = C_CN if g == 'CN' else C_PD
        ax.bar(i, count, color=c, alpha=0.8, width=0.5)
        ax.text(i, count + 2, str(count), ha='center', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CN (Control)', 'PD (Parkinson\'s)'], fontsize=11)
    ax.set_ylabel('Number of Target recordings')
    ax.set_title('Total Target Recordings per Group')
    ax.grid(axis='y', alpha=0.2)

    # (b) Recordings per subject distribution
    ax = axes[0, 1]
    for g, c in [('CN', C_CN), ('PD', C_PD)]:
        vals = subj_df[subj_df['group'] == g]['n_recordings']
        ax.hist(vals, bins=range(0, int(vals.max()) + 5, 3),
                alpha=0.6, color=c, label=g, edgecolor='white')
    ax.set_xlabel('Number of Target recordings per subject')
    ax.set_ylabel('Number of subjects')
    ax.set_title('Distribution of Recordings per Subject')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)

    # (c) Years from diagnosis distribution
    ax = axes[1, 0]
    if has_years:
        for g, c in [('CN', C_CN), ('PD', C_PD)]:
            yrs = target_df[(target_df['group'] == g)]['years_num'].dropna()
            if len(yrs) > 0:
                ax.hist(yrs, bins=30, alpha=0.6, color=c, label=g, edgecolor='white')
        ax.axvline(0, color='black', ls='--', lw=1.5, alpha=0.7, label='Diagnosis year')
        ax.set_xlabel('Years from diagnosis (negative = before)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Years from Diagnosis')
        ax.legend()
        ax.grid(axis='y', alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No years_from_diagnosis data', ha='center', va='center',
                fontsize=12, transform=ax.transAxes, color='gray')
        ax.set_title('Years from Diagnosis')

    # (d) Before/After breakdown
    ax = axes[1, 1]
    if has_ba:
        ba_data = {}
        for g in ['CN', 'PD']:
            g_df = target_df[target_df['group'] == g]
            ba = g_df[before_after_col].astype(str).str.strip().str.lower()
            ba_data[g] = ba.value_counts()

        all_phases = sorted(set().union(*[set(v.index) for v in ba_data.values()]))
        phase_colors = {'before': C_BEFORE, 'after': C_AFTER}

        x = np.arange(len(['CN', 'PD']))
        width = 0.25
        for i, phase in enumerate(all_phases):
            vals = [ba_data[g].get(phase, 0) for g in ['CN', 'PD']]
            color = phase_colors.get(phase, '#95A5A6')
            ax.bar(x + i * width - width * len(all_phases) / 2 + width / 2,
                   vals, width * 0.9, label=phase.capitalize(),
                   color=color, alpha=0.8, edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(['CN', 'PD'], fontsize=11)
        ax.set_ylabel('Number of recordings')
        ax.set_title('Before / After Diagnosis Breakdown')
        ax.legend()
        ax.grid(axis='y', alpha=0.2)
    else:
        ax.text(0.5, 0.5, 'No before/after data', ha='center', va='center',
                fontsize=12, transform=ax.transAxes, color='gray')
        ax.set_title('Before / After Diagnosis')

    plt.suptitle('ParkCeleb Temporal Analysis — Overview',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, 'temporal_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved: {path}')

    # ---- Figure 2: Per-subject timeline (PD only) ----
    if has_years:
        for g, c_main in [('PD', C_PD), ('CN', C_CN)]:
            g_target = target_df[target_df['group'] == g].copy()
            g_subjs = sorted(g_target['subject'].unique())

            if len(g_subjs) == 0:
                continue

            fig, ax = plt.subplots(figsize=(14, max(4, len(g_subjs) * 0.4)))

            for i, subj in enumerate(g_subjs):
                sdf = g_target[g_target['subject'] == subj]
                yrs = sdf['years_num'].dropna()
                if len(yrs) == 0:
                    ax.plot(0, i, 'x', color='gray', markersize=6)
                    continue

                # Color by before/after if available
                if has_ba:
                    for _, row in sdf.iterrows():
                        yr = row.get('years_num', np.nan)
                        if pd.isna(yr):
                            continue
                        ba = str(row.get(before_after_col, '')).strip().lower()
                        if ba == 'before':
                            mc = C_BEFORE
                        elif ba == 'after':
                            mc = C_AFTER
                        else:
                            mc = '#95A5A6'
                        ax.plot(yr, i, 'o', color=mc, markersize=5, alpha=0.7)
                else:
                    ax.scatter(yrs, [i] * len(yrs), s=15, color=c_main, alpha=0.6)

                # Draw span line
                if len(yrs) > 1:
                    ax.plot([yrs.min(), yrs.max()], [i, i],
                            '-', color=c_main, alpha=0.3, linewidth=2)

            ax.axvline(0, color='black', ls='--', lw=1.5, alpha=0.5, label='Diagnosis year')
            ax.set_yticks(range(len(g_subjs)))
            ax.set_yticklabels(g_subjs, fontsize=8)
            ax.set_xlabel('Years from diagnosis (negative = before)')
            ax.set_ylabel('Subject')
            ax.set_title(f'{g} Group — Recording Timeline per Subject')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

            if has_ba:
                before_p = mpatches.Patch(color=C_BEFORE, label='Before diagnosis')
                after_p = mpatches.Patch(color=C_AFTER, label='After diagnosis')
                diag_line = plt.Line2D([0], [0], color='black', ls='--', label='Diagnosis year')
                ax.legend(handles=[before_p, after_p, diag_line], loc='lower right', fontsize=9)

            plt.tight_layout()
            path = os.path.join(out_dir, f'temporal_timeline_{g}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'Saved: {path}')

    # ---- Figure 3: Coverage heatmap (years × subjects) ----
    if has_years:
        for g, c_main in [('PD', C_PD), ('CN', C_CN)]:
            g_target = target_df[target_df['group'] == g].copy()
            g_subjs = sorted(g_target['subject'].unique())
            yrs_all = g_target['years_num'].dropna()

            if len(yrs_all) == 0 or len(g_subjs) == 0:
                continue

            yr_min = int(np.floor(yrs_all.min()))
            yr_max = int(np.ceil(yrs_all.max()))
            year_bins = range(yr_min, yr_max + 1)

            # Build matrix: subjects × year bins
            matrix = np.zeros((len(g_subjs), len(year_bins)))
            for i, subj in enumerate(g_subjs):
                sdf = g_target[g_target['subject'] == subj]
                for yr in sdf['years_num'].dropna():
                    bin_idx = int(np.floor(yr)) - yr_min
                    if 0 <= bin_idx < len(year_bins):
                        matrix[i, bin_idx] += 1

            fig, ax = plt.subplots(figsize=(max(10, len(year_bins) * 0.4),
                                            max(4, len(g_subjs) * 0.35)))

            cmap = 'Blues' if g == 'CN' else 'Reds'
            im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='nearest')

            ax.set_xticks(range(len(year_bins)))
            ax.set_xticklabels(year_bins, fontsize=7, rotation=90)
            ax.set_yticks(range(len(g_subjs)))
            ax.set_yticklabels(g_subjs, fontsize=7)
            ax.set_xlabel('Year relative to diagnosis')
            ax.set_ylabel('Subject')
            ax.set_title(f'{g} Group — Recording Density Heatmap')

            # Mark diagnosis year
            if -yr_min < len(year_bins):
                ax.axvline(-yr_min - 0.5, color='white', ls='--', lw=2, alpha=0.8)

            fig.colorbar(im, ax=ax, label='Number of recordings', shrink=0.8)
            plt.tight_layout()
            path = os.path.join(out_dir, f'temporal_heatmap_{g}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'Saved: {path}')

    # ---- Figure 4: Per-subject bar chart (before vs after counts) ----
    if has_ba and 'n_before' in subj_df.columns:
        for g, c_main in [('PD', C_PD), ('CN', C_CN)]:
            g_subj = subj_df[subj_df['group'] == g].copy()
            if len(g_subj) == 0:
                continue

            g_subj = g_subj.sort_values('subject')

            fig, ax = plt.subplots(figsize=(14, max(4, len(g_subj) * 0.35)))

            y = range(len(g_subj))
            ax.barh(y, -g_subj['n_before'].fillna(0), color=C_BEFORE, alpha=0.8,
                    label='Before diagnosis')
            ax.barh(y, g_subj['n_after'].fillna(0), color=C_AFTER, alpha=0.8,
                    label='After diagnosis')

            ax.set_yticks(y)
            ax.set_yticklabels(g_subj['subject'], fontsize=8)
            ax.set_xlabel('← Before          Number of recordings          After →')
            ax.set_title(f'{g} Group — Before vs After Diagnosis per Subject')
            ax.axvline(0, color='black', lw=1)
            ax.legend(loc='lower right')
            ax.grid(axis='x', alpha=0.2)

            # Annotate
            for i, (_, r) in enumerate(g_subj.iterrows()):
                b = int(r.get('n_before', 0))
                a = int(r.get('n_after', 0))
                if b > 0:
                    ax.text(-b - 0.3, i, str(b), va='center', ha='right', fontsize=7)
                if a > 0:
                    ax.text(a + 0.3, i, str(a), va='center', ha='left', fontsize=7)

            plt.tight_layout()
            path = os.path.join(out_dir, f'temporal_before_after_{g}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f'Saved: {path}')

    # ---- Figure 5: PD vs CN years distribution comparison ----
    if has_years:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Overlaid histogram
        ax = axes[0]
        for g, c in [('CN', C_CN), ('PD', C_PD)]:
            yrs = target_df[target_df['group'] == g]['years_num'].dropna()
            if len(yrs) > 0:
                ax.hist(yrs, bins=np.arange(yrs.min() - 0.5, yrs.max() + 1.5, 1),
                        alpha=0.5, color=c, label=f'{g} (n={len(yrs)})', edgecolor='white')
        ax.axvline(0, color='black', ls='--', lw=1.5, alpha=0.6)
        ax.set_xlabel('Years from diagnosis')
        ax.set_ylabel('Count')
        ax.set_title('Recording Year Distribution (Overlaid)')
        ax.legend()
        ax.grid(axis='y', alpha=0.2)

        # Box plot comparison
        ax = axes[1]
        data_to_plot = []
        labels = []
        colors_bp = []
        for g, c in [('CN', C_CN), ('PD', C_PD)]:
            yrs = target_df[target_df['group'] == g]['years_num'].dropna()
            if len(yrs) > 0:
                data_to_plot.append(yrs)
                labels.append(f'{g}\n(n={len(yrs)})')
                colors_bp.append(c)

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.5)
            for patch, c in zip(bp['boxes'], colors_bp):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
            ax.set_ylabel('Years from diagnosis')
            ax.set_title('Years from Diagnosis — Box Plot')
            ax.grid(axis='y', alpha=0.2)

        plt.suptitle('ParkCeleb — Temporal Distribution: CN vs PD',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        path = os.path.join(out_dir, 'temporal_cn_vs_pd.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Saved: {path}')

    # ========================================================
    # Key observations
    # ========================================================
    print()
    print('KEY OBSERVATIONS')
    print('=' * 70)

    if has_years:
        for g in ['PD', 'CN']:
            yrs = target_df[target_df['group'] == g]['years_num'].dropna()
            if len(yrs) > 0:
                n_before = (yrs < 0).sum()
                n_after = (yrs >= 0).sum()
                print(f'\n  {g} group:')
                print(f'    Recordings before diagnosis (yr < 0) : {n_before} ({n_before/len(yrs):.0%})')
                print(f'    Recordings after diagnosis (yr >= 0) : {n_after} ({n_after/len(yrs):.0%})')
                print(f'    Earliest recording                   : {yrs.min():.1f} years before diagnosis')
                print(f'    Latest recording                     : {yrs.max():.1f} years after diagnosis')
                print(f'    Temporal span                        : {yrs.max() - yrs.min():.1f} years')

        pd_yrs = target_df[target_df['group'] == 'PD']['years_num'].dropna()
        if len(pd_yrs) > 0:
            # ParkCeleb time intervals
            intervals = {
                'Interval -2 (6-10yr pre)': (pd_yrs <= -6) & (pd_yrs >= -10),
                'Interval -1 (1-5yr pre)':  (pd_yrs <= -1) & (pd_yrs >= -5),
                'Interval 1 (0-5yr post)':  (pd_yrs >= 0) & (pd_yrs <= 5),
                'Interval 2 (6-10yr post)': (pd_yrs >= 6) & (pd_yrs <= 10),
                'Interval 3 (11-30yr post)': (pd_yrs >= 11) & (pd_yrs <= 30),
            }
            print(f'\n  PD recordings by ParkCeleb time intervals:')
            for name, mask in intervals.items():
                count = mask.sum()
                pct = count / len(pd_yrs) * 100
                print(f'    {name:<30s}: {count:>5d} recordings ({pct:.1f}%)')

    # ========================================================
    # Save CSVs
    # ========================================================
    if save_csv:
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.join(out_dir, 'target_recordings.csv')
        target_df.to_csv(path, index=False)
        print(f'\nSaved: {path} ({len(target_df)} rows)')

        path = os.path.join(out_dir, 'subject_temporal_stats.csv')
        subj_df.to_csv(path, index=False)
        print(f'Saved: {path} ({len(subj_df)} rows)')

    print()
    print(f'All visuals saved to {out_dir}/')
    print('=' * 70)
    print('  Analysis complete.')
    print('=' * 70)


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyse ParkCeleb temporal metadata (speakers_info.csv).'
    )
    parser.add_argument('--root', default='.',
                        help='Path to ParkCeleb directory (default: current dir)')
    parser.add_argument('--csv', action='store_true',
                        help='Save extracted data as CSV')
    args = parser.parse_args()

    root = args.root
    if not os.path.isdir(os.path.join(root, 'CN')) and not os.path.isdir(os.path.join(root, 'PD')):
        print(f'ERROR: Neither CN/ nor PD/ found under {os.path.abspath(root)}')
        print(f'  Expected: {root}/CN/cn_01/, {root}/PD/pd_01/')
        sys.exit(1)

    run_analysis(root, save_csv=args.csv)
