# RWTH Aachen UROP 2026 — Speech-Based Detection of Parkinson's Disease

YUQI SUN

**Detection and prediction of Parkinson's disease using speech-based biomarkers and acoustic features**
*Erkennung und Vorhersage von Parkinson mittels sprachbasierter Biomarker und akustischer Merkmale*

---

## Project Overview / Projektübersicht

This research develops machine learning frameworks for binary classification of Parkinson's disease (PD) versus healthy controls (CN) using linguistic and acoustic features derived from ParkCeleb and MDVR-KCL datasets. The work is conducted as part of UROP at RWTH Aachen University and Exaia Technologies. 

Im Rahmen dieser Forschungsarbeit werden Frameworks für maschinelles Lernen zur binären Klassifizierung von Parkinson (PD) und gesunden Kontrollpersonen (CN) entwickelt, wobei auf linguistischen und akustischen Merkmalen aus den Datensätzen ParkCeleb und MDVR-KCL aufgebaut wird. Die Arbeit erfolgt im Rahmen des UROP-Programms an der RWTH Aachen und bei Exaia Technologies.

## Datasets / Datensätze

| Dataset | Subjects | Type | Source |
|---------|----------|------|--------|
| **MDVR-KCL** | 37 (21 CN, 16 PD) | Controlled clinical recordings (reading + dialogue) | King's College London |
| **ParkCeleb** | 80 (40 CN, 40 PD) | Longitudinal celebrity speech from YouTube | Favaro et al., 2024 |

## Methods / Methoden

- **Acoustic feature extraction** — F0, energy, jitter, shimmer, MFCCs, pause analysis (librosa, Praat/parselmouth)
- **Linguistic feature extraction** — 344 metrics via CYMO (syntactic complexity, lexical diversity, discourse features)
- **Feature selection** — mRMR (Minimum Redundancy Maximum Relevance)
- **Classification** — Logistic Regression, Random Forest
- **Evaluation** — Subject-level majority vote, cross-corpus generalisation (ParkCeleb → KCL)

---

## Weekly Progress / Wöchentlicher Fortschritt

### Week 1 — Literature Review & Dataset Overview
*Literaturrecherche und Datensatzübersicht*

Overview of two prior studies (KCL and ParkCeleb) and the datasets to be used.

[Presentation Slides](https://docs.google.com/presentation/d/1BauPEvf2RkpHf8-d7AzpVN6rI-8K8TEFhfqoIeKJrUc/edit?usp=sharing)

---

### Week 2 — Acoustic Feature Extraction (KCL)
*Akustische Merkmalsextraktion (KCL)*

Initial audio processing, extraction and visualisation of low-level acoustic features (F0, energy, ZCR, MFCCs, spectral centroid) on the KCL dataset.

[Presentation Slides](https://docs.google.com/presentation/d/111RI7L360mERA5Jzn_CGGbxtOIioYqF0RPK9yckkKHk/edit?usp=drive_link)

---

### Week 3 — ParkCeleb Dataset Analysis
*Analyse des ParkCeleb-Datensatzes*

Download of the ParkCeleb dataset, acoustic feature extraction, SNR analysis, and temporal distribution visualisation.

[Presentation Slides](https://docs.google.com/presentation/d/1Jbqn55i-gB6XapPBwuVQBQW1y5KaaPznfB6MjdOeTTI/edit?usp=drive_link)

---

### Week 4 — Linguistic Feature Extraction (CYMO)
*Linguistische Merkmalsextraktion (CYMO)*

Transcript processing into CYMO-compatible format, extraction of 344 linguistic features, and initial feature set analysis.

[Presentation Slides](https://docs.google.com/presentation/d/1W1xnYkXYFDsoaT9lFyhMtX64pBS3rKf9RRghqBHfw-s/edit?usp=drive_link)

---

### Week 5 — Feature Selection, Model Training, and Cross-Corpus Evaluation
*Merkmalsauswahl, Modelltraining, und Korpusübergreifende Evaluation*

mRMR feature selection, logistic regression and random forest model training and evaluation, comparison of mRMR-selected features against baseline feature set.

[Presentation Slides](https://docs.google.com/presentation/d/1FTK9GCR42gorRZuKtLAtITvUxuTk2Eu9argD-qqYYng/edit?usp=drive_link)

---

### Week 6 — 

*(in progress / in Bearbeitung)*

---

<!--
### Week N — Title
*German Title*

Description.

[Presentation Slides](URL)

---
-->

## Repository Structure / Verzeichnisstruktur

```
└── README.md
```

## Key Results / Wichtigste Ergebnisse



## References / Literatur

- Reference listFavaro, A., Butala, A., Thebaud, T., Villalba, J., Dehak, N. and Moro-Velázquez, L. (2024). Unveiling early signs of Parkinson’s disease via a longitudinal analysis of celebrity speech recordings. npj Parkinson’s Disease, 10(1). doi:https://doi.org/10.1038/s41531-024-00817-9.
- Jaeger, H., Trivedi, D. and Stadtschnitzer, M. (2019). Mobile Device Voice Recordings at King’s College London (MDVR-KCL) from both early and advanced Parkinson’s disease patients and healthy controls. OPAL (Open@LaTrobe) (La Trobe University), [online] 1(1). doi:https://doi.org/10.5281/zenodo.2867216.

## Tools & Libraries

`librosa` · `parselmouth (Praat)` · `openSMILE` · `OpenAI Whisper` · `CYMO` · `scikit-learn` · `pandas` · `matplotlib`

---

*RWTH Aachen University · UROP · Summer 2026*
