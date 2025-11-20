# Model Card — Titanic AutoGluon

## Problem & Intended Use
- Cel: przewidywanie `Survived` dla pasażerów Titanica (binary classification).
- Użytkownicy: edukacyjny / demonstracyjny projekt.

## Data
- Źródło: Kaggle Titanic Dataset (pobrane 14.10.2025).
- Rozmiar: <tu liczba wierszy> (wstaw wartości).
- PII: brak danych identyfikujących osoby.

## Metrics
- Main: ROC AUC
- Secondary: F1-score, precision, recall
- Best model (production): W&B run: `<link do runu>`; ROC_AUC = 0.XXX

## Limitations & Risks
- Dane historyczne z 1912 — ograniczona przydatność do współczesnych prognoz.
- Bias: rozkład klas / brak niektórych cech.

## Versioning
- Model artifact: `titanic-team/ag_model:production` (W&B).
- Commit: `<git hash>`
- Env: Python 3.11, AutoGluon 1.x
