"""Diagnostic: audit training pipeline + retrain + verify."""
from pathlib import Path
import pandas as pd
from fib_pattern_engine_v4 import (
    FibPatternEngineV4, train_and_save_model_v4,
    FEATURE_CATEGORICAL, FEATURE_NUMERIC, RANK_COLUMNS, HIT_COLUMNS,
)

excel = Path("Dataset Analisis Trading.xlsx")
print("=" * 70)
print("STEP 1 — Audit raw dataset")
print("=" * 70)
df_raw = pd.read_excel(excel)
print(f"Total rows in xlsx        : {len(df_raw)}")
print(f"Required FEATURE_CATEGORICAL: {FEATURE_CATEGORICAL}")
print(f"Required FEATURE_NUMERIC   : {FEATURE_NUMERIC}")
print(f"Required RANK columns      : {list(RANK_COLUMNS.values())}")

# NaN per kolom yang penting
need = FEATURE_CATEGORICAL + FEATURE_NUMERIC + ["Date", "Clock"] + list(RANK_COLUMNS.values())
nan_summary = df_raw[need].isna().sum()
print("\nNaN per kolom wajib:")
for c, n in nan_summary.items():
    if n > 0:
        print(f"  - {c:<25}: {n} NaN")

# Cek unik categorical
print("\nNilai unik tiap categorical (untuk deteksi inkonsistensi case/typo):")
for col in FEATURE_CATEGORICAL:
    unique = sorted({str(x).strip() for x in df_raw[col].dropna()})
    print(f"  - {col:<22}: {unique}")

print()
print("=" * 70)
print("STEP 2 — Fresh training")
print("=" * 70)
engine = train_and_save_model_v4(
    excel_path=excel,
    model_path="fib_pattern_engine_v4.pkl",
    first_hit_summary_csv="fib_pattern_first_hit_summary_v4.csv",
    reach_summary_csv="fib_pattern_reach_summary_v4.csv",
)
print(f"Training rows (post-cleanup)    : {len(engine.train_df)}")
print(f"Unique exact_key (pattern groups): {len(engine.pattern_counts)}")
print(f"kNN neighbors                   : {engine.nn_model.n_neighbors}")
print(f"Feature matrix shape            : {engine.feature_matrix.shape}")

print("\nDistribusi first_hit_target di training:")
print(engine.train_df["first_hit_target"].value_counts().to_string())

print("\nGlobal first-hit probs:")
for k, v in sorted(engine.global_first_hit_probs.items(), key=lambda x: -x[1]):
    print(f"  - {k:<14}: {v:.2%}")

print("\nGlobal reach probs:")
for k, v in sorted(engine.global_reach_probs.items(), key=lambda x: -x[1]):
    print(f"  - {k:<10}: {v:.2%}")

# Pattern dengan match_count terbesar
print("\nTop-5 exact_key dengan jumlah baris terbanyak:")
top_keys = sorted(engine.pattern_counts.items(), key=lambda x: -x[1])[:5]
for key, cnt in top_keys:
    print(f"  cnt={cnt} | {key}")

print()
print("=" * 70)
print("STEP 3 — Verifikasi exact match per baris training")
print("=" * 70)
# Setiap baris training, kalau di-predict pakai featurnya sendiri,
# harus dapat exact_match_count >= 1
sample = engine.train_df.sample(min(5, len(engine.train_df)), random_state=42)
for _, row in sample.iterrows():
    setup = {col: row[col] for col in FEATURE_CATEGORICAL + FEATURE_NUMERIC}
    result = engine.predict(setup, top_k_matches=1)
    print(f"  Date={row.get('Date')} jam={row.get('Clock')} -> "
          f"exact_match={result.source_summary['exact_match_count']:.0f}")
