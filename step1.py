import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# =============================================================================
# Paths
# =============================================================================

PHE_PATH = "../data/phecode_definitions1.2.csv"
VAR_PATH = "../data/whi_mesa_v2.csv"

OUT_CAND_PATH = "../output/phecode_variable_step1_bge_adaptive_threshold_by_database.csv"
OUT_COUNT_PATH = "../output/phecode_variable_step1_bge_adaptive_threshold_by_database_count_summary.csv"
OUT_GROUP_COUNT_PATH = "../output/phecode_variable_step1_bge_adaptive_threshold_by_database_group_summary.csv"


# =============================================================================
# Model
# =============================================================================

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32


# =============================================================================
# Candidate constraints
# =============================================================================
# Key change:
#   We apply min/max candidate constraints within each phecode x Database group.
#
# Why Database?
#   In your file, Study looks like dbGaP accession:
#       phs000209, phs000200, phs001334, ...
#   Database is more likely the cohort/database level:
#       mesa, whi
#
# If you really want accession-level grouping, change this to "Study".
# Recommended default: "Database".

CONSTRAINT_GROUP_COL = "Database"

MIN_CANDIDATES_PER_GROUP = 10
MAX_CANDIDATES_PER_GROUP = 100

DELTA_METHOD = "mad"   # "mad" or "sd"
DELTA_SCALE = 3.0

DEBUG_N_PHECODES = None
# DEBUG_N_PHECODES = 3

TARGET_PHECODES = None
# TARGET_PHECODES = ["250", "401", "296", "8", "8.5", "10"]

INSPECT_PHECODES = ["8", "8.5", "10", "250", "401", "296"]


# =============================================================================
# Basic utilities
# =============================================================================

def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[col].fillna("").astype(str)


def safe_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_phecode_str(x: Any) -> str:
    s = safe_str(x)
    if s == "":
        return ""

    if re.fullmatch(r"\d+", s):
        return str(int(s))

    if re.fullmatch(r"\d+\.\d+", s):
        val = float(s)
        if val.is_integer():
            return str(int(val))
        return str(val)

    return s


def keep_target_phecode(x: str) -> bool:
    if pd.isna(x):
        return False
    x = str(x).strip()
    return bool(re.fullmatch(r"\d+|\d+\.\d", x))


def normalize_group_value(x: Any, group_col: str) -> str:
    s = safe_str(x)
    if s == "":
        return f"UNKNOWN_{group_col.upper()}"
    return s


# =============================================================================
# Build phecode table
# =============================================================================

def build_phecode_text(row: pd.Series) -> str:
    phecode = safe_str(row.get("phecode", ""))
    phenotype = safe_str(row.get("phenotype", ""))
    category = safe_str(row.get("category", ""))

    return (
        f"Phecode: {phecode}. "
        f"Phenotype: {phenotype}. "
        f"Category: {category}."
    )


def build_phecode_table(phe_path: str) -> pd.DataFrame:
    df = pd.read_csv(phe_path, dtype={"phecode": str})

    df["phecode"] = safe_col(df, "phecode").str.strip()
    df["phenotype"] = safe_col(df, "phenotype").str.strip()
    df["category"] = safe_col(df, "category").str.strip()

    df = df[df["phecode"].apply(keep_target_phecode)].copy()
    df["phecode"] = df["phecode"].apply(normalize_phecode_str)

    if TARGET_PHECODES is not None:
        target_norm = {normalize_phecode_str(x) for x in TARGET_PHECODES}
        df = df[df["phecode"].isin(target_norm)].copy()

    df["phe_text"] = df.apply(build_phecode_text, axis=1)
    df = df.reset_index(drop=True)

    if DEBUG_N_PHECODES is not None:
        df = df.head(DEBUG_N_PHECODES).copy()

    return df


# =============================================================================
# Build variable table
# =============================================================================

def build_variable_text(row: pd.Series) -> str:
    accession = safe_str(row.get("Variable accession", ""))
    name = safe_str(row.get("Variable name", ""))
    desc = safe_str(row.get("Variable description", ""))
    vtype = safe_str(row.get("Type", ""))
    study = safe_str(row.get("Study", ""))
    database = safe_str(row.get("Database", ""))
    dataset_name = safe_str(row.get("Dataset name", ""))
    table_name = safe_str(row.get("Table name", ""))
    visit = safe_str(row.get("Visit", ""))
    visit_name = safe_str(row.get("Visit name", ""))
    collection_event = safe_str(row.get("Collection event", ""))

    return (
        f"Variable accession: {accession}. "
        f"Variable name: {name}. "
        f"Variable description: {desc}. "
        f"Variable type: {vtype}. "
        f"Study accession: {study}. "
        f"Database/Cohort: {database}. "
        f"Dataset name: {dataset_name}. "
        f"Table name: {table_name}. "
        f"Visit: {visit}. "
        f"Visit name: {visit_name}. "
        f"Collection event: {collection_event}."
    )


def build_variable_table(var_path: str) -> pd.DataFrame:
    df = pd.read_csv(var_path)

    needed_cols = [
        "Variable accession",
        "Variable name",
        "Variable description",
        "Type",
        "Dataset accession",
        "Dataset name",
        "Study",
        "Database",
        "Form name",
        "Table name",
        "Visit",
        "Visit name",
        "Collection event",
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    for col in needed_cols:
        df[col] = safe_col(df, col).str.strip()

    if CONSTRAINT_GROUP_COL not in df.columns:
        raise ValueError(
            f"CONSTRAINT_GROUP_COL={CONSTRAINT_GROUP_COL!r} does not exist in variable table. "
            f"Available columns: {list(df.columns)}"
        )

    df[CONSTRAINT_GROUP_COL] = df[CONSTRAINT_GROUP_COL].apply(
        lambda x: normalize_group_value(x, CONSTRAINT_GROUP_COL)
    )

    df["var_text"] = df.apply(build_variable_text, axis=1)

    return df.reset_index(drop=True)


# =============================================================================
# Embedding
# =============================================================================

def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


# =============================================================================
# Candidate selection
# =============================================================================

def compute_delta(
    scores: np.ndarray,
    method: str = "mad",
    scale: float = 3.0,
) -> float:
    if len(scores) == 0:
        return 0.0

    if method == "sd":
        return float(scale * np.std(scores))

    if method == "mad":
        med = np.median(scores)
        mad = np.median(np.abs(scores - med))
        return float(scale * mad)

    raise ValueError(f"Unsupported DELTA_METHOD: {method}")


def select_candidates_adaptive(
    scores: np.ndarray,
    min_candidates: int,
    max_candidates: int,
    delta_method: str = "mad",
    delta_scale: float = 3.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Select candidates from one constraint group score vector.

    This is called separately within each phecode x constraint group.
    Therefore min/max constraints are not global per phecode.
    """

    if len(scores) == 0:
        meta = {
            "max_score": np.nan,
            "delta": np.nan,
            "threshold": np.nan,
            "n_pool": 0,
            "n_above_threshold": 0,
            "n_selected": 0,
            "min_padded": False,
            "max_capped": False,
        }
        return np.array([], dtype=int), meta

    order = np.argsort(-scores)
    max_score = float(scores[order[0]])

    delta = compute_delta(scores, method=delta_method, scale=delta_scale)
    threshold = max_score - delta

    keep_idx = np.where(scores >= threshold)[0]
    keep_idx = keep_idx[np.argsort(-scores[keep_idx])]

    n_above_threshold = len(keep_idx)
    min_padded = False
    max_capped = False

    if len(keep_idx) < min_candidates:
        keep_idx = order[:min(min_candidates, len(order))]
        min_padded = True

    if len(keep_idx) > max_candidates:
        keep_idx = keep_idx[:max_candidates]
        max_capped = True

    meta = {
        "max_score": max_score,
        "delta": float(delta),
        "threshold": float(threshold),
        "n_pool": int(len(scores)),
        "n_above_threshold": int(n_above_threshold),
        "n_selected": int(len(keep_idx)),
        "min_padded": bool(min_padded),
        "max_capped": bool(max_capped),
    }

    return keep_idx, meta


def build_constraint_group_index_map(
    df_var: pd.DataFrame,
    group_col: str,
) -> Dict[str, np.ndarray]:
    """
    Build mapping:
        constraint_group -> variable row indices

    Recommended:
        group_col = "Database"

    Alternative:
        group_col = "Study"
    """

    if group_col not in df_var.columns:
        raise ValueError(f"Group column {group_col!r} not found in df_var.")

    group_index_map: Dict[str, np.ndarray] = {}

    for group_value, g in df_var.groupby(group_col, dropna=False, sort=True):
        group_name = normalize_group_value(group_value, group_col)
        group_index_map[group_name] = g.index.to_numpy(dtype=int)

    return group_index_map


def retrieve_adaptive_threshold_by_group(
    df_phe: pd.DataFrame,
    df_var: pd.DataFrame,
    phe_emb: np.ndarray,
    var_emb: np.ndarray,
    group_col: str = "Database",
    min_candidates_per_group: int = 10,
    max_candidates_per_group: int = 100,
    delta_method: str = "mad",
    delta_scale: float = 3.0,
) -> pd.DataFrame:
    """
    New Step 1 retrieval.

    For each phecode:
        1. Compute similarity against all variables.
        2. Split variables by group_col, default Database.
        3. Within each phecode x group:
            - apply adaptive threshold
            - enforce min_candidates_per_group
            - enforce max_candidates_per_group
        4. Combine selected candidates from all groups.
        5. Assign global rank within phecode after combining.

    This satisfies the requirement:
        Apply minimum and maximum candidate constraints by study/cohort.
    """

    results = []
    group_index_map = build_constraint_group_index_map(df_var, group_col=group_col)

    for i, phe_row in df_phe.iterrows():
        scores = var_emb @ phe_emb[i]
        scores = scores.astype(np.float32, copy=False)

        global_order = np.argsort(-scores)
        global_max_score = float(scores[global_order[0]])

        phe_records = []

        for group_name, group_indices in group_index_map.items():
            if len(group_indices) == 0:
                continue

            group_scores = scores[group_indices]

            keep_local_idx, meta = select_candidates_adaptive(
                scores=group_scores,
                min_candidates=min_candidates_per_group,
                max_candidates=max_candidates_per_group,
                delta_method=delta_method,
                delta_scale=delta_scale,
            )

            if len(keep_local_idx) == 0:
                continue

            keep_global_idx = group_indices[keep_local_idx]
            keep_global_idx = keep_global_idx[np.argsort(-scores[keep_global_idx])]

            for rank_within_group, j in enumerate(keep_global_idx, start=1):
                var_row = df_var.iloc[j]

                phe_records.append(
                    {
                        "phecode": phe_row["phecode"],
                        "phenotype": phe_row["phenotype"],
                        "category": phe_row["category"],

                        "Variable accession": var_row["Variable accession"],
                        "Variable name": var_row["Variable name"],
                        "Variable description": var_row["Variable description"],
                        "Type": var_row["Type"],
                        "Study": var_row["Study"],
                        "Database": var_row["Database"],
                        "Dataset accession": var_row["Dataset accession"],
                        "Dataset name": var_row["Dataset name"],
                        "Form name": var_row["Form name"],
                        "Table name": var_row["Table name"],
                        "Visit": var_row["Visit"],
                        "Visit name": var_row["Visit name"],
                        "Collection event": var_row["Collection event"],

                        "cosine_similarity": float(scores[j]),

                        # Global rank within phecode, filled after combining groups.
                        "step1_rank": None,

                        # Rank within phecode x constraint group.
                        "step1_rank_within_constraint_group": rank_within_group,

                        "step1_pass": 1,
                        "step1_label": "not_hard_negative",

                        # Backward-compatible global max.
                        "max_similarity_for_phecode": global_max_score,

                        # New group-level threshold metadata.
                        "constraint_scope": f"phecode_{group_col}",
                        "constraint_group_col": group_col,
                        "constraint_group": group_name,

                        "max_similarity_for_phecode_group": meta["max_score"],
                        "delta": meta["delta"],
                        "adaptive_threshold": meta["threshold"],
                        "delta_method": delta_method,
                        "delta_scale": delta_scale,

                        "min_candidates_per_group": min_candidates_per_group,
                        "max_candidates_per_group": max_candidates_per_group,
                        "n_variables_in_group_pool": meta["n_pool"],
                        "n_above_threshold_in_group": meta["n_above_threshold"],
                        "n_selected_in_group": meta["n_selected"],
                        "group_min_padded": int(meta["min_padded"]),
                        "group_max_capped": int(meta["max_capped"]),
                    }
                )

        # Assign global rank within this phecode after combining all groups.
        phe_records = sorted(
            phe_records,
            key=lambda x: (
                -x["cosine_similarity"],
                x["constraint_group"],
                x["step1_rank_within_constraint_group"],
            ),
        )

        for global_rank, rec in enumerate(phe_records, start=1):
            rec["step1_rank"] = global_rank

        results.extend(phe_records)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(df_phe)} phecodes")

    return pd.DataFrame(results)


# =============================================================================
# Summary
# =============================================================================

def summarize_candidate_counts(
    df_phe: pd.DataFrame,
    df_out: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summary per phecode after by-group constraints.
    """

    cnt = df_out.groupby("phecode").size().reset_index(name="n_candidates")

    group_stats = (
        df_out.groupby(["phecode", "constraint_group"])
        .size()
        .reset_index(name="n_candidates_in_group")
        .groupby("phecode", as_index=False)
        .agg(
            n_constraint_groups_with_candidates=("constraint_group", "nunique"),
            min_candidates_in_one_group=("n_candidates_in_group", "min"),
            median_candidates_in_one_group=("n_candidates_in_group", "median"),
            max_candidates_in_one_group=("n_candidates_in_group", "max"),
        )
    )

    meta = (
        df_out.groupby("phecode", as_index=False)
        .agg(
            max_similarity_for_phecode=("max_similarity_for_phecode", "first"),
            constraint_scope=("constraint_scope", "first"),
            constraint_group_col=("constraint_group_col", "first"),
            min_candidates_per_group=("min_candidates_per_group", "first"),
            max_candidates_per_group=("max_candidates_per_group", "first"),
            n_group_min_padded=("group_min_padded", "sum"),
            n_group_max_capped=("group_max_capped", "sum"),
        )
    )

    cnt = df_phe[["phecode", "phenotype", "category"]].merge(
        cnt,
        on="phecode",
        how="left",
    )

    cnt = cnt.merge(group_stats, on="phecode", how="left")
    cnt = cnt.merge(meta, on="phecode", how="left")

    cnt["n_candidates"] = cnt["n_candidates"].fillna(0).astype(int)
    cnt["n_constraint_groups_with_candidates"] = (
        cnt["n_constraint_groups_with_candidates"]
        .fillna(0)
        .astype(int)
    )

    return cnt


def summarize_candidate_counts_by_group(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Summary per phecode x constraint group.
    """

    if df_out.empty:
        return pd.DataFrame()

    group_summary = (
        df_out.groupby(
            [
                "phecode",
                "phenotype",
                "category",
                "constraint_group_col",
                "constraint_group",
            ],
            as_index=False,
        )
        .agg(
            n_candidates=("phecode", "size"),
            max_similarity_for_phecode_group=("max_similarity_for_phecode_group", "first"),
            adaptive_threshold=("adaptive_threshold", "first"),
            delta=("delta", "first"),
            n_variables_in_group_pool=("n_variables_in_group_pool", "first"),
            n_above_threshold_in_group=("n_above_threshold_in_group", "first"),
            n_selected_in_group=("n_selected_in_group", "first"),
            group_min_padded=("group_min_padded", "first"),
            group_max_capped=("group_max_capped", "first"),
            min_candidates_per_group=("min_candidates_per_group", "first"),
            max_candidates_per_group=("max_candidates_per_group", "first"),
        )
    )

    return group_summary


def print_distribution_stats(
    cnt: pd.DataFrame,
    group_cnt: pd.DataFrame,
) -> None:
    x = cnt["n_candidates"]

    print("\n" + "=" * 100)
    print("Candidate count summary per phecode")
    print("=" * 100)
    print(x.describe())

    print("\nQuantiles per phecode:")
    for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
        print(f"{q:>5.2f}: {x.quantile(q):.2f}")

    print(f"\nPhecodes with >20 candidates: {(x > 20).sum()}")
    print(f"Phecodes with >50 candidates: {(x > 50).sum()}")
    print(f"Phecodes with >100 candidates: {(x > 100).sum()}")
    print(f"Phecodes with >200 candidates: {(x > 200).sum()}")

    print("\nTop 20 largest candidate counts per phecode:")
    print(
        cnt.sort_values(["n_candidates", "phecode"], ascending=[False, True])
        .head(20)
        .to_string(index=False)
    )

    if not group_cnt.empty:
        y = group_cnt["n_candidates"]

        print("\n" + "=" * 100)
        print("Candidate count summary per phecode x constraint group")
        print("=" * 100)
        print(y.describe())

        print("\nQuantiles per phecode x constraint group:")
        for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
            print(f"{q:>5.2f}: {y.quantile(q):.2f}")

        print(
            f"\nPhecode-group rows with exactly {MIN_CANDIDATES_PER_GROUP} candidates: "
            f"{(y == MIN_CANDIDATES_PER_GROUP).sum()}"
        )
        print(
            f"Phecode-group rows with >= {MAX_CANDIDATES_PER_GROUP} candidates: "
            f"{(y >= MAX_CANDIDATES_PER_GROUP).sum()}"
        )

        print("\nTop 20 largest candidate counts per phecode x constraint group:")
        print(
            group_cnt.sort_values(
                ["n_candidates", "phecode", "constraint_group"],
                ascending=[False, True, True],
            )
            .head(20)
            .to_string(index=False)
        )


# =============================================================================
# Inspection
# =============================================================================

def inspect_results(
    df_out: pd.DataFrame,
    phecodes: List[str],
    top_n: int = 20,
) -> None:
    print("\n" + "=" * 100)
    print("Sample step1 results")
    print("=" * 100)

    for phe in phecodes:
        phe_norm = normalize_phecode_str(phe)
        sub = (
            df_out[df_out["phecode"] == phe_norm]
            .sort_values("step1_rank")
            .head(top_n)
        )

        if sub.empty:
            print(f"\nPhecode {phe_norm}: no candidates")
            continue

        print(f"\nPhecode {phe_norm}: {sub['phenotype'].iloc[0]}")
        print(
            sub[
                [
                    "step1_rank",
                    "step1_rank_within_constraint_group",
                    "constraint_group",
                    "Variable name",
                    "Variable description",
                    "Study",
                    "Database",
                    "Dataset name",
                    "Table name",
                    "Visit",
                    "cosine_similarity",
                    "adaptive_threshold",
                    "delta",
                    "n_selected_in_group",
                    "group_min_padded",
                    "group_max_capped",
                ]
            ].to_string(index=False)
        )


def inspect_results_by_group(
    df_out: pd.DataFrame,
    phecodes: List[str],
    top_n_per_group: int = 10,
) -> None:
    print("\n" + "=" * 100)
    print("Sample step1 results by constraint group")
    print("=" * 100)

    for phe in phecodes:
        phe_norm = normalize_phecode_str(phe)
        sub = df_out[df_out["phecode"] == phe_norm].copy()

        if sub.empty:
            print(f"\nPhecode {phe_norm}: no candidates")
            continue

        print(f"\nPhecode {phe_norm}: {sub['phenotype'].iloc[0]}")

        for group, g in sub.groupby("constraint_group", sort=True):
            g = g.sort_values("step1_rank_within_constraint_group").head(top_n_per_group)

            print(f"\n  Constraint group: {group}")
            print(
                g[
                    [
                        "step1_rank_within_constraint_group",
                        "step1_rank",
                        "Variable name",
                        "Variable description",
                        "Study",
                        "Database",
                        "Dataset name",
                        "cosine_similarity",
                        "adaptive_threshold",
                    ]
                ].to_string(index=False)
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Loading phecode file...")
    df_phe = build_phecode_table(PHE_PATH)
    print(f"Phecodes kept: {len(df_phe)}")

    print("Loading cohort variable file...")
    df_var = build_variable_table(VAR_PATH)
    print(f"Variables kept after filtering + dedup: {len(df_var)}")

    print("\nStudy accession counts:")
    print(df_var["Study"].value_counts(dropna=False).to_string())

    print(f"\nConstraint group column: {CONSTRAINT_GROUP_COL}")
    print("Constraint groups found:")
    print(df_var[CONSTRAINT_GROUP_COL].value_counts(dropna=False).to_string())

    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding phecodes...")
    phe_emb = encode_texts(
        model,
        df_phe["phe_text"].tolist(),
        batch_size=BATCH_SIZE,
    )

    print("Encoding variables...")
    var_emb = encode_texts(
        model,
        df_var["var_text"].tolist(),
        batch_size=BATCH_SIZE,
    )

    print("\nRunning step1 adaptive threshold filtering by constraint group...")
    print(f"CONSTRAINT_GROUP_COL = {CONSTRAINT_GROUP_COL}")
    print(f"MIN_CANDIDATES_PER_GROUP = {MIN_CANDIDATES_PER_GROUP}")
    print(f"MAX_CANDIDATES_PER_GROUP = {MAX_CANDIDATES_PER_GROUP}")
    print(f"DELTA_METHOD = {DELTA_METHOD}")
    print(f"DELTA_SCALE = {DELTA_SCALE}")

    df_out = retrieve_adaptive_threshold_by_group(
        df_phe=df_phe,
        df_var=df_var,
        phe_emb=phe_emb,
        var_emb=var_emb,
        group_col=CONSTRAINT_GROUP_COL,
        min_candidates_per_group=MIN_CANDIDATES_PER_GROUP,
        max_candidates_per_group=MAX_CANDIDATES_PER_GROUP,
        delta_method=DELTA_METHOD,
        delta_scale=DELTA_SCALE,
    )

    cnt = summarize_candidate_counts(df_phe, df_out)
    group_cnt = summarize_candidate_counts_by_group(df_out)

    out_dir = os.path.dirname(OUT_CAND_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(OUT_CAND_PATH, index=False)
    cnt.to_csv(OUT_COUNT_PATH, index=False)
    group_cnt.to_csv(OUT_GROUP_COUNT_PATH, index=False)

    print(f"\nSaved candidate-level output to: {OUT_CAND_PATH}")
    print(f"Saved phecode count summary to: {OUT_COUNT_PATH}")
    print(f"Saved phecode-group count summary to: {OUT_GROUP_COUNT_PATH}")

    print_distribution_stats(cnt, group_cnt)
    inspect_results(df_out, INSPECT_PHECODES, top_n=20)
    inspect_results_by_group(df_out, INSPECT_PHECODES, top_n_per_group=10)


if __name__ == "__main__":
    main()