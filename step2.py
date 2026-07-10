# step2.py

import os
import re
import json
import time
import hashlib
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI


# =============================================================================
# Config
# =============================================================================

# Recommended input if you used the new Step 1 by Database.
INPUT_CSV = "../output/phecode_variable_step1_bge_adaptive_threshold_by_database.csv"

# If you are still using the old Step 1, use this instead:
# INPUT_CSV = "../output/phecode_variable_step1_bge_adaptive_threshold.csv"

OUTPUT_CSV = "../output/phecode_variable_step2_clinical_roles.csv"
OUTPUT_RANKED_CSV = "../output/phecode_variable_step2_clinical_roles_ranked.csv"

CHECKPOINT_CSV = "../output/phecode_variable_step2_clinical_roles_checkpoint.csv"

RESUME_FROM_CHECKPOINT = True

# If True, one failed LLM batch stops the whole script.
# If False, failed batches are marked llm_failed and the script continues.
FAIL_ON_LLM_ERROR = False

DEBUG_N_PHECODES = None
# DEBUG_N_PHECODES = 3

TARGET_PHECODES = None
# TARGET_PHECODES = ["272", "272.1", "250", "401"]

INSPECT_PHECODES = ["272", "272.1", "250", "401", "296"]

# If Step 1 already capped candidates, keep this as None.
MAX_CANDIDATES_PER_PHECODE = None

# Smaller batch reduces the chance of server-side 500 errors.
BATCH_MAX_VARS = 25

MAX_COMPLETION_TOKENS = 8000
MAX_RETRIES = 5

SLEEP_BETWEEN_CALLS = 0.3
BASE_RETRY_SLEEP = 5

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "gpt-5.2-chat")


VALID_CLINICAL_CATEGORIES = {
    "diagnosis",
    "lab",
    "medication",
    "procedure",
    "vital",
    "imaging",
    "symptom",
    "lifestyle",
    "survey",
    "demographic",
    "administrative",
    "other",
    "unrelated",
}

VALID_RELATION_LEVELS = {
    "exact_same",
    "directly_related",
    "clinically_relevant",
    "unrelated",
}


CATEGORY_RANK = {
    "diagnosis": 1,
    "lab": 2,
    "medication": 3,
    "procedure": 4,
    "vital": 5,
    "imaging": 6,
    "symptom": 7,
    "lifestyle": 8,
    "survey": 9,
    "demographic": 10,
    "other": 50,
    "administrative": 98,
    "unrelated": 99,
}

RELATION_RANK = {
    "exact_same": 1,
    "directly_related": 2,
    "clinically_relevant": 3,
    "unrelated": 99,
}


# =============================================================================
# Client
# =============================================================================

def make_client() -> OpenAI:
    if not OPENAI_ENDPOINT:
        raise ValueError("Missing environment variable: OPENAI_ENDPOINT")

    if not OPENAI_API_KEY:
        raise ValueError("Missing environment variable: OPENAI_API_KEY")

    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_ENDPOINT,
    )


client = make_client()


# =============================================================================
# Utilities
# =============================================================================

def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_space(x: str) -> str:
    return re.sub(r"\s+", " ", x).strip()


def normalize_for_key(x: Any) -> str:
    x = clean_text(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = normalize_space(x)
    return x


def conservative_base_name(name: Any) -> str:
    """
    Conservative variable-name normalization.

    This is only used to provide grouping context to the LLM.
    It is not used to automatically promote labels.
    """

    x = clean_text(name).lower()
    x = re.sub(r"[^a-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")

    patterns = [
        r"(_?visit_?\d+)$",
        r"(_?v_?\d+)$",
        r"(_?fu_?\d+)$",
        r"(_?followup_?\d+)$",
        r"(_?year_?\d+)$",
        r"(_?yr_?\d+)$",
        r"(_?month_?\d+)$",
        r"(_?mo_?\d+)$",
        r"(_?day_?\d+)$",
    ]

    for pat in patterns:
        x = re.sub(pat, "", x)

    return x.strip("_")


def short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def add_context_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add context_group_id for LLM reference.

    The group is context only.
    It does not cause automatic group-level promotion.
    """

    df = df.copy()

    df["_norm_study"] = df.get("Study", "").apply(normalize_for_key)
    df["_norm_dataset"] = df.get("Dataset name", "").apply(normalize_for_key)
    df["_norm_var_name"] = df.get("Variable name", "").apply(normalize_for_key)
    df["_base_name"] = df.get("Variable name", "").apply(conservative_base_name)
    df["_norm_desc"] = df.get("Variable description", "").apply(normalize_for_key)

    group_ids = []

    for _, row in df.iterrows():
        study = row["_norm_study"]
        dataset = row["_norm_dataset"]
        base = row["_base_name"]
        desc = row["_norm_desc"]

        if study and base:
            key = f"same_study_base::{study}::{base}"
        elif dataset and base:
            key = f"same_dataset_base::{dataset}::{base}"
        elif len(desc) >= 12:
            key = f"same_description::{desc}"
        else:
            key = f"singleton::{row['_row_id']}"

        group_ids.append("G_" + short_hash(key))

    df["context_group_id"] = group_ids

    return df


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Parse LLM JSON robustly.
    """

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


# =============================================================================
# Result validation and fallback
# =============================================================================

def validate_one_result(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate one LLM result row.
    """

    row_id = str(item.get("row_id", "")).strip()

    clinical_category = str(
        item.get("clinical_category", "unrelated")
    ).strip().lower()

    relation_level = str(
        item.get("relation_level", "unrelated")
    ).strip().lower()

    if clinical_category not in VALID_CLINICAL_CATEGORIES:
        clinical_category = "other"

    if relation_level not in VALID_RELATION_LEVELS:
        relation_level = "unrelated"

    if relation_level == "unrelated":
        clinical_category = "unrelated"

    if relation_level == "exact_same":
        priority = 1
        keep = True
    elif relation_level == "directly_related":
        priority = 2
        keep = True
    elif relation_level == "clinically_relevant":
        priority = 3
        keep = True
    else:
        priority = 4
        keep = False

    reason = normalize_space(str(item.get("reason", "")))

    return {
        "_row_id": row_id,
        "step2_clinical_category": clinical_category,
        "step2_relation_level": relation_level,
        "step2_keep": keep,
        "step2_priority": priority,
        "step2_reason": reason,
        "step2_status": "ok",
        "step2_error": "",
    }


def make_missing_output_fallback(row_id: str) -> Dict[str, Any]:
    return {
        "_row_id": str(row_id),
        "step2_clinical_category": "unrelated",
        "step2_relation_level": "unrelated",
        "step2_keep": False,
        "step2_priority": 4,
        "step2_reason": "Missing from LLM output; assigned unrelated fallback.",
        "step2_status": "missing_from_llm_output",
        "step2_error": "",
    }


def make_llm_failure_fallback(
    batch_df: pd.DataFrame,
    error_message: str,
) -> pd.DataFrame:
    """
    If the LLM fails after all retries, return fallback rows instead of crashing.
    """

    rows = []

    for _, row in batch_df.iterrows():
        rows.append(
            {
                "_row_id": str(row["_row_id"]),
                "step2_clinical_category": "unrelated",
                "step2_relation_level": "unrelated",
                "step2_keep": False,
                "step2_priority": 4,
                "step2_reason": (
                    "LLM failed after all retries; fallback assigned. "
                    f"Error: {error_message}"
                ),
                "step2_status": "llm_failed",
                "step2_error": error_message,
            }
        )

    return pd.DataFrame(rows)


def make_step2_label(row: pd.Series) -> str:
    cat = row["step2_clinical_category"]
    rel = row["step2_relation_level"]

    if rel == "exact_same" and cat == "diagnosis":
        return "exact_diagnosis"

    if rel == "exact_same":
        return f"exact_{cat}"

    if rel == "directly_related":
        return f"related_{cat}"

    if rel == "clinically_relevant":
        return f"relevant_{cat}"

    return "unrelated"


# =============================================================================
# Prompt
# =============================================================================

def build_prompt(
    phecode: str,
    phenotype: str,
    phecode_category: str,
    variables: List[Dict[str, Any]],
) -> str:
    variables_json = json.dumps(variables, ensure_ascii=False, indent=2)

    prompt = f"""
You are a careful biomedical variable classifier.

The target is a DIAGNOSIS phenotype.

Target diagnosis:
- phecode: {phecode}
- phenotype: {phenotype}
- phecode category: {phecode_category}

You will receive candidate cohort variables retrieved by embedding similarity.

For each candidate variable, classify:
1. clinical_category
2. relation_level
3. keep
4. priority
5. short reason

Clinical categories:
- diagnosis:
  disease status, diagnosis history, ICD-based diagnosis, self-reported disease diagnosis,
  physician diagnosis, clinical disease indicator.

- lab:
  laboratory measurement, biomarker, blood test, urine test, chemistry test,
  molecular assay, measured biological quantity.
  Examples: LDL, HDL, glucose, HbA1c, CRP, IL-6, creatinine.

- medication:
  drug exposure, prescription, medication use, treatment drug class.
  Examples: statin, insulin, antihypertensive medication.

- procedure:
  surgery, intervention, treatment procedure, clinical procedure.

- vital:
  blood pressure, BMI, height, weight, heart rate, waist circumference.

- imaging:
  imaging result, imaging finding, scan-derived measurement.

- symptom:
  symptom, sign, complaint, pain, physical manifestation.

- lifestyle:
  smoking, alcohol, diet, exercise, sleep, physical activity.

- survey:
  questionnaire response not clearly diagnosis/lab/medication.

- demographic:
  age, sex, race, ethnicity, education, income.

- administrative:
  ID, accession ID, visit number, site, date, form name, table metadata,
  sample ID, record number, data collection metadata.

- other:
  medically meaningful but not fitting above.

- unrelated:
  no meaningful clinical relationship to the target diagnosis.

Relation levels:
- exact_same:
  The variable directly records the target diagnosis itself.
  Example:
  Target = hyperlipidemia.
  Variable = history of hyperlipidemia, high cholesterol diagnosis,
  self-reported high cholesterol, physician-diagnosed hyperlipidemia.

- directly_related:
  The variable is not the diagnosis itself, but is a standard clinical measurement,
  medication, procedure, or core disease-defining variable for this diagnosis.
  Example:
  Target = hyperlipidemia.
  Variables = LDL, HDL, total cholesterol, triglycerides, statin use,
  lipid-lowering medication.

- clinically_relevant:
  The variable is clinically associated with the diagnosis, risk, complication,
  or broad context, but should not be treated as disease-defining evidence.
  Example:
  Target = hyperlipidemia.
  Variables = BMI, diabetes, smoking, diet, cardiovascular risk factors.

- unrelated:
  No meaningful clinical relationship.

Priority:
- 1: exact same diagnosis variable
- 2: directly related lab / medication / procedure / core clinical evidence
- 3: clinically relevant contextual variable
- 4: unrelated

Important decision rules:
- Do NOT mark a lab as exact_same just because it helps define or monitor the disease.
  LDL for hyperlipidemia is lab + directly_related, not exact_same.
- Do NOT mark a medication as exact_same.
  Statin for hyperlipidemia is medication + directly_related.
- Do NOT over-include broad risk factors.
  Keep broad variables only if they are clearly clinically useful for the target diagnosis.
- Administrative variables are usually unrelated.
- context_group_id is only grouping context. Do not assign the same label to all variables
  in a group automatically.
- Use variable_description first, then variable_name, dataset_name, table_name, study,
  database, and type as supporting context.
- Be conservative when evidence is weak.

Return strict JSON only.
Do not include markdown.
Do not include comments outside JSON.

Required output schema:
{{
  "results": [
    {{
      "row_id": "string",
      "clinical_category": "diagnosis | lab | medication | procedure | vital | imaging | symptom | lifestyle | survey | demographic | administrative | other | unrelated",
      "relation_level": "exact_same | directly_related | clinically_relevant | unrelated",
      "keep": true,
      "priority": 1,
      "reason": "short reason"
    }}
  ]
}}

Candidate variables:
{variables_json}
"""
    return prompt.strip()


# =============================================================================
# LLM call
# =============================================================================

def call_llm_for_batch(
    phecode: str,
    phenotype: str,
    phecode_category: str,
    batch_df: pd.DataFrame,
) -> pd.DataFrame:
    variables = []

    for _, row in batch_df.iterrows():
        variables.append(
            {
                "row_id": str(row["_row_id"]),
                "context_group_id": clean_text(row.get("context_group_id", "")),
                "variable_accession": clean_text(row.get("Variable accession", "")),
                "variable_name": clean_text(row.get("Variable name", "")),
                "variable_description": clean_text(row.get("Variable description", "")),
                "type": clean_text(row.get("Type", "")),
                "study": clean_text(row.get("Study", "")),
                "database": clean_text(row.get("Database", "")),
                "dataset_accession": clean_text(row.get("Dataset accession", "")),
                "dataset_name": clean_text(row.get("Dataset name", "")),
                "form_name": clean_text(row.get("Form name", "")),
                "table_name": clean_text(row.get("Table name", "")),
                "visit": clean_text(row.get("Visit", "")),
                "visit_name": clean_text(row.get("Visit name", "")),
                "collection_event": clean_text(row.get("Collection event", "")),
                "cosine_similarity": (
                    float(row["cosine_similarity"])
                    if "cosine_similarity" in row and pd.notna(row["cosine_similarity"])
                    else None
                ),
                "step1_rank": (
                    int(row["step1_rank"])
                    if "step1_rank" in row and pd.notna(row["step1_rank"])
                    else None
                ),
                "step1_label": clean_text(row.get("step1_label", "")),
            }
        )

    prompt = build_prompt(
        phecode=phecode,
        phenotype=phenotype,
        phecode_category=phecode_category,
        variables=variables,
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a biomedical variable classification assistant. "
                            "Return strict valid JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )

            content = response.choices[0].message.content
            parsed = safe_json_loads(content)

            raw_results = parsed.get("results", [])
            if not isinstance(raw_results, list):
                raise ValueError("JSON field 'results' is not a list.")

            results = [validate_one_result(x) for x in raw_results]
            result_df = pd.DataFrame(results)

            expected_ids = set(batch_df["_row_id"].astype(str))
            returned_ids = set(result_df["_row_id"].astype(str))

            missing_ids = expected_ids - returned_ids

            if missing_ids:
                fallback_rows = [
                    make_missing_output_fallback(row_id)
                    for row_id in missing_ids
                ]

                result_df = pd.concat(
                    [result_df, pd.DataFrame(fallback_rows)],
                    ignore_index=True,
                )

            return result_df

        except Exception as e:
            last_error = e
            wait_seconds = BASE_RETRY_SLEEP * attempt

            print(
                f"[WARN] LLM call failed for phecode={phecode}, "
                f"attempt={attempt}/{MAX_RETRIES}: {e}"
            )
            print(f"       Sleeping {wait_seconds} seconds before retry...")

            time.sleep(wait_seconds)

    error_message = str(last_error)

    if FAIL_ON_LLM_ERROR:
        raise RuntimeError(
            f"LLM failed for phecode={phecode} after {MAX_RETRIES} attempts. "
            f"Last error: {error_message}"
        )

    print(
        f"[ERROR] LLM failed for phecode={phecode} after {MAX_RETRIES} attempts. "
        "Using fallback rows and continuing."
    )

    return make_llm_failure_fallback(
        batch_df=batch_df,
        error_message=error_message,
    )


# =============================================================================
# Checkpoint
# =============================================================================

def append_checkpoint(result_df: pd.DataFrame, checkpoint_csv: str) -> None:
    out_dir = os.path.dirname(checkpoint_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    write_header = not os.path.exists(checkpoint_csv)

    result_df.to_csv(
        checkpoint_csv,
        mode="a",
        header=write_header,
        index=False,
    )


def load_checkpoint(checkpoint_csv: str) -> pd.DataFrame:
    if not os.path.exists(checkpoint_csv):
        return pd.DataFrame()

    ckpt = pd.read_csv(checkpoint_csv, dtype={"_row_id": str})
    ckpt["_row_id"] = ckpt["_row_id"].astype(str)

    ckpt = ckpt.drop_duplicates(subset=["_row_id"], keep="last")

    return ckpt


# =============================================================================
# Main Step 2
# =============================================================================

def load_step1_candidates(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required = [
        "phecode",
        "phenotype",
        "category",
        "Variable accession",
        "Variable name",
        "Variable description",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    optional_cols = [
        "Type",
        "Study",
        "Dataset accession",
        "Dataset name",
        "Database",
        "Form name",
        "Table name",
        "Visit",
        "Visit name",
        "Collection event",
        "cosine_similarity",
        "step1_rank",
        "step1_pass",
        "step1_label",
    ]

    df = ensure_columns(df, optional_cols)

    if "step1_pass" in df.columns:
        if df["step1_pass"].notna().any():
            pass_mask = df["step1_pass"].astype(str).str.lower().isin(
                ["true", "1", "yes"]
            )
            if pass_mask.any():
                df = df.loc[pass_mask].copy()

    if TARGET_PHECODES is not None:
        target = [str(x) for x in TARGET_PHECODES]
        df = df[df["phecode"].astype(str).isin(target)].copy()

    if "cosine_similarity" in df.columns:
        df["cosine_similarity"] = pd.to_numeric(
            df["cosine_similarity"],
            errors="coerce",
        )

    if "step1_rank" in df.columns:
        df["step1_rank"] = pd.to_numeric(
            df["step1_rank"],
            errors="coerce",
        )

    if MAX_CANDIDATES_PER_PHECODE is not None:
        if "cosine_similarity" in df.columns:
            df = (
                df.sort_values(
                    ["phecode", "cosine_similarity"],
                    ascending=[True, False],
                )
                .groupby("phecode", as_index=False)
                .head(MAX_CANDIDATES_PER_PHECODE)
                .copy()
            )
        elif "step1_rank" in df.columns:
            df = (
                df.sort_values(
                    ["phecode", "step1_rank"],
                    ascending=[True, True],
                )
                .groupby("phecode", as_index=False)
                .head(MAX_CANDIDATES_PER_PHECODE)
                .copy()
            )

    df = df.reset_index(drop=True)
    df["_row_id"] = [str(i) for i in range(len(df))]

    df = add_context_groups(df)

    return df


def classify_all_phecodes(df: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    processed_row_ids = set()

    if RESUME_FROM_CHECKPOINT:
        ckpt = load_checkpoint(CHECKPOINT_CSV)
        if not ckpt.empty:
            print(f"Loaded checkpoint: {CHECKPOINT_CSV}")
            print(f"Already processed rows: {len(ckpt)}")

            all_results.append(ckpt)
            processed_row_ids = set(ckpt["_row_id"].astype(str))

    phecode_groups = list(df.groupby("phecode", sort=False))

    if DEBUG_N_PHECODES is not None:
        phecode_groups = phecode_groups[:DEBUG_N_PHECODES]

    print("=" * 80)
    print("Step 2: LLM Clinical Role Classification")
    print("=" * 80)
    print(f"Model: {OPENAI_DEPLOYMENT}")
    print(f"Input rows: {len(df)}")
    print(f"Rows already processed from checkpoint: {len(processed_row_ids)}")
    print(f"Phecodes to process: {len(phecode_groups)}")
    print(f"Batch max vars: {BATCH_MAX_VARS}")
    print("=" * 80)

    for i, (phecode, g) in enumerate(phecode_groups, start=1):
        g = g.copy()

        g = g[~g["_row_id"].astype(str).isin(processed_row_ids)].copy()

        if g.empty:
            print(
                f"[{i}/{len(phecode_groups)}] "
                f"Phecode {phecode}: already processed, skipping."
            )
            continue

        if "cosine_similarity" in g.columns:
            g = g.sort_values("cosine_similarity", ascending=False)
        elif "step1_rank" in g.columns:
            g = g.sort_values("step1_rank", ascending=True)

        phenotype = clean_text(g["phenotype"].iloc[0])
        phecode_category = clean_text(g["category"].iloc[0])

        print(
            f"[{i}/{len(phecode_groups)}] "
            f"Phecode {phecode}: {phenotype} "
            f"({len(g)} remaining candidates)"
        )

        n_batches = (len(g) + BATCH_MAX_VARS - 1) // BATCH_MAX_VARS

        for batch_idx, start in enumerate(range(0, len(g), BATCH_MAX_VARS), start=1):
            end = min(start + BATCH_MAX_VARS, len(g))
            batch_df = g.iloc[start:end].copy()

            batch_df = batch_df[
                ~batch_df["_row_id"].astype(str).isin(processed_row_ids)
            ].copy()

            if batch_df.empty:
                print(
                    f"    Batch {batch_idx}/{n_batches}: already processed, skipping."
                )
                continue

            print(
                f"    Batch {batch_idx}/{n_batches}: "
                f"{len(batch_df)} variables"
            )

            result_df = call_llm_for_batch(
                phecode=str(phecode),
                phenotype=phenotype,
                phecode_category=phecode_category,
                batch_df=batch_df,
            )

            result_df["_row_id"] = result_df["_row_id"].astype(str)

            append_checkpoint(result_df, CHECKPOINT_CSV)

            all_results.append(result_df)
            processed_row_ids.update(result_df["_row_id"].astype(str))

            time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_results:
        raise ValueError("No LLM results generated.")

    step2_results = pd.concat(all_results, ignore_index=True)
    step2_results["_row_id"] = step2_results["_row_id"].astype(str)

    step2_results = step2_results.drop_duplicates(
        subset=["_row_id"],
        keep="last",
    )

    return step2_results


def merge_results(df: pd.DataFrame, step2_results: pd.DataFrame) -> pd.DataFrame:
    base_keep_cols = [
        "_row_id",
        "step2_clinical_category",
        "step2_relation_level",
        "step2_keep",
        "step2_priority",
        "step2_reason",
        "step2_status",
        "step2_error",
    ]

    keep_cols = [c for c in base_keep_cols if c in step2_results.columns]

    merged = df.merge(
        step2_results[keep_cols],
        on="_row_id",
        how="left",
    )

    merged["step2_clinical_category"] = merged["step2_clinical_category"].fillna(
        "unrelated"
    )
    merged["step2_relation_level"] = merged["step2_relation_level"].fillna(
        "unrelated"
    )
    merged["step2_keep"] = merged["step2_keep"].fillna(False)
    merged["step2_priority"] = (
        pd.to_numeric(merged["step2_priority"], errors="coerce")
        .fillna(4)
        .astype(int)
    )
    merged["step2_reason"] = merged["step2_reason"].fillna(
        "Missing classification; assigned unrelated fallback."
    )

    if "step2_status" not in merged.columns:
        merged["step2_status"] = "unknown"
    else:
        merged["step2_status"] = merged["step2_status"].fillna("missing")

    if "step2_error" not in merged.columns:
        merged["step2_error"] = ""
    else:
        merged["step2_error"] = merged["step2_error"].fillna("")

    merged["step2_label"] = merged.apply(make_step2_label, axis=1)

    merged["is_exact_diagnosis"] = (
        (merged["step2_clinical_category"] == "diagnosis")
        & (merged["step2_relation_level"] == "exact_same")
    )

    merged["is_related_lab"] = (
        (merged["step2_clinical_category"] == "lab")
        & (
            merged["step2_relation_level"].isin(
                ["directly_related", "clinically_relevant"]
            )
        )
    )

    merged["is_related_medication"] = (
        (merged["step2_clinical_category"] == "medication")
        & (
            merged["step2_relation_level"].isin(
                ["directly_related", "clinically_relevant"]
            )
        )
    )

    merged["relation_rank"] = (
        merged["step2_relation_level"]
        .map(RELATION_RANK)
        .fillna(99)
        .astype(int)
    )

    merged["category_rank"] = (
        merged["step2_clinical_category"]
        .map(CATEGORY_RANK)
        .fillna(99)
        .astype(int)
    )

    if "cosine_similarity" in merged.columns:
        merged["cosine_similarity"] = pd.to_numeric(
            merged["cosine_similarity"],
            errors="coerce",
        ).fillna(-999.0)
    else:
        merged["cosine_similarity"] = -999.0

    return merged


def make_ranked_output(merged: pd.DataFrame) -> pd.DataFrame:
    ranked = merged.copy()

    ranked = ranked.sort_values(
        by=[
            "phecode",
            "step2_priority",
            "relation_rank",
            "category_rank",
            "cosine_similarity",
        ],
        ascending=[True, True, True, True, False],
    )

    ranked["step2_rank"] = ranked.groupby("phecode").cumcount() + 1

    return ranked


def save_inspection_files(ranked: pd.DataFrame) -> None:
    inspect_dir = "../output/step2_inspect"
    os.makedirs(inspect_dir, exist_ok=True)

    for phe in INSPECT_PHECODES:
        sub = ranked[ranked["phecode"].astype(str) == str(phe)].copy()
        if len(sub) == 0:
            continue

        out = os.path.join(inspect_dir, f"phecode_{phe}_clinical_roles.csv")
        sub.to_csv(out, index=False)
        print(f"Saved inspect file: {out}")


def print_summary(ranked: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("Step 2 Summary")
    print("=" * 80)

    print("\nClinical category counts:")
    print(ranked["step2_clinical_category"].value_counts(dropna=False))

    print("\nRelation level counts:")
    print(ranked["step2_relation_level"].value_counts(dropna=False))

    print("\nStep2 label counts:")
    print(ranked["step2_label"].value_counts(dropna=False))

    print("\nKeep counts:")
    print(ranked["step2_keep"].value_counts(dropna=False))

    print("\nStep2 status counts:")
    print(ranked["step2_status"].value_counts(dropna=False))

    exact_count = ranked["is_exact_diagnosis"].sum()
    lab_count = ranked["is_related_lab"].sum()
    med_count = ranked["is_related_medication"].sum()

    print(f"\nExact diagnosis variables: {exact_count}")
    print(f"Related lab variables: {lab_count}")
    print(f"Related medication variables: {med_count}")

    failed = ranked[ranked["step2_status"] == "llm_failed"]
    print(f"\nLLM failed rows: {len(failed)}")

    if len(failed) > 0:
        print("\nFailed phecodes:")
        print(
            failed[["phecode", "phenotype"]]
            .drop_duplicates()
            .head(30)
            .to_string(index=False)
        )

    per_phecode = (
        ranked.groupby("phecode")
        .agg(
            n_candidates=("phecode", "size"),
            n_keep=("step2_keep", "sum"),
            n_exact_diagnosis=("is_exact_diagnosis", "sum"),
            n_related_lab=("is_related_lab", "sum"),
            n_related_medication=("is_related_medication", "sum"),
            n_llm_failed=("step2_status", lambda x: (x == "llm_failed").sum()),
        )
        .reset_index()
    )

    print("\nPer-phecode summary preview:")
    print(per_phecode.head(20))


def main() -> None:
    df = load_step1_candidates(INPUT_CSV)

    step2_results = classify_all_phecodes(df)

    merged = merge_results(df, step2_results)

    internal_cols = [
        "_norm_study",
        "_norm_dataset",
        "_norm_var_name",
        "_base_name",
        "_norm_desc",
    ]

    save_df = merged.drop(columns=[c for c in internal_cols if c in merged.columns])
    save_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved Step 2 output to: {OUTPUT_CSV}")

    ranked = make_ranked_output(merged)
    ranked_save = ranked.drop(columns=[c for c in internal_cols if c in ranked.columns])
    ranked_save.to_csv(OUTPUT_RANKED_CSV, index=False)
    print(f"Saved ranked Step 2 output to: {OUTPUT_RANKED_CSV}")

    save_inspection_files(ranked_save)
    print_summary(ranked_save)


if __name__ == "__main__":
    main()