# SONNET Silver-Standard Label Pipeline

This folder contains the two core scripts used to generate silver-standard phenotype-variable labels for SONNET.

The pipeline has two stages:

1. **Step 1: Broad embedding-based candidate filtering**
   - Script: `step1.py`
   - Retrieves candidate cohort variables for each phecode using embedding similarity.
   - The current implementation uses `BAAI/bge-base-en-v1.5` and applies adaptive thresholding within each `phecode × Database` group.

2. **Step 2: Stringent LLM-based clinical matching**
   - Script: `step2.py`
   - Classifies the Step 1 candidate variables into clinical relationship categories using an LLM.
   - Produces clinical labels such as `exact_same`, `directly_related`, `clinically_relevant`, and `unrelated`.

## Expected input files

By default, the scripts expect the following input paths:

```text
../data/phecode_definitions1.2.csv
../data/whi_mesa_v2.csv
```

`step1.py` expects:

- a phecode definition file with columns such as `phecode`, `phenotype`, and `category`
- a variable metadata file with columns such as `Variable accession`, `Variable name`, `Variable description`, `Study`, `Database`, `Dataset name`, and visit/table metadata

`step2.py` expects the Step 1 output:

```text
../output/phecode_variable_step1_bge_adaptive_threshold_by_database.csv
```

## Output files

Step 1 writes:

```text
../output/phecode_variable_step1_bge_adaptive_threshold_by_database.csv
../output/phecode_variable_step1_bge_adaptive_threshold_by_database_count_summary.csv
../output/phecode_variable_step1_bge_adaptive_threshold_by_database_group_summary.csv
```

Step 2 writes:

```text
../output/phecode_variable_step2_clinical_roles.csv
../output/phecode_variable_step2_clinical_roles_ranked.csv
../output/phecode_variable_step2_clinical_roles_checkpoint.csv
```

The ranked Step 2 output is the main silver-standard label file used for downstream review and visualization.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Required environment variables for Step 2

Before running `step2.py`, set the OpenAI-compatible endpoint variables:

```bash
export OPENAI_ENDPOINT="<your_endpoint>"
export OPENAI_API_KEY="<your_api_key>"
export OPENAI_DEPLOYMENT="<your_deployment_name>"
```

Do not commit or share real API keys.

## How to run

Run Step 1 first:

```bash
python step1.py
```

Then run Step 2:

```bash
python step2.py
```

## Step 1 summary

`step1.py` builds text descriptions for each phecode and each cohort variable, embeds both with `BAAI/bge-base-en-v1.5`, computes cosine similarity, and selects candidate variables.

Candidate selection is intentionally broad. It is designed to preserve recall before the stricter LLM matching step.

The current configuration applies candidate constraints within each `phecode × Database` group:

```text
MIN_CANDIDATES_PER_GROUP = 10
MAX_CANDIDATES_PER_GROUP = 100
DELTA_METHOD = "mad"
DELTA_SCALE = 3.0
```

## Step 2 summary

`step2.py` takes the Step 1 candidates and asks the LLM to classify each candidate variable according to:

- `step2_clinical_category`
- `step2_relation_level`
- `step2_keep`
- `step2_priority`
- `step2_reason`

The relation levels are:

```text
exact_same
strictly/directly_related
clinically_relevant
unrelated
```

In the current script, the valid stored relation level is `directly_related`, not `strictly_related`.

Priority is interpreted as:

```text
1 = exact same diagnosis variable
2 = directly related core clinical evidence
3 = clinically relevant contextual variable
4 = unrelated
```

## Important note on failed LLM rows

If an LLM batch fails after all retries and `FAIL_ON_LLM_ERROR = False`, the script continues and assigns fallback labels:

```text
step2_status = llm_failed
step2_relation_level = unrelated
step2_keep = False
```

Rows with `step2_status == "llm_failed"` should not be interpreted as true negative labels. They indicate failed classification and should be excluded or rerun before final analysis.

## Checkpointing

Step 2 writes a checkpoint file:

```text
../output/phecode_variable_step2_clinical_roles_checkpoint.csv
```

If `RESUME_FROM_CHECKPOINT = True`, rerunning `step2.py` will reuse already processed rows and continue from the checkpoint.

## Suggested clean package contents

For sharing the core silver-label generation pipeline, include:

```text
step1.py
step2.py
README.md
requirements.txt
```

Do not include raw cohort data, real output files, API keys, or local credential/configuration files.
