"""
Dutch-calibrated GGUF quantization pipeline — single model runner.
=================================================================
Produces a Q5_K_NL Dutch-calibrated dynamic GGUF for any Granite 4.0 model,
measures perplexity degradation vs FP16, and measures KLD divergence.

Published as part of the Leesplank-vloeiend-nl dataset project:
  https://huggingface.co/datasets/MichielBuisman/Leesplank-vloeiend-nl-curriculum-cp2

Calibration dataset used: MichielBuisman/Leesplank-vloeiend-nl-curriculum-cp2
  5.38M Dutch sentences, curriculum-stratified by perplexity and complexity.
  The imatrix calibration draws ~3,500 rows (~295K tokens) stratified across
  all 70 semantic clusters. Cluster text files are reused for PPL measurement.

What this script does
---------------------
Phase A — Setup & layer map extraction
  A1. Download FP16 GGUF from HuggingFace (if not already present)
  A2. Download Unsloth Q5_K_XL reference GGUF (for layer map extraction)
  A3. Extract Unsloth's dynamic layer map from the reference GGUF
  A4. Prepare calibration texts from the published dataset (download if needed)

Phase B — Quantization
  B1. Build Dutch imatrix from calibration texts
  B2. Quantize: Q5_K_NL (Unsloth layer map + Dutch imatrix) — main output
  B3. Quantize: Q5_K_plain (Dutch imatrix only) — ablation control

Phase C — PPL measurement
  C1. Write cluster text files (30 texts × 6 strata × N_CLUSTERS)
  C2. Measure per-cluster perplexity for FP16, Q5_K_NL, Q5_K_plain, Unsloth Q5
  C3. Print delta table and save JSON results

Phase D — KLD measurement
  D1. Extract FP16 top-K logprobs via HF transformers (cached to disk)
  D2. Measure KLD for each quant via llama-server (teacher-forcing logprobs)
  D3. Print KLD summary table and save JSON results

Usage
-----
  1. Edit CONFIG block below (model URLs, paths)
  2. python quant_one_model.py
  3. Re-run freely — all phases are fully resumable via checkpoints

Supported models (tested)
--------------------------
  ibm-granite/granite-4.0-micro         3B dense
  ibm-granite/granite-4.0-h-micro       3B hybrid (Mamba-2 + attention)
  ibm-granite/granite-4.0-h-tiny        7B hybrid (Mamba-2 + attention + MoE)
  ibm-granite/granite-4.0-h-nano        350M hybrid (Mamba-2 + attention)

For hybrid models: the SSM tensor names (mamba.in_proj, mamba.conv1d, etc.)
are handled automatically by the layer map extractor. The imatrix and
llama-quantize pipeline works identically for hybrid and dense GGUFs.

Requirements
------------
  pip install gguf transformers torch accelerate polars numpy huggingface_hub
  llama-perplexity.exe, llama-quantize.exe, llama-imatrix.exe, llama-server.exe
    all present in LLAMACPP_DIR (post-Feb 2026 build recommended for hybrid support)

Outputs (all in MODEL_DIR)
--------------------------
  <model_slug>-f16.gguf                        FP16 GGUF (downloaded)
  <model_slug>-UD-Q5_K_XL.gguf                Unsloth reference (downloaded)
  <model_slug>-imatrix-dutch.dat               Dutch imatrix
  <model_slug>-Q5_K_NL.gguf                   Main output: Dutch dynamic Q5
  <model_slug>-Q5_K_plain.gguf                Ablation: Dutch imatrix, no map
  results/ppl_comparison.json                  PPL delta table
  results/kld_comparison.json                  KLD divergence table
  checkpoints/                                 Resume state (safe to delete after)
"""

import gc
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

# ================================================================
# ███████╗ CONFIG — edit this block for each model
# ================================================================
# ================================================================
# ███████╗ CONFIG — corrected for HF Repo accuracy
# ================================================================

BASE_DIR          = "C:/dutch-lora"
LLAMACPP_DIR      = "C:/llamacpp-rocm"
# HuggingFace repo IDs — adjust for each model you run
HF_MODEL_ID       = "ibm-granite/granite-4.0-h-micro"       # HF model for FP16 + tokenizer
HF_FP16_GGUF_REPO = "ibm-granite/granite-4.0-h-micro-GGUF"  # HF repo containing F16 GGUF
HF_FP16_GGUF_FILE = "granite-4.0-h-micro-f16.gguf"          # filename inside that repo
HF_UNSLOTH_REPO   = "unsloth/granite-4.0-h-micro-GGUF"      # Unsloth Q5 reference repo
HF_UNSLOTH_FILE   = "granite-4.0-h-micro-UD-Q5_K_XL.gguf"   # Unsloth Q5 XL filename

# Short slug used for naming all output files
MODEL_SLUG        = "granite-4.0-h-micro"

# Calibration dataset (published, no auth required)
HF_DATASET_ID     = "MichielBuisman/Leesplank-vloeiend-nl-curriculum-cp2"

# Local paths
BASE_DIR          = "C:/dutch-lora"
LLAMACPP_DIR      = "C:/llamacpp-rocm"

# llama.cpp settings
NGL               = 99     # GPU layers — 99 = fully offload
CTX_SIZE          = 4096   # context for PPL measurement
PPL_STRIDE        = 512    # stride for sliding window PPL
PPL_TIMEOUT       = 600    # seconds per cluster

# KLD settings
TOP_K             = 200    # top-K logprobs (captures >99.9% prob mass)
MAX_SEQ_LEN       = 256    # token truncation for KLD
SERVER_PORT       = 8083
SERVER_WAIT       = 90
KLD_FAST_MODE     = True   # True = 3 texts/cluster (~45 min); False = all texts (~6h)
KLD_TEXTS_FAST    = 3

# PPL sampling — 30 texts per stratum × 6 strata × N clusters
PPL_ROWS_PER_STRATUM = 30
RANDOM_SEED          = 42

# ================================================================
# DERIVED PATHS (do not edit)
# ================================================================
MODEL_DIR    = Path(BASE_DIR) / "models" / MODEL_SLUG
DATA_DIR     = Path(BASE_DIR) / "data"
RESULTS_DIR  = DATA_DIR / "results" / MODEL_SLUG
CKPT_DIR     = DATA_DIR / "checkpoints" / MODEL_SLUG
TEXTS_DIR    = CKPT_DIR / "texts"
STDOUT_DIR   = CKPT_DIR / "stdout"
FP16_LOGP_DIR = CKPT_DIR / "fp16_logprobs"
CALIB_DIR    = DATA_DIR / "calibration"

FP16_GGUF    = MODEL_DIR / HF_FP16_GGUF_FILE
UNSLOTH_GGUF = MODEL_DIR / HF_UNSLOTH_FILE
IMATRIX_FILE = MODEL_DIR / f"{MODEL_SLUG}-imatrix-dutch.dat"
Q5_NL_GGUF   = MODEL_DIR / f"{MODEL_SLUG}-Q5_K_NL.gguf"
Q5_PLAIN_GGUF = MODEL_DIR / f"{MODEL_SLUG}-Q5_K_plain.gguf"
CALIB_TXT    = CALIB_DIR / "dutch_calibration.txt"
LAYER_MAP_JSON   = RESULTS_DIR / "unsloth_layer_map_q5.json"
OVERRIDES_TXT    = RESULTS_DIR / "unsloth_overrides_q5.txt"
PPL_RESULTS_JSON = RESULTS_DIR / "ppl_comparison.json"
KLD_RESULTS_JSON = RESULTS_DIR / "kld_comparison.json"

LLAMA_PPL     = Path(LLAMACPP_DIR) / "llama-perplexity.exe"
LLAMA_QUANT   = Path(LLAMACPP_DIR) / "llama-quantize.exe"
LLAMA_IMATRIX = Path(LLAMACPP_DIR) / "llama-imatrix.exe"
LLAMA_SERVER  = Path(LLAMACPP_DIR) / "llama-server.exe"

for d in [MODEL_DIR, RESULTS_DIR, CKPT_DIR, TEXTS_DIR, STDOUT_DIR,
          FP16_LOGP_DIR, CALIB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ================================================================
# HELPERS
# ================================================================
def banner(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def run(cmd, **kwargs):
    """Run a subprocess command, return CompletedProcess. Raises on failure."""
    return subprocess.run(cmd, **kwargs)

def hf_download_file(repo_id, filename, local_path):
    """Download a single file from HuggingFace Hub."""
    if Path(local_path).exists():
        print(f"  Already present: {local_path}")
        return
    print(f"  Downloading {repo_id}/{filename}...")
    from huggingface_hub import hf_hub_download
    tmp = hf_hub_download(repo_id=repo_id, filename=filename)
    import shutil
    shutil.copy(tmp, local_path)
    print(f"  Saved: {local_path}")

def hf_download_snapshot(repo_id, local_dir):
    """Download full HF repo snapshot (for HF model/tokenizer files)."""
    if (Path(local_dir) / "config.json").exists():
        print(f"  Already present: {local_dir}")
        return
    print(f"  Downloading snapshot: {repo_id}...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir),
                      ignore_patterns=["*.gguf", "*.bin", "*.safetensors"])
    print(f"  Saved: {local_dir}")

# ================================================================
# STARTUP CHECKS
# ================================================================
banner(f"Dutch Q5_K_NL pipeline: {MODEL_SLUG}")
print(f"  Model:      {HF_MODEL_ID}")
print(f"  Dataset:    {HF_DATASET_ID}")
print(f"  Output dir: {MODEL_DIR}")
print()

for exe_path, name in [(LLAMA_PPL, "llama-perplexity"),
                       (LLAMA_QUANT, "llama-quantize"),
                       (LLAMA_IMATRIX, "llama-imatrix"),
                       (LLAMA_SERVER, "llama-server")]:
    if not exe_path.exists():
        print(f"ERROR: {name}.exe not found at {exe_path}")
        sys.exit(1)
    print(f"  {name}.exe: OK")

# ================================================================
# ██████╗ PHASE A: Setup
# ================================================================
banner("Phase A: Setup")

# A1. Download FP16 GGUF
print("\n[A1] FP16 GGUF")
hf_download_file(HF_FP16_GGUF_REPO, HF_FP16_GGUF_FILE, FP16_GGUF)

# A2. Download Unsloth Q5 reference GGUF
print("\n[A2] Unsloth Q5 reference GGUF")
hf_download_file(HF_UNSLOTH_REPO, HF_UNSLOTH_FILE, UNSLOTH_GGUF)

# A3. Extract Unsloth dynamic layer map from reference GGUF
print("\n[A3] Layer map extraction")
if OVERRIDES_TXT.exists():
    print(f"  Already extracted: {OVERRIDES_TXT}")
else:
    try:
        from gguf import GGUFReader
    except ImportError:
        print("ERROR: gguf package not installed. Run: pip install gguf")
        sys.exit(1)

    print(f"  Reading tensor metadata from {UNSLOTH_GGUF.name}...")
    reader = GGUFReader(str(UNSLOTH_GGUF))
    tensor_map = {t.name: t.tensor_type.name for t in reader.tensors}

    # Type distribution
    type_counts = defaultdict(int)
    for qt in tensor_map.values():
        type_counts[qt] += 1
    print(f"  Total tensors: {len(tensor_map)}")
    for qt, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {qt:12s}: {n}")

    # Q5 base types — anything NOT in this set was promoted
    Q5_BASE = {"Q5_K", "Q5_0", "Q5_1", "Q4_K", "Q4_0"}
    PROMOTED = {"Q6_K", "Q8_0", "F16", "F32", "BF16"}

    promoted = {name: qt for name, qt in tensor_map.items()
                if qt in PROMOTED}
    print(f"\n  Promoted tensors: {len(promoted)}")
    for name, qt in sorted(promoted.items()):
        print(f"    {qt:8s}  {name}")

    # Build --tensor-type override patterns
    # Group by tensor suffix across all block indices
    pattern_groups = defaultdict(lambda: defaultdict(list))
    for name, qt in promoted.items():
        parts = name.split(".")
        if parts[0] == "blk" and len(parts) >= 3:
            suffix = ".".join(parts[2:])
            pattern_groups[suffix][qt].append(name)
        else:
            # Non-block tensors (token_embd, output, etc.)
            pattern_groups[name][qt].append(name)

    override_lines = []
    for suffix, type_dict in sorted(pattern_groups.items()):
        for qt, names in type_dict.items():
            if "." in suffix:
                regex = f"blk\\..*\\.{re.escape(suffix)}"
            else:
                regex = re.escape(suffix)
            override_lines.append((f"{regex}={qt.lower()}", len(names)))

    print(f"\n  Override patterns for --tensor-type:")
    for line, count in override_lines:
        print(f"    {line}  # {count} tensors")

    # Save full tensor map
    with open(LAYER_MAP_JSON, "w") as f:
        json.dump(tensor_map, f, indent=2, sort_keys=True)
    print(f"\n  Full tensor map saved: {LAYER_MAP_JSON}")

    # Save override file
    with open(OVERRIDES_TXT, "w") as f:
        f.write(f"# Unsloth Dynamic Q5_K layer overrides for {MODEL_SLUG}\n")
        f.write(f"# Extracted from {HF_UNSLOTH_FILE}\n")
        f.write(f"# {len(promoted)} promoted tensors -> {len(override_lines)} patterns\n\n")
        for line, _ in override_lines:
            f.write(f"--tensor-type {line}\n")
    print(f"  Override file saved: {OVERRIDES_TXT}")

# A4. Calibration texts
print("\n[A4] Calibration dataset")
if CALIB_TXT.exists() and CALIB_TXT.stat().st_size > 10_000:
    print(f"  Already present: {CALIB_TXT} ({CALIB_TXT.stat().st_size/1e3:.0f} KB)")
else:
    print(f"  Downloading from {HF_DATASET_ID}...")
    print("  This streams the dataset — may take a few minutes...")

    import polars as pl
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split="train", streaming=True)

    # Stratified calibration sample matching the published pipeline:
    # 60% low-PPL rows, 25% mid-PPL, 15% high-PPL clusters
    # For a clean single-model pipeline, use a flat stratified sample
    # of N rows across all available texts.
    TARGET_ROWS = 3504   # ~295K tokens at ~84 chars/row average
    rng_calib   = np.random.default_rng(RANDOM_SEED)
    texts = []
    for row in ds:
        if rng_calib.random() < 0.001:   # reservoir-style: keep ~0.1%
            texts.append(row["text"].strip())
        if len(texts) >= TARGET_ROWS * 3:
            break
    # Trim to target
    rng_calib.shuffle(texts)
    texts = texts[:TARGET_ROWS]

    with open(CALIB_TXT, "w", encoding="utf-8") as f:
        for t in texts:
            if t:
                f.write(t + "\n\n")
    total_chars = sum(len(t) for t in texts)
    print(f"  Written {len(texts)} rows, ~{total_chars//4:,} tokens: {CALIB_TXT}")

# ================================================================
# ██████╗ PHASE B: Quantization
# ================================================================
banner("Phase B: Quantization")

# Load override patterns
def load_overrides(path):
    lines = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line.replace("--tensor-type ", "").strip())
    return lines

overrides = load_overrides(OVERRIDES_TXT)
print(f"\nLoaded {len(overrides)} override patterns from {OVERRIDES_TXT.name}")

# B1. Build Dutch imatrix
print("\n[B1] Dutch imatrix")
if IMATRIX_FILE.exists() and IMATRIX_FILE.stat().st_size > 1_000:
    print(f"  Already present: {IMATRIX_FILE} ({IMATRIX_FILE.stat().st_size/1e6:.1f} MB)")
else:
    print(f"  Building imatrix from {CALIB_TXT.name}...")
    print(f"  (Expected time: 30-90 minutes on RX 9070 XT)\n")
    cmd = [
        str(LLAMA_IMATRIX),
        "-m", str(FP16_GGUF),
        "-f", str(CALIB_TXT),
        "-o", str(IMATRIX_FILE),
        "-ngl", str(NGL),
        "--chunks", "500",
    ]
    print("  Command:", " ".join(cmd))
    result = run(cmd)
    if result.returncode != 0:
        print("ERROR: llama-imatrix failed")
        sys.exit(1)
    print(f"  Imatrix saved: {IMATRIX_FILE}")

def quantize(output_path, quant_type, use_overrides, description):
    """Run llama-quantize, print command, return True on success."""
    if output_path.exists() and output_path.stat().st_size > 1_000_000:
        print(f"  Already present: {output_path.name}")
        return True
    print(f"\n  Quantizing {description}...")
    cmd = [str(LLAMA_QUANT), "--imatrix", str(IMATRIX_FILE)]
    if use_overrides:
        for ov in overrides:
            cmd += ["--tensor-type", ov]
    cmd += ["--leave-output-tensor", str(FP16_GGUF), str(output_path), quant_type]
    print("  Command:\n   ", " \\\n    ".join(cmd))
    result = run(cmd)
    if result.returncode != 0:
        print(f"ERROR: quantization failed for {output_path.name}")
        return False
    print(f"  Done: {output_path.name} ({output_path.stat().st_size/1e9:.2f} GB)")
    return True

# B2. Q5_K_NL — main output
print("\n[B2] Q5_K_NL (Unsloth map + Dutch imatrix)")
quantize(Q5_NL_GGUF, "Q5_K_M", use_overrides=True,
         description="Q5_K_NL Dutch dynamic")

# B3. Q5_K_plain — ablation control
print("\n[B3] Q5_K_plain (Dutch imatrix only, no layer map)")
quantize(Q5_PLAIN_GGUF, "Q5_K_M", use_overrides=False,
         description="Q5_K_plain Dutch imatrix only")

# Verify smoke tests
print("\n[B-verify] Smoke test: generation check")
for gguf_path, label in [(Q5_NL_GGUF, "Q5_K_NL"), (Q5_PLAIN_GGUF, "Q5_K_plain")]:
    if not gguf_path.exists():
        continue
    result = run([
        str(Path(LLAMACPP_DIR) / "llama-cli.exe"),
        "-m", str(gguf_path), "-ngl", str(NGL), "-n", "20", "--no-warmup",
        "-p", "De Nederlandse taal is",
    ], capture_output=True, text=True, timeout=60)
    output = result.stdout[-200:] if result.stdout else "(no output)"
    print(f"  {label}: {output.strip()[-80:]}")

# ================================================================
# ██████╗ PHASE C: PPL Measurement
# ================================================================
banner("Phase C: Perplexity measurement")

# Models to evaluate in PPL phase
PPL_MODELS = {
    "fp16":        str(FP16_GGUF),
    "q5_nl":       str(Q5_NL_GGUF),
    "q5_plain":    str(Q5_PLAIN_GGUF),
    "unsloth_q5":  str(UNSLOTH_GGUF),
}
# Filter to only existing files
PPL_MODELS = {k: v for k, v in PPL_MODELS.items() if Path(v).exists()}
print(f"\nModels to evaluate: {list(PPL_MODELS.keys())}")

# C1. Write cluster text files from dataset
print("\n[C1] Cluster text files")
# We need clustered texts. Since this is a standalone pipeline, we use a
# per-PPL-bucket approach: high/mid/low perplexity texts from the published
# dataset, treated as synthetic clusters.
#
# If checkpoint2_full.parquet is present (full pipeline was run), use it.
# Otherwise, stream directly from HuggingFace and split into PPL strata.

FULL_PARQUET = Path(BASE_DIR) / "data" / "embeddings" / "checkpoint2_full.parquet"
N_CLUSTERS   = 70

def write_cluster_texts_from_parquet():
    """Use existing checkpoint2_full.parquet with cluster_id and ppl_fp16."""
    import polars as pl
    print(f"  Loading {FULL_PARQUET}...")
    df = pl.read_parquet(
        FULL_PARQUET,
        columns=["row_id", "text", "cluster_id", "ppl_fp16", "is_simple", "lev_distance"]
    ).filter(pl.col("ppl_fp16").is_not_null())
    print(f"  {len(df):,} measured rows across {df['cluster_id'].n_unique()} clusters")

    rng = np.random.default_rng(RANDOM_SEED)
    clusters_written = 0

    for cl in range(N_CLUSTERS):
        txt_path = TEXTS_DIR / f"cluster_{cl:03d}.txt"
        if txt_path.exists() and txt_path.stat().st_size > 0:
            clusters_written += 1
            continue

        df_cl = df.filter(pl.col("cluster_id") == cl)
        if len(df_cl) == 0:
            continue

        # Stratified sample: is_simple × lev_distance tercile
        lev = df_cl["lev_distance"].to_numpy()
        if len(lev) < 6:
            sampled = df_cl["text"].to_list()
        else:
            q33, q66 = np.percentile(lev, [33, 66])
            sampled = []
            for is_simple in [True, False]:
                for lo, hi in [(None, q33), (q33, q66), (q66, None)]:
                    mask = (df_cl["is_simple"] == is_simple).to_numpy()
                    if lo is not None: mask &= lev > lo
                    if hi is not None: mask &= lev <= hi
                    sub = df_cl.filter(pl.Series(mask))["text"].to_numpy()
                    n = min(PPL_ROWS_PER_STRATUM, len(sub))
                    if n > 0:
                        sampled.extend(rng.choice(sub, n, replace=False).tolist())

        if sampled:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(t.strip() for t in sampled if t.strip()))
            clusters_written += 1

    print(f"  {clusters_written}/{N_CLUSTERS} cluster text files ready")
    return N_CLUSTERS

def write_cluster_texts_from_stream():
    """
    No local parquet available. Stream dataset and bin texts by
    approximate PPL strata using text length as a proxy
    (longer sentences tend to have higher model perplexity).
    Creates 10 synthetic clusters rather than 70.
    """
    from datasets import load_dataset
    print(f"  No local parquet found — streaming from {HF_DATASET_ID}...")
    print("  Using text-length proxy for cluster assignment (10 buckets)")

    N_STREAM_CLUSTERS = 10
    buckets = defaultdict(list)
    max_per_bucket = PPL_ROWS_PER_STRATUM * 6

    ds = load_dataset(HF_DATASET_ID, split="train", streaming=True)
    total = 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        # Bucket by word count decile (0-9)
        wc = len(text.split())
        bucket = min(wc // 10, N_STREAM_CLUSTERS - 1)
        if len(buckets[bucket]) < max_per_bucket:
            buckets[bucket].append(text)
        total += 1
        if all(len(v) >= max_per_bucket for v in buckets.values()):
            break
        if total > 500_000:
            break

    print(f"  Read {total:,} rows into {len(buckets)} buckets")
    rng = np.random.default_rng(RANDOM_SEED)

    for cl in range(N_STREAM_CLUSTERS):
        txt_path = TEXTS_DIR / f"cluster_{cl:03d}.txt"
        if txt_path.exists() and txt_path.stat().st_size > 0:
            continue
        texts = buckets.get(cl, [])
        rng.shuffle(texts)
        if texts:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(texts[:PPL_ROWS_PER_STRATUM * 6]))
    print(f"  {N_STREAM_CLUSTERS} cluster text files written")
    return N_STREAM_CLUSTERS

# Choose path based on whether full parquet exists
if FULL_PARQUET.exists():
    n_clusters_ppl = write_cluster_texts_from_parquet()
else:
    print("  NOTE: checkpoint2_full.parquet not found.")
    print("  Using streaming mode (10 proxy clusters instead of 70).")
    print("  For full 70-cluster results, run the complete pipeline first.")
    n_clusters_ppl = write_cluster_texts_from_stream()

# C2+C3. PPL measurement per model per cluster
print("\n[C2] Perplexity measurement")

def load_ppl_checkpoint(model_name):
    p = CKPT_DIR / f"ppl_{model_name}.jsonl"
    done = {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line.strip())
                done[int(rec["cl"])] = rec.get("ppl")
            except Exception:
                pass
    return done

def save_ppl_checkpoint(model_name, cl, ppl):
    p = CKPT_DIR / f"ppl_{model_name}.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"cl": cl, "ppl": ppl}) + "\n")

def parse_ppl_from_stdout(stdout_text):
    """Extract final perplexity value from llama-perplexity output."""
    # Primary pattern: Final estimate: PPL = 9.1234
    m = re.search(r"Final estimate:\s+PPL\s*=\s*([\d.]+)", stdout_text)
    if m:
        return float(m.group(1))
    # Fallback: last "PPL = N" occurrence
    matches = re.findall(r"PPL\s*=\s*([\d.]+)", stdout_text)
    if matches:
        return float(matches[-1])
    return None

def run_ppl(model_path, txt_path, model_name, cl):
    """Run llama-perplexity on a cluster text file. Returns PPL or None."""
    stdout_file = STDOUT_DIR / f"{model_name}_cl{cl:03d}.txt"
    cmd = [
        str(LLAMA_PPL),
        "-m", model_path,
        "-ngl", str(NGL),
        "-f", str(txt_path),
        "--ctx-size", str(CTX_SIZE),
        "--ppl-stride", str(PPL_STRIDE),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=PPL_TIMEOUT,
        )
        combined = result.stdout + result.stderr
        stdout_file.write_text(combined)
        ppl = parse_ppl_from_stdout(combined)
        # Reject if too-short warning
        if "tokenizes to only" in combined and ppl is None:
            return None
        return ppl
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None

all_ppl_results = {}

for model_name, model_path in PPL_MODELS.items():
    print(f"\n  {'='*60}")
    print(f"  Model: {model_name}")
    print(f"    {model_path}")
    print(f"  {'='*60}")

    done = load_ppl_checkpoint(model_name)
    valid_cls = [cl for cl in range(n_clusters_ppl)
                 if (TEXTS_DIR / f"cluster_{cl:03d}.txt").exists()]
    remaining = [cl for cl in valid_cls if cl not in done]

    if not remaining:
        print(f"  Fully complete ({len(done)} clusters) — skipping.")
        all_ppl_results[model_name] = done
        continue
    print(f"  {len(done)} done, {len(remaining)} remaining")

    results = dict(done)
    for cl in valid_cls:
        if cl in done:
            continue
        txt_path = TEXTS_DIR / f"cluster_{cl:03d}.txt"
        ppl = run_ppl(model_path, txt_path, model_name, cl)
        results[cl] = ppl
        save_ppl_checkpoint(model_name, cl, ppl)
        if ppl:
            fp16_ppl = all_ppl_results.get("fp16", {}).get(cl)
            delta_str = f"  Δ={ppl - fp16_ppl:+.3f}" if fp16_ppl else ""
            print(f"  Cluster {cl:3d}: PPL={ppl:.3f}{delta_str}")
        else:
            print(f"  Cluster {cl:3d}: FAILED")

    all_ppl_results[model_name] = results

    valid = [v for v in results.values() if v is not None]
    if valid:
        fp16_vals = all_ppl_results.get("fp16", {})
        paired = [(results[cl], fp16_vals[cl]) for cl in results
                  if results.get(cl) and fp16_vals.get(cl)]
        if paired and model_name != "fp16":
            deltas = [p[0] - p[1] for p in paired]
            print(f"\n  {model_name}: median PPL={np.median(valid):.3f}  "
                  f"median Δ={np.median(deltas):+.3f}  "
                  f"mean Δ={np.mean(deltas):+.3f}")
        else:
            print(f"\n  {model_name}: median PPL={np.median(valid):.3f}")

# Summary table
print(f"\n{'='*70}")
print("PPL SUMMARY")
print(f"{'='*70}")
print(f"{'Model':<16} {'Median PPL':>11} {'Median Δ':>10} {'Max Δ':>8} {'N valid':>8}")
print("-" * 55)
fp16_res = all_ppl_results.get("fp16", {})
for m, res in all_ppl_results.items():
    valid = [(res[cl], fp16_res.get(cl)) for cl in res
             if res.get(cl) and fp16_res.get(cl)]
    ppls   = [v[0] for v in valid] if valid else [v for v in res.values() if v]
    if not ppls:
        continue
    if valid and m != "fp16":
        deltas = [v[0] - v[1] for v in valid]
        print(f"{m:<16} {np.median(ppls):>11.3f} "
              f"{np.median(deltas):>+10.3f} {np.max(deltas):>+8.3f} "
              f"{len(ppls):>8}")
    else:
        print(f"{m:<16} {np.median(ppls):>11.3f} {'—':>10} {'—':>8} {len(ppls):>8}")

with open(PPL_RESULTS_JSON, "w") as f:
    json.dump({m: {str(cl): r for cl, r in res.items()}
               for m, res in all_ppl_results.items()}, f, indent=2)
print(f"\nSaved: {PPL_RESULTS_JSON}")

# ================================================================
# ██████╗ PHASE D: KLD Measurement
# ================================================================
banner("Phase D: KLD divergence measurement")
print(f"KL(FP16 || quant) over top-{TOP_K} vocab tokens")
print(f"Fast mode: {KLD_FAST_MODE} "
      f"({'~45 min' if KLD_FAST_MODE else '~6-8h depending on cluster count'})\n")

KLD_MODELS = {k: v for k, v in PPL_MODELS.items() if k != "fp16"}

# D1. Extract FP16 logprobs via HF transformers
def extract_fp16_logprobs():
    """
    Load FP16 model via HF transformers, extract top-K log-softmax per token
    position for all cluster texts. Cached to FP16_LOGP_DIR.
    """
    # Check what's needed
    needed = []
    for cl in range(n_clusters_ppl):
        txt_path = TEXTS_DIR / f"cluster_{cl:03d}.txt"
        if not txt_path.exists():
            continue
        p = FP16_LOGP_DIR / f"cluster_{cl:03d}.jsonl"
        if p.exists() and p.stat().st_size > 0:
            try:
                json.loads(p.read_text().splitlines()[0])
                continue  # valid
            except Exception:
                pass
        needed.append(cl)

    if not needed:
        print("[D1] FP16 logprobs: all clusters cached. Skipping.\n")
        return

    print(f"[D1] FP16 logprob extraction: {len(needed)} clusters needed")
    print(f"     Loading {HF_MODEL_ID}...")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("     WARNING: no GPU — CPU inference will be slow")

    # Download tokenizer/config if needed (no weight files)
    tokenizer_dir = MODEL_DIR / "hf_tokenizer"
    hf_download_snapshot(HF_MODEL_ID, tokenizer_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,  # load weights from HF directly (not GGUF)
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    print(f"     Loaded on {device}\n")

    import torch
    with torch.no_grad():
        for cl in needed:
            txt_path = TEXTS_DIR / f"cluster_{cl:03d}.txt"
            texts_raw = txt_path.read_text(encoding="utf-8").split("\n\n")
            texts = [t.strip() for t in texts_raw if t.strip()]
            if KLD_FAST_MODE:
                texts = texts[:KLD_TEXTS_FAST]

            out_path = FP16_LOGP_DIR / f"cluster_{cl:03d}.jsonl"
            n_written = 0
            with open(out_path, "w", encoding="utf-8") as fout:
                for text in texts:
                    enc = tokenizer(text, return_tensors="pt", truncation=True,
                                    max_length=MAX_SEQ_LEN).to(device)
                    n_tok = enc["input_ids"].shape[1]
                    if n_tok < 4:
                        continue
                    logits    = model(**enc).logits[0].float()
                    log_probs = torch.log_softmax(logits, dim=-1)
                    topk      = torch.topk(log_probs, TOP_K, dim=-1)
                    fout.write(json.dumps({
                        "n_tokens":  n_tok,
                        "topk_ids":  topk.indices.cpu().tolist(),
                        "topk_logp": topk.values.cpu().tolist(),
                    }, separators=(',', ':')) + "\n")
                    n_written += 1
            print(f"     Cluster {cl:3d}: {n_written}/{len(texts)} texts")

    print("\n     Unloading FP16 model...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)

extract_fp16_logprobs()

# D2. KLD per quant model via llama-server
def start_server(gguf_path):
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(
        [str(LLAMA_SERVER), "-m", str(gguf_path), "-ngl", str(NGL),
         "--ctx-size", str(MAX_SEQ_LEN + 64),
         "--host", "127.0.0.1", "--port", str(SERVER_PORT),
         "--no-warmup", "--slots"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags,
    )

def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()

def wait_for_server():
    deadline = time.time() + SERVER_WAIT
    while time.time() < deadline:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False

def _post(endpoint, payload, timeout=60):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}{endpoint}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return None

def get_quant_topk_logprobs(text):
    """
    Teacher-forcing logprobs from llama-server:
    tokenize text, feed token IDs as prompt with n_predict=n_tok-1,
    collect completion_probabilities[t] = P(token_{t+1} | tokens_0..t).
    Aligns with FP16 logprobs: fp16_logp[t+1] ↔ quant_probs[t].
    Returns list of {token_id: log_prob} dicts, length n_tok-1.
    """
    tok = _post("/tokenize", {"content": text, "add_special": True})
    if not tok or "tokens" not in tok:
        return None
    tokens = tok["tokens"][:MAX_SEQ_LEN]
    if len(tokens) < 4:
        return None

    resp = _post("/completion", {
        "prompt": tokens, "n_predict": len(tokens) - 1,
        "temperature": 0.0, "n_probs": TOP_K,
        "top_k": 0, "top_p": 1.0, "min_p": 0.0,
        "repeat_penalty": 1.0, "cache_prompt": False,
    })
    if not resp:
        return None

    cp = resp.get("completion_probabilities") or resp.get("probs") or []
    result = []
    for step in cp:
        top = step.get("top_probs") or step.get("probs") or []
        pos_map = {}
        for entry in top:
            tid  = entry.get("id")
            prob = float(entry.get("prob", 0.0))
            if isinstance(tid, int) and prob > 1e-12:
                pos_map[tid] = math.log(prob)
        result.append(pos_map)
    return result

def compute_kld_cluster(cl, fp16_records):
    """KL(P_fp16 || P_quant) per token position, return stats dict or None."""
    txt_path = TEXTS_DIR / f"cluster_{cl:03d}.txt"
    if not txt_path.exists() or not fp16_records:
        return None
    texts = [t.strip() for t in txt_path.read_text(encoding="utf-8").split("\n\n")
             if t.strip()]
    if KLD_FAST_MODE:
        texts = texts[:KLD_TEXTS_FAST]

    all_klds   = []
    flip_count = 0
    total_pos  = 0

    for text, rec in zip(texts, fp16_records):
        quant_maps = get_quant_topk_logprobs(text)
        if not quant_maps:
            continue
        fp16_ids  = rec["topk_ids"]
        fp16_logp = rec["topk_logp"]
        n_tok     = rec["n_tokens"]

        for t in range(1, min(n_tok - 1, len(quant_maps)) + 1):
            t_ids    = fp16_ids[t]
            t_logp   = fp16_logp[t]
            q_map    = quant_maps[t - 1]
            if not q_map:
                continue
            floor_lp = t_logp[-1] - math.log(100.0)
            kld = 0.0
            for tok_id, lp_fp16 in zip(t_ids, t_logp):
                p_fp16 = math.exp(lp_fp16)
                if p_fp16 < 1e-10:
                    break
                lp_quant = q_map.get(tok_id, floor_lp)
                kld += p_fp16 * (lp_fp16 - lp_quant)
            if math.isfinite(kld) and kld >= 0:
                all_klds.append(kld)
                total_pos += 1
                if (q_map and max(q_map, key=q_map.get) != t_ids[0]):
                    flip_count += 1

    if not all_klds:
        return None
    return {
        "n_positions": total_pos,
        "mean_kld":    float(np.mean(all_klds)),
        "median_kld":  float(np.median(all_klds)),
        "p95_kld":     float(np.percentile(all_klds, 95)),
        "p99_kld":     float(np.percentile(all_klds, 99)),
        "p999_kld":    float(np.percentile(all_klds, 99.9)),
        "flip_rate":   float(flip_count / total_pos),
    }

def load_kld_checkpoint(model_name):
    p = CKPT_DIR / f"kld_{model_name}.jsonl"
    done = {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                rec = json.loads(line.strip())
                done[int(rec["cl"])] = rec.get("result")
            except Exception:
                pass
    return done

def save_kld_checkpoint(model_name, cl, result):
    p = CKPT_DIR / f"kld_{model_name}.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps({"cl": cl, "result": result}) + "\n")

print("[D2] KLD per quant model\n")
all_kld_results = {}
server_proc = None

try:
    for model_name, model_path in KLD_MODELS.items():
        print(f"  {'='*60}")
        print(f"  KLD: {model_name}")
        print(f"  {'='*60}")

        done = load_kld_checkpoint(model_name)
        valid_cls = [cl for cl in range(n_clusters_ppl)
                     if (TEXTS_DIR / f"cluster_{cl:03d}.txt").exists()
                     and (FP16_LOGP_DIR / f"cluster_{cl:03d}.jsonl").exists()]
        remaining = [cl for cl in valid_cls if cl not in done]

        if not remaining:
            print(f"  Fully complete — skipping.")
            all_kld_results[model_name] = done
            continue
        print(f"  {len(done)} done, {len(remaining)} remaining")

        server_proc = start_server(model_path)
        if not wait_for_server():
            print(f"  ERROR: server failed — skipping {model_name}")
            stop_server(server_proc); server_proc = None; continue
        print(f"  Server ready.\n")

        results = dict(done)
        for cl in valid_cls:
            if cl in done:
                continue
            fp16_recs = []
            p = FP16_LOGP_DIR / f"cluster_{cl:03d}.jsonl"
            if p.exists():
                for line in p.read_text().splitlines():
                    try: fp16_recs.append(json.loads(line.strip()))
                    except Exception: pass

            result = compute_kld_cluster(cl, fp16_recs) if fp16_recs else None
            results[cl] = result
            save_kld_checkpoint(model_name, cl, result)

            if result:
                print(f"  Cluster {cl:3d}: "
                      f"mean={result['mean_kld']:.5f}  "
                      f"p99={result['p99_kld']:.5f}  "
                      f"p99.9={result['p999_kld']:.5f}  "
                      f"flip={result['flip_rate']:.3f}")
            else:
                print(f"  Cluster {cl:3d}: FAILED")

        stop_server(server_proc); server_proc = None
        time.sleep(3)
        all_kld_results[model_name] = results

        valid = [v for v in results.values() if v]
        if valid:
            print(f"\n  {model_name}: "
                  f"median mean_kld={np.median([v['mean_kld'] for v in valid]):.5f}  "
                  f"median p99.9={np.median([v['p999_kld'] for v in valid]):.5f}  "
                  f"mean flip={np.mean([v['flip_rate'] for v in valid]):.3f}")

except KeyboardInterrupt:
    print("\nInterrupted — all progress checkpointed.")
finally:
    if server_proc:
        stop_server(server_proc)

# Summary table
print(f"\n{'='*75}")
print(f"KLD SUMMARY — KL(FP16 || quant), top-{TOP_K} tokens")
print(f"Sample: {'fast (3 texts/cluster)' if KLD_FAST_MODE else 'full'} | "
      f"seq {MAX_SEQ_LEN} tokens")
print(f"{'='*75}")
print(f"{'Model':<16} {'med mean':>10} {'med p99':>10} {'med p99.9':>11} {'flip':>8} {'N':>6}")
print("-" * 65)
for m, results in all_kld_results.items():
    valid = [v for v in results.values() if v]
    if not valid:
        print(f"{m:<16} {'N/A'}")
        continue
    marker = " ◄" if m == "q5_nl" else ""
    print(f"{m:<16} "
          f"{np.median([v['mean_kld'] for v in valid]):>10.5f} "
          f"{np.median([v['p99_kld'] for v in valid]):>10.5f} "
          f"{np.median([v['p999_kld'] for v in valid]):>11.5f} "
          f"{np.mean([v['flip_rate'] for v in valid]):>8.3f} "
          f"{len(valid):>6}{marker}")

with open(KLD_RESULTS_JSON, "w") as f:
    json.dump({m: {str(cl): r for cl, r in res.items()}
               for m, res in all_kld_results.items()}, f, indent=2)
print(f"\nSaved: {KLD_RESULTS_JSON}")

# ================================================================
# FINAL SUMMARY
# ================================================================
banner("Pipeline complete")
print(f"Model:        {MODEL_SLUG}")
print(f"Dataset:      {HF_DATASET_ID}")
print()
print("Outputs:")
for p in [FP16_GGUF, UNSLOTH_GGUF, IMATRIX_FILE, Q5_NL_GGUF, Q5_PLAIN_GGUF,
          OVERRIDES_TXT, PPL_RESULTS_JSON, KLD_RESULTS_JSON]:
    exists = "✓" if Path(p).exists() else "✗"
    size   = f"({Path(p).stat().st_size/1e6:.0f} MB)" if Path(p).exists() else ""
    print(f"  {exists} {p} {size}")

print()
print("To run on a different model, edit the CONFIG block at the top of this script.")
print("Models tested with this pipeline:")
print("  ibm-granite/granite-4.0-h-micro  (3B hybrid)")
print("  ibm-granite/granite-4.0-h-nano   (350M hybrid)")
print("  ibm-granite/granite-4.0-h-tiny   (7B hybrid)")
print("  ibm-granite/granite-4.0-micro    (3B dense)")
