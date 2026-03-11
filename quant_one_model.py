"""
Dutch-calibrated GGUF quantization pipeline — single model runner.
=================================================================
Phase A: Setup (Downloads, Layer Maps, Dataset Sampling)
Phase B: Quantize (Imatrix -> Dynamic NL GGUF + Plain GGUF)
Phase C: Perplexity (PPL) per cluster vs FP16 baseline
Phase D: KL-Divergence (KLD) per cluster via llama-server logprobs

All phases fully resumable. Use SKIP_* / RESET_* flags in CONFIG.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

# ================================================================
# CONFIG
# ================================================================

# TARGET MODEL
HF_MODEL_ID       = "ibm-granite/granite-4.0-h-tiny"
HF_FP16_GGUF_REPO = "ibm-granite/granite-4.0-h-tiny-GGUF"
HF_FP16_GGUF_FILE = "granite-4.0-h-tiny-f16.gguf"

# REFERENCE MODEL (for Unsloth dynamic maps)
HF_UNSLOTH_REPO   = "unsloth/granite-4.0-h-tiny-GGUF"

# ----------------------------------------------------------------
# CHANGE ONLY THIS
# ----------------------------------------------------------------
QUANT_BITS = 4  # Target bit-depth
NGL        = 99 # GPU layers

# Execution Control
SKIP_PPL    = False
SKIP_KLD    = False
RESET_PPL   = False
RESET_KLD   = False
RESET_QUANT = False

# KLD Settings
TOP_K       = 100
KLD_SAMPLES = 100
# ----------------------------------------------------------------

# Derived values
MODEL_NAME  = HF_MODEL_ID.split("/")[-1]
MODEL_SLUG  = f"{MODEL_NAME}-q{QUANT_BITS}"
QUANT_TYPE  = f"Q{QUANT_BITS}_K_M"
UNSLOTH_FILE = f"{MODEL_NAME}-UD-Q{QUANT_BITS}_K_XL.gguf"

# Dataset Settings
HF_DATASET_ID = "MichielBuisman/Leesplank-vloeiend-nl-curriculum-cp2"
N_CLUSTERS    = 70
PPL_SAMPLE    = 250

# Paths
BASE_DIR     = Path("C:/dutch-lora")
MODELS_DIR   = BASE_DIR / "models" / MODEL_NAME
RESULTS_DIR  = BASE_DIR / "data" / "results" / MODEL_SLUG
STDOUT_DIR   = RESULTS_DIR / "logs"
TEXTS_DIR    = RESULTS_DIR / "cluster_texts"
CALIB_DIR    = BASE_DIR / "data" / "calibration"

for d in [MODELS_DIR, RESULTS_DIR, STDOUT_DIR, TEXTS_DIR, CALIB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Binaries
LLAMA_BIN_DIR = Path("C:/llamacpp-rocm")
LLAMA_QUANT   = LLAMA_BIN_DIR / "llama-quantize.exe"
LLAMA_IMATRIX = LLAMA_BIN_DIR / "llama-imatrix.exe"
LLAMA_PPL     = LLAMA_BIN_DIR / "llama-perplexity.exe"
LLAMA_SERVER  = LLAMA_BIN_DIR / "llama-server.exe"

# Files
FP16_GGUF        = MODELS_DIR / HF_FP16_GGUF_FILE
UNSLOTH_REF_GGUF = MODELS_DIR / UNSLOTH_FILE
IMATRIX_FILE     = RESULTS_DIR / f"{MODEL_SLUG}-imatrix-dutch.dat"
OVERRIDES_TXT    = RESULTS_DIR / f"unsloth_overrides_q{QUANT_BITS}.txt"
Q_NL_GGUF        = RESULTS_DIR / f"{MODEL_SLUG}-NL.gguf"
Q_PLAIN_GGUF     = RESULTS_DIR / f"{MODEL_SLUG}-plain.gguf"
CALIB_TEXT_FILE  = CALIB_DIR / "dutch_calibration.txt"
PPL_RESULTS_JSON = RESULTS_DIR / f"{MODEL_SLUG}_ppl.json"
KLD_RESULTS_JSON = RESULTS_DIR / f"{MODEL_SLUG}_kld.json"

# ================================================================
# UTILS
# ================================================================

def run_cmd(cmd, label, log_name=None):
    print(f"  {label} ... ", end="", flush=True)
    logfile = STDOUT_DIR / f"{log_name}.log" if log_name else None
    try:
        if logfile:
            with open(logfile, "w", encoding="utf-8") as f:
                subprocess.run(cmd, stdout=f, stderr=f, check=True)
        else:
            subprocess.run(cmd, capture_output=True, check=True)
        print("OK")
    except subprocess.CalledProcessError:
        print(f"FAILED (See logs: {log_name}.log)")
        sys.exit(1)

def start_server(model_path, port=8080):
    cmd = [str(LLAMA_SERVER), "-m", str(model_path), "-ngl", str(NGL), "--port", str(port), "--log-disable"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
                if r.status == 200: return proc
        except: pass
        time.sleep(1)
    proc.terminate()
    return None

def stop_server(proc):
    if proc:
        proc.terminate()
        proc.wait()

def get_probs(prompt, port=8080):
    """Fetches probabilities, supporting multiple llama-server API versions."""
    data = json.dumps({"prompt": prompt, "n_predict": 0, "n_probs": TOP_K}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as r:
            res = json.loads(r.read().decode())
            if "completion_probabilities" in res:
                return res["completion_probabilities"][0].get("probs", [])
            if "choices" in res:
                return res["choices"][0].get("logprobs", {}).get("top_logprobs", [])
            return []
    except:
        return []

# ================================================================
# MAIN PIPELINE
# ================================================================
print(f"\n{'='*70}\nPipeline: {MODEL_SLUG}\n{'='*70}")

if RESET_QUANT:
    print(f"[!] RESET_QUANT: Cleaning {MODEL_SLUG}...")
    for f in [Q_NL_GGUF, Q_PLAIN_GGUF, OVERRIDES_TXT]:
        if f.exists(): f.unlink()

# [A1] Downloads
from huggingface_hub import hf_hub_download
for rid, fn, dest in [(HF_FP16_GGUF_REPO, HF_FP16_GGUF_FILE, FP16_GGUF),
                      (HF_UNSLOTH_REPO, UNSLOTH_FILE, UNSLOTH_REF_GGUF)]:
    if not dest.exists() or dest.stat().st_size == 0:
        print(f"[A1] Downloading {fn}...")
        path = hf_hub_download(repo_id=rid, filename=fn)
        shutil.copy(path, dest)
    else:
        print(f"[A1] Found existing {fn}")

# [A2] Layer Map Extraction
if not OVERRIDES_TXT.exists() or OVERRIDES_TXT.stat().st_size == 0:
    print(f"[A2] Extracting dynamic layer map from {UNSLOTH_FILE}...")
    import gguf
    reader = gguf.GGUFReader(str(UNSLOTH_REF_GGUF))
    with open(OVERRIDES_TXT, "w", encoding="utf-8") as f:
        for tensor in reader.tensors:
            f.write(f"{tensor.name}={tensor.tensor_type.name}\n")
    print(f"  [OK] Extracted {len(reader.tensors)} tensors.")

# [A3] Dataset Sampling
if not CALIB_TEXT_FILE.exists() or not any(TEXTS_DIR.iterdir()):
    print(f"[A3] Preparing Dutch data...")
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET_ID, split="train")
    valid_ds = ds.filter(lambda x: x['ppl_fp16'] is not None)
    
    with open(CALIB_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(valid_ds['text'][:2500]))
    
    for cid in range(N_CLUSTERS):
        c_ds = valid_ds.filter(lambda x: x['cluster_id'] == cid)
        if len(c_ds) > 0:
            with open(TEXTS_DIR / f"cluster_{cid:03d}.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(c_ds['text'][:PPL_SAMPLE]))
    print("  [OK] Data prepared.")

# [B] Quantization
print(f"\nPhase B: Building GGUFs")

if not IMATRIX_FILE.exists():
    run_cmd([str(LLAMA_IMATRIX), "-m", str(FP16_GGUF), "-f", str(CALIB_TEXT_FILE), 
             "-o", str(IMATRIX_FILE), "-ngl", str(NGL)], "Imatrix", "imatrix")

if not Q_NL_GGUF.exists():
    run_cmd([str(LLAMA_QUANT), "--imatrix", str(IMATRIX_FILE), "--tensor-type-file", 
             str(OVERRIDES_TXT), str(FP16_GGUF), str(Q_NL_GGUF), QUANT_TYPE], 
             "Quant (NL-Dynamic)", "quant_nl")

if not Q_PLAIN_GGUF.exists():
    run_cmd([str(LLAMA_QUANT), "--imatrix", str(IMATRIX_FILE), 
             str(FP16_GGUF), str(Q_PLAIN_GGUF), QUANT_TYPE], 
             "Quant (Plain)", "quant_plain")

# [C] Perplexity (PPL)
if not SKIP_PPL:
    print(f"\n{'='*70}\nPhase C: Perplexity (PPL)\n{'='*70}")
    if RESET_PPL and PPL_RESULTS_JSON.exists(): PPL_RESULTS_JSON.unlink()
    all_ppl = json.load(open(PPL_RESULTS_JSON)) if PPL_RESULTS_JSON.exists() else {}

    models_to_test = {"fp16": FP16_GGUF, "q_nl": Q_NL_GGUF, "q_plain": Q_PLAIN_GGUF}
    for m_key, m_path in models_to_test.items():
        if m_key not in all_ppl: all_ppl[m_key] = {}
        print(f"  Model: {m_key}")
        for cluster_file in sorted(TEXTS_DIR.glob("cluster_*.txt")):
            cid = cluster_file.stem.split("_")[1]
            if cid in all_ppl[m_key]: continue
            print(f"    Cluster {cid} ... ", end="", flush=True)
            log_file = STDOUT_DIR / f"ppl_{m_key}_{cid}.log"
            cmd = [str(LLAMA_PPL), "-m", str(m_path), "-f", str(cluster_file), "-ngl", str(NGL)]
            try:
                with open(log_file, "w", encoding="utf-8") as f:
                    subprocess.run(cmd, stdout=f, stderr=f, check=True)
                with open(log_file, "r", encoding="utf-8") as f:
                    m = re.search(r"Final estimate: PPL = ([\d\.]+)", f.read())
                    if m: 
                        val = float(m.group(1))
                        all_ppl[m_key][cid] = val
                        print(f"{val:.4f}")
                    else: print("PARSE ERROR")
            except: print("FAILED")
        with open(PPL_RESULTS_JSON, "w") as f: json.dump(all_ppl, f, indent=2)

# [D] KL-Divergence (KLD)
if not SKIP_KLD:
    print(f"\n{'='*70}\nPhase D: KL-Divergence\n{'='*70}")
    if RESET_KLD and KLD_RESULTS_JSON.exists(): KLD_RESULTS_JSON.unlink()
    all_kld = json.load(open(KLD_RESULTS_JSON)) if KLD_RESULTS_JSON.exists() else {}

    for m_key, m_path in [("q_nl", Q_NL_GGUF), ("q_plain", Q_PLAIN_GGUF)]:
        if m_key not in all_kld: all_kld[m_key] = {}
        print(f"  Testing {m_key} vs FP16...")

        srv_fp16 = start_server(FP16_GGUF, 8080)
        if not srv_fp16:
            print("  [!] FP16 server startup failed."); continue
        srv_q = start_server(m_path, 8081)
        if not srv_q:
            print("  [!] Quant server startup failed."); stop_server(srv_fp16); continue
        time.sleep(3)

        for cluster_file in sorted(TEXTS_DIR.glob("cluster_*.txt")):
            cid = cluster_file.stem.split("_")[1]
            if cid in all_kld[m_key]: continue

            prompts = [p.strip() for p in cluster_file.read_text(encoding='utf-8').split("\n\n") if p.strip()][:KLD_SAMPLES]
            kls, flips = [], 0

            for p in prompts:
                def clean_probs(p_list):
                    return {i.get("tok_str", i.get("token")): i.get("prob", i.get("p", 0)) for i in p_list}

                p_f16 = clean_probs(get_probs(p, 8080))
                p_q   = clean_probs(get_probs(p, 8081))
                if not p_f16 or not p_q: continue

                kl = 0
                for t, prob_a in p_f16.items():
                    prob_b = p_q.get(t, 1e-7)
                    kl += prob_a * math.log(prob_a / prob_b)
                kls.append(kl)
                if list(p_f16.keys())[0] != list(p_q.keys())[0]: flips += 1

            if kls:
                res = {"mean": float(np.mean(kls)), "median": float(np.median(kls)), "p99": float(np.percentile(kls, 99)), "flip_rate": flips/len(kls), "n": len(kls)}
                all_kld[m_key][cid] = res
                print(f"    Cluster {cid}: KL={res['mean']:.5f} Flip={res['flip_rate']:.2%}")
                with open(KLD_RESULTS_JSON, "w") as f:
                    json.dump(all_kld, f, indent=2)

        stop_server(srv_fp16); stop_server(srv_q)

    # FINAL SUMMARY
    print(f"\n{'='*70}\nKLD SUMMARY — KL(FP16 || quant)\n{'='*70}")
    print(f"{'Model':<16} {'w.mean':>10} {'w.median':>10} {'flip':>8} {'N':>5}")
    for m, results in all_kld.items():
        v = [val for val in results.values() if val]
        if not v: continue
        tn = sum(i["n"] for i in v)
        wm = sum(i["mean"] * i["n"] for i in v) / tn
        wf = sum(i["flip_rate"] * i["n"] for i in v) / tn
        print(f"{m:<16} {wm:>10.5f} {np.median([i['median'] for i in v]):>10.5f} {wf:>8.2%} {len(v):>5}")

print("\nPipeline Finished.")
