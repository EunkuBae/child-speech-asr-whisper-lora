from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from jiwer import wer, cer
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


@dataclass
class EvalConfig:
    manifest_path: str
    model_id: str = "openai/whisper-small"
    device: str = "cpu"                # cpu | cuda
    language: str = "en"               # "en" recommended for this dataset
    task: str = "transcribe"           # transcribe | translate
    batch_size: int = 4
    max_items: int = 200               # -1 for full
    seed: int = 42
    out_dir: str = "artifacts/eval/baseline_whisper_small"
    write_predictions: bool = True
    append_results_md: bool = True
    results_md_path: str = "reports/RESULTS.md"


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def normalize_text(s: str) -> str:
    # Light normalization only (avoid over-normalizing; keep it honest)
    return " ".join((s or "").strip().split())


def load_audio(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32)
    if sr != target_sr:
        # Avoid extra deps; simple resample via torch if needed
        # but our preprocessing already forces 16k, so this is a safeguard.
        x_t = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        x_t = torch.nn.functional.interpolate(
            x_t.unsqueeze(0), size=int(len(x) * (target_sr / sr)), mode="linear", align_corners=False
        ).squeeze(0).squeeze(0)
        x = x_t.numpy().astype(np.float32)
        sr = target_sr
    x = np.clip(x, -1.0, 1.0)
    return x, sr


def pick_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model_and_processor(model_id: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_existing_preds(pred_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load previous predictions for resume. Keyed by utt_id.
    """
    if not pred_path.exists():
        return {}
    done = {}
    for ln in pred_path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        done[obj["utt_id"]] = obj
    return done


@torch.inference_mode()
def transcribe_batch(
    model,
    processor,
    device: torch.device,
    audio_arrays: List[np.ndarray],
    sampling_rate: int,
    language: str,
    task: str,
) -> List[str]:
    inputs = processor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    input_features = inputs["input_features"].to(device)

    gen_kwargs = {}
    # Use language/task forcing if supported by the processor/tokenizer
    # This avoids auto language detection variability.
    if language:
        try:
            forced_ids = processor.get_decoder_prompt_ids(language=language, task=task)
            gen_kwargs["forced_decoder_ids"] = forced_ids
        except Exception:
            # Fallback: rely on model defaults
            pass

    predicted_ids = model.generate(input_features, **gen_kwargs)
    texts = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    texts = [normalize_text(t) for t in texts]
    return texts


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    refs = df["ref"].tolist()
    hyps = df["hyp"].tolist()
    return {
        "wer": float(wer(refs, hyps)),
        "cer": float(cer(refs, hyps)),
        "n": int(len(df)),
    }


def append_results_md(
    md_path: Path,
    title: str,
    config: EvalConfig,
    overall: Dict[str, float],
    by_subset: Dict[str, Dict[str, float]],
):
    ensure_dir(md_path.parent)

    lines = []
    lines.append(f"\n## {title}\n")
    lines.append(f"- model: `{config.model_id}`\n")
    lines.append(f"- device: `{config.device}`\n")
    lines.append(f"- language: `{config.language}` / task: `{config.task}`\n")
    lines.append(f"- manifest: `{config.manifest_path}`\n")
    lines.append(f"- evaluated: n={overall['n']}\n")
    lines.append(f"- **WER**: {overall['wer']:.4f}\n")
    lines.append(f"- **CER**: {overall['cer']:.4f}\n")
    lines.append("\n### Breakdown by subset\n\n")
    lines.append("| subset | n | WER | CER |\n")
    lines.append("|---|---:|---:|---:|\n")
    for subset, m in by_subset.items():
        lines.append(f"| {subset} | {m['n']} | {m['wer']:.4f} | {m['cer']:.4f} |\n")

    with md_path.open("a", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/eval_baseline.yaml")
    ap.add_argument("--max_items", type=int, default=None, help="Override config max_items. Use -1 for full.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing preds.jsonl if available.")
    args = ap.parse_args()

    cfg_dict = load_yaml(Path(args.config))
    cfg = EvalConfig(**cfg_dict)

    if args.max_items is not None:
        cfg.max_items = args.max_items

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    manifest_path = Path(cfg.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path.resolve()}")

    out_dir = Path(cfg.out_dir)
    ensure_dir(out_dir)

    pred_path = out_dir / "preds.jsonl"
    existing = load_existing_preds(pred_path) if args.resume else {}

    rows = read_jsonl(manifest_path)

    # Shuffle for quick partial evaluation (so it isn't biased to early files)
    random.shuffle(rows)

    if cfg.max_items != -1:
        rows = rows[: cfg.max_items]

    device = pick_device(cfg.device)
    model, processor = build_model_and_processor(cfg.model_id, device)

    to_process = [r for r in rows if r["utt_id"] not in existing]
    done_preds: List[Dict[str, Any]] = list(existing.values()) if args.resume else []

    pbar = tqdm(total=len(to_process), desc="Evaluating", unit="utt")
    batch = []
    meta = []

    def flush_batch():
        nonlocal batch, meta, done_preds
        if not batch:
            return
        hyps = transcribe_batch(
            model=model,
            processor=processor,
            device=device,
            audio_arrays=batch,
            sampling_rate=16000,
            language=cfg.language,
            task=cfg.task,
        )
        for m, hyp in zip(meta, hyps):
            obj = {
                "utt_id": m["utt_id"],
                "subset": m.get("subset", ""),
                "split": m.get("split", ""),
                "speaker_id": m.get("speaker_id", ""),
                "audio_path": m["audio_path"],
                "ref": normalize_text(m.get("text", "")),
                "hyp": hyp,
            }
            done_preds.append(obj)
            if cfg.write_predictions:
                with pred_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        batch, meta = [], []

    for r in to_process:
        audio_path = Path(r["audio_path"])
        x, sr = load_audio(audio_path, target_sr=16000)
        batch.append(x)
        meta.append(r)

        if len(batch) >= cfg.batch_size:
            flush_batch()
            pbar.update(cfg.batch_size)

    # Flush remainder
    if batch:
        flush_batch()
        pbar.update(len(batch))
    pbar.close()

    # Build dataframe
    df = pd.DataFrame(done_preds)
    if df.empty:
        raise RuntimeError("No predictions generated. Check input manifest and audio paths.")

    overall = compute_metrics(df)

    by_subset = {}
    for subset, g in df.groupby("subset"):
        by_subset[subset] = compute_metrics(g)

    # Save summary JSON
    summary = {
        "model_id": cfg.model_id,
        "device": cfg.device,
        "language": cfg.language,
        "task": cfg.task,
        "manifest": cfg.manifest_path,
        "overall": overall,
        "by_subset": by_subset,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Evaluation complete")
    print(f"     overall: n={overall['n']} WER={overall['wer']:.4f} CER={overall['cer']:.4f}")
    for subset, m in by_subset.items():
        print(f"     subset={subset}: n={m['n']} WER={m['wer']:.4f} CER={m['cer']:.4f}")

    if cfg.append_results_md:
        title = f"{time.strftime('%Y-%m-%d')} Baseline (Whisper-small)"
        append_results_md(
            md_path=Path(cfg.results_md_path),
            title=title,
            config=cfg,
            overall=overall,
            by_subset=by_subset,
        )
        print(f"[OK] Appended results to {cfg.results_md_path}")


if __name__ == "__main__":
    main()
