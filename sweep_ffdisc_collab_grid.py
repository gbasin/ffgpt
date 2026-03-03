from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from ffgpt import (
    FFDiscriminativeTrainer,
    FFTransformer,
    TransformerConfig,
    Vocab,
    coverage_preserving_sum_split,
    generate_all_problems,
)
from ffgpt.utils import set_seed


@dataclass
class GridResult:
    mode: str
    seed: int
    steps: int
    collab_weight: float
    kl_sync_weight: float
    aux_weight: float
    nonfinal_aux_weight: float
    final_aux_weight: float
    inter_block_norm: str
    goodness_aggregation: str
    fit_goodness_weights: bool
    run_tag: str
    train_good_exact: float
    test_good_exact: float
    train_logit_exact: float
    test_logit_exact: float
    checkpoint_path: str


def parse_float_list(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"no numeric values parsed from '{raw}'")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"no integer values parsed from '{raw}'")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid sweep for FF-discriminative D1/D2 collaboration settings")
    parser.add_argument("--mode", type=str, default="combined", choices=["perblockaux", "combined"])
    parser.add_argument("--steps-coarse", type=int, default=1000)
    parser.add_argument("--steps-refine", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold-momentum", type=float, default=0.9)
    parser.add_argument("--final-aux-weight", type=float, default=1.0)
    parser.add_argument("--aux-weight", type=float, default=0.05)
    parser.add_argument("--nonfinal-aux-weight", type=float, default=0.25)
    parser.add_argument("--collab-weights", type=str, default="0.0,0.05,0.1,0.2")
    parser.add_argument("--kl-weights", type=str, default="0.0,0.05,0.1,0.2")
    parser.add_argument("--coarse-seeds", type=str, default="42,43")
    parser.add_argument("--refine-seeds", type=str, default="42,43,44")
    parser.add_argument("--top-k-refine", type=int, default=6)
    parser.add_argument("--run-prefix", type=str, default="coverage_collab")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sweeps")
    return parser.parse_args()


def to_rows(results: list[GridResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def build_model(vocab: Vocab, args: argparse.Namespace) -> FFTransformer:
    config = TransformerConfig(
        vocab_size=vocab.size,
        max_seq_len=6,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        mlp_hidden=args.mlp_hidden,
        dropout=0.0,
    )
    return FFTransformer(config)


def mode_settings(mode: str) -> tuple[str, str, bool]:
    if mode == "combined":
        return "layernorm", "weighted_sum", True
    return "none", "uniform_sum", False


def train_one(
    *,
    mode: str,
    steps: int,
    seed: int,
    collab_weight: float,
    kl_sync_weight: float,
    aux_weight: float,
    nonfinal_aux_weight: float,
    final_aux_weight: float,
    split_seed: int,
    test_size: int,
    checkpoint_dir: str,
    log_every: int,
    batch_size: int,
    lr: float,
    threshold_momentum: float,
    run_tag: str,
    args: argparse.Namespace,
) -> GridResult:
    set_seed(seed)
    vocab = Vocab()
    problems = generate_all_problems()
    train_problems, test_problems = coverage_preserving_sum_split(
        problems=problems,
        test_size=test_size,
        seed=split_seed,
    )
    model = build_model(vocab=vocab, args=args)
    inter_block_norm, goodness_aggregation, fit_goodness_weights = mode_settings(mode)

    trainer = FFDiscriminativeTrainer(
        model=model,
        vocab=vocab,
        train_problems=train_problems,
        test_problems=test_problems,
        lr=lr,
        batch_size=batch_size,
        num_steps=steps,
        checkpoint_every=max(steps, 1),
        checkpoint_dir=checkpoint_dir,
        threshold_momentum=threshold_momentum,
        logit_aux_weight=aux_weight,
        use_per_block_logit_aux=True,
        final_block_logit_aux_weight=final_aux_weight,
        nonfinal_block_logit_aux_weight=nonfinal_aux_weight,
        collaborative_global_offset_weight=collab_weight,
        kl_sync_weight=kl_sync_weight,
        inter_block_norm=inter_block_norm,
        goodness_aggregation=goodness_aggregation,
        fit_goodness_block_weights=fit_goodness_weights,
        sequence_length=6,
        max_answer_tokens=2,
        max_answer_value=18,
        run_tag=run_tag,
        seed=seed,
    )
    result = trainer.train(log_every=max(log_every, 1))
    train_good = result["train_goodness"]
    test_good = result["test_goodness"]
    train_log = result["train_logits"]
    test_log = result["test_logits"]

    return GridResult(
        mode=mode,
        seed=seed,
        steps=steps,
        collab_weight=collab_weight,
        kl_sync_weight=kl_sync_weight,
        aux_weight=aux_weight,
        nonfinal_aux_weight=nonfinal_aux_weight,
        final_aux_weight=final_aux_weight,
        inter_block_norm=inter_block_norm,
        goodness_aggregation=goodness_aggregation,
        fit_goodness_weights=fit_goodness_weights,
        run_tag=run_tag,
        train_good_exact=float(train_good.sequence_exact_match),
        test_good_exact=float(test_good.sequence_exact_match),
        train_logit_exact=float(train_log.sequence_exact_match),
        test_logit_exact=float(test_log.sequence_exact_match),
        checkpoint_path=str(result["checkpoint"]),
    )


def summarize(results: list[GridResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[GridResult]] = {}
    for result in results:
        key = (
            result.mode,
            int(round(result.collab_weight * 1000)),
            int(round(result.kl_sync_weight * 1000)),
        )
        grouped.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for _, group in grouped.items():
        row = {
            "mode": group[0].mode,
            "collab_weight": group[0].collab_weight,
            "kl_sync_weight": group[0].kl_sync_weight,
            "aux_weight": group[0].aux_weight,
            "nonfinal_aux_weight": group[0].nonfinal_aux_weight,
            "seeds": [r.seed for r in group],
            "steps": group[0].steps,
            "mean_test_good_exact": mean(r.test_good_exact for r in group),
            "mean_test_logit_exact": mean(r.test_logit_exact for r in group),
            "mean_train_good_exact": mean(r.train_good_exact for r in group),
            "mean_train_logit_exact": mean(r.train_logit_exact for r in group),
        }
        row["balanced_score"] = 0.5 * (row["mean_test_good_exact"] + row["mean_test_logit_exact"])
        rows.append(row)
    rows.sort(key=lambda x: (x["balanced_score"], x["mean_test_good_exact"], x["mean_test_logit_exact"]), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            row_out = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            writer.writerow(row_out)


def main() -> None:
    args = parse_args()
    collab_weights = parse_float_list(args.collab_weights)
    kl_weights = parse_float_list(args.kl_weights)
    coarse_seeds = parse_int_list(args.coarse_seeds)
    refine_seeds = parse_int_list(args.refine_seeds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coarse_results: list[GridResult] = []
    total = len(collab_weights) * len(kl_weights) * len(coarse_seeds)
    idx = 0
    for collab_weight in collab_weights:
        for kl_sync_weight in kl_weights:
            for seed in coarse_seeds:
                idx += 1
                run_tag = (
                    f"{args.run_prefix}_{args.mode}_coarse_s{seed}"
                    f"_c{collab_weight:g}_k{kl_sync_weight:g}"
                )
                print(
                    f"[coarse {idx}/{total}] mode={args.mode} seed={seed} "
                    f"collab={collab_weight} kl={kl_sync_weight}"
                )
                coarse_results.append(
                    train_one(
                        mode=args.mode,
                        steps=args.steps_coarse,
                        seed=seed,
                        collab_weight=collab_weight,
                        kl_sync_weight=kl_sync_weight,
                        aux_weight=args.aux_weight,
                        nonfinal_aux_weight=args.nonfinal_aux_weight,
                        final_aux_weight=args.final_aux_weight,
                        split_seed=args.split_seed,
                        test_size=args.test_size,
                        checkpoint_dir=args.checkpoint_dir,
                        log_every=args.log_every,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        threshold_momentum=args.threshold_momentum,
                        run_tag=run_tag,
                        args=args,
                    )
                )

    coarse_summary = summarize(coarse_results)
    top = coarse_summary[: max(args.top_k_refine, 0)]
    print("\n[coarse top]")
    for row in top:
        print(json.dumps(row, indent=2))

    refine_results: list[GridResult] = []
    if top:
        total_refine = len(top) * len(refine_seeds)
        refine_idx = 0
        for row in top:
            collab_weight = float(row["collab_weight"])
            kl_sync_weight = float(row["kl_sync_weight"])
            for seed in refine_seeds:
                refine_idx += 1
                run_tag = (
                    f"{args.run_prefix}_{args.mode}_refine_s{seed}"
                    f"_c{collab_weight:g}_k{kl_sync_weight:g}"
                )
                print(
                    f"[refine {refine_idx}/{total_refine}] mode={args.mode} seed={seed} "
                    f"collab={collab_weight} kl={kl_sync_weight}"
                )
                refine_results.append(
                    train_one(
                        mode=args.mode,
                        steps=args.steps_refine,
                        seed=seed,
                        collab_weight=collab_weight,
                        kl_sync_weight=kl_sync_weight,
                        aux_weight=args.aux_weight,
                        nonfinal_aux_weight=args.nonfinal_aux_weight,
                        final_aux_weight=args.final_aux_weight,
                        split_seed=args.split_seed,
                        test_size=args.test_size,
                        checkpoint_dir=args.checkpoint_dir,
                        log_every=args.log_every,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        threshold_momentum=args.threshold_momentum,
                        run_tag=run_tag,
                        args=args,
                    )
                )

    refine_summary = summarize(refine_results)
    print("\n[refine top]")
    for row in refine_summary[:10]:
        print(json.dumps(row, indent=2))

    coarse_rows = to_rows(coarse_results)
    refine_rows = to_rows(refine_results)
    result_blob = {
        "timestamp": timestamp,
        "config": vars(args),
        "coarse_results": coarse_rows,
        "coarse_summary": coarse_summary,
        "refine_results": refine_rows,
        "refine_summary": refine_summary,
    }

    json_path = out_dir / f"{args.run_prefix}_{timestamp}.json"
    coarse_csv = out_dir / f"{args.run_prefix}_{timestamp}_coarse_summary.csv"
    refine_csv = out_dir / f"{args.run_prefix}_{timestamp}_refine_summary.csv"

    json_path.write_text(json.dumps(result_blob, indent=2), encoding="utf-8")
    if coarse_summary:
        write_csv(coarse_csv, coarse_summary)
    if refine_summary:
        write_csv(refine_csv, refine_summary)

    print(f"\n[done] wrote {json_path}")
    if coarse_summary:
        print(f"[done] wrote {coarse_csv}")
    else:
        print("[done] coarse summary is empty; no coarse CSV written")
    if refine_summary:
        print(f"[done] wrote {refine_csv}")
    else:
        print("[done] refine summary is empty; no refine CSV written")


if __name__ == "__main__":
    main()
