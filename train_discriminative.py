from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffgpt import (
    FFDiscriminativeTrainer,
    FFTransformer,
    TransformerConfig,
    Vocab,
    coverage_preserving_sum_split,
    generate_all_problems,
    generate_problems_for_operand_digits,
    max_seq_len_for_operand_digits,
    max_sum_for_operand_digits,
    run_roundtrip_tests,
    summarize_answer_token_coverage,
    train_test_split,
)
from ffgpt.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FF discriminative transformer")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--operand-digits", type=int, default=1)
    parser.add_argument("--samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--exhaustive-limit", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-train-max-samples", type=int, default=None)
    parser.add_argument("--eval-test-max-samples", type=int, default=None)
    parser.add_argument("--skip-goodness-eval", action="store_true")
    parser.add_argument("--enable-logit-rank-diagnostics", action="store_true")
    parser.add_argument("--logit-rank-eval-max-candidates", type=int, default=512)
    parser.add_argument("--split", type=str, default="mod5", choices=["mod5", "coverage", "random"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional checkpoint suffix tag")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--threshold-momentum", type=float, default=0.9)
    parser.add_argument("--logit-aux-weight", type=float, default=1.0)
    parser.add_argument("--max-full-candidate-answers", type=int, default=2048)
    parser.add_argument("--diagnose-logits", action="store_true")
    parser.add_argument("--diagnose-top-k", type=int, default=5)
    parser.add_argument("--diagnose-max-examples", type=int, default=20)
    parser.add_argument("--diagnose-output", type=str, default=None)
    parser.add_argument(
        "--near-miss-start-step",
        type=int,
        default=None,
        help="Step to switch from random negatives to near-miss negatives (default: half of total steps)",
    )
    parser.add_argument(
        "--near-miss-offsets",
        type=str,
        default="1",
        help="Comma-separated offsets for near-miss negatives, e.g. '1' or '1,2'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.operand_digits == 1:
        run_roundtrip_tests()

    near_miss_offsets = tuple(int(x.strip()) for x in args.near_miss_offsets.split(",") if x.strip())
    near_miss_start_step = args.near_miss_start_step
    if near_miss_start_step is None:
        near_miss_start_step = max(1, args.steps // 2)

    vocab = Vocab()
    if args.operand_digits == 1 and args.samples == 0:
        problems = generate_all_problems()
    else:
        problems = generate_problems_for_operand_digits(
            operand_digits=args.operand_digits,
            num_samples=None if args.samples == 0 else args.samples,
            seed=args.seed,
            exhaustive_limit=args.exhaustive_limit,
        )
    if args.test_size <= 0 or args.test_size >= len(problems):
        raise ValueError(f"--test-size must be in [1, {len(problems)-1}], got {args.test_size}")

    if args.split == "mod5":
        train_problems, test_problems = train_test_split(problems)
    elif args.split == "coverage":
        train_problems, test_problems = coverage_preserving_sum_split(
            problems=problems,
            test_size=args.test_size,
            seed=args.split_seed,
        )
    else:
        import random

        rng = random.Random(args.split_seed)
        shuffled = list(problems)
        rng.shuffle(shuffled)
        n_test = args.test_size
        n_train = len(shuffled) - n_test
        train_problems, test_problems = shuffled[:n_train], shuffled[n_train:]

    print(
        f"[split] strategy={args.split} train={len(train_problems)} test={len(test_problems)} "
        f"split_seed={args.split_seed} operand_digits={args.operand_digits}"
    )
    run_tag = (
        args.run_tag
        if args.run_tag is not None
        else f"d{args.operand_digits}_{args.split}_s{args.split_seed}"
    )
    print(f"[run] run_tag={run_tag}")
    max_answer_tokens = args.operand_digits + 1
    max_answer_value = max_sum_for_operand_digits(args.operand_digits)
    max_seq_len = max_seq_len_for_operand_digits(args.operand_digits)
    coverage = summarize_answer_token_coverage(
        train_problems=train_problems,
        test_problems=test_problems,
        max_answer_tokens=max_answer_tokens,
    )
    missing = coverage["missing_test_tokens_in_train_by_position"]
    if any(missing):
        print("[warn] test answer tokens missing from train targets by output position:")
        for idx, missing_tokens in enumerate(missing):
            print(f"  position_{idx}: {missing_tokens}")

    config = TransformerConfig(
        vocab_size=vocab.size,
        max_seq_len=max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        mlp_hidden=args.mlp_hidden,
        dropout=0.0,
    )

    model = FFTransformer(config)

    trainer = FFDiscriminativeTrainer(
        model=model,
        vocab=vocab,
        train_problems=train_problems,
        test_problems=test_problems,
        lr=args.lr,
        batch_size=args.batch_size,
        num_steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        threshold_momentum=args.threshold_momentum,
        logit_aux_weight=args.logit_aux_weight,
        sequence_length=max_seq_len,
        max_answer_tokens=max_answer_tokens,
        max_answer_value=max_answer_value,
        max_full_candidate_answers=args.max_full_candidate_answers,
        eval_every=args.eval_every,
        eval_train_max_samples=args.eval_train_max_samples,
        eval_test_max_samples=args.eval_test_max_samples,
        enable_goodness_eval=not args.skip_goodness_eval,
        enable_logit_rank_diagnostics=args.enable_logit_rank_diagnostics,
        logit_rank_eval_max_candidates=args.logit_rank_eval_max_candidates,
        near_miss_start_step=near_miss_start_step,
        near_miss_offsets=near_miss_offsets,
        device=args.device,
        seed=args.seed,
        run_tag=run_tag,
    )

    result = trainer.train(log_every=args.log_every)

    train_good = result["train_goodness"]
    test_good = result["test_goodness"]
    train_log = result["train_logits"]
    test_log = result["test_logits"]

    print("\nDiscriminative FF final metrics (goodness inference)")
    print(f"train token_accuracy={train_good.token_accuracy:.4f}")
    print(f"train sequence_exact_match={train_good.sequence_exact_match:.4f}")
    print(f"train candidate_ranking_accuracy={train_good.candidate_ranking_accuracy:.4f}")
    print(f"test token_accuracy={test_good.token_accuracy:.4f}")
    print(f"test sequence_exact_match={test_good.sequence_exact_match:.4f}")
    print(f"test candidate_ranking_accuracy={test_good.candidate_ranking_accuracy:.4f}")

    print("\nDiscriminative FF final metrics (logit inference)")
    print(f"train sequence_exact_match={train_log.sequence_exact_match:.4f}")
    print(f"test sequence_exact_match={test_log.sequence_exact_match:.4f}")

    if args.diagnose_logits:
        train_diag = trainer.evaluate_logits_detailed(
            train_problems,
            top_k=args.diagnose_top_k,
            collect_examples=True,
            max_examples=args.diagnose_max_examples,
            max_candidates=args.logit_rank_eval_max_candidates,
        )
        test_diag = trainer.evaluate_logits_detailed(
            test_problems,
            top_k=args.diagnose_top_k,
            collect_examples=True,
            max_examples=args.diagnose_max_examples,
            max_candidates=args.logit_rank_eval_max_candidates,
        )

        print("\nDiscriminative logit diagnostics")
        print(f"train mean_correct_rank={train_diag['mean_correct_rank']:.3f}")
        print(f"test mean_correct_rank={test_diag['mean_correct_rank']:.3f}")
        print(f"test per_sum_accuracy={test_diag['per_sum_accuracy']}")
        if bool(train_diag.get("skipped", False)) or bool(test_diag.get("skipped", False)):
            print(
                "[warn] rank diagnostics skipped because candidate pool exceeded "
                f"--logit-rank-eval-max-candidates={args.logit_rank_eval_max_candidates}"
            )

        diagnostics = {
            "run_tag": run_tag,
            "train": train_diag,
            "test": test_diag,
        }
        output_path = (
            Path(args.diagnose_output)
            if args.diagnose_output is not None
            else Path(args.checkpoint_dir) / f"discriminative_{run_tag}_logit_diagnostics.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"saved_diagnostics={output_path}")


if __name__ == "__main__":
    main()
