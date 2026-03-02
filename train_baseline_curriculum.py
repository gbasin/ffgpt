from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffgpt import (
    BackpropTrainer,
    BaselineTransformer,
    TransformerConfig,
    Vocab,
    generate_problems_for_operand_digits,
    max_seq_len_for_operand_digits,
    max_sum_for_operand_digits,
)
from ffgpt.data import Problem
from ffgpt.utils import ensure_dir, set_seed


def parse_int_list(raw: str) -> list[int]:
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in '{raw}'")
    return [int(value) for value in values]


def align_stage_values(stage_count: int, values: list[int], name: str) -> list[int]:
    if len(values) == 1:
        return values * stage_count
    if len(values) != stage_count:
        raise ValueError(
            f"{name} must have either one value or exactly one value per stage. "
            f"Got {len(values)} values for {stage_count} stages."
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curriculum training: baseline model over increasing operand digit lengths")
    parser.add_argument("--digit-stages", type=str, default="1,2,3", help="Comma-separated operand digit lengths")
    parser.add_argument("--stage-steps", type=str, default="1200", help="One value or comma list aligned to stages")
    parser.add_argument(
        "--stage-samples",
        type=str,
        default="0,10000,20000",
        help="One value or comma list aligned to stages. 0 means exhaustive when feasible.",
    )
    parser.add_argument("--holdout-mod", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--exhaustive-limit", type=int, default=100_000)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/curriculum")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-train-max-samples", type=int, default=None)
    parser.add_argument("--eval-test-max-samples", type=int, default=None)
    parser.add_argument("--no-eval-step-one", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    return parser.parse_args()


def split_modulo(problems: list[Problem], mod: int, remainder: int) -> tuple[list[Problem], list[Problem]]:
    test = [problem for problem in problems if (problem.answer % mod) == remainder]
    train = [problem for problem in problems if (problem.answer % mod) != remainder]
    return train, test


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    stages = parse_int_list(args.digit_stages)
    if any(stage <= 0 for stage in stages):
        raise ValueError(f"All digit stages must be >=1. Got: {stages}")

    stage_steps = align_stage_values(len(stages), parse_int_list(args.stage_steps), "stage_steps")
    stage_samples = align_stage_values(len(stages), parse_int_list(args.stage_samples), "stage_samples")

    vocab = Vocab()
    max_digits = max(stages)
    model = BaselineTransformer(
        TransformerConfig(
            vocab_size=vocab.size,
            max_seq_len=max_seq_len_for_operand_digits(max_digits),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_blocks=args.n_blocks,
            mlp_hidden=args.mlp_hidden,
            dropout=0.0,
        )
    )

    summary: dict[str, object] = {
        "stages": [],
        "model_max_seq_len": model.config.max_seq_len,
        "checkpoints_root": str(args.checkpoint_dir),
    }

    for stage_idx, (digits, steps, samples) in enumerate(zip(stages, stage_steps, stage_samples), start=1):
        sample_count = None if samples == 0 else int(samples)
        problems = generate_problems_for_operand_digits(
            operand_digits=digits,
            num_samples=sample_count,
            seed=args.seed + stage_idx,
            exhaustive_limit=args.exhaustive_limit,
        )
        train_problems, test_problems = split_modulo(problems, args.holdout_mod, args.holdout_remainder)

        if not train_problems or not test_problems:
            raise RuntimeError(
                f"Stage {digits} digits has empty split: train={len(train_problems)} test={len(test_problems)}. "
                f"Adjust --stage-samples or holdout settings."
            )

        stage_seq_len = max_seq_len_for_operand_digits(digits)
        stage_answer_tokens = digits + 1
        stage_answer_max = max_sum_for_operand_digits(digits)

        stage_ckpt_dir = ensure_dir(Path(args.checkpoint_dir) / f"d{digits}")

        trainer = BackpropTrainer(
            model=model,
            vocab=vocab,
            train_problems=train_problems,
            test_problems=test_problems,
            lr=args.lr,
            batch_size=args.batch_size,
            num_steps=steps,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=str(stage_ckpt_dir),
            device=args.device,
            sequence_length=stage_seq_len,
            max_answer_tokens=stage_answer_tokens,
            max_answer_value=stage_answer_max,
            eval_every=args.eval_every,
            eval_train_max_samples=args.eval_train_max_samples,
            eval_test_max_samples=args.eval_test_max_samples,
            eval_at_step_one=not args.no_eval_step_one,
        )

        result = trainer.train(log_every=args.log_every)
        train_eval = result["train"]
        test_eval = result["test"]

        stage_summary = {
            "digits": digits,
            "steps": steps,
            "samples": len(problems),
            "train_size": len(train_problems),
            "test_size": len(test_problems),
            "sequence_length": stage_seq_len,
            "max_answer_tokens": stage_answer_tokens,
            "max_answer_value": stage_answer_max,
            "train_token_accuracy": train_eval.token_accuracy,
            "train_sequence_exact_match": train_eval.sequence_exact_match,
            "test_token_accuracy": test_eval.token_accuracy,
            "test_sequence_exact_match": test_eval.sequence_exact_match,
            "checkpoint": result["checkpoint"],
        }
        summary["stages"].append(stage_summary)

        print(
            f"[curriculum stage d={digits}] train_exact={train_eval.sequence_exact_match:.4f} "
            f"test_exact={test_eval.sequence_exact_match:.4f}"
        )

    out_dir = ensure_dir(args.checkpoint_dir)
    summary_path = out_dir / "curriculum_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nCurriculum summary")
    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
