from __future__ import annotations

import argparse
import json
from pathlib import Path

from ffgpt import (
    BackpropTrainer,
    BaselineTransformer,
    FFAutoregressiveTrainer,
    FFDiscriminativeTrainer,
    FFTransformer,
    TransformerConfig,
    Vocab,
    coverage_preserving_sum_split,
    generate_all_problems,
    generate_problems_for_operand_digits,
    max_seq_len_for_operand_digits,
    max_sum_for_operand_digits,
    summarize_answer_token_coverage,
    train_test_split,
)
from ffgpt.utils import (
    compute_confusion_matrix,
    ensure_dir,
    latest_checkpoint,
    load_checkpoint,
    plot_confusion_matrix,
    plot_curve,
    plot_multi_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline + FF models on train/test sets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--discriminative-checkpoint", type=str, default=None)
    parser.add_argument("--autoregressive-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints/eval")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--operand-digits", type=int, default=1)
    parser.add_argument("--samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--exhaustive-limit", type=int, default=100_000)
    parser.add_argument("--split", type=str, default="mod5", choices=["mod5", "coverage", "random"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--run-tag", type=str, default=None, help="Checkpoint run tag (defaults to split-based tag)")
    parser.add_argument("--eval-train-max-samples", type=int, default=None)
    parser.add_argument("--eval-test-max-samples", type=int, default=None)
    parser.add_argument("--skip-goodness-eval", action="store_true")
    parser.add_argument("--logit-rank-eval-max-candidates", type=int, default=512)
    return parser.parse_args()


def resolve_checkpoint_path(explicit: str | None, checkpoint_dir: str, mode: str, run_tag: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)

    path = latest_checkpoint(checkpoint_dir, mode, run_tag=run_tag)
    if path is None and run_tag is not None:
        # Backward compatibility for older checkpoints without run tags.
        path = latest_checkpoint(checkpoint_dir, mode, run_tag=None)
    if path is None:
        raise FileNotFoundError(
            f"No checkpoint found for mode={mode} in {checkpoint_dir} "
            f"(run_tag={run_tag!r})"
        )
    return path


def config_from_checkpoint(config_dict: dict) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=int(config_dict["vocab_size"]),
        max_seq_len=int(config_dict["max_seq_len"]),
        d_model=int(config_dict["d_model"]),
        n_heads=int(config_dict["n_heads"]),
        n_blocks=int(config_dict["n_blocks"]),
        mlp_hidden=int(config_dict["mlp_hidden"]),
        dropout=float(config_dict.get("dropout", 0.0)),
    )


def main() -> None:
    args = parse_args()

    out_dir = ensure_dir(args.output_dir)
    vocab = Vocab()
    if args.operand_digits == 1 and args.samples == 0:
        problems = generate_all_problems()
    else:
        problems = generate_problems_for_operand_digits(
            operand_digits=args.operand_digits,
            num_samples=None if args.samples == 0 else args.samples,
            seed=args.split_seed,
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

    def sample_subset(problems: list, max_samples: int | None, seed: int) -> list:
        if max_samples is None or max_samples >= len(problems):
            return problems
        if max_samples <= 0:
            return []
        import random

        rng = random.Random(seed)
        indices = rng.sample(range(len(problems)), max_samples)
        return [problems[idx] for idx in indices]

    train_eval_problems = sample_subset(train_problems, args.eval_train_max_samples, args.split_seed + 101)
    test_eval_problems = sample_subset(test_problems, args.eval_test_max_samples, args.split_seed + 202)

    baseline_ckpt_path = resolve_checkpoint_path(
        args.baseline_checkpoint,
        args.checkpoint_dir,
        "baseline",
        run_tag=run_tag,
    )
    disc_ckpt_path = resolve_checkpoint_path(
        args.discriminative_checkpoint,
        args.checkpoint_dir,
        "discriminative",
        run_tag=run_tag,
    )
    ar_ckpt_path = resolve_checkpoint_path(
        args.autoregressive_checkpoint,
        args.checkpoint_dir,
        "autoregressive",
        run_tag=run_tag,
    )

    baseline_ckpt = load_checkpoint(baseline_ckpt_path, map_location=args.device)
    disc_ckpt = load_checkpoint(disc_ckpt_path, map_location=args.device)
    ar_ckpt = load_checkpoint(ar_ckpt_path, map_location=args.device)

    baseline_model = BaselineTransformer(config_from_checkpoint(baseline_ckpt["config"]))
    baseline_model.load_state_dict(baseline_ckpt["model_state"])

    disc_model = FFTransformer(config_from_checkpoint(disc_ckpt["config"]))
    disc_model.load_state_dict(disc_ckpt["model_state"])

    ar_model = FFTransformer(config_from_checkpoint(ar_ckpt["config"]))
    ar_model.load_state_dict(ar_ckpt["model_state"])

    baseline_trainer = BackpropTrainer(
        model=baseline_model,
        vocab=vocab,
        train_problems=train_problems,
        test_problems=test_problems,
        num_steps=0,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        sequence_length=baseline_ckpt.get("sequence_length", max_seq_len),
        max_answer_tokens=baseline_ckpt.get("max_answer_tokens", max_answer_tokens),
        max_answer_value=baseline_ckpt.get("max_answer_value", max_answer_value),
    )
    disc_trainer = FFDiscriminativeTrainer(
        model=disc_model,
        vocab=vocab,
        train_problems=train_problems,
        test_problems=test_problems,
        num_steps=0,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        sequence_length=disc_ckpt.get("sequence_length", max_seq_len),
        max_answer_tokens=disc_ckpt.get("max_answer_tokens", max_answer_tokens),
        max_answer_value=disc_ckpt.get("max_answer_value", max_answer_value),
        max_full_candidate_answers=disc_ckpt.get("max_full_candidate_answers", 2048),
        candidate_answers=disc_ckpt.get("candidate_answers"),
        eval_every=disc_ckpt.get("eval_every"),
        eval_train_max_samples=disc_ckpt.get("eval_train_max_samples"),
        eval_test_max_samples=disc_ckpt.get("eval_test_max_samples"),
        enable_goodness_eval=disc_ckpt.get("enable_goodness_eval", True),
        enable_logit_rank_diagnostics=disc_ckpt.get("enable_logit_rank_diagnostics", False),
        logit_rank_eval_max_candidates=disc_ckpt.get("logit_rank_eval_max_candidates", 512),
    )
    ar_trainer = FFAutoregressiveTrainer(
        model=ar_model,
        vocab=vocab,
        train_problems=train_problems,
        test_problems=test_problems,
        num_steps=0,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        sequence_length=ar_ckpt.get("sequence_length", max_seq_len),
        max_answer_tokens=ar_ckpt.get("max_answer_tokens", max_answer_tokens),
        max_answer_value=ar_ckpt.get("max_answer_value", max_answer_value),
        max_full_candidate_answers=ar_ckpt.get("max_full_candidate_answers", 2048),
        candidate_answers=ar_ckpt.get("candidate_answers"),
        eval_every=ar_ckpt.get("eval_every"),
        eval_train_max_samples=ar_ckpt.get("eval_train_max_samples"),
        eval_test_max_samples=ar_ckpt.get("eval_test_max_samples"),
        enable_goodness_eval=ar_ckpt.get("enable_goodness_eval", True),
    )

    baseline_train = baseline_trainer.evaluate(train_eval_problems)
    baseline_test = baseline_trainer.evaluate(test_eval_problems)

    if args.skip_goodness_eval:
        disc_train_good = None
        disc_test_good = None
    else:
        disc_train_good = disc_trainer.evaluate_goodness(train_eval_problems, causal=False)
        disc_test_good = disc_trainer.evaluate_goodness(test_eval_problems, causal=False)
    disc_train_log = disc_trainer.evaluate_logits(train_eval_problems)
    disc_test_log = disc_trainer.evaluate_logits(test_eval_problems)
    disc_train_log_diag = disc_trainer.evaluate_logits_detailed(
        train_eval_problems,
        top_k=5,
        collect_examples=False,
        max_candidates=args.logit_rank_eval_max_candidates,
    )
    disc_test_log_diag = disc_trainer.evaluate_logits_detailed(
        test_eval_problems,
        top_k=5,
        collect_examples=False,
        max_candidates=args.logit_rank_eval_max_candidates,
    )
    if bool(disc_train_log_diag.get("skipped", False)) or bool(disc_test_log_diag.get("skipped", False)):
        print(
            "[warn] discriminative logit rank diagnostics skipped because candidate pool exceeded "
            f"--logit-rank-eval-max-candidates={args.logit_rank_eval_max_candidates}"
        )

    ar_train_logits = ar_trainer.evaluate_logits(train_eval_problems)
    ar_test_logits = ar_trainer.evaluate_logits(test_eval_problems)
    if args.skip_goodness_eval:
        ar_train_good = None
        ar_test_good = None
    else:
        ar_train_good = ar_trainer.evaluate_goodness(train_eval_problems)
        ar_test_good = ar_trainer.evaluate_goodness(test_eval_problems)

    num_classes = max_answer_value + 1
    if num_classes <= 512:
        disc_targets = disc_test_good.targets if disc_test_good is not None else disc_test_log.targets
        disc_preds = disc_test_good.predictions if disc_test_good is not None else disc_test_log.predictions
        baseline_train_cm = compute_confusion_matrix(baseline_train.targets, baseline_train.predictions, num_classes=num_classes)
        baseline_test_cm = compute_confusion_matrix(baseline_test.targets, baseline_test.predictions, num_classes=num_classes)
        disc_test_cm = compute_confusion_matrix(disc_targets, disc_preds, num_classes=num_classes)
        ar_test_cm = compute_confusion_matrix(ar_test_logits.targets, ar_test_logits.predictions, num_classes=num_classes)

        plot_confusion_matrix(baseline_train_cm, out_dir / "baseline_train_confusion.png", "Baseline Train Confusion")
        plot_confusion_matrix(baseline_test_cm, out_dir / "baseline_test_confusion.png", "Baseline Test Confusion")
        plot_confusion_matrix(disc_test_cm, out_dir / "discriminative_test_confusion.png", "Discriminative FF Test Confusion")
        plot_confusion_matrix(ar_test_cm, out_dir / "autoregressive_test_confusion.png", "Autoregressive FF Test Confusion")
    else:
        print(f"[warn] skipping confusion matrices because num_classes={num_classes} > 512")

    for name, ckpt in [
        ("baseline", baseline_ckpt),
        ("discriminative", disc_ckpt),
        ("autoregressive", ar_ckpt),
    ]:
        history = ckpt.get("history", {})
        steps = history.get("step", [])
        losses = history.get("loss", [])

        if steps and losses and len(steps) == len(losses):
            plot_curve(steps, losses, out_dir / f"{name}_loss.png", f"{name} Loss", "Loss")

        train_exact = history.get("train_sequence_exact_match", [])
        test_exact = history.get("test_sequence_exact_match", [])
        if steps and len(train_exact) == len(steps) and len(test_exact) == len(steps):
            plot_multi_curve(
                steps,
                {"train_exact": train_exact, "test_exact": test_exact},
                out_dir / f"{name}_exact_match.png",
                f"{name} Sequence Exact Match",
                "Exact Match",
            )

    def plot_block_diagnostics(prefix: str, history: dict) -> None:
        step_count = len(history.get("block_separation", [[]])[0]) if history.get("block_separation") else 0
        if step_count == 0:
            return
        steps = list(range(1, step_count + 1))

        sep_series = {
            f"block_{idx}": values
            for idx, values in enumerate(history.get("block_separation", []))
            if len(values) == step_count
        }
        if sep_series:
            plot_multi_curve(
                steps,
                sep_series,
                out_dir / f"{prefix}_block_separation.png",
                f"{prefix} Block Goodness Separation",
                "g_pos - g_neg",
            )

        gpos_series = {
            f"block_{idx}": values
            for idx, values in enumerate(history.get("block_g_pos", []))
            if len(values) == step_count
        }
        gneg_series = {
            f"block_{idx}": values
            for idx, values in enumerate(history.get("block_g_neg", []))
            if len(values) == step_count
        }
        if gpos_series:
            plot_multi_curve(
                steps,
                gpos_series,
                out_dir / f"{prefix}_block_g_pos.png",
                f"{prefix} Block Positive Goodness",
                "g_pos",
            )
        if gneg_series:
            plot_multi_curve(
                steps,
                gneg_series,
                out_dir / f"{prefix}_block_g_neg.png",
                f"{prefix} Block Negative Goodness",
                "g_neg",
            )

    plot_block_diagnostics("discriminative", disc_ckpt.get("history", {}))
    plot_block_diagnostics("autoregressive", ar_ckpt.get("history", {}))

    summary = {
        "split": {
            "strategy": args.split,
            "split_seed": args.split_seed,
            "train_size": len(train_problems),
            "test_size": len(test_problems),
            "eval_train_size": len(train_eval_problems),
            "eval_test_size": len(test_eval_problems),
            "missing_test_tokens_in_train_by_position": missing,
        },
        "checkpoints": {
            "baseline": str(baseline_ckpt_path),
            "discriminative": str(disc_ckpt_path),
            "autoregressive": str(ar_ckpt_path),
        },
        "run_tag": run_tag,
        "baseline": {
            "train_token_accuracy": baseline_train.token_accuracy,
            "train_sequence_exact_match": baseline_train.sequence_exact_match,
            "test_token_accuracy": baseline_test.token_accuracy,
            "test_sequence_exact_match": baseline_test.sequence_exact_match,
        },
        "discriminative": {
            "goodness_train_token_accuracy": None if disc_train_good is None else disc_train_good.token_accuracy,
            "goodness_train_sequence_exact_match": None if disc_train_good is None else disc_train_good.sequence_exact_match,
            "goodness_train_candidate_ranking_accuracy": None
            if disc_train_good is None
            else disc_train_good.candidate_ranking_accuracy,
            "goodness_test_token_accuracy": None if disc_test_good is None else disc_test_good.token_accuracy,
            "goodness_test_sequence_exact_match": None if disc_test_good is None else disc_test_good.sequence_exact_match,
            "goodness_test_candidate_ranking_accuracy": None
            if disc_test_good is None
            else disc_test_good.candidate_ranking_accuracy,
            "logit_train_sequence_exact_match": disc_train_log.sequence_exact_match,
            "logit_test_sequence_exact_match": disc_test_log.sequence_exact_match,
            "logit_train_mean_correct_rank": disc_train_log_diag["mean_correct_rank"],
            "logit_test_mean_correct_rank": disc_test_log_diag["mean_correct_rank"],
            "logit_test_per_sum_accuracy": disc_test_log_diag["per_sum_accuracy"],
            "logit_rank_diagnostics_skipped": bool(disc_test_log_diag.get("skipped", False)),
        },
        "autoregressive": {
            "logit_train_token_accuracy": ar_train_logits.token_accuracy,
            "logit_train_sequence_exact_match": ar_train_logits.sequence_exact_match,
            "logit_test_token_accuracy": ar_test_logits.token_accuracy,
            "logit_test_sequence_exact_match": ar_test_logits.sequence_exact_match,
            "goodness_train_candidate_ranking_accuracy": None
            if ar_train_good is None
            else ar_train_good.candidate_ranking_accuracy,
            "goodness_test_candidate_ranking_accuracy": None
            if ar_test_good is None
            else ar_test_good.candidate_ranking_accuracy,
        },
        "artifacts_dir": str(out_dir),
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
