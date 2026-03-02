from __future__ import annotations

import argparse
import json

from ffgpt import (
    BackpropTrainer,
    BaselineTransformer,
    FFAutoregressiveTrainer,
    FFTransformer,
    TransformerConfig,
    Vocab,
    generate_problems_for_operand_digits,
    max_seq_len_for_operand_digits,
    max_sum_for_operand_digits,
    summarize_answer_token_coverage,
)
from ffgpt.utils import ensure_dir, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train on one digit length and evaluate OOD on another (e.g., d->d+1)"
    )
    parser.add_argument("--mode", type=str, choices=["baseline", "ff-ar"], required=True)

    parser.add_argument("--train-operand-digits", type=int, required=True)
    parser.add_argument("--test-operand-digits", type=int, required=True)
    parser.add_argument("--train-samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--test-samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--exhaustive-limit", type=int, default=100_000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed-offset", type=int, default=0)
    parser.add_argument("--test-seed-offset", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-train-max-samples", type=int, default=None)
    parser.add_argument("--eval-test-max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-tag", type=str, default=None)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=256)

    # FF-AR specific knobs.
    parser.add_argument("--k-negatives", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-min", type=float, default=0.1)
    parser.add_argument("--goodness-aux-weight", type=float, default=1.0)
    parser.add_argument("--threshold-momentum", type=float, default=0.9)
    parser.add_argument("--max-full-candidate-answers", type=int, default=2048)
    parser.add_argument("--skip-goodness-eval", action="store_true")
    parser.add_argument("--no-detach-output-embedding", action="store_true")
    parser.add_argument("--use-per-block-output-heads", action="store_true")
    parser.add_argument("--final-block-loss-weight", type=float, default=1.0)
    parser.add_argument("--nonfinal-block-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.train_operand_digits <= 0 or args.test_operand_digits <= 0:
        raise ValueError("train/test operand digits must both be >= 1")

    train_seed = args.seed + args.train_seed_offset
    test_seed = args.seed + args.test_seed_offset
    train_num_samples = None if args.train_samples == 0 else args.train_samples
    test_num_samples = None if args.test_samples == 0 else args.test_samples

    train_problems = generate_problems_for_operand_digits(
        operand_digits=args.train_operand_digits,
        num_samples=train_num_samples,
        seed=train_seed,
        exhaustive_limit=args.exhaustive_limit,
    )
    test_problems = generate_problems_for_operand_digits(
        operand_digits=args.test_operand_digits,
        num_samples=test_num_samples,
        seed=test_seed,
        exhaustive_limit=args.exhaustive_limit,
    )

    if not train_problems or not test_problems:
        raise RuntimeError(
            f"empty dataset: train={len(train_problems)} test={len(test_problems)}"
        )

    max_digits = max(args.train_operand_digits, args.test_operand_digits)
    max_seq_len = max_seq_len_for_operand_digits(max_digits)
    max_answer_tokens = max_digits + 1
    max_answer_value = max_sum_for_operand_digits(max_digits)

    vocab = Vocab()
    config = TransformerConfig(
        vocab_size=vocab.size,
        max_seq_len=max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks,
        mlp_hidden=args.mlp_hidden,
        dropout=0.0,
    )

    run_tag = (
        args.run_tag
        if args.run_tag is not None
        else (
            f"ood_{args.mode}_d{args.train_operand_digits}_to_d{args.test_operand_digits}"
            f"_n{len(train_problems)}v{len(test_problems)}_s{args.seed}"
        )
    )

    print(
        f"[ood] mode={args.mode} train_digits={args.train_operand_digits} test_digits={args.test_operand_digits} "
        f"train={len(train_problems)} test={len(test_problems)} train_seed={train_seed} test_seed={test_seed}"
    )
    print(f"[run] run_tag={run_tag}")

    coverage = summarize_answer_token_coverage(
        train_problems=train_problems,
        test_problems=test_problems,
        max_answer_tokens=max_answer_tokens,
    )
    missing = coverage["missing_test_tokens_in_train_by_position"]
    if any(missing):
        print("[warn] OOD test answer tokens missing from train targets by output position:")
        for idx, missing_tokens in enumerate(missing):
            print(f"  position_{idx}: {missing_tokens}")

    summary: dict[str, object] = {
        "mode": args.mode,
        "run_tag": run_tag,
        "train_operand_digits": args.train_operand_digits,
        "test_operand_digits": args.test_operand_digits,
        "train_size": len(train_problems),
        "test_size": len(test_problems),
        "sequence_length": max_seq_len,
        "max_answer_tokens": max_answer_tokens,
        "max_answer_value": max_answer_value,
        "missing_test_tokens_in_train_by_position": missing,
        "train_seed": train_seed,
        "test_seed": test_seed,
    }

    if args.mode == "baseline":
        model = BaselineTransformer(config)
        trainer = BackpropTrainer(
            model=model,
            vocab=vocab,
            train_problems=train_problems,
            test_problems=test_problems,
            lr=args.lr,
            batch_size=args.batch_size,
            num_steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
            sequence_length=max_seq_len,
            max_answer_tokens=max_answer_tokens,
            max_answer_value=max_answer_value,
            eval_every=args.eval_every,
            eval_train_max_samples=args.eval_train_max_samples,
            eval_test_max_samples=args.eval_test_max_samples,
            run_tag=run_tag,
        )
        result = trainer.train(log_every=args.log_every)
        train_eval = result["train"]
        test_eval = result["test"]
        summary.update(
            {
                "train_token_accuracy": train_eval.token_accuracy,
                "train_sequence_exact_match": train_eval.sequence_exact_match,
                "test_token_accuracy": test_eval.token_accuracy,
                "test_sequence_exact_match": test_eval.sequence_exact_match,
                "checkpoint": result["checkpoint"],
            }
        )
        print("\nOOD baseline final metrics")
        print(f"train sequence_exact_match={train_eval.sequence_exact_match:.4f}")
        print(f"test sequence_exact_match={test_eval.sequence_exact_match:.4f}")

    else:
        model = FFTransformer(config)
        trainer = FFAutoregressiveTrainer(
            model=model,
            vocab=vocab,
            train_problems=train_problems,
            test_problems=test_problems,
            lr=args.lr,
            batch_size=args.batch_size,
            num_steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            k_negatives=args.k_negatives,
            temperature=args.temperature,
            temperature_min=args.temperature_min,
            goodness_aux_weight=args.goodness_aux_weight,
            threshold_momentum=args.threshold_momentum,
            sequence_length=max_seq_len,
            max_answer_tokens=max_answer_tokens,
            max_answer_value=max_answer_value,
            max_full_candidate_answers=args.max_full_candidate_answers,
            eval_every=args.eval_every,
            eval_train_max_samples=args.eval_train_max_samples,
            eval_test_max_samples=args.eval_test_max_samples,
            enable_goodness_eval=not args.skip_goodness_eval,
            output_embedding_detached=not args.no_detach_output_embedding,
            use_per_block_output_heads=args.use_per_block_output_heads,
            final_block_loss_weight=args.final_block_loss_weight,
            nonfinal_block_loss_weight=args.nonfinal_block_loss_weight,
            device=args.device,
            seed=args.seed,
            run_tag=run_tag,
        )
        result = trainer.train(log_every=args.log_every)
        train_logits = result["train_logits"]
        test_logits = result["test_logits"]
        train_good = result["train_goodness"]
        test_good = result["test_goodness"]

        summary.update(
            {
                "train_logit_token_accuracy": train_logits.token_accuracy,
                "train_logit_sequence_exact_match": train_logits.sequence_exact_match,
                "test_logit_token_accuracy": test_logits.token_accuracy,
                "test_logit_sequence_exact_match": test_logits.sequence_exact_match,
                "train_goodness_candidate_ranking_accuracy": train_good.candidate_ranking_accuracy,
                "test_goodness_candidate_ranking_accuracy": test_good.candidate_ranking_accuracy,
                "checkpoint": result["checkpoint"],
                "output_embedding_detached": not args.no_detach_output_embedding,
                "use_per_block_output_heads": args.use_per_block_output_heads,
                "final_block_loss_weight": args.final_block_loss_weight,
                "nonfinal_block_loss_weight": args.nonfinal_block_loss_weight,
            }
        )
        print("\nOOD FF-AR final metrics")
        print(f"train logit sequence_exact_match={train_logits.sequence_exact_match:.4f}")
        print(f"test logit sequence_exact_match={test_logits.sequence_exact_match:.4f}")

    out_dir = ensure_dir(f"{args.checkpoint_dir}/ood_eval")
    summary_path = out_dir / f"{run_tag}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
