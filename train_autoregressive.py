from __future__ import annotations

import argparse

from ffgpt import (
    FFAutoregressiveTrainer,
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
    parser = argparse.ArgumentParser(description="Train FF autoregressive transformer")
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
    parser.add_argument("--split", type=str, default="mod5", choices=["mod5", "coverage", "random"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional checkpoint suffix tag")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--k-negatives", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-min", type=float, default=0.1)
    parser.add_argument("--goodness-aux-weight", type=float, default=1.0)
    parser.add_argument("--threshold-momentum", type=float, default=0.9)
    parser.add_argument("--max-full-candidate-answers", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.operand_digits == 1:
        run_roundtrip_tests()

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
        device=args.device,
        seed=args.seed,
        run_tag=run_tag,
    )

    result = trainer.train(log_every=args.log_every)

    train_logits = result["train_logits"]
    test_logits = result["test_logits"]
    train_good = result["train_goodness"]
    test_good = result["test_goodness"]

    print("\nAutoregressive FF final metrics (logit inference)")
    print(f"train token_accuracy={train_logits.token_accuracy:.4f}")
    print(f"train sequence_exact_match={train_logits.sequence_exact_match:.4f}")
    print(f"test token_accuracy={test_logits.token_accuracy:.4f}")
    print(f"test sequence_exact_match={test_logits.sequence_exact_match:.4f}")

    print("\nAutoregressive FF final metrics (goodness inference)")
    print(f"train candidate_ranking_accuracy={train_good.candidate_ranking_accuracy:.4f}")
    print(f"test candidate_ranking_accuracy={test_good.candidate_ranking_accuracy:.4f}")


if __name__ == "__main__":
    main()
