from __future__ import annotations

import argparse

from ffgpt import (
    FFDiscriminativeTrainer,
    FFTransformer,
    TransformerConfig,
    Vocab,
    coverage_preserving_sum_split,
    generate_all_problems,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--split", type=str, default="mod5", choices=["mod5", "coverage", "random"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--threshold-momentum", type=float, default=0.9)
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
    run_roundtrip_tests()

    near_miss_offsets = tuple(int(x.strip()) for x in args.near_miss_offsets.split(",") if x.strip())
    near_miss_start_step = args.near_miss_start_step
    if near_miss_start_step is None:
        near_miss_start_step = max(1, args.steps // 2)

    vocab = Vocab()
    problems = generate_all_problems()
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
        f"split_seed={args.split_seed}"
    )
    coverage = summarize_answer_token_coverage(
        train_problems=train_problems,
        test_problems=test_problems,
        max_answer_tokens=2,
    )
    missing = coverage["missing_test_tokens_in_train_by_position"]
    if any(missing):
        print("[warn] test answer tokens missing from train targets by output position:")
        for idx, missing_tokens in enumerate(missing):
            print(f"  position_{idx}: {missing_tokens}")

    config = TransformerConfig(
        vocab_size=vocab.size,
        max_seq_len=6,
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
        near_miss_start_step=near_miss_start_step,
        near_miss_offsets=near_miss_offsets,
        device=args.device,
        seed=args.seed,
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


if __name__ == "__main__":
    main()
