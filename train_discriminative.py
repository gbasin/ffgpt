from __future__ import annotations

import argparse

from ffgpt import FFDiscriminativeTrainer, FFTransformer, TransformerConfig, Vocab, generate_all_problems, run_roundtrip_tests, train_test_split
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
    train_problems, test_problems = train_test_split(problems)

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
