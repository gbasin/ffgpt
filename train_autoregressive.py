from __future__ import annotations

import argparse

from ffgpt import (
    FFAutoregressiveTrainer,
    FFTransformer,
    TransformerConfig,
    Vocab,
    generate_all_problems,
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-blocks", type=int, default=2)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--k-negatives", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-min", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_roundtrip_tests()

    vocab = Vocab()
    problems = generate_all_problems()
    train_problems, test_problems = train_test_split(problems)
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
        device=args.device,
        seed=args.seed,
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
