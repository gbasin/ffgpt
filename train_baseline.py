from __future__ import annotations

import argparse

from ffgpt import BackpropTrainer, BaselineTransformer, TransformerConfig, Vocab, generate_all_problems, run_roundtrip_tests, train_test_split
from ffgpt.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train backprop baseline autoregressive transformer")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_roundtrip_tests()

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
    )

    result = trainer.train(log_every=args.log_every)

    train_eval = result["train"]
    test_eval = result["test"]

    print("\nBaseline final metrics")
    print(f"train token_accuracy={train_eval.token_accuracy:.4f}")
    print(f"train sequence_exact_match={train_eval.sequence_exact_match:.4f}")
    print(f"test token_accuracy={test_eval.token_accuracy:.4f}")
    print(f"test sequence_exact_match={test_eval.sequence_exact_match:.4f}")


if __name__ == "__main__":
    main()
