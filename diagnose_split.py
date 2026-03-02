from __future__ import annotations

import argparse
import random
from statistics import mean

import torch

from ffgpt import (
    BaselineTransformer,
    TransformerConfig,
    Vocab,
    coverage_preserving_sum_split,
    generate_all_problems,
    generate_problems_for_operand_digits,
    summarize_answer_token_coverage,
    train_test_split,
)
from ffgpt.data import Problem, max_seq_len_for_operand_digits
from ffgpt.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose split coverage and optional checkpoint behavior")
    parser.add_argument("--operand-digits", type=int, default=1)
    parser.add_argument("--samples", type=int, default=0, help="0 uses full grid when feasible")
    parser.add_argument("--split", type=str, default="mod5", choices=["mod5", "coverage", "random"])
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def split_random(problems: list[Problem], seed: int) -> tuple[list[Problem], list[Problem]]:
    rng = random.Random(seed)
    shuffled = list(problems)
    rng.shuffle(shuffled)
    n_train = int(0.8 * len(shuffled))
    return shuffled[:n_train], shuffled[n_train:]


def build_dataset(operand_digits: int, samples: int, seed: int) -> list[Problem]:
    if operand_digits == 1 and samples == 0:
        return generate_all_problems()
    return generate_problems_for_operand_digits(
        operand_digits=operand_digits,
        num_samples=None if samples == 0 else samples,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    vocab = Vocab()

    problems = build_dataset(args.operand_digits, args.samples, args.seed)
    if args.split == "mod5":
        train_problems, test_problems = train_test_split(problems)
    elif args.split == "coverage":
        train_problems, test_problems = coverage_preserving_sum_split(
            problems=problems,
            test_size=args.test_size,
            seed=args.seed,
        )
    else:
        train_problems, test_problems = split_random(problems, args.seed)

    max_answer_tokens = args.operand_digits + 1
    coverage = summarize_answer_token_coverage(
        train_problems=train_problems,
        test_problems=test_problems,
        max_answer_tokens=max_answer_tokens,
    )

    print(f"dataset size={len(problems)} train={len(train_problems)} test={len(test_problems)}")
    print(f"split={args.split} operand_digits={args.operand_digits} max_answer_tokens={max_answer_tokens}")
    print("missing_test_tokens_in_train_by_position:")
    for pos, missing in enumerate(coverage["missing_test_tokens_in_train_by_position"]):
        print(f"  position_{pos}: {missing}")

    if args.checkpoint is None:
        return

    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    config = TransformerConfig(**ckpt["config"])
    model = BaselineTransformer(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    prompt_len = max_seq_len_for_operand_digits(args.operand_digits) - max_answer_tokens
    if prompt_len < 2:
        print("[warn] prompt length too small for probe")
        return

    digit_tokens = [str(i) for i in range(10)]
    token_ids = {d: vocab.token_to_id[d] for d in digit_tokens}
    first_step_logits = {d: [] for d in digit_tokens}

    with torch.no_grad():
        for p in test_problems[: min(500, len(test_problems))]:
            prompt_tokens = [*list(str(p.a)), "+", *list(str(p.b)), "="]
            input_ids = torch.tensor([vocab.encode_tokens(prompt_tokens)], dtype=torch.long, device=args.device)
            pad_mask = input_ids != vocab.pad_id
            logits, _, _ = model(input_ids, pad_mask=pad_mask, causal=True)
            step_logits = logits[0, -1]
            for d, tid in token_ids.items():
                first_step_logits[d].append(float(step_logits[tid].item()))

    print("avg first-answer logits on sampled test prompts:")
    for d in digit_tokens:
        vals = first_step_logits[d]
        print(f"  {d}: {mean(vals):.4f}")


if __name__ == "__main__":
    main()
