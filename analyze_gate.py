from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn

from ffgpt import (
    FFTransformer,
    TransformerConfig,
    Vocab,
    coverage_preserving_sum_split,
    generate_all_problems,
    generate_problems_for_operand_digits,
    max_seq_len_for_operand_digits,
    max_sum_for_operand_digits,
    train_test_split,
)
from ffgpt.model import GatedResidualGate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze gate activations from a gated staged checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to gated staged checkpoint")
    parser.add_argument("--operand-digits", type=int, required=True)
    parser.add_argument("--samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--exhaustive-limit", type=int, default=100_000)
    parser.add_argument("--split", type=str, default="random", choices=["mod5", "coverage", "random"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="gate_analysis")
    parser.add_argument("--max-problems", type=int, default=200, help="Max problems to analyze (for heatmap clarity)")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def has_carry(a: int, b: int, position: int) -> bool:
    """Check if adding a+b produces a carry at a given digit position (0=ones)."""
    carry = 0
    for pos in range(position + 1):
        d_a = (a // (10 ** pos)) % 10
        d_b = (b // (10 ** pos)) % 10
        total = d_a + d_b + carry
        carry = total // 10
        if pos == position:
            return carry > 0
    return False


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config_dict = checkpoint["config"]
    config = TransformerConfig(**config_dict)
    vocab = Vocab()

    model = FFTransformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()

    # Reconstruct gates
    gate_state = checkpoint.get("gate_state")
    if gate_state is None:
        print("ERROR: checkpoint does not contain gate state (not a phase-2 checkpoint)")
        return

    n_gates = len(model.blocks) - 1
    gates = nn.ModuleList(
        [GatedResidualGate(config.d_model, checkpoint.get("gate_init_bias", -5.0)) for _ in range(n_gates)]
    ).to(args.device)
    gates.load_state_dict(gate_state)
    gates.eval()

    # Generate test problems
    if args.operand_digits == 1 and args.samples == 0:
        problems = generate_all_problems()
    else:
        problems = generate_problems_for_operand_digits(
            operand_digits=args.operand_digits,
            num_samples=None if args.samples == 0 else args.samples,
            seed=args.seed,
            exhaustive_limit=args.exhaustive_limit,
        )

    if args.split == "mod5":
        _, test_problems = train_test_split(problems)
    elif args.split == "coverage":
        _, test_problems = coverage_preserving_sum_split(
            problems=problems, test_size=args.test_size, seed=args.split_seed,
        )
    else:
        rng = random.Random(args.split_seed)
        shuffled = list(problems)
        rng.shuffle(shuffled)
        test_problems = shuffled[len(shuffled) - args.test_size :]

    if len(test_problems) > args.max_problems:
        test_problems = test_problems[: args.max_problems]

    print(f"Analyzing {len(test_problems)} test problems with {n_gates} gate(s)")

    max_seq_len = max_seq_len_for_operand_digits(args.operand_digits)
    max_answer_tokens = args.operand_digits + 1

    # Collect gate values
    all_gate_vals: list[list[list[float]]] = []  # [problem][gate_idx][position]
    problem_labels: list[str] = []
    carry_masks: list[list[bool]] = []

    with torch.no_grad():
        for problem in test_problems:
            tokens = vocab.encode_equation(problem.a, problem.b, problem.answer, max_len=max_seq_len)
            input_ids = torch.tensor([tokens[:-1]], dtype=torch.long, device=args.device)
            pad_mask = input_ids != vocab.pad_id

            # Run forward and capture gate values manually
            x = model.embed(input_ids)
            seq_len = x.shape[1]

            gate_vals_per_gate: list[list[float]] = []
            for block_idx, block in enumerate(model.blocks):
                if block_idx > 0:
                    x = x.detach()
                block_input = x
                x, _ = block(x, pad_mask=pad_mask, causal=True)
                if block_idx > 0:
                    delta = x - block_input
                    x, gv = gates[block_idx - 1](block_input, delta)
                    # Average gate value across d_model for each position
                    gate_means = gv[0].mean(dim=-1).tolist()  # [seq_len]
                    gate_vals_per_gate.append(gate_means)

            all_gate_vals.append(gate_vals_per_gate)
            problem_labels.append(f"{problem.a}+{problem.b}={problem.answer}")

            # Compute carry mask for answer positions
            # Answer positions start after prompt (a digits + '+' + b digits + '=')
            prompt_len = len(str(problem.a)) + 1 + len(str(problem.b)) + 1
            carries: list[bool] = [False] * seq_len
            for ans_pos in range(max_answer_tokens):
                seq_pos = prompt_len + ans_pos - 1  # -1 because input is shifted
                if 0 <= seq_pos < seq_len:
                    # Answer digit position (from right): max_answer_tokens - 1 - ans_pos
                    digit_pos = max_answer_tokens - 1 - ans_pos
                    carries[seq_pos] = has_carry(problem.a, problem.b, digit_pos)
            carry_masks.append(carries)

    # Compute statistics
    n_positions = len(all_gate_vals[0][0]) if all_gate_vals and all_gate_vals[0] else 0
    for gate_idx in range(n_gates):
        print(f"\n--- Gate {gate_idx} (between block {gate_idx} and block {gate_idx + 1}) ---")

        # Per-position mean
        pos_means = []
        for pos in range(n_positions):
            vals = [all_gate_vals[p][gate_idx][pos] for p in range(len(test_problems))]
            pos_means.append(sum(vals) / len(vals))
        print(f"Per-position mean gate activation: {[f'{v:.4f}' for v in pos_means]}")

        # Split by carry vs no-carry at answer positions
        prompt_len_example = len(str(test_problems[0].a)) + 1 + len(str(test_problems[0].b)) + 1
        for ans_pos in range(max_answer_tokens):
            seq_pos = prompt_len_example + ans_pos - 1
            if 0 <= seq_pos < n_positions:
                carry_vals = []
                no_carry_vals = []
                for p in range(len(test_problems)):
                    if carry_masks[p][seq_pos]:
                        carry_vals.append(all_gate_vals[p][gate_idx][seq_pos])
                    else:
                        no_carry_vals.append(all_gate_vals[p][gate_idx][seq_pos])
                carry_mean = sum(carry_vals) / len(carry_vals) if carry_vals else float("nan")
                no_carry_mean = sum(no_carry_vals) / len(no_carry_vals) if no_carry_vals else float("nan")
                print(
                    f"  answer_pos={ans_pos} (seq_pos={seq_pos}): "
                    f"carry={carry_mean:.4f} (n={len(carry_vals)}) "
                    f"no_carry={no_carry_mean:.4f} (n={len(no_carry_vals)})"
                )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "n_problems": len(test_problems),
        "n_gates": n_gates,
        "n_positions": n_positions,
        "problem_labels": problem_labels,
        "gate_values": all_gate_vals,
        "carry_masks": carry_masks,
        "checkpoint": args.checkpoint,
    }
    json_path = output_dir / "gate_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Generate heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        for gate_idx in range(n_gates):
            data = np.array([[all_gate_vals[p][gate_idx][pos] for pos in range(n_positions)] for p in range(len(test_problems))])

            fig, ax = plt.subplots(figsize=(max(8, n_positions), max(6, len(test_problems) * 0.15)))
            im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.1, data.max()))
            ax.set_xlabel("Sequence position")
            ax.set_ylabel("Problem")
            ax.set_title(f"Gate {gate_idx} activation (block {gate_idx}→{gate_idx + 1})")
            plt.colorbar(im, ax=ax, label="Gate value (mean over d_model)")

            # Mark carry positions
            for p_idx in range(len(test_problems)):
                for pos in range(n_positions):
                    if carry_masks[p_idx][pos]:
                        ax.plot(pos, p_idx, "x", color="blue", markersize=3, markeredgewidth=0.5)

            ax.set_yticks(range(0, len(test_problems), max(1, len(test_problems) // 20)))

            png_path = output_dir / f"gate_{gate_idx}_heatmap.png"
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"Saved heatmap: {png_path}")

    except ImportError:
        print("matplotlib not available, skipping heatmap generation")


if __name__ == "__main__":
    main()
