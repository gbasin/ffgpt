from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import torch

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
)
from ffgpt.data import Problem, answer_to_token_ids_variable
from ffgpt.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze greedy vs teacher-forced behavior on OOD checkpoints.")
    parser.add_argument("--mode", type=str, choices=["baseline", "ff-ar"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--train-operand-digits", type=int, required=True)
    parser.add_argument("--test-operand-digits", type=int, required=True)
    parser.add_argument("--train-samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--test-samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--exhaustive-limit", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed-offset", type=int, default=0)
    parser.add_argument("--test-seed-offset", type=int, default=1)
    parser.add_argument("--eval-train-max-samples", type=int, default=512)
    parser.add_argument("--eval-test-max-samples", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def maybe_subsample(problems: list[Problem], max_samples: int | None, seed: int) -> list[Problem]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(problems):
        return list(problems)
    import random

    rng = random.Random(seed)
    return rng.sample(problems, max_samples)


def _build_full_tensor(problems: Sequence[Problem], vocab: Vocab, seq_len: int, device: torch.device) -> torch.Tensor:
    rows = [vocab.encode_equation(problem.a, problem.b, problem.answer, max_len=seq_len) for problem in problems]
    return torch.tensor(rows, dtype=torch.long, device=device)


def _teacher_forced_from_logits(
    *,
    full_tokens: torch.Tensor,
    logits: torch.Tensor,
    vocab: Vocab,
    max_answer_tokens: int,
) -> dict[str, Any]:
    # full_tokens: [B, T], logits: [B, T-1, V]
    y = full_tokens[:, 1:]
    pred = torch.argmax(logits, dim=-1)
    bsz = full_tokens.shape[0]

    token_correct = 0
    token_total = 0
    seq_correct = 0
    per_pos_correct = [0 for _ in range(max_answer_tokens)]
    per_pos_total = [0 for _ in range(max_answer_tokens)]

    for b in range(bsz):
        eq_positions = (full_tokens[b] == vocab.equals_id).nonzero(as_tuple=False)
        if eq_positions.numel() == 0:
            continue
        eq_pos = int(eq_positions[0].item())

        seq_ok = True
        for j in range(max_answer_tokens):
            y_idx = eq_pos + j
            if y_idx >= y.shape[1]:
                seq_ok = False
                continue
            p_tok = int(pred[b, y_idx].item())
            t_tok = int(y[b, y_idx].item())
            per_pos_total[j] += 1
            token_total += 1
            if p_tok == t_tok:
                token_correct += 1
                per_pos_correct[j] += 1
            else:
                seq_ok = False
        if seq_ok:
            seq_correct += 1

    return {
        "token_accuracy": float(token_correct / max(token_total, 1)),
        "sequence_exact_match": float(seq_correct / max(bsz, 1)),
        "per_position_accuracy": [
            float(c / max(t, 1)) for c, t in zip(per_pos_correct, per_pos_total)
        ],
    }


@torch.no_grad()
def baseline_teacher_forced_metrics(
    trainer: BackpropTrainer,
    problems: list[Problem],
    batch_size: int,
) -> dict[str, Any]:
    trainer.model.eval()
    metrics_accum = {"token_correct": 0, "token_total": 0, "seq_correct": 0, "count": 0}
    per_pos_correct = [0 for _ in range(trainer.max_answer_tokens)]
    per_pos_total = [0 for _ in range(trainer.max_answer_tokens)]

    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        full = _build_full_tensor(chunk, trainer.vocab, trainer.sequence_length, trainer.device)
        x = full[:, :-1]
        pad_mask = x != trainer.vocab.pad_id
        logits, _, _ = trainer.model(x, pad_mask=pad_mask, causal=True)
        out = _teacher_forced_from_logits(
            full_tokens=full,
            logits=logits,
            vocab=trainer.vocab,
            max_answer_tokens=trainer.max_answer_tokens,
        )

        y = full[:, 1:]
        pred = torch.argmax(logits, dim=-1)
        for b in range(full.shape[0]):
            eq_positions = (full[b] == trainer.vocab.equals_id).nonzero(as_tuple=False)
            if eq_positions.numel() == 0:
                continue
            eq_pos = int(eq_positions[0].item())
            seq_ok = True
            for j in range(trainer.max_answer_tokens):
                y_idx = eq_pos + j
                if y_idx >= y.shape[1]:
                    seq_ok = False
                    continue
                t_tok = int(y[b, y_idx].item())
                p_tok = int(pred[b, y_idx].item())
                per_pos_total[j] += 1
                metrics_accum["token_total"] += 1
                if p_tok == t_tok:
                    per_pos_correct[j] += 1
                    metrics_accum["token_correct"] += 1
                else:
                    seq_ok = False
            metrics_accum["count"] += 1
            if seq_ok:
                metrics_accum["seq_correct"] += 1

    return {
        "token_accuracy": float(metrics_accum["token_correct"] / max(metrics_accum["token_total"], 1)),
        "sequence_exact_match": float(metrics_accum["seq_correct"] / max(metrics_accum["count"], 1)),
        "per_position_accuracy": [
            float(c / max(t, 1)) for c, t in zip(per_pos_correct, per_pos_total)
        ],
    }


@torch.no_grad()
def ff_teacher_forced_metrics(
    trainer: FFAutoregressiveTrainer,
    problems: list[Problem],
    batch_size: int,
) -> dict[str, Any]:
    trainer.model.eval()
    metrics_accum = {"token_correct": 0, "token_total": 0, "seq_correct": 0, "count": 0}
    per_pos_correct = [0 for _ in range(trainer.max_answer_tokens)]
    per_pos_total = [0 for _ in range(trainer.max_answer_tokens)]

    for start in range(0, len(problems), batch_size):
        chunk = problems[start : start + batch_size]
        full = _build_full_tensor(chunk, trainer.vocab, trainer.sequence_length, trainer.device)
        x = full[:, :-1]
        pad_mask = x != trainer.vocab.pad_id
        block_outputs, _ = trainer.model(x, pad_mask=pad_mask, causal=True, detach_between_blocks=True)
        logits = trainer._project_block_logits(len(block_outputs) - 1, block_outputs[-1])

        y = full[:, 1:]
        pred = torch.argmax(logits, dim=-1)
        for b in range(full.shape[0]):
            eq_positions = (full[b] == trainer.vocab.equals_id).nonzero(as_tuple=False)
            if eq_positions.numel() == 0:
                continue
            eq_pos = int(eq_positions[0].item())
            seq_ok = True
            for j in range(trainer.max_answer_tokens):
                y_idx = eq_pos + j
                if y_idx >= y.shape[1]:
                    seq_ok = False
                    continue
                t_tok = int(y[b, y_idx].item())
                p_tok = int(pred[b, y_idx].item())
                per_pos_total[j] += 1
                metrics_accum["token_total"] += 1
                if p_tok == t_tok:
                    per_pos_correct[j] += 1
                    metrics_accum["token_correct"] += 1
                else:
                    seq_ok = False
            metrics_accum["count"] += 1
            if seq_ok:
                metrics_accum["seq_correct"] += 1

    return {
        "token_accuracy": float(metrics_accum["token_correct"] / max(metrics_accum["token_total"], 1)),
        "sequence_exact_match": float(metrics_accum["seq_correct"] / max(metrics_accum["count"], 1)),
        "per_position_accuracy": [
            float(c / max(t, 1)) for c, t in zip(per_pos_correct, per_pos_total)
        ],
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    vocab = Vocab()

    train = generate_problems_for_operand_digits(
        operand_digits=args.train_operand_digits,
        num_samples=None if args.train_samples == 0 else args.train_samples,
        seed=args.seed + args.train_seed_offset,
        exhaustive_limit=args.exhaustive_limit,
    )
    test = generate_problems_for_operand_digits(
        operand_digits=args.test_operand_digits,
        num_samples=None if args.test_samples == 0 else args.test_samples,
        seed=args.seed + args.test_seed_offset,
        exhaustive_limit=args.exhaustive_limit,
    )
    eval_train = maybe_subsample(train, args.eval_train_max_samples, seed=args.seed + 3001)
    eval_test = maybe_subsample(test, args.eval_test_max_samples, seed=args.seed + 3002)

    ckpt = load_checkpoint(args.checkpoint, map_location=args.device)
    max_digits = max(args.train_operand_digits, args.test_operand_digits)
    default_seq_len = max_seq_len_for_operand_digits(max_digits)
    default_answer_tokens = max_digits + 1
    default_answer_value = max_sum_for_operand_digits(max_digits)

    summary: dict[str, Any] = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "train_size": len(train),
        "test_size": len(test),
        "eval_train_size": len(eval_train),
        "eval_test_size": len(eval_test),
    }

    if args.mode == "baseline":
        model = BaselineTransformer(TransformerConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state"])
        trainer = BackpropTrainer(
            model=model,
            vocab=vocab,
            train_problems=train,
            test_problems=test,
            num_steps=0,
            checkpoint_dir="checkpoints",
            device=args.device,
            sequence_length=ckpt.get("sequence_length", default_seq_len),
            max_answer_tokens=ckpt.get("max_answer_tokens", default_answer_tokens),
            max_answer_value=ckpt.get("max_answer_value", default_answer_value),
        )
        greedy_train = trainer.evaluate(eval_train)
        greedy_test = trainer.evaluate(eval_test)
        tf_train = baseline_teacher_forced_metrics(trainer, eval_train, batch_size=args.batch_size)
        tf_test = baseline_teacher_forced_metrics(trainer, eval_test, batch_size=args.batch_size)
        summary["greedy_train"] = {
            "token_accuracy": greedy_train.token_accuracy,
            "sequence_exact_match": greedy_train.sequence_exact_match,
        }
        summary["greedy_test"] = {
            "token_accuracy": greedy_test.token_accuracy,
            "sequence_exact_match": greedy_test.sequence_exact_match,
        }
        summary["teacher_forced_train"] = tf_train
        summary["teacher_forced_test"] = tf_test
    else:
        model = FFTransformer(TransformerConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model_state"])
        trainer = FFAutoregressiveTrainer(
            model=model,
            vocab=vocab,
            train_problems=train,
            test_problems=test,
            num_steps=0,
            checkpoint_dir="checkpoints",
            device=args.device,
            sequence_length=ckpt.get("sequence_length", default_seq_len),
            max_answer_tokens=ckpt.get("max_answer_tokens", default_answer_tokens),
            max_answer_value=ckpt.get("max_answer_value", default_answer_value),
            max_full_candidate_answers=ckpt.get("max_full_candidate_answers", 2048),
            candidate_answers=ckpt.get("candidate_answers"),
            eval_every=ckpt.get("eval_every"),
            eval_train_max_samples=ckpt.get("eval_train_max_samples"),
            eval_test_max_samples=ckpt.get("eval_test_max_samples"),
            enable_goodness_eval=ckpt.get("enable_goodness_eval", True),
            output_embedding_detached=ckpt.get("output_embedding_detached", True),
            use_per_block_output_heads=ckpt.get("use_per_block_output_heads", False),
            final_block_loss_weight=ckpt.get("final_block_loss_weight", 1.0),
            nonfinal_block_loss_weight=ckpt.get("nonfinal_block_loss_weight", 1.0),
            block_output_head_states=ckpt.get("block_output_head_states"),
        )
        greedy_train = trainer.evaluate_logits(eval_train)
        greedy_test = trainer.evaluate_logits(eval_test)
        tf_train = ff_teacher_forced_metrics(trainer, eval_train, batch_size=args.batch_size)
        tf_test = ff_teacher_forced_metrics(trainer, eval_test, batch_size=args.batch_size)
        summary["greedy_train"] = {
            "token_accuracy": greedy_train.token_accuracy,
            "sequence_exact_match": greedy_train.sequence_exact_match,
        }
        summary["greedy_test"] = {
            "token_accuracy": greedy_test.token_accuracy,
            "sequence_exact_match": greedy_test.sequence_exact_match,
        }
        summary["teacher_forced_train"] = tf_train
        summary["teacher_forced_test"] = tf_test
        summary["config_flags"] = {
            "output_embedding_detached": ckpt.get("output_embedding_detached", True),
            "use_per_block_output_heads": ckpt.get("use_per_block_output_heads", False),
            "final_block_loss_weight": ckpt.get("final_block_loss_weight", 1.0),
            "nonfinal_block_loss_weight": ckpt.get("nonfinal_block_loss_weight", 1.0),
            "goodness_aux_weight": ckpt.get("goodness_aux_weight", None),
        }

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"saved_summary={args.output_json}")


if __name__ == "__main__":
    main()
