from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

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
    train_test_split,
)
from ffgpt.data import Problem, parse_answer_tokens_variable
from ffgpt.utils import ensure_dir, latest_checkpoint, load_checkpoint


@dataclass
class TransitionStats:
    from_block: int
    to_block: int
    rescue_rate: float
    degrade_rate: float
    rescue_count: int
    rescue_denominator: int
    degrade_count: int
    degrade_denominator: int


def config_from_checkpoint(config_dict: dict[str, Any]) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=int(config_dict["vocab_size"]),
        max_seq_len=int(config_dict["max_seq_len"]),
        d_model=int(config_dict["d_model"]),
        n_heads=int(config_dict["n_heads"]),
        n_blocks=int(config_dict["n_blocks"]),
        mlp_hidden=int(config_dict["mlp_hidden"]),
        dropout=float(config_dict.get("dropout", 0.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-block help/hurt and compare FF models to baseline.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--baseline-checkpoint", type=str, default=None)
    parser.add_argument("--discriminative-checkpoint", type=str, default=None)
    parser.add_argument("--autoregressive-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints/analysis")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--operand-digits", type=int, default=1)
    parser.add_argument("--samples", type=int, default=0, help="0 => exhaustive when feasible")
    parser.add_argument("--exhaustive-limit", type=int, default=100_000)
    parser.add_argument("--split", type=str, default="coverage", choices=["mod5", "coverage", "random"])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--run-tag", type=str, default=None)
    return parser.parse_args()


def resolve_checkpoint_path(explicit: str | None, checkpoint_dir: str, mode: str, run_tag: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)

    candidate_tags: list[str | None] = []
    if run_tag is not None:
        candidate_tags.append(run_tag)
        if run_tag.startswith("d1_"):
            candidate_tags.append(run_tag.replace("d1_", "", 1))
    candidate_tags.append(None)

    for tag in candidate_tags:
        path = latest_checkpoint(checkpoint_dir, mode, run_tag=tag)
        if path is not None:
            return path

    raise FileNotFoundError(
        f"No checkpoint found for mode={mode} in {checkpoint_dir} with run_tag candidates={candidate_tags}"
    )


def build_split(args: argparse.Namespace) -> tuple[list[Problem], list[Problem]]:
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
        return train_test_split(problems)
    if args.split == "coverage":
        return coverage_preserving_sum_split(problems=problems, test_size=args.test_size, seed=args.split_seed)

    import random

    rng = random.Random(args.split_seed)
    shuffled = list(problems)
    rng.shuffle(shuffled)
    n_test = args.test_size
    n_train = len(shuffled) - n_test
    return shuffled[:n_train], shuffled[n_train:]


def _prompt_token_ids(vocab: Vocab, a: int, b: int) -> list[int]:
    prompt_tokens = [*list(str(a)), "+", *list(str(b)), "="]
    return vocab.encode_tokens(prompt_tokens)


@torch.no_grad()
def baseline_predict_with_block_limit(
    model: BaselineTransformer,
    vocab: Vocab,
    a: int,
    b: int,
    block_limit: int,
    max_answer_tokens: int,
    max_answer_value: int,
    device: torch.device,
) -> int | None:
    context = _prompt_token_ids(vocab, a, b)
    generated: list[int] = []
    model.eval()

    for _ in range(max_answer_tokens):
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        pad_mask = input_ids != vocab.pad_id
        x = model.embed(input_ids)
        for block_idx, block in enumerate(model.blocks):
            x, _ = block(x, pad_mask=pad_mask, causal=True)
            if block_idx >= block_limit:
                break
        logits = model.lm_head(x)
        next_token = int(torch.argmax(logits[0, -1]).item())
        generated.append(next_token)
        context.append(next_token)
        if next_token == vocab.pad_id:
            break

    if len(generated) < max_answer_tokens:
        generated.extend([vocab.pad_id] * (max_answer_tokens - len(generated)))

    pred_answer, _ = parse_answer_tokens_variable(
        generated[:max_answer_tokens],
        vocab,
        expected_length=max_answer_tokens,
        max_answer_value=max_answer_value,
    )
    return pred_answer


@torch.no_grad()
def ffar_predict_with_block(
    trainer: FFAutoregressiveTrainer,
    a: int,
    b: int,
    block_idx: int,
) -> int | None:
    context = trainer._prompt_token_ids(a, b)
    generated: list[int] = []
    trainer.model.eval()

    for _ in range(trainer.max_answer_tokens):
        input_ids = torch.tensor([context], dtype=torch.long, device=trainer.device)
        pad_mask = input_ids != trainer.vocab.pad_id
        block_outputs, _ = trainer.model(input_ids, pad_mask=pad_mask, causal=True, detach_between_blocks=True)
        hidden = block_outputs[block_idx][0, -1]
        logits = trainer._project_block_logits(block_idx, hidden)
        next_token = int(torch.argmax(logits).item())
        generated.append(next_token)
        context.append(next_token)
        if next_token == trainer.vocab.pad_id:
            break

    if len(generated) < trainer.max_answer_tokens:
        generated.extend([trainer.vocab.pad_id] * (trainer.max_answer_tokens - len(generated)))

    pred_answer, _ = parse_answer_tokens_variable(
        generated[: trainer.max_answer_tokens],
        trainer.vocab,
        expected_length=trainer.max_answer_tokens,
        max_answer_value=trainer.max_answer_value,
    )
    return pred_answer


@torch.no_grad()
def ffdisc_predict_with_logits_block(
    trainer: FFDiscriminativeTrainer,
    a: int,
    b: int,
    block_idx: int,
) -> int | None:
    context = trainer._prompt_token_ids(a, b)
    generated: list[int] = []
    trainer.model.eval()

    for _ in range(trainer.max_answer_tokens):
        input_ids = torch.tensor([context], dtype=torch.long, device=trainer.device)
        pad_mask = input_ids != trainer.vocab.pad_id
        block_outputs, _ = trainer.model(
            input_ids,
            pad_mask=pad_mask,
            causal=True,
            detach_between_blocks=True,
            inter_block_norm=trainer.inter_block_norm,
            inter_block_norm_eps=trainer.inter_block_norm_eps,
        )
        hidden = block_outputs[block_idx][0, -1]
        logits = torch.matmul(hidden, trainer.model.embedding_weight.detach().T)
        next_token = int(torch.argmax(logits).item())
        generated.append(next_token)
        context.append(next_token)
        if next_token == trainer.vocab.pad_id:
            break

    if len(generated) < trainer.max_answer_tokens:
        generated.extend([trainer.vocab.pad_id] * (trainer.max_answer_tokens - len(generated)))

    pred_answer, _ = parse_answer_tokens_variable(
        generated[: trainer.max_answer_tokens],
        trainer.vocab,
        expected_length=trainer.max_answer_tokens,
        max_answer_value=trainer.max_answer_value,
    )
    return pred_answer


def compute_transition_stats(
    preds_by_block: list[list[int | None]],
    targets: list[int],
) -> tuple[list[float], list[TransitionStats]]:
    def is_correct(pred: int | None, target: int) -> bool:
        return pred is not None and int(pred) == target

    n_blocks = len(preds_by_block)
    n = len(targets)
    block_acc: list[float] = []
    for block_preds in preds_by_block:
        correct = sum(int(is_correct(p, t)) for p, t in zip(block_preds, targets))
        block_acc.append(float(correct / max(n, 1)))

    transitions: list[TransitionStats] = []
    for block_idx in range(n_blocks - 1):
        from_preds = preds_by_block[block_idx]
        to_preds = preds_by_block[block_idx + 1]
        from_wrong = 0
        from_correct = 0
        rescue = 0
        degrade = 0
        for p_from, p_to, target in zip(from_preds, to_preds, targets):
            from_ok = int(is_correct(p_from, target))
            to_ok = int(is_correct(p_to, target))
            if from_ok == 0:
                from_wrong += 1
                rescue += int(to_ok == 1)
            else:
                from_correct += 1
                degrade += int(to_ok == 0)
        transitions.append(
            TransitionStats(
                from_block=block_idx,
                to_block=block_idx + 1,
                rescue_rate=float(rescue / max(from_wrong, 1)),
                degrade_rate=float(degrade / max(from_correct, 1)),
                rescue_count=rescue,
                rescue_denominator=from_wrong,
                degrade_count=degrade,
                degrade_denominator=from_correct,
            )
        )
    return block_acc, transitions


def to_dict_transitions(transitions: list[TransitionStats]) -> list[dict[str, Any]]:
    return [
        {
            "from_block": t.from_block,
            "to_block": t.to_block,
            "rescue_rate": t.rescue_rate,
            "degrade_rate": t.degrade_rate,
            "rescue_count": t.rescue_count,
            "rescue_denominator": t.rescue_denominator,
            "degrade_count": t.degrade_count,
            "degrade_denominator": t.degrade_denominator,
        }
        for t in transitions
    ]


def analyze_baseline_logits(
    model: BaselineTransformer,
    vocab: Vocab,
    problems: list[Problem],
    device: torch.device,
    max_answer_tokens: int,
    max_answer_value: int,
) -> dict[str, Any]:
    n_blocks = len(model.blocks)
    preds_by_block: list[list[int | None]] = [[] for _ in range(n_blocks)]
    targets = [int(problem.answer) for problem in problems]
    for problem in problems:
        for block_idx in range(n_blocks):
            preds_by_block[block_idx].append(
                baseline_predict_with_block_limit(
                    model=model,
                    vocab=vocab,
                    a=problem.a,
                    b=problem.b,
                    block_limit=block_idx,
                    max_answer_tokens=max_answer_tokens,
                    max_answer_value=max_answer_value,
                    device=device,
                )
            )

    block_acc, transitions = compute_transition_stats(preds_by_block=preds_by_block, targets=targets)
    return {
        "n_samples": len(problems),
        "block_accuracy": block_acc,
        "transitions": to_dict_transitions(transitions),
        "final_exact": block_acc[-1] if block_acc else float("nan"),
    }


def analyze_ffar_logits(
    trainer: FFAutoregressiveTrainer,
    problems: list[Problem],
) -> dict[str, Any]:
    n_blocks = len(trainer.model.blocks)
    preds_by_block: list[list[int | None]] = [[] for _ in range(n_blocks)]
    targets = [int(problem.answer) for problem in problems]
    for problem in problems:
        for block_idx in range(n_blocks):
            preds_by_block[block_idx].append(ffar_predict_with_block(trainer, problem.a, problem.b, block_idx))

    block_acc, transitions = compute_transition_stats(preds_by_block=preds_by_block, targets=targets)
    return {
        "n_samples": len(problems),
        "block_accuracy": block_acc,
        "transitions": to_dict_transitions(transitions),
        "final_exact": block_acc[-1] if block_acc else float("nan"),
    }


@torch.no_grad()
def analyze_ffdisc_goodness(
    trainer: FFDiscriminativeTrainer,
    problems: list[Problem],
) -> dict[str, Any]:
    n_blocks = len(trainer.model.blocks)
    preds_by_block: list[list[int | None]] = [[] for _ in range(n_blocks)]
    combined_preds: list[int] = []
    targets = [int(problem.answer) for problem in problems]

    for problem in problems:
        total_scores, block_scores = trainer._score_candidates_goodness(problem.a, problem.b, causal=False)
        final_idx = int(torch.argmax(total_scores).item())
        final_pred = int(trainer.candidate_answers[final_idx])
        combined_preds.append(final_pred)
        for block_idx in range(n_blocks):
            idx = int(torch.argmax(block_scores[block_idx]).item())
            preds_by_block[block_idx].append(int(trainer.candidate_answers[idx]))

    block_acc, transitions = compute_transition_stats(preds_by_block=preds_by_block, targets=targets)
    combined_acc = float(sum(int(p == t) for p, t in zip(combined_preds, targets)) / max(len(targets), 1))
    rescue_from_first = 0
    degrade_from_first = 0
    first_wrong = 0
    first_correct = 0
    if n_blocks >= 1:
        first_preds = preds_by_block[0]
        for p0, p_combined, target in zip(first_preds, combined_preds, targets):
            if p0 is not None and int(p0) == target:
                first_correct += 1
                degrade_from_first += int(p_combined != target)
            else:
                first_wrong += 1
                rescue_from_first += int(p_combined == target)
    return {
        "n_samples": len(problems),
        "block_accuracy": block_acc,
        "transitions": to_dict_transitions(transitions),
        "combined_final_exact": combined_acc,
        "combined_from_block0": {
            "rescue_rate": float(rescue_from_first / max(first_wrong, 1)),
            "degrade_rate": float(degrade_from_first / max(first_correct, 1)),
            "rescue_count": rescue_from_first,
            "rescue_denominator": first_wrong,
            "degrade_count": degrade_from_first,
            "degrade_denominator": first_correct,
        },
        "final_exact": combined_acc,
    }


def analyze_ffdisc_logits(
    trainer: FFDiscriminativeTrainer,
    problems: list[Problem],
) -> dict[str, Any]:
    n_blocks = len(trainer.model.blocks)
    preds_by_block: list[list[int | None]] = [[] for _ in range(n_blocks)]
    targets = [int(problem.answer) for problem in problems]
    for problem in problems:
        for block_idx in range(n_blocks):
            preds_by_block[block_idx].append(ffdisc_predict_with_logits_block(trainer, problem.a, problem.b, block_idx))

    block_acc, transitions = compute_transition_stats(preds_by_block=preds_by_block, targets=targets)
    return {
        "n_samples": len(problems),
        "block_accuracy": block_acc,
        "transitions": to_dict_transitions(transitions),
        "final_exact": block_acc[-1] if block_acc else float("nan"),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_problems, test_problems = build_split(args)
    vocab = Vocab()
    max_answer_tokens = args.operand_digits + 1
    max_answer_value = max_sum_for_operand_digits(args.operand_digits)
    max_seq_len = max_seq_len_for_operand_digits(args.operand_digits)
    run_tag = args.run_tag if args.run_tag is not None else f"d{args.operand_digits}_{args.split}_s{args.split_seed}"

    baseline_path = resolve_checkpoint_path(args.baseline_checkpoint, args.checkpoint_dir, "baseline", run_tag=run_tag)
    disc_path = resolve_checkpoint_path(args.discriminative_checkpoint, args.checkpoint_dir, "discriminative", run_tag=run_tag)
    ar_path = resolve_checkpoint_path(args.autoregressive_checkpoint, args.checkpoint_dir, "autoregressive", run_tag=run_tag)

    baseline_ckpt = load_checkpoint(baseline_path, map_location=device)
    disc_ckpt = load_checkpoint(disc_path, map_location=device)
    ar_ckpt = load_checkpoint(ar_path, map_location=device)

    baseline_model = BaselineTransformer(config_from_checkpoint(baseline_ckpt["config"]))
    baseline_model.load_state_dict(baseline_ckpt["model_state"])
    baseline_model.to(device)
    baseline_model.eval()

    disc_model = FFTransformer(config_from_checkpoint(disc_ckpt["config"]))
    disc_model.load_state_dict(disc_ckpt["model_state"])
    disc_model.to(device)
    disc_model.eval()

    ar_model = FFTransformer(config_from_checkpoint(ar_ckpt["config"]))
    ar_model.load_state_dict(ar_ckpt["model_state"])
    ar_model.to(device)
    ar_model.eval()

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
        near_miss_start_step=disc_ckpt.get("near_miss_start_step"),
        near_miss_offsets=tuple(disc_ckpt.get("near_miss_offsets", (1,))),
        inter_block_norm=disc_ckpt.get("inter_block_norm", "none"),
        inter_block_norm_eps=disc_ckpt.get("inter_block_norm_eps", 1e-5),
        use_per_block_logit_aux=disc_ckpt.get("use_per_block_logit_aux", False),
        final_block_logit_aux_weight=disc_ckpt.get("final_block_logit_aux_weight", 1.0),
        nonfinal_block_logit_aux_weight=disc_ckpt.get("nonfinal_block_logit_aux_weight", 1.0),
        collaborative_global_offset_weight=disc_ckpt.get("collaborative_global_offset_weight", 0.0),
        kl_sync_weight=disc_ckpt.get("kl_sync_weight", 0.0),
        goodness_aggregation=disc_ckpt.get("goodness_aggregation", "uniform_sum"),
        goodness_block_weights=disc_ckpt.get("goodness_block_weights"),
        fit_goodness_block_weights=disc_ckpt.get("fit_goodness_block_weights", False),
        layerwise_train_single_block=disc_ckpt.get("layerwise_train_single_block", False),
        layerwise_phase_steps=disc_ckpt.get("layerwise_phase_steps"),
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
        output_embedding_detached=ar_ckpt.get("output_embedding_detached", True),
        use_per_block_output_heads=ar_ckpt.get("use_per_block_output_heads", False),
        final_block_loss_weight=ar_ckpt.get("final_block_loss_weight", 1.0),
        nonfinal_block_loss_weight=ar_ckpt.get("nonfinal_block_loss_weight", 1.0),
        block_output_head_states=ar_ckpt.get("block_output_head_states"),
    )

    baseline_train = analyze_baseline_logits(
        model=baseline_model,
        vocab=vocab,
        problems=train_problems,
        device=device,
        max_answer_tokens=baseline_ckpt.get("max_answer_tokens", max_answer_tokens),
        max_answer_value=baseline_ckpt.get("max_answer_value", max_answer_value),
    )
    baseline_test = analyze_baseline_logits(
        model=baseline_model,
        vocab=vocab,
        problems=test_problems,
        device=device,
        max_answer_tokens=baseline_ckpt.get("max_answer_tokens", max_answer_tokens),
        max_answer_value=baseline_ckpt.get("max_answer_value", max_answer_value),
    )

    disc_train_good = analyze_ffdisc_goodness(disc_trainer, train_problems)
    disc_test_good = analyze_ffdisc_goodness(disc_trainer, test_problems)
    disc_train_logit = analyze_ffdisc_logits(disc_trainer, train_problems)
    disc_test_logit = analyze_ffdisc_logits(disc_trainer, test_problems)

    ar_train_logit = analyze_ffar_logits(ar_trainer, train_problems)
    ar_test_logit = analyze_ffar_logits(ar_trainer, test_problems)

    results = {
        "split": {
            "strategy": args.split,
            "split_seed": args.split_seed,
            "train_size": len(train_problems),
            "test_size": len(test_problems),
            "operand_digits": args.operand_digits,
        },
        "checkpoints": {
            "baseline": str(baseline_path),
            "discriminative": str(disc_path),
            "autoregressive": str(ar_path),
        },
        "baseline_logits": {
            "train": baseline_train,
            "test": baseline_test,
        },
        "ff_discriminative_goodness": {
            "train": disc_train_good,
            "test": disc_test_good,
        },
        "ff_discriminative_logits": {
            "train": disc_train_logit,
            "test": disc_test_logit,
        },
        "ff_autoregressive_logits": {
            "train": ar_train_logit,
            "test": ar_test_logit,
        },
    }

    summary = {
        "baseline_test_final_exact": baseline_test["final_exact"],
        "ff_disc_goodness_test_final_exact": disc_test_good["final_exact"],
        "ff_disc_logit_test_final_exact": disc_test_logit["final_exact"],
        "ff_ar_logit_test_final_exact": ar_test_logit["final_exact"],
        "baseline_test_block0_to_block1": baseline_test["transitions"][0] if baseline_test["transitions"] else None,
        "ff_disc_goodness_test_block0_to_block1": disc_test_good["transitions"][0] if disc_test_good["transitions"] else None,
        "ff_disc_logit_test_block0_to_block1": disc_test_logit["transitions"][0] if disc_test_logit["transitions"] else None,
        "ff_ar_logit_test_block0_to_block1": ar_test_logit["transitions"][0] if ar_test_logit["transitions"] else None,
    }
    results["summary"] = summary

    out_dir = ensure_dir(args.output_dir)
    default_name = f"block_help_hurt_{run_tag}.json"
    output_name = args.output_name if args.output_name is not None else default_name
    out_path = out_dir / output_name
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
