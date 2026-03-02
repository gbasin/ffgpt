from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from .data import (
    ANSWER_MAX,
    ANSWER_MIN,
    Problem,
    Vocab,
    answer_to_token_ids_variable,
    parse_answer_tokens_variable,
    sample_negative_answer,
    sample_negatives,
)
from .goodness import candidate_set_ce_loss, ff_loss_bce, mean_goodness, update_threshold_ema
from .model import FFTransformer
from .utils import build_checkpoint_prefix, capture_rng_states, save_checkpoint


@dataclass
class FFGoodnessEvalResult:
    token_accuracy: float
    sequence_exact_match: float
    candidate_ranking_accuracy: float
    block_candidate_accuracy: list[float]
    predictions: list[int]
    targets: list[int]


@dataclass
class FFLogitEvalResult:
    token_accuracy: float
    sequence_exact_match: float
    predictions: list[int]
    targets: list[int]


class FFDiscriminativeTrainer:
    def __init__(
        self,
        model: FFTransformer,
        vocab: Vocab,
        train_problems: list[Problem],
        test_problems: list[Problem],
        lr: float = 1e-3,
        batch_size: int = 64,
        num_steps: int = 5000,
        checkpoint_every: int = 1000,
        checkpoint_dir: str = "checkpoints",
        threshold_momentum: float = 0.9,
        logit_aux_weight: float = 1.0,
        sequence_length: int | None = None,
        max_answer_tokens: int | None = None,
        max_answer_value: int | None = None,
        max_full_candidate_answers: int = 2048,
        candidate_answers: list[int] | None = None,
        near_miss_start_step: int | None = None,
        near_miss_offsets: tuple[int, ...] = (1,),
        device: str = "cpu",
        seed: int = 42,
        run_tag: str | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.train_problems = list(train_problems)
        self.test_problems = list(test_problems)
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.threshold_momentum = threshold_momentum
        self.logit_aux_weight = float(logit_aux_weight)
        self.sequence_length = int(sequence_length) if sequence_length is not None else int(self.model.config.max_seq_len)
        self.near_miss_start_step = near_miss_start_step
        self.near_miss_offsets = tuple(int(x) for x in near_miss_offsets)
        self.device = torch.device(device)
        self.seed = seed
        self.run_tag = run_tag
        self.checkpoint_prefix = build_checkpoint_prefix(mode="discriminative", run_tag=run_tag)

        if self.sequence_length > self.model.config.max_seq_len:
            raise ValueError(
                f"sequence_length={self.sequence_length} exceeds model max_seq_len={self.model.config.max_seq_len}"
            )

        all_problems = self.train_problems + self.test_problems
        inferred_max_answer_tokens = max((len(str(problem.answer)) for problem in all_problems), default=2)
        self.max_answer_tokens = int(max_answer_tokens) if max_answer_tokens is not None else inferred_max_answer_tokens
        inferred_max_answer_value = max((problem.answer for problem in all_problems), default=ANSWER_MAX)
        self.max_answer_value = int(max_answer_value) if max_answer_value is not None else inferred_max_answer_value
        self.max_full_candidate_answers = int(max_full_candidate_answers)

        if candidate_answers is not None:
            cands = sorted(set(int(x) for x in candidate_answers if ANSWER_MIN <= int(x) <= self.max_answer_value))
        elif (self.max_answer_value - ANSWER_MIN + 1) <= self.max_full_candidate_answers:
            cands = list(range(ANSWER_MIN, self.max_answer_value + 1))
        else:
            cands = sorted(set(int(problem.answer) for problem in all_problems))
        if not cands:
            raise ValueError("candidate answer pool is empty")
        self.candidate_answers = cands
        self.answer_to_candidate_index = {answer: idx for idx, answer in enumerate(self.candidate_answers)}

        self.model.to(self.device)

        self.block_optimizers = [torch.optim.Adam(block.parameters(), lr=self.lr) for block in self.model.blocks]
        self.embedding_optimizer = torch.optim.Adam(
            list(self.model.token_embedding.parameters()) + list(self.model.position_embedding.parameters()),
            lr=self.lr,
        )

        n_blocks = len(self.model.blocks)
        self.thresholds: list[torch.Tensor | None] = [None for _ in range(n_blocks)]
        self.collapse_counts: list[int] = [0 for _ in range(n_blocks)]
        self.loss_increase_streak = 0
        self.prev_loss: float | None = None
        self.rng = random.Random(seed)

        self.history: dict[str, Any] = {
            "step": [],
            "loss": [],
            "train_token_accuracy": [],
            "test_token_accuracy": [],
            "train_sequence_exact_match": [],
            "test_sequence_exact_match": [],
            "train_candidate_ranking_accuracy": [],
            "test_candidate_ranking_accuracy": [],
            "train_logit_sequence_exact_match": [],
            "test_logit_sequence_exact_match": [],
            "train_logit_mean_correct_rank": [],
            "test_logit_mean_correct_rank": [],
            "logit_aux_loss": [],
            "block_g_pos": [[] for _ in range(n_blocks)],
            "block_g_neg": [[] for _ in range(n_blocks)],
            "block_threshold": [[] for _ in range(n_blocks)],
            "block_separation": [[] for _ in range(n_blocks)],
            "block_separation_ratio": [[] for _ in range(n_blocks)],
            "block_accuracy": [[] for _ in range(n_blocks)],
        }

    def _negative_strategy_for_step(self, step: int) -> str:
        if self.near_miss_start_step is not None and step >= self.near_miss_start_step:
            return "near_miss"
        return "random"

    def _sample_problem_batch(self) -> list[Problem]:
        return [self.rng.choice(self.train_problems) for _ in range(self.batch_size)]

    def _build_discriminative_batch(self, problems: list[Problem], strategy: str) -> tuple[torch.Tensor, torch.Tensor]:
        pos: list[list[int]] = []
        neg: list[list[int]] = []
        for problem in problems:
            negative_answer = sample_negative_answer(
                problem.answer,
                strategy=strategy,
                near_miss_offsets=self.near_miss_offsets,
                rng=self.rng,
                answer_max=self.max_answer_value,
            )
            pos.append(self.vocab.encode_equation(problem.a, problem.b, problem.answer, max_len=self.sequence_length))
            neg.append(self.vocab.encode_equation(problem.a, problem.b, negative_answer, max_len=self.sequence_length))

        pos_tokens = torch.tensor(pos, dtype=torch.long, device=self.device)
        neg_tokens = torch.tensor(neg, dtype=torch.long, device=self.device)
        return pos_tokens, neg_tokens

    def _prompt_token_ids(self, a: int, b: int) -> list[int]:
        prompt_tokens = [*list(str(a)), "+", *list(str(b)), "="]
        return self.vocab.encode_tokens(prompt_tokens)

    def _target_tokens(self, answer: int) -> list[int]:
        return answer_to_token_ids_variable(
            answer=answer,
            vocab=self.vocab,
            total_answer_tokens=self.max_answer_tokens,
            max_answer_value=self.max_answer_value,
        )

    def _answer_token_ce_loss(self, block_hidden: torch.Tensor, x_tokens: torch.Tensor, y_tokens: torch.Tensor) -> torch.Tensor:
        """Cross-entropy on answer-generation positions only (after '=')."""
        logits = torch.matmul(block_hidden, self.model.embedding_weight.detach().T)
        selected_logits: list[torch.Tensor] = []
        selected_targets: list[torch.Tensor] = []
        batch_size, seq_len, _ = logits.shape

        for batch_idx in range(batch_size):
            eq_positions = (x_tokens[batch_idx] == self.vocab.equals_id).nonzero(as_tuple=False)
            if eq_positions.numel() == 0:
                continue
            eq_pos = int(eq_positions[0].item())
            for pos in range(eq_pos, seq_len):
                selected_logits.append(logits[batch_idx, pos])
                selected_targets.append(y_tokens[batch_idx, pos])

        if not selected_logits:
            return torch.tensor(0.0, device=self.device)

        flat_logits = torch.stack(selected_logits, dim=0)
        flat_targets = torch.stack(selected_targets, dim=0)
        return F.cross_entropy(flat_logits, flat_targets)

    def _score_candidates_goodness(
        self,
        a: int,
        b: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_answers
        token_batch = torch.tensor(
            [self.vocab.encode_equation(a, b, candidate, max_len=self.sequence_length) for candidate in candidates],
            dtype=torch.long,
            device=self.device,
        )
        pad_mask = token_batch != self.vocab.pad_id
        _, acts = self.model(token_batch, pad_mask=pad_mask, causal=causal, detach_between_blocks=True)

        block_scores: list[torch.Tensor] = []
        for block_acts in acts:
            g = mean_goodness(block_acts, pad_mask)
            block_scores.append(g)

        stacked = torch.stack(block_scores, dim=0)
        total = stacked.sum(dim=0)
        return total, stacked

    @torch.no_grad()
    def _score_candidates_logits(self, a: int, b: int) -> torch.Tensor:
        """Autoregressive candidate scoring using final-block logits."""
        candidates = self.candidate_answers
        scores = torch.zeros(len(candidates), device=self.device)

        for cand_idx, answer in enumerate(candidates):
            answer_tokens = self._target_tokens(answer)
            context = self._prompt_token_ids(a, b)
            score = 0.0

            for target_token in answer_tokens:
                input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
                pad_mask = input_ids != self.vocab.pad_id
                block_outputs, _ = self.model(input_ids, pad_mask=pad_mask, causal=True, detach_between_blocks=True)
                hidden = block_outputs[-1][0, -1]
                logits = torch.matmul(hidden, self.model.embedding_weight.detach().T)
                log_probs = F.log_softmax(logits, dim=-1)
                score += float(log_probs[int(target_token)].item())
                context.append(int(target_token))

            scores[cand_idx] = score
        return scores

    @torch.no_grad()
    def predict_with_goodness(self, a: int, b: int, causal: bool = False) -> tuple[int, list[int]]:
        total_scores, block_scores = self._score_candidates_goodness(a=a, b=b, causal=causal)
        pred_idx = int(torch.argmax(total_scores).item())
        pred = int(self.candidate_answers[pred_idx])
        block_preds = [
            int(self.candidate_answers[int(torch.argmax(block_scores[block_idx]).item())])
            for block_idx in range(block_scores.shape[0])
        ]
        return pred, block_preds

    @torch.no_grad()
    def predict_with_logits(self, a: int, b: int) -> tuple[int | None, list[int]]:
        scores = self._score_candidates_logits(a=a, b=b)
        pred_idx = int(torch.argmax(scores).item())
        pred_answer = int(self.candidate_answers[pred_idx])
        return pred_answer, self._target_tokens(pred_answer)

    @torch.no_grad()
    def evaluate_goodness(self, problems: list[Problem], causal: bool = False) -> FFGoodnessEvalResult:
        token_correct = 0
        token_total = 0
        sequence_correct = 0
        candidate_correct = 0
        predictions: list[int] = []
        targets: list[int] = []

        n_blocks = len(self.model.blocks)
        block_hits = [0 for _ in range(n_blocks)]

        for problem in problems:
            pred_answer, block_preds = self.predict_with_goodness(problem.a, problem.b, causal=causal)
            target_tokens = self._target_tokens(problem.answer)
            pred_tokens = self._target_tokens(pred_answer)

            token_correct += sum(int(p == t) for p, t in zip(pred_tokens, target_tokens))
            token_total += len(target_tokens)
            sequence_correct += int(pred_tokens == target_tokens)
            candidate_correct += int(pred_answer == problem.answer)

            for block_idx, block_pred in enumerate(block_preds):
                block_hits[block_idx] += int(block_pred == problem.answer)

            predictions.append(pred_answer)
            targets.append(problem.answer)

        return FFGoodnessEvalResult(
            token_accuracy=float(token_correct / max(token_total, 1)),
            sequence_exact_match=float(sequence_correct / max(len(problems), 1)),
            candidate_ranking_accuracy=float(candidate_correct / max(len(problems), 1)),
            block_candidate_accuracy=[float(hit / max(len(problems), 1)) for hit in block_hits],
            predictions=predictions,
            targets=targets,
        )

    @torch.no_grad()
    def evaluate_logits(self, problems: list[Problem]) -> FFLogitEvalResult:
        token_correct = 0
        token_total = 0
        sequence_correct = 0
        predictions: list[int] = []
        targets: list[int] = []

        for problem in problems:
            pred_answer, pred_tokens = self.predict_with_logits(problem.a, problem.b)
            target_tokens = self._target_tokens(problem.answer)

            token_correct += sum(int(p == t) for p, t in zip(pred_tokens, target_tokens))
            token_total += len(target_tokens)
            sequence_correct += int(pred_tokens == target_tokens)

            predictions.append(int(pred_answer) if pred_answer is not None else -1)
            targets.append(problem.answer)

        return FFLogitEvalResult(
            token_accuracy=float(token_correct / max(token_total, 1)),
            sequence_exact_match=float(sequence_correct / max(len(problems), 1)),
            predictions=predictions,
            targets=targets,
        )

    @torch.no_grad()
    def evaluate_logits_detailed(
        self,
        problems: list[Problem],
        top_k: int = 5,
        collect_examples: bool = True,
        max_examples: int = 20,
    ) -> dict[str, Any]:
        candidates = self.candidate_answers
        per_sum: dict[int, dict[str, int]] = {}
        correct_ranks: list[int] = []
        examples: list[dict[str, Any]] = []
        exact_matches = 0

        for problem in problems:
            scores = self._score_candidates_logits(problem.a, problem.b)
            pred_idx = int(torch.argmax(scores).item())
            pred_answer = candidates[pred_idx]
            exact_matches += int(pred_answer == problem.answer)

            target_idx = self.answer_to_candidate_index.get(int(problem.answer))
            if target_idx is None:
                target_score = float("-inf")
                rank = len(candidates) + 1
            else:
                target_score = float(scores[target_idx].item())
                rank = int((scores > target_score).sum().item()) + 1
            correct_ranks.append(rank)

            stats = per_sum.setdefault(problem.answer, {"count": 0, "correct": 0})
            stats["count"] += 1
            stats["correct"] += int(pred_answer == problem.answer)

            if collect_examples and len(examples) < max_examples:
                k = min(top_k, len(candidates))
                top_scores, top_indices = torch.topk(scores, k=k)
                top = [
                    {
                        "answer": int(candidates[int(idx.item())]),
                        "score": float(score.item()),
                    }
                    for score, idx in zip(top_scores, top_indices)
                ]
                examples.append(
                    {
                        "a": int(problem.a),
                        "b": int(problem.b),
                        "target": int(problem.answer),
                        "prediction": int(pred_answer),
                        "correct_rank": rank,
                        "target_score": target_score,
                        "top_candidates": top,
                    }
                )

        per_sum_accuracy = {
            str(answer): (stats["correct"] / stats["count"] if stats["count"] > 0 else 0.0)
            for answer, stats in sorted(per_sum.items())
        }
        mean_rank = float(sum(correct_ranks) / max(len(correct_ranks), 1))

        return {
            "sequence_exact_match": float(exact_matches / max(len(problems), 1)),
            "mean_correct_rank": mean_rank,
            "per_sum_accuracy": per_sum_accuracy,
            "examples": examples,
        }

    def _save_checkpoint(self, step: int) -> Path:
        state: dict[str, Any] = {
            "mode": "discriminative",
            "step": step,
            "model_state": self.model.state_dict(),
            "block_optimizer_states": [optimizer.state_dict() for optimizer in self.block_optimizers],
            "embedding_optimizer_state": self.embedding_optimizer.state_dict(),
            "thresholds": [float(th.item()) if th is not None else None for th in self.thresholds],
            "temperature": None,
            "history": self.history,
            "config": self.model.config.__dict__,
            "run_tag": self.run_tag,
            "checkpoint_prefix": self.checkpoint_prefix,
            "logit_aux_weight": self.logit_aux_weight,
            "sequence_length": self.sequence_length,
            "max_answer_tokens": self.max_answer_tokens,
            "max_answer_value": self.max_answer_value,
            "candidate_answers": self.candidate_answers,
            "max_full_candidate_answers": self.max_full_candidate_answers,
            "rng_states": capture_rng_states(),
        }
        path = self.checkpoint_dir / f"{self.checkpoint_prefix}_step{step}.pt"
        return save_checkpoint(path, state)

    def train(self, log_every: int = 100) -> dict[str, Any]:
        self.model.train()

        for step in range(1, self.num_steps + 1):
            strategy = self._negative_strategy_for_step(step)
            batch_problems = self._sample_problem_batch()
            pos_tokens, neg_tokens = self._build_discriminative_batch(batch_problems, strategy)

            pos_mask = pos_tokens != self.vocab.pad_id
            neg_mask = neg_tokens != self.vocab.pad_id

            _, pos_acts = self.model(pos_tokens, pad_mask=pos_mask, causal=False, detach_between_blocks=True)
            _, neg_acts = self.model(neg_tokens, pad_mask=neg_mask, causal=False, detach_between_blocks=True)

            block_losses: list[torch.Tensor] = []
            block_stats: list[dict[str, float]] = []

            for block_idx, (p_acts, n_acts) in enumerate(zip(pos_acts, neg_acts)):
                g_pos = mean_goodness(p_acts, pos_mask)
                g_neg = mean_goodness(n_acts, neg_mask)

                self.thresholds[block_idx] = update_threshold_ema(
                    g_pos=g_pos,
                    g_neg=g_neg,
                    current_threshold=self.thresholds[block_idx],
                    momentum=self.threshold_momentum,
                )
                threshold = self.thresholds[block_idx]
                assert threshold is not None

                loss_k = ff_loss_bce(g_pos, g_neg, threshold)
                block_losses.append(loss_k)

                g_pos_mean = float(g_pos.mean().item())
                g_neg_mean = float(g_neg.mean().item())
                threshold_value = float(threshold.item())
                separation = g_pos_mean - g_neg_mean
                separation_ratio = separation / (abs(g_neg_mean) + 1e-8)
                block_accuracy = (
                    float((g_pos > threshold).float().mean().item())
                    + float((g_neg <= threshold).float().mean().item())
                ) * 0.5

                block_stats.append(
                    {
                        "g_pos": g_pos_mean,
                        "g_neg": g_neg_mean,
                        "threshold": threshold_value,
                        "separation": separation,
                        "separation_ratio": separation_ratio,
                        "accuracy": block_accuracy,
                    }
                )

            ff_loss_total = sum(block_losses)
            logit_aux_loss = torch.tensor(0.0, device=self.device)
            if self.logit_aux_weight > 0.0:
                x_pos = pos_tokens[:, :-1]
                y_pos = pos_tokens[:, 1:]
                x_pos_mask = x_pos != self.vocab.pad_id
                pos_block_outputs_causal, _ = self.model(
                    x_pos,
                    pad_mask=x_pos_mask,
                    causal=True,
                    detach_between_blocks=True,
                )
                logit_aux_loss = self._answer_token_ce_loss(
                    block_hidden=pos_block_outputs_causal[-1],
                    x_tokens=x_pos,
                    y_tokens=y_pos,
                )

            total_loss = ff_loss_total + (self.logit_aux_weight * logit_aux_loss)

            finite = torch.isfinite(total_loss)
            if not bool(finite.item()):
                raise RuntimeError(f"NaN/Inf detected in loss at step={step}")

            for stats in block_stats:
                if not all(torch.isfinite(torch.tensor(v)) for v in stats.values()):
                    raise RuntimeError(f"NaN/Inf detected in block diagnostics at step={step}: {stats}")

            if self.prev_loss is not None and total_loss.item() > self.prev_loss:
                self.loss_increase_streak += 1
            else:
                self.loss_increase_streak = 0
            self.prev_loss = float(total_loss.item())

            if self.loss_increase_streak == 500:
                print(f"[warn] loss has increased for 500 consecutive steps at step={step} (possible divergence)")

            for block_idx, stats in enumerate(block_stats):
                if abs(stats["separation"]) < 1e-6:
                    self.collapse_counts[block_idx] += 1
                else:
                    self.collapse_counts[block_idx] = 0

                if self.collapse_counts[block_idx] == 100:
                    print(f"[warn] goodness collapsed at block={block_idx} step={step}")

                self.history["block_g_pos"][block_idx].append(stats["g_pos"])
                self.history["block_g_neg"][block_idx].append(stats["g_neg"])
                self.history["block_threshold"][block_idx].append(stats["threshold"])
                self.history["block_separation"][block_idx].append(stats["separation"])
                self.history["block_separation_ratio"][block_idx].append(stats["separation_ratio"])
                self.history["block_accuracy"][block_idx].append(stats["accuracy"])

            for optimizer in self.block_optimizers:
                optimizer.zero_grad(set_to_none=True)
            self.embedding_optimizer.zero_grad(set_to_none=True)

            total_loss.backward()

            for optimizer in self.block_optimizers:
                optimizer.step()
            self.embedding_optimizer.step()

            if step % log_every == 0 or step == 1 or step == self.num_steps:
                train_good = self.evaluate_goodness(self.train_problems, causal=False)
                test_good = self.evaluate_goodness(self.test_problems, causal=False)
                train_log = self.evaluate_logits(self.train_problems)
                test_log = self.evaluate_logits(self.test_problems)
                train_log_detail = self.evaluate_logits_detailed(
                    self.train_problems,
                    top_k=3,
                    collect_examples=False,
                )
                test_log_detail = self.evaluate_logits_detailed(
                    self.test_problems,
                    top_k=3,
                    collect_examples=False,
                )

                self.history["step"].append(step)
                self.history["loss"].append(float(total_loss.item()))
                self.history["logit_aux_loss"].append(float(logit_aux_loss.item()))
                self.history["train_token_accuracy"].append(train_good.token_accuracy)
                self.history["test_token_accuracy"].append(test_good.token_accuracy)
                self.history["train_sequence_exact_match"].append(train_good.sequence_exact_match)
                self.history["test_sequence_exact_match"].append(test_good.sequence_exact_match)
                self.history["train_candidate_ranking_accuracy"].append(train_good.candidate_ranking_accuracy)
                self.history["test_candidate_ranking_accuracy"].append(test_good.candidate_ranking_accuracy)
                self.history["train_logit_sequence_exact_match"].append(train_log.sequence_exact_match)
                self.history["test_logit_sequence_exact_match"].append(test_log.sequence_exact_match)
                self.history["train_logit_mean_correct_rank"].append(float(train_log_detail["mean_correct_rank"]))
                self.history["test_logit_mean_correct_rank"].append(float(test_log_detail["mean_correct_rank"]))

                block_msg = " ".join(
                    [
                        (
                            f"b{idx}:g+={stats['g_pos']:.4f},g-={stats['g_neg']:.4f},"
                            f"th={stats['threshold']:.4f},sep={stats['separation']:.4f}"
                        )
                        for idx, stats in enumerate(block_stats)
                    ]
                )
                print(
                    f"[ff-disc step {step}] loss={total_loss.item():.4f} "
                    f"ff_loss={ff_loss_total.item():.4f} "
                    f"logit_aux={logit_aux_loss.item():.4f} "
                    f"train_exact={train_good.sequence_exact_match:.3f} "
                    f"test_exact={test_good.sequence_exact_match:.3f} "
                    f"train_rank={train_good.candidate_ranking_accuracy:.3f} "
                    f"test_rank={test_good.candidate_ranking_accuracy:.3f} "
                    f"logit_test_rank={float(test_log_detail['mean_correct_rank']):.2f} {block_msg}"
                )

            if step % self.checkpoint_every == 0:
                ckpt_path = self._save_checkpoint(step)
                print(f"Saved checkpoint: {ckpt_path}")

        final_ckpt = self._save_checkpoint(self.num_steps)
        print(f"Saved final checkpoint: {final_ckpt}")

        final_train_good = self.evaluate_goodness(self.train_problems, causal=False)
        final_test_good = self.evaluate_goodness(self.test_problems, causal=False)
        final_train_log = self.evaluate_logits(self.train_problems)
        final_test_log = self.evaluate_logits(self.test_problems)

        return {
            "train_goodness": final_train_good,
            "test_goodness": final_test_good,
            "train_logits": final_train_log,
            "test_logits": final_test_log,
            "history": self.history,
            "checkpoint": str(final_ckpt),
        }


class FFAutoregressiveTrainer:
    def __init__(
        self,
        model: FFTransformer,
        vocab: Vocab,
        train_problems: list[Problem],
        test_problems: list[Problem],
        lr: float = 1e-3,
        batch_size: int = 64,
        num_steps: int = 5000,
        checkpoint_every: int = 1000,
        checkpoint_dir: str = "checkpoints",
        k_negatives: int = 12,
        temperature: float = 1.0,
        temperature_min: float = 0.1,
        goodness_aux_weight: float = 1.0,
        threshold_momentum: float = 0.9,
        sequence_length: int | None = None,
        max_answer_tokens: int | None = None,
        max_answer_value: int | None = None,
        max_full_candidate_answers: int = 2048,
        candidate_answers: list[int] | None = None,
        device: str = "cpu",
        seed: int = 42,
        run_tag: str | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.train_problems = list(train_problems)
        self.test_problems = list(test_problems)
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.checkpoint_every = checkpoint_every
        self.checkpoint_dir = Path(checkpoint_dir)
        self.k_negatives = int(k_negatives)
        self.temperature = float(temperature)
        self.temperature_min = float(temperature_min)
        self.goodness_aux_weight = float(goodness_aux_weight)
        self.threshold_momentum = float(threshold_momentum)
        self.sequence_length = int(sequence_length) if sequence_length is not None else int(self.model.config.max_seq_len)
        self.device = torch.device(device)
        self.seed = seed
        self.run_tag = run_tag
        self.checkpoint_prefix = build_checkpoint_prefix(mode="autoregressive", run_tag=run_tag)

        if self.sequence_length > self.model.config.max_seq_len:
            raise ValueError(
                f"sequence_length={self.sequence_length} exceeds model max_seq_len={self.model.config.max_seq_len}"
            )

        all_problems = self.train_problems + self.test_problems
        inferred_max_answer_tokens = max((len(str(problem.answer)) for problem in all_problems), default=2)
        self.max_answer_tokens = int(max_answer_tokens) if max_answer_tokens is not None else inferred_max_answer_tokens
        inferred_max_answer_value = max((problem.answer for problem in all_problems), default=ANSWER_MAX)
        self.max_answer_value = int(max_answer_value) if max_answer_value is not None else inferred_max_answer_value
        self.max_full_candidate_answers = int(max_full_candidate_answers)

        if candidate_answers is not None:
            cands = sorted(set(int(x) for x in candidate_answers if ANSWER_MIN <= int(x) <= self.max_answer_value))
        elif (self.max_answer_value - ANSWER_MIN + 1) <= self.max_full_candidate_answers:
            cands = list(range(ANSWER_MIN, self.max_answer_value + 1))
        else:
            cands = sorted(set(int(problem.answer) for problem in all_problems))
        if not cands:
            raise ValueError("candidate answer pool is empty")
        self.candidate_answers = cands

        self.model.to(self.device)

        self.block_optimizers = [torch.optim.Adam(block.parameters(), lr=self.lr) for block in self.model.blocks]
        self.embedding_optimizer = torch.optim.Adam(
            list(self.model.token_embedding.parameters()) + list(self.model.position_embedding.parameters()),
            lr=self.lr,
        )

        n_blocks = len(self.model.blocks)
        self.thresholds: list[torch.Tensor | None] = [None for _ in range(n_blocks)]
        self.collapse_counts: list[int] = [0 for _ in range(n_blocks)]
        self.prev_loss: float | None = None
        self.loss_increase_streak = 0
        self.best_loss = float("inf")
        self.plateau_steps = 0
        self.rng = random.Random(seed)

        self.history: dict[str, Any] = {
            "step": [],
            "loss": [],
            "candidate_loss": [],
            "goodness_aux_loss": [],
            "temperature": [],
            "train_token_accuracy": [],
            "test_token_accuracy": [],
            "train_sequence_exact_match": [],
            "test_sequence_exact_match": [],
            "train_goodness_candidate_accuracy": [],
            "test_goodness_candidate_accuracy": [],
            "block_g_pos": [[] for _ in range(n_blocks)],
            "block_g_neg": [[] for _ in range(n_blocks)],
            "block_threshold": [[] for _ in range(n_blocks)],
            "block_separation": [[] for _ in range(n_blocks)],
            "block_separation_ratio": [[] for _ in range(n_blocks)],
            "block_accuracy": [[] for _ in range(n_blocks)],
        }

    def _sample_problem_batch(self) -> list[Problem]:
        return [self.rng.choice(self.train_problems) for _ in range(self.batch_size)]

    def _build_autoregressive_batch(self, problems: list[Problem]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        full = torch.tensor(
            [self.vocab.encode_equation(problem.a, problem.b, problem.answer, max_len=self.sequence_length) for problem in problems],
            dtype=torch.long,
            device=self.device,
        )
        x = full[:, :-1]
        y = full[:, 1:]
        return full, x, y

    def _build_negative_full_batch(self, problems: list[Problem]) -> torch.Tensor:
        negatives = []
        for problem in problems:
            negative_answer = sample_negative_answer(
                problem.answer,
                strategy="random",
                rng=self.rng,
                answer_max=self.max_answer_value,
            )
            negatives.append(self.vocab.encode_equation(problem.a, problem.b, negative_answer, max_len=self.sequence_length))
        return torch.tensor(negatives, dtype=torch.long, device=self.device)

    def _prompt_token_ids(self, a: int, b: int) -> list[int]:
        prompt_tokens = [*list(str(a)), "+", *list(str(b)), "="]
        return self.vocab.encode_tokens(prompt_tokens)

    def _target_tokens(self, answer: int) -> list[int]:
        return answer_to_token_ids_variable(
            answer=answer,
            vocab=self.vocab,
            total_answer_tokens=self.max_answer_tokens,
            max_answer_value=self.max_answer_value,
        )

    def _candidate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]

        if self.k_negatives >= vocab_size - 1:
            return F.cross_entropy((logits / self.temperature).reshape(-1, vocab_size), targets.reshape(-1))

        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        candidate_rows: list[torch.Tensor] = []

        for idx in range(flat_logits.shape[0]):
            target = int(flat_targets[idx].item())
            negatives = sample_negatives(
                correct_token=target,
                vocab_size=vocab_size,
                k=self.k_negatives,
                rng=self.rng,
            )
            candidate_ids = [target, *negatives]
            candidate_rows.append(flat_logits[idx, candidate_ids])

        candidate_logits = torch.stack(candidate_rows, dim=0)
        candidate_targets = torch.zeros(candidate_logits.shape[0], dtype=torch.long, device=self.device)
        return candidate_set_ce_loss(candidate_logits, candidate_targets, temperature=self.temperature)

    def _score_candidates_goodness(
        self,
        a: int,
        b: int,
        causal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_answers
        token_batch = torch.tensor(
            [self.vocab.encode_equation(a, b, candidate, max_len=self.sequence_length) for candidate in candidates],
            dtype=torch.long,
            device=self.device,
        )
        pad_mask = token_batch != self.vocab.pad_id
        _, acts = self.model(token_batch, pad_mask=pad_mask, causal=causal, detach_between_blocks=True)

        block_scores: list[torch.Tensor] = []
        for block_acts in acts:
            block_scores.append(mean_goodness(block_acts, pad_mask))

        stacked = torch.stack(block_scores, dim=0)
        total = stacked.sum(dim=0)
        return total, stacked

    @torch.no_grad()
    def predict_with_goodness(self, a: int, b: int) -> tuple[int, list[int]]:
        total, block_scores = self._score_candidates_goodness(a, b, causal=True)
        pred_idx = int(torch.argmax(total).item())
        pred = int(self.candidate_answers[pred_idx])
        block_preds = [
            int(self.candidate_answers[int(torch.argmax(block_scores[block_idx]).item())])
            for block_idx in range(block_scores.shape[0])
        ]
        return pred, block_preds

    @torch.no_grad()
    def predict_with_logits(self, a: int, b: int) -> tuple[int | None, list[int]]:
        context = self._prompt_token_ids(a, b)
        generated: list[int] = []

        for _ in range(self.max_answer_tokens):
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            pad_mask = input_ids != self.vocab.pad_id
            block_outputs, _ = self.model(input_ids, pad_mask=pad_mask, causal=True, detach_between_blocks=True)
            hidden = block_outputs[-1][0, -1]
            logits = torch.matmul(hidden, self.model.embedding_weight.detach().T)
            next_token = int(torch.argmax(logits).item())
            generated.append(next_token)
            context.append(next_token)
            if next_token == self.vocab.pad_id:
                break

        if len(generated) < self.max_answer_tokens:
            generated.extend([self.vocab.pad_id] * (self.max_answer_tokens - len(generated)))

        pred_answer, normalized_tokens = parse_answer_tokens_variable(
            generated[: self.max_answer_tokens],
            self.vocab,
            expected_length=self.max_answer_tokens,
            max_answer_value=self.max_answer_value,
        )
        return pred_answer, normalized_tokens

    @torch.no_grad()
    def evaluate_logits(self, problems: list[Problem]) -> FFLogitEvalResult:
        token_correct = 0
        token_total = 0
        sequence_correct = 0
        predictions: list[int] = []
        targets: list[int] = []

        for problem in problems:
            pred_answer, pred_tokens = self.predict_with_logits(problem.a, problem.b)
            target_tokens = self._target_tokens(problem.answer)

            token_correct += sum(int(p == t) for p, t in zip(pred_tokens, target_tokens))
            token_total += len(target_tokens)
            sequence_correct += int(pred_tokens == target_tokens)

            predictions.append(int(pred_answer) if pred_answer is not None else -1)
            targets.append(problem.answer)

        return FFLogitEvalResult(
            token_accuracy=float(token_correct / max(token_total, 1)),
            sequence_exact_match=float(sequence_correct / max(len(problems), 1)),
            predictions=predictions,
            targets=targets,
        )

    @torch.no_grad()
    def evaluate_goodness(self, problems: list[Problem]) -> FFGoodnessEvalResult:
        token_correct = 0
        token_total = 0
        sequence_correct = 0
        candidate_correct = 0
        predictions: list[int] = []
        targets: list[int] = []

        n_blocks = len(self.model.blocks)
        block_hits = [0 for _ in range(n_blocks)]

        for problem in problems:
            pred_answer, block_preds = self.predict_with_goodness(problem.a, problem.b)
            target_tokens = self._target_tokens(problem.answer)
            pred_tokens = self._target_tokens(pred_answer)

            token_correct += sum(int(p == t) for p, t in zip(pred_tokens, target_tokens))
            token_total += len(target_tokens)
            sequence_correct += int(pred_tokens == target_tokens)
            candidate_correct += int(pred_answer == problem.answer)

            for block_idx, block_pred in enumerate(block_preds):
                block_hits[block_idx] += int(block_pred == problem.answer)

            predictions.append(pred_answer)
            targets.append(problem.answer)

        return FFGoodnessEvalResult(
            token_accuracy=float(token_correct / max(token_total, 1)),
            sequence_exact_match=float(sequence_correct / max(len(problems), 1)),
            candidate_ranking_accuracy=float(candidate_correct / max(len(problems), 1)),
            block_candidate_accuracy=[float(hit / max(len(problems), 1)) for hit in block_hits],
            predictions=predictions,
            targets=targets,
        )

    def _save_checkpoint(self, step: int) -> Path:
        state: dict[str, Any] = {
            "mode": "autoregressive",
            "step": step,
            "model_state": self.model.state_dict(),
            "block_optimizer_states": [optimizer.state_dict() for optimizer in self.block_optimizers],
            "embedding_optimizer_state": self.embedding_optimizer.state_dict(),
            "thresholds": [float(th.item()) if th is not None else None for th in self.thresholds],
            "temperature": self.temperature,
            "history": self.history,
            "config": self.model.config.__dict__,
            "run_tag": self.run_tag,
            "checkpoint_prefix": self.checkpoint_prefix,
            "goodness_aux_weight": self.goodness_aux_weight,
            "threshold_momentum": self.threshold_momentum,
            "sequence_length": self.sequence_length,
            "max_answer_tokens": self.max_answer_tokens,
            "max_answer_value": self.max_answer_value,
            "candidate_answers": self.candidate_answers,
            "max_full_candidate_answers": self.max_full_candidate_answers,
            "rng_states": capture_rng_states(),
        }
        path = self.checkpoint_dir / f"{self.checkpoint_prefix}_step{step}.pt"
        return save_checkpoint(path, state)

    def train(self, log_every: int = 100) -> dict[str, Any]:
        self.model.train()

        for step in range(1, self.num_steps + 1):
            batch_problems = self._sample_problem_batch()
            full_tokens, x, y = self._build_autoregressive_batch(batch_problems)
            x_mask = x != self.vocab.pad_id

            block_outputs, _ = self.model(x, pad_mask=x_mask, causal=True, detach_between_blocks=True)
            detached_embeddings = self.model.embedding_weight.detach()

            candidate_block_losses: list[torch.Tensor] = []
            for block_hidden in block_outputs:
                logits = torch.matmul(block_hidden, detached_embeddings.T)
                candidate_block_losses.append(self._candidate_loss(logits, y))
            candidate_loss_total = sum(candidate_block_losses)

            pos_mask = full_tokens != self.vocab.pad_id
            neg_tokens = self._build_negative_full_batch(batch_problems)
            neg_mask = neg_tokens != self.vocab.pad_id
            _, pos_acts = self.model(full_tokens, pad_mask=pos_mask, causal=True, detach_between_blocks=True)
            _, neg_acts = self.model(neg_tokens, pad_mask=neg_mask, causal=True, detach_between_blocks=True)

            goodness_block_losses: list[torch.Tensor] = []
            block_stats: list[dict[str, float]] = []
            for block_idx, (pos_block_acts, neg_block_acts) in enumerate(zip(pos_acts, neg_acts)):
                g_pos = mean_goodness(pos_block_acts, pos_mask)
                g_neg = mean_goodness(neg_block_acts, neg_mask)

                self.thresholds[block_idx] = update_threshold_ema(
                    g_pos=g_pos,
                    g_neg=g_neg,
                    current_threshold=self.thresholds[block_idx],
                    momentum=self.threshold_momentum,
                )
                threshold = self.thresholds[block_idx]
                assert threshold is not None

                ff_loss_k = ff_loss_bce(g_pos, g_neg, threshold)
                goodness_block_losses.append(ff_loss_k)

                g_pos_mean = float(g_pos.mean().item())
                g_neg_mean = float(g_neg.mean().item())
                separation = g_pos_mean - g_neg_mean
                separation_ratio = separation / (abs(g_neg_mean) + 1e-8)
                threshold_value = float(threshold.item())
                block_accuracy = (
                    float((g_pos > threshold).float().mean().item())
                    + float((g_neg <= threshold).float().mean().item())
                ) * 0.5

                block_stats.append(
                    {
                        "g_pos": g_pos_mean,
                        "g_neg": g_neg_mean,
                        "threshold": threshold_value,
                        "separation": separation,
                        "separation_ratio": separation_ratio,
                        "accuracy": block_accuracy,
                    }
                )

            goodness_aux_loss_total = sum(goodness_block_losses)
            total_loss = candidate_loss_total + (self.goodness_aux_weight * goodness_aux_loss_total)

            if not bool(torch.isfinite(total_loss).item()):
                raise RuntimeError(f"NaN/Inf detected in loss at step={step}")

            for stats in block_stats:
                if not all(torch.isfinite(torch.tensor(v)) for v in stats.values()):
                    raise RuntimeError(f"NaN/Inf detected in block diagnostics at step={step}: {stats}")

            if self.prev_loss is not None and total_loss.item() > self.prev_loss:
                self.loss_increase_streak += 1
            else:
                self.loss_increase_streak = 0
            self.prev_loss = float(total_loss.item())

            if self.loss_increase_streak == 500:
                print(f"[warn] loss has increased for 500 consecutive steps at step={step} (possible divergence)")

            for block_idx, stats in enumerate(block_stats):
                if abs(stats["separation"]) < 1e-6:
                    self.collapse_counts[block_idx] += 1
                else:
                    self.collapse_counts[block_idx] = 0
                if self.collapse_counts[block_idx] == 100:
                    print(f"[warn] goodness collapsed at block={block_idx} step={step}")

                self.history["block_g_pos"][block_idx].append(stats["g_pos"])
                self.history["block_g_neg"][block_idx].append(stats["g_neg"])
                self.history["block_threshold"][block_idx].append(stats["threshold"])
                self.history["block_separation"][block_idx].append(stats["separation"])
                self.history["block_separation_ratio"][block_idx].append(stats["separation_ratio"])
                self.history["block_accuracy"][block_idx].append(stats["accuracy"])

            if total_loss.item() < self.best_loss - 1e-8:
                self.best_loss = float(total_loss.item())
                self.plateau_steps = 0
            else:
                self.plateau_steps += 1

            if self.plateau_steps >= 500 and self.temperature > self.temperature_min:
                self.temperature = max(self.temperature_min, self.temperature * 0.5)
                self.plateau_steps = 0
                print(f"[info] reducing temperature to {self.temperature:.4f} at step={step}")

            for optimizer in self.block_optimizers:
                optimizer.zero_grad(set_to_none=True)
            self.embedding_optimizer.zero_grad(set_to_none=True)

            total_loss.backward()

            for optimizer in self.block_optimizers:
                optimizer.step()
            self.embedding_optimizer.step()

            if step % log_every == 0 or step == 1 or step == self.num_steps:
                train_logits = self.evaluate_logits(self.train_problems)
                test_logits = self.evaluate_logits(self.test_problems)
                train_good = self.evaluate_goodness(self.train_problems)
                test_good = self.evaluate_goodness(self.test_problems)

                self.history["step"].append(step)
                self.history["loss"].append(float(total_loss.item()))
                self.history["candidate_loss"].append(float(candidate_loss_total.item()))
                self.history["goodness_aux_loss"].append(float(goodness_aux_loss_total.item()))
                self.history["temperature"].append(float(self.temperature))
                self.history["train_token_accuracy"].append(train_logits.token_accuracy)
                self.history["test_token_accuracy"].append(test_logits.token_accuracy)
                self.history["train_sequence_exact_match"].append(train_logits.sequence_exact_match)
                self.history["test_sequence_exact_match"].append(test_logits.sequence_exact_match)
                self.history["train_goodness_candidate_accuracy"].append(train_good.candidate_ranking_accuracy)
                self.history["test_goodness_candidate_accuracy"].append(test_good.candidate_ranking_accuracy)

                block_msg = " ".join(
                    [
                        (
                            f"b{idx}:g+={stats['g_pos']:.4f},g-={stats['g_neg']:.4f},"
                            f"th={stats['threshold']:.4f},sep={stats['separation']:.4f}"
                        )
                        for idx, stats in enumerate(block_stats)
                    ]
                )
                print(
                    f"[ff-ar step {step}] loss={total_loss.item():.4f} temp={self.temperature:.3f} "
                    f"cand={candidate_loss_total.item():.4f} "
                    f"good_aux={goodness_aux_loss_total.item():.4f} "
                    f"train_exact={train_logits.sequence_exact_match:.3f} "
                    f"test_exact={test_logits.sequence_exact_match:.3f} "
                    f"train_good_rank={train_good.candidate_ranking_accuracy:.3f} "
                    f"test_good_rank={test_good.candidate_ranking_accuracy:.3f} {block_msg}"
                )

            if step % self.checkpoint_every == 0:
                path = self._save_checkpoint(step)
                print(f"Saved checkpoint: {path}")

        final_path = self._save_checkpoint(self.num_steps)
        print(f"Saved final checkpoint: {final_path}")

        final_train_logits = self.evaluate_logits(self.train_problems)
        final_test_logits = self.evaluate_logits(self.test_problems)
        final_train_good = self.evaluate_goodness(self.train_problems)
        final_test_good = self.evaluate_goodness(self.test_problems)

        return {
            "train_logits": final_train_logits,
            "test_logits": final_test_logits,
            "train_goodness": final_train_good,
            "test_goodness": final_test_good,
            "history": self.history,
            "checkpoint": str(final_path),
            "temperature": self.temperature,
        }
