from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
import torch.nn as nn

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
        eval_every: int | None = None,
        eval_train_max_samples: int | None = None,
        eval_test_max_samples: int | None = None,
        enable_goodness_eval: bool = True,
        enable_logit_rank_diagnostics: bool = False,
        logit_rank_eval_max_candidates: int = 512,
        near_miss_start_step: int | None = None,
        near_miss_offsets: tuple[int, ...] = (1,),
        inter_block_norm: str = "none",
        inter_block_norm_eps: float = 1e-5,
        use_per_block_logit_aux: bool = False,
        final_block_logit_aux_weight: float = 1.0,
        nonfinal_block_logit_aux_weight: float = 1.0,
        collaborative_global_offset_weight: float = 0.0,
        kl_sync_weight: float = 0.0,
        goodness_aggregation: str = "uniform_sum",
        goodness_block_weights: list[float] | None = None,
        fit_goodness_block_weights: bool = False,
        layerwise_train_single_block: bool = False,
        layerwise_phase_steps: list[int] | None = None,
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
        self.inter_block_norm = str(inter_block_norm).lower()
        self.inter_block_norm_eps = float(inter_block_norm_eps)
        if self.inter_block_norm not in {"none", "layernorm", "rmsnorm", "l2"}:
            raise ValueError(
                f"inter_block_norm must be one of ['none', 'layernorm', 'rmsnorm', 'l2'], got {inter_block_norm}"
            )
        if self.inter_block_norm_eps <= 0.0:
            raise ValueError(f"inter_block_norm_eps must be > 0, got {inter_block_norm_eps}")
        self.use_per_block_logit_aux = bool(use_per_block_logit_aux)
        self.final_block_logit_aux_weight = float(final_block_logit_aux_weight)
        self.nonfinal_block_logit_aux_weight = float(nonfinal_block_logit_aux_weight)
        if self.final_block_logit_aux_weight <= 0.0 or self.nonfinal_block_logit_aux_weight < 0.0:
            raise ValueError(
                "final_block_logit_aux_weight must be > 0 and nonfinal_block_logit_aux_weight must be >= 0. "
                f"Got {self.final_block_logit_aux_weight}, {self.nonfinal_block_logit_aux_weight}"
            )
        self.collaborative_global_offset_weight = float(collaborative_global_offset_weight)
        self.kl_sync_weight = float(kl_sync_weight)
        if self.collaborative_global_offset_weight < 0.0:
            raise ValueError(
                "collaborative_global_offset_weight must be >= 0. "
                f"Got {self.collaborative_global_offset_weight}"
            )
        if self.kl_sync_weight < 0.0:
            raise ValueError(f"kl_sync_weight must be >= 0. Got {self.kl_sync_weight}")
        self.goodness_aggregation = str(goodness_aggregation).lower()
        if self.goodness_aggregation not in {"uniform_sum", "weighted_sum"}:
            raise ValueError(
                f"goodness_aggregation must be one of ['uniform_sum', 'weighted_sum'], got {goodness_aggregation}"
            )
        self.fit_goodness_block_weights = bool(fit_goodness_block_weights)
        if self.fit_goodness_block_weights and self.goodness_aggregation != "weighted_sum":
            raise ValueError("fit_goodness_block_weights=True requires goodness_aggregation='weighted_sum'")
        self.goodness_block_weights_raw = None if goodness_block_weights is None else [float(x) for x in goodness_block_weights]
        self.layerwise_train_single_block = bool(layerwise_train_single_block)
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
        self.eval_every = int(eval_every) if eval_every is not None else None
        self.eval_train_max_samples = int(eval_train_max_samples) if eval_train_max_samples is not None else None
        self.eval_test_max_samples = int(eval_test_max_samples) if eval_test_max_samples is not None else None
        self.enable_goodness_eval = bool(enable_goodness_eval)
        self.enable_logit_rank_diagnostics = bool(enable_logit_rank_diagnostics)
        self.logit_rank_eval_max_candidates = int(logit_rank_eval_max_candidates)

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
        self.layerwise_phase_steps = self._normalize_layerwise_phase_steps(
            layerwise_phase_steps=layerwise_phase_steps,
            n_blocks=n_blocks,
        )
        running = 0
        self.layerwise_phase_boundaries: list[int] = []
        for phase_steps in self.layerwise_phase_steps:
            running += phase_steps
            self.layerwise_phase_boundaries.append(running)
        if self.goodness_block_weights_raw is None:
            self.goodness_block_weights = [1.0 for _ in range(n_blocks)]
        else:
            if len(self.goodness_block_weights_raw) != n_blocks:
                raise ValueError(
                    f"goodness_block_weights length must match n_blocks ({n_blocks}), got {len(self.goodness_block_weights_raw)}"
                )
            self.goodness_block_weights = list(self.goodness_block_weights_raw)
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
            "eval_train_size": [],
            "eval_test_size": [],
            "active_block": [],
            "goodness_block_weights": [],
            "logit_aux_loss": [],
            "kl_sync_loss": [],
            "block_g_pos": [[] for _ in range(n_blocks)],
            "block_g_neg": [[] for _ in range(n_blocks)],
            "block_threshold": [[] for _ in range(n_blocks)],
            "block_separation": [[] for _ in range(n_blocks)],
            "block_separation_ratio": [[] for _ in range(n_blocks)],
            "block_accuracy": [[] for _ in range(n_blocks)],
            "block_collab_pos_offset": [[] for _ in range(n_blocks)],
            "block_collab_neg_offset": [[] for _ in range(n_blocks)],
        }

    def _normalize_layerwise_phase_steps(
        self,
        layerwise_phase_steps: list[int] | None,
        n_blocks: int,
    ) -> list[int]:
        if not self.layerwise_train_single_block:
            return [self.num_steps]

        if n_blocks <= 0:
            raise ValueError("model must contain at least one block")

        if layerwise_phase_steps is None:
            if self.num_steps == 0:
                return [0 for _ in range(n_blocks)]
            if self.num_steps < n_blocks:
                raise ValueError(
                    "layerwise single-block training requires num_steps >= n_blocks when phase steps are automatic. "
                    f"Got num_steps={self.num_steps}, n_blocks={n_blocks}"
                )
            base = self.num_steps // n_blocks
            remainder = self.num_steps % n_blocks
            steps = [base for _ in range(n_blocks)]
            for idx in range(remainder):
                steps[idx] += 1
            return steps

        steps = [int(x) for x in layerwise_phase_steps]
        if len(steps) != n_blocks:
            raise ValueError(
                f"layerwise_phase_steps length must match n_blocks ({n_blocks}), got {len(steps)}"
            )
        if any(step < 0 for step in steps):
            raise ValueError(f"layerwise_phase_steps must be non-negative, got {steps}")
        if self.num_steps > 0 and sum(steps) != self.num_steps:
            raise ValueError(
                "sum(layerwise_phase_steps) must equal num_steps for layerwise single-block training. "
                f"Got sum={sum(steps)}, num_steps={self.num_steps}"
            )
        if all(step == 0 for step in steps):
            raise ValueError("layerwise_phase_steps cannot be all zeros")
        return steps

    def _active_blocks_for_step(self, step: int) -> list[int]:
        if not self.layerwise_train_single_block:
            return list(range(len(self.model.blocks)))

        phase_idx = 0
        while phase_idx < (len(self.layerwise_phase_boundaries) - 1) and step > self.layerwise_phase_boundaries[phase_idx]:
            phase_idx += 1
        return [phase_idx]

    def _logit_aux_block_weight(self, block_idx: int, n_blocks: int) -> float:
        if block_idx == (n_blocks - 1):
            return self.final_block_logit_aux_weight
        return self.nonfinal_block_logit_aux_weight

    def _aggregate_goodness_scores(self, stacked: torch.Tensor) -> torch.Tensor:
        if self.goodness_aggregation == "uniform_sum":
            return stacked.sum(dim=0)

        weights = torch.tensor(self.goodness_block_weights, dtype=stacked.dtype, device=stacked.device).unsqueeze(1)
        return (stacked * weights).sum(dim=0)

    def _collaborative_offsets(self, cached_goodness: list[torch.Tensor]) -> list[torch.Tensor]:
        """For each block l, return sum of cached goodness from all blocks r != l."""
        if not cached_goodness:
            return []
        total = torch.stack(cached_goodness, dim=0).sum(dim=0)
        return [total - cached_goodness[idx] for idx in range(len(cached_goodness))]

    @torch.no_grad()
    def _fit_goodness_weights(self, problems: list[Problem], causal: bool = False) -> list[float]:
        n_blocks = len(self.model.blocks)
        if n_blocks == 0:
            return []
        if not problems:
            return list(self.goodness_block_weights)
        if n_blocks == 1:
            self.goodness_block_weights = [1.0]
            return list(self.goodness_block_weights)

        cached_scores: list[tuple[torch.Tensor, int]] = []
        for problem in problems:
            _, stacked = self._score_candidates_goodness(problem.a, problem.b, causal=causal)
            target_idx = self.answer_to_candidate_index.get(int(problem.answer))
            if target_idx is None:
                continue
            cached_scores.append((stacked.detach(), int(target_idx)))
        if not cached_scores:
            return list(self.goodness_block_weights)

        grid = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
        max_combos = 4096
        total_combos = len(grid) ** n_blocks
        if total_combos <= max_combos:
            combos = list(product(grid, repeat=n_blocks))
        else:
            combos = [
                tuple(self.rng.choice(grid) for _ in range(n_blocks))
                for _ in range(max_combos)
            ]

        best_weights = tuple(self.goodness_block_weights)
        best_acc = float("-inf")
        best_l1 = float("inf")
        for weights in combos:
            correct = 0
            weights_t = None
            for stacked, target_idx in cached_scores:
                if weights_t is None:
                    weights_t = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).unsqueeze(1)
                scores = (stacked * weights_t).sum(dim=0)
                pred_idx = int(torch.argmax(scores).item())
                correct += int(pred_idx == target_idx)
            acc = float(correct / max(len(cached_scores), 1))
            l1 = float(sum(abs(w) for w in weights))
            if (acc > best_acc + 1e-12) or (abs(acc - best_acc) <= 1e-12 and l1 < best_l1):
                best_acc = acc
                best_l1 = l1
                best_weights = weights

        self.goodness_block_weights = [float(w) for w in best_weights]
        return list(self.goodness_block_weights)

    def _negative_strategy_for_step(self, step: int) -> str:
        if self.near_miss_start_step is not None and step >= self.near_miss_start_step:
            return "near_miss"
        return "random"

    def _sample_problem_batch(self) -> list[Problem]:
        return [self.rng.choice(self.train_problems) for _ in range(self.batch_size)]

    def _sample_eval_subset(self, problems: list[Problem], max_samples: int | None) -> list[Problem]:
        if max_samples is None or max_samples >= len(problems):
            return problems
        if max_samples <= 0:
            return []
        indices = self.rng.sample(range(len(problems)), max_samples)
        return [problems[idx] for idx in indices]

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

    def _answer_token_kl_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        x_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """KL(teacher || student) on answer-generation positions only (after '=')."""
        embedding = self.model.embedding_weight.detach()
        student_logits = torch.matmul(student_hidden, embedding.T)
        teacher_logits = torch.matmul(teacher_hidden.detach(), embedding.T)

        selected_student: list[torch.Tensor] = []
        selected_teacher: list[torch.Tensor] = []
        batch_size, seq_len, _ = student_logits.shape

        for batch_idx in range(batch_size):
            eq_positions = (x_tokens[batch_idx] == self.vocab.equals_id).nonzero(as_tuple=False)
            if eq_positions.numel() == 0:
                continue
            eq_pos = int(eq_positions[0].item())
            for pos in range(eq_pos, seq_len):
                selected_student.append(student_logits[batch_idx, pos])
                selected_teacher.append(teacher_logits[batch_idx, pos])

        if not selected_student:
            return torch.tensor(0.0, device=self.device)

        flat_student = torch.stack(selected_student, dim=0)
        flat_teacher = torch.stack(selected_teacher, dim=0)
        log_probs_student = F.log_softmax(flat_student, dim=-1)
        probs_teacher = F.softmax(flat_teacher, dim=-1)
        return F.kl_div(log_probs_student, probs_teacher, reduction="batchmean")

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
        _, acts = self.model(
            token_batch,
            pad_mask=pad_mask,
            causal=causal,
            detach_between_blocks=True,
            inter_block_norm=self.inter_block_norm,
            inter_block_norm_eps=self.inter_block_norm_eps,
        )

        block_scores: list[torch.Tensor] = []
        for block_acts in acts:
            g = mean_goodness(block_acts, pad_mask)
            block_scores.append(g)

        stacked = torch.stack(block_scores, dim=0)
        total = self._aggregate_goodness_scores(stacked)
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
                block_outputs, _ = self.model(
                    input_ids,
                    pad_mask=pad_mask,
                    causal=True,
                    detach_between_blocks=True,
                    inter_block_norm=self.inter_block_norm,
                    inter_block_norm_eps=self.inter_block_norm_eps,
                )
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
        context = self._prompt_token_ids(a, b)
        generated: list[int] = []

        for _ in range(self.max_answer_tokens):
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            pad_mask = input_ids != self.vocab.pad_id
            block_outputs, _ = self.model(
                input_ids,
                pad_mask=pad_mask,
                causal=True,
                detach_between_blocks=True,
                inter_block_norm=self.inter_block_norm,
                inter_block_norm_eps=self.inter_block_norm_eps,
            )
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
        max_candidates: int | None = None,
    ) -> dict[str, Any]:
        candidates = self.candidate_answers
        if max_candidates is not None and len(candidates) > max_candidates:
            coarse = self.evaluate_logits(problems)
            return {
                "sequence_exact_match": coarse.sequence_exact_match,
                "mean_correct_rank": float("nan"),
                "per_sum_accuracy": {},
                "examples": [],
                "skipped": True,
                "candidate_count": len(candidates),
                "max_candidates": int(max_candidates),
            }

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
            "use_per_block_logit_aux": self.use_per_block_logit_aux,
            "final_block_logit_aux_weight": self.final_block_logit_aux_weight,
            "nonfinal_block_logit_aux_weight": self.nonfinal_block_logit_aux_weight,
            "collaborative_global_offset_weight": self.collaborative_global_offset_weight,
            "kl_sync_weight": self.kl_sync_weight,
            "sequence_length": self.sequence_length,
            "max_answer_tokens": self.max_answer_tokens,
            "max_answer_value": self.max_answer_value,
            "candidate_answers": self.candidate_answers,
            "max_full_candidate_answers": self.max_full_candidate_answers,
            "inter_block_norm": self.inter_block_norm,
            "inter_block_norm_eps": self.inter_block_norm_eps,
            "goodness_aggregation": self.goodness_aggregation,
            "goodness_block_weights": self.goodness_block_weights,
            "fit_goodness_block_weights": self.fit_goodness_block_weights,
            "eval_every": self.eval_every,
            "eval_train_max_samples": self.eval_train_max_samples,
            "eval_test_max_samples": self.eval_test_max_samples,
            "enable_goodness_eval": self.enable_goodness_eval,
            "enable_logit_rank_diagnostics": self.enable_logit_rank_diagnostics,
            "logit_rank_eval_max_candidates": self.logit_rank_eval_max_candidates,
            "layerwise_train_single_block": self.layerwise_train_single_block,
            "layerwise_phase_steps": self.layerwise_phase_steps,
            "rng_states": capture_rng_states(),
        }
        path = self.checkpoint_dir / f"{self.checkpoint_prefix}_step{step}.pt"
        return save_checkpoint(path, state)

    def train(self, log_every: int = 100) -> dict[str, Any]:
        self.model.train()
        start_time = time.perf_counter()

        for step in range(1, self.num_steps + 1):
            strategy = self._negative_strategy_for_step(step)
            batch_problems = self._sample_problem_batch()
            pos_tokens, neg_tokens = self._build_discriminative_batch(batch_problems, strategy)

            pos_mask = pos_tokens != self.vocab.pad_id
            neg_mask = neg_tokens != self.vocab.pad_id

            collaborative_offsets_pos: list[torch.Tensor] | None = None
            collaborative_offsets_neg: list[torch.Tensor] | None = None
            collaborative_enabled = self.collaborative_global_offset_weight > 0.0 and len(self.model.blocks) > 1
            if collaborative_enabled:
                # Two-phase collaborative training:
                # 1) cache per-block goodness with no grad
                # 2) apply other-block goodness as stop-grad offsets in local block losses
                with torch.no_grad():
                    _, pos_acts_cached = self.model(
                        pos_tokens,
                        pad_mask=pos_mask,
                        causal=False,
                        detach_between_blocks=True,
                        inter_block_norm=self.inter_block_norm,
                        inter_block_norm_eps=self.inter_block_norm_eps,
                    )
                    _, neg_acts_cached = self.model(
                        neg_tokens,
                        pad_mask=neg_mask,
                        causal=False,
                        detach_between_blocks=True,
                        inter_block_norm=self.inter_block_norm,
                        inter_block_norm_eps=self.inter_block_norm_eps,
                    )
                    cached_pos_goodness = [mean_goodness(block_acts, pos_mask) for block_acts in pos_acts_cached]
                    cached_neg_goodness = [mean_goodness(block_acts, neg_mask) for block_acts in neg_acts_cached]
                collaborative_offsets_pos = self._collaborative_offsets(cached_pos_goodness)
                collaborative_offsets_neg = self._collaborative_offsets(cached_neg_goodness)

            _, pos_acts = self.model(
                pos_tokens,
                pad_mask=pos_mask,
                causal=False,
                detach_between_blocks=True,
                inter_block_norm=self.inter_block_norm,
                inter_block_norm_eps=self.inter_block_norm_eps,
            )
            _, neg_acts = self.model(
                neg_tokens,
                pad_mask=neg_mask,
                causal=False,
                detach_between_blocks=True,
                inter_block_norm=self.inter_block_norm,
                inter_block_norm_eps=self.inter_block_norm_eps,
            )

            block_losses: list[torch.Tensor] = []
            block_stats: list[dict[str, float]] = []

            for block_idx, (p_acts, n_acts) in enumerate(zip(pos_acts, neg_acts)):
                g_pos_raw = mean_goodness(p_acts, pos_mask)
                g_neg_raw = mean_goodness(n_acts, neg_mask)
                collab_pos_offset = torch.zeros_like(g_pos_raw)
                collab_neg_offset = torch.zeros_like(g_neg_raw)
                if collaborative_offsets_pos is not None and collaborative_offsets_neg is not None:
                    collab_pos_offset = collaborative_offsets_pos[block_idx]
                    collab_neg_offset = collaborative_offsets_neg[block_idx]

                g_pos = g_pos_raw + (self.collaborative_global_offset_weight * collab_pos_offset)
                g_neg = g_neg_raw + (self.collaborative_global_offset_weight * collab_neg_offset)

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

                g_pos_mean_raw = float(g_pos_raw.mean().item())
                g_neg_mean_raw = float(g_neg_raw.mean().item())
                g_pos_mean_eff = float(g_pos.mean().item())
                g_neg_mean_eff = float(g_neg.mean().item())
                collab_pos_offset_mean = float(collab_pos_offset.mean().item())
                collab_neg_offset_mean = float(collab_neg_offset.mean().item())
                threshold_value = float(threshold.item())
                separation = g_pos_mean_raw - g_neg_mean_raw
                separation_ratio = separation / (abs(g_neg_mean_raw) + 1e-8)
                separation_eff = g_pos_mean_eff - g_neg_mean_eff
                block_accuracy = (
                    float((g_pos > threshold).float().mean().item())
                    + float((g_neg <= threshold).float().mean().item())
                ) * 0.5

                block_stats.append(
                    {
                        "g_pos": g_pos_mean_raw,
                        "g_neg": g_neg_mean_raw,
                        "g_pos_eff": g_pos_mean_eff,
                        "g_neg_eff": g_neg_mean_eff,
                        "collab_pos_offset": collab_pos_offset_mean,
                        "collab_neg_offset": collab_neg_offset_mean,
                        "threshold": threshold_value,
                        "separation": separation,
                        "separation_eff": separation_eff,
                        "separation_ratio": separation_ratio,
                        "accuracy": block_accuracy,
                    }
                )

            active_blocks = self._active_blocks_for_step(step)
            active_block_set = set(active_blocks)
            ff_loss_total = torch.stack([block_losses[idx] for idx in active_blocks], dim=0).sum()
            logit_aux_loss = torch.tensor(0.0, device=self.device)
            kl_sync_loss = torch.tensor(0.0, device=self.device)
            if self.logit_aux_weight > 0.0 or self.kl_sync_weight > 0.0:
                x_pos = pos_tokens[:, :-1]
                y_pos = pos_tokens[:, 1:]
                x_pos_mask = x_pos != self.vocab.pad_id
                pos_block_outputs_causal, _ = self.model(
                    x_pos,
                    pad_mask=x_pos_mask,
                    causal=True,
                    detach_between_blocks=True,
                    inter_block_norm=self.inter_block_norm,
                    inter_block_norm_eps=self.inter_block_norm_eps,
                )
                n_blocks = len(pos_block_outputs_causal)
                if self.logit_aux_weight > 0.0:
                    if self.use_per_block_logit_aux:
                        aux_terms: list[torch.Tensor] = []
                        for block_idx, block_hidden in enumerate(pos_block_outputs_causal):
                            if self.layerwise_train_single_block and block_idx not in active_block_set:
                                continue
                            weight = self._logit_aux_block_weight(block_idx, n_blocks)
                            if weight <= 0.0:
                                continue
                            ce_k = self._answer_token_ce_loss(
                                block_hidden=block_hidden,
                                x_tokens=x_pos,
                                y_tokens=y_pos,
                            )
                            aux_terms.append(ce_k * weight)
                        if aux_terms:
                            logit_aux_loss = torch.stack(aux_terms, dim=0).sum()
                    else:
                        enable_final_block_aux = (not self.layerwise_train_single_block) or ((n_blocks - 1) in active_block_set)
                        if enable_final_block_aux:
                            logit_aux_loss = self._answer_token_ce_loss(
                                block_hidden=pos_block_outputs_causal[-1],
                                x_tokens=x_pos,
                                y_tokens=y_pos,
                            )

                if self.kl_sync_weight > 0.0 and n_blocks > 1:
                    teacher_hidden = pos_block_outputs_causal[-1]
                    kl_terms: list[torch.Tensor] = []
                    for block_idx, student_hidden in enumerate(pos_block_outputs_causal[:-1]):
                        if self.layerwise_train_single_block and block_idx not in active_block_set:
                            continue
                        kl_terms.append(
                            self._answer_token_kl_loss(
                                student_hidden=student_hidden,
                                teacher_hidden=teacher_hidden,
                                x_tokens=x_pos,
                            )
                        )
                    if kl_terms:
                        kl_sync_loss = torch.stack(kl_terms, dim=0).mean()

            total_loss = (
                ff_loss_total
                + (self.logit_aux_weight * logit_aux_loss)
                + (self.kl_sync_weight * kl_sync_loss)
            )

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
                self.history["block_collab_pos_offset"][block_idx].append(stats["collab_pos_offset"])
                self.history["block_collab_neg_offset"][block_idx].append(stats["collab_neg_offset"])

            for optimizer in self.block_optimizers:
                optimizer.zero_grad(set_to_none=True)
            self.embedding_optimizer.zero_grad(set_to_none=True)

            total_loss.backward()

            for block_idx, optimizer in enumerate(self.block_optimizers):
                if block_idx in active_block_set:
                    optimizer.step()
            if 0 in active_block_set:
                self.embedding_optimizer.step()

            if step % log_every == 0 or step == 1 or step == self.num_steps:
                should_eval = (
                    self.eval_every is None
                    or step == 1
                    or step == self.num_steps
                    or step % self.eval_every == 0
                )
                elapsed = max(time.perf_counter() - start_time, 1e-8)
                steps_per_sec = step / elapsed

                block_parts: list[str] = []
                for idx, stats in enumerate(block_stats):
                    part = (
                        f"b{idx}:g+={stats['g_pos']:.4f},g-={stats['g_neg']:.4f},"
                        f"th={stats['threshold']:.4f},sep={stats['separation']:.4f}"
                    )
                    if collaborative_enabled:
                        part += (
                            f",co+={stats['collab_pos_offset']:.4f},"
                            f"co-={stats['collab_neg_offset']:.4f}"
                        )
                    block_parts.append(part)
                block_msg = " ".join(block_parts)
                active_blocks_msg = ",".join(str(idx) for idx in active_blocks)

                if not should_eval:
                    print(
                        f"[ff-disc step {step}] loss={total_loss.item():.4f} "
                        f"ff_loss={ff_loss_total.item():.4f} "
                        f"logit_aux={logit_aux_loss.item():.4f} "
                        f"kl_sync={kl_sync_loss.item():.4f} "
                        f"active_blocks=[{active_blocks_msg}] "
                        f"steps_per_sec={steps_per_sec:.2f} "
                        f"(eval skipped) {block_msg}"
                    )
                else:
                    train_eval = self._sample_eval_subset(self.train_problems, self.eval_train_max_samples)
                    test_eval = self._sample_eval_subset(self.test_problems, self.eval_test_max_samples)
                    if self.enable_goodness_eval and self.fit_goodness_block_weights:
                        fitted = self._fit_goodness_weights(train_eval, causal=False)
                        fitted_msg = ",".join(f"{w:.2f}" for w in fitted)
                        print(f"[ff-disc step {step}] fitted_goodness_weights=[{fitted_msg}]")

                    if self.enable_goodness_eval:
                        train_good = self.evaluate_goodness(train_eval, causal=False)
                        test_good = self.evaluate_goodness(test_eval, causal=False)
                    else:
                        train_good = FFGoodnessEvalResult(
                            token_accuracy=float("nan"),
                            sequence_exact_match=float("nan"),
                            candidate_ranking_accuracy=float("nan"),
                            block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                            predictions=[],
                            targets=[],
                        )
                        test_good = FFGoodnessEvalResult(
                            token_accuracy=float("nan"),
                            sequence_exact_match=float("nan"),
                            candidate_ranking_accuracy=float("nan"),
                            block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                            predictions=[],
                            targets=[],
                        )
                    train_log = self.evaluate_logits(train_eval)
                    test_log = self.evaluate_logits(test_eval)
                    if self.enable_logit_rank_diagnostics:
                        train_log_detail = self.evaluate_logits_detailed(
                            train_eval,
                            top_k=3,
                            collect_examples=False,
                            max_candidates=self.logit_rank_eval_max_candidates,
                        )
                        test_log_detail = self.evaluate_logits_detailed(
                            test_eval,
                            top_k=3,
                            collect_examples=False,
                            max_candidates=self.logit_rank_eval_max_candidates,
                        )
                        train_log_rank = float(train_log_detail["mean_correct_rank"])
                        test_log_rank = float(test_log_detail["mean_correct_rank"])
                    else:
                        train_log_rank = float("nan")
                        test_log_rank = float("nan")

                    self.history["step"].append(step)
                    self.history["loss"].append(float(total_loss.item()))
                    self.history["logit_aux_loss"].append(float(logit_aux_loss.item()))
                    self.history["kl_sync_loss"].append(float(kl_sync_loss.item()))
                    self.history["train_token_accuracy"].append(train_good.token_accuracy)
                    self.history["test_token_accuracy"].append(test_good.token_accuracy)
                    self.history["train_sequence_exact_match"].append(train_good.sequence_exact_match)
                    self.history["test_sequence_exact_match"].append(test_good.sequence_exact_match)
                    self.history["train_candidate_ranking_accuracy"].append(train_good.candidate_ranking_accuracy)
                    self.history["test_candidate_ranking_accuracy"].append(test_good.candidate_ranking_accuracy)
                    self.history["train_logit_sequence_exact_match"].append(train_log.sequence_exact_match)
                    self.history["test_logit_sequence_exact_match"].append(test_log.sequence_exact_match)
                    self.history["train_logit_mean_correct_rank"].append(train_log_rank)
                    self.history["test_logit_mean_correct_rank"].append(test_log_rank)
                    self.history["eval_train_size"].append(len(train_eval))
                    self.history["eval_test_size"].append(len(test_eval))
                    self.history["active_block"].append(active_blocks[0] if self.layerwise_train_single_block else -1)
                    self.history["goodness_block_weights"].append(list(self.goodness_block_weights))

                    weights_msg = ",".join(f"{w:.2f}" for w in self.goodness_block_weights)

                    print(
                        f"[ff-disc step {step}] loss={total_loss.item():.4f} "
                        f"ff_loss={ff_loss_total.item():.4f} "
                        f"logit_aux={logit_aux_loss.item():.4f} "
                        f"kl_sync={kl_sync_loss.item():.4f} "
                        f"active_blocks=[{active_blocks_msg}] "
                        f"good_w=[{weights_msg}] "
                        f"steps_per_sec={steps_per_sec:.2f} "
                        f"train_logit_exact={train_log.sequence_exact_match:.3f} "
                        f"test_logit_exact={test_log.sequence_exact_match:.3f} "
                        f"train_good_rank={train_good.candidate_ranking_accuracy:.3f} "
                        f"test_good_rank={test_good.candidate_ranking_accuracy:.3f} "
                        f"logit_test_rank={test_log_rank:.2f} "
                        f"eval_train={len(train_eval)} eval_test={len(test_eval)} {block_msg}"
                    )

            if step % self.checkpoint_every == 0:
                ckpt_path = self._save_checkpoint(step)
                print(f"Saved checkpoint: {ckpt_path}")

        final_ckpt = self._save_checkpoint(self.num_steps)
        print(f"Saved final checkpoint: {final_ckpt}")

        final_train_eval = self._sample_eval_subset(self.train_problems, self.eval_train_max_samples)
        final_test_eval = self._sample_eval_subset(self.test_problems, self.eval_test_max_samples)
        if self.enable_goodness_eval and self.fit_goodness_block_weights:
            self._fit_goodness_weights(final_train_eval, causal=False)
        if self.enable_goodness_eval:
            final_train_good = self.evaluate_goodness(final_train_eval, causal=False)
            final_test_good = self.evaluate_goodness(final_test_eval, causal=False)
        else:
            final_train_good = FFGoodnessEvalResult(
                token_accuracy=float("nan"),
                sequence_exact_match=float("nan"),
                candidate_ranking_accuracy=float("nan"),
                block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                predictions=[],
                targets=[],
            )
            final_test_good = FFGoodnessEvalResult(
                token_accuracy=float("nan"),
                sequence_exact_match=float("nan"),
                candidate_ranking_accuracy=float("nan"),
                block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                predictions=[],
                targets=[],
            )
        final_train_log = self.evaluate_logits(final_train_eval)
        final_test_log = self.evaluate_logits(final_test_eval)

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
        eval_every: int | None = None,
        eval_train_max_samples: int | None = None,
        eval_test_max_samples: int | None = None,
        enable_goodness_eval: bool = True,
        output_embedding_detached: bool = True,
        use_per_block_output_heads: bool = False,
        final_block_loss_weight: float = 1.0,
        nonfinal_block_loss_weight: float = 1.0,
        block_output_head_states: list[dict[str, Any]] | None = None,
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
        self.eval_every = int(eval_every) if eval_every is not None else None
        self.eval_train_max_samples = int(eval_train_max_samples) if eval_train_max_samples is not None else None
        self.eval_test_max_samples = int(eval_test_max_samples) if eval_test_max_samples is not None else None
        self.enable_goodness_eval = bool(enable_goodness_eval)
        self.output_embedding_detached = bool(output_embedding_detached)
        self.use_per_block_output_heads = bool(use_per_block_output_heads)
        self.final_block_loss_weight = float(final_block_loss_weight)
        self.nonfinal_block_loss_weight = float(nonfinal_block_loss_weight)
        if self.final_block_loss_weight <= 0.0 or self.nonfinal_block_loss_weight < 0.0:
            raise ValueError(
                "final_block_loss_weight must be > 0 and nonfinal_block_loss_weight must be >= 0. "
                f"Got {self.final_block_loss_weight}, {self.nonfinal_block_loss_weight}"
            )

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
        self.block_output_heads: nn.ModuleList | None = None
        if self.use_per_block_output_heads:
            self.block_output_heads = nn.ModuleList(
                [nn.Linear(self.model.config.d_model, self.vocab.size, bias=True) for _ in self.model.blocks]
            ).to(self.device)
            if block_output_head_states is not None:
                if len(block_output_head_states) != len(self.block_output_heads):
                    raise ValueError(
                        "block_output_head_states length mismatch: "
                        f"expected {len(self.block_output_heads)}, got {len(block_output_head_states)}"
                    )
                for head, state in zip(self.block_output_heads, block_output_head_states):
                    head.load_state_dict(state)
            else:
                # Start from embedding-tied weights to reduce cold-start mismatch.
                embed = self.model.embedding_weight.detach()
                for head in self.block_output_heads:
                    head.weight.data.copy_(embed)
                    head.bias.data.zero_()
        elif block_output_head_states is not None:
            raise ValueError("block_output_head_states provided but use_per_block_output_heads=False")

        self.block_optimizers = []
        for block_idx, block in enumerate(self.model.blocks):
            params = list(block.parameters())
            if self.block_output_heads is not None:
                params.extend(self.block_output_heads[block_idx].parameters())
            self.block_optimizers.append(torch.optim.Adam(params, lr=self.lr))
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
            "candidate_loss_unweighted": [],
            "candidate_loss_final_block": [],
            "candidate_loss_nonfinal_mean": [],
            "goodness_aux_loss": [],
            "temperature": [],
            "train_token_accuracy": [],
            "test_token_accuracy": [],
            "train_sequence_exact_match": [],
            "test_sequence_exact_match": [],
            "train_goodness_candidate_accuracy": [],
            "test_goodness_candidate_accuracy": [],
            "eval_train_size": [],
            "eval_test_size": [],
            "block_g_pos": [[] for _ in range(n_blocks)],
            "block_g_neg": [[] for _ in range(n_blocks)],
            "block_threshold": [[] for _ in range(n_blocks)],
            "block_separation": [[] for _ in range(n_blocks)],
            "block_separation_ratio": [[] for _ in range(n_blocks)],
            "block_accuracy": [[] for _ in range(n_blocks)],
        }

    def _sample_problem_batch(self) -> list[Problem]:
        return [self.rng.choice(self.train_problems) for _ in range(self.batch_size)]

    def _sample_eval_subset(self, problems: list[Problem], max_samples: int | None) -> list[Problem]:
        if max_samples is None or max_samples >= len(problems):
            return problems
        if max_samples <= 0:
            return []
        indices = self.rng.sample(range(len(problems)), max_samples)
        return [problems[idx] for idx in indices]

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

    def _project_block_logits(self, block_idx: int, block_hidden: torch.Tensor) -> torch.Tensor:
        if self.block_output_heads is not None:
            return self.block_output_heads[block_idx](block_hidden)

        embedding = self.model.embedding_weight
        if self.output_embedding_detached:
            embedding = embedding.detach()
        return torch.matmul(block_hidden, embedding.T)

    def _block_loss_weight(self, block_idx: int, n_blocks: int) -> float:
        if block_idx == (n_blocks - 1):
            return self.final_block_loss_weight
        return self.nonfinal_block_loss_weight

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
            final_block_idx = len(block_outputs) - 1
            hidden = block_outputs[final_block_idx][:, -1, :]
            logits = self._project_block_logits(final_block_idx, hidden)[0]
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
            "eval_every": self.eval_every,
            "eval_train_max_samples": self.eval_train_max_samples,
            "eval_test_max_samples": self.eval_test_max_samples,
            "enable_goodness_eval": self.enable_goodness_eval,
            "output_embedding_detached": self.output_embedding_detached,
            "use_per_block_output_heads": self.use_per_block_output_heads,
            "final_block_loss_weight": self.final_block_loss_weight,
            "nonfinal_block_loss_weight": self.nonfinal_block_loss_weight,
            "block_output_head_states": (
                [head.state_dict() for head in self.block_output_heads]
                if self.block_output_heads is not None
                else None
            ),
            "rng_states": capture_rng_states(),
        }
        path = self.checkpoint_dir / f"{self.checkpoint_prefix}_step{step}.pt"
        return save_checkpoint(path, state)

    def train(self, log_every: int = 100) -> dict[str, Any]:
        self.model.train()
        start_time = time.perf_counter()

        for step in range(1, self.num_steps + 1):
            batch_problems = self._sample_problem_batch()
            full_tokens, x, y = self._build_autoregressive_batch(batch_problems)
            x_mask = x != self.vocab.pad_id

            block_outputs, _ = self.model(x, pad_mask=x_mask, causal=True, detach_between_blocks=True)
            n_blocks = len(block_outputs)

            candidate_block_losses_unweighted: list[torch.Tensor] = []
            candidate_block_losses_weighted: list[torch.Tensor] = []
            for block_idx, block_hidden in enumerate(block_outputs):
                logits = self._project_block_logits(block_idx, block_hidden)
                loss_k = self._candidate_loss(logits, y)
                weight_k = self._block_loss_weight(block_idx, n_blocks)
                candidate_block_losses_unweighted.append(loss_k)
                candidate_block_losses_weighted.append(loss_k * weight_k)

            candidate_loss_unweighted_total = sum(candidate_block_losses_unweighted)
            candidate_loss_total = sum(candidate_block_losses_weighted)
            candidate_loss_final_block = candidate_block_losses_unweighted[-1]
            if n_blocks > 1:
                candidate_loss_nonfinal_mean = torch.stack(candidate_block_losses_unweighted[:-1], dim=0).mean()
            else:
                candidate_loss_nonfinal_mean = torch.tensor(float("nan"), device=self.device)

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
                should_eval = (
                    self.eval_every is None
                    or step == 1
                    or step == self.num_steps
                    or step % self.eval_every == 0
                )
                elapsed = max(time.perf_counter() - start_time, 1e-8)
                steps_per_sec = step / elapsed

                block_msg = " ".join(
                    [
                        (
                            f"b{idx}:g+={stats['g_pos']:.4f},g-={stats['g_neg']:.4f},"
                            f"th={stats['threshold']:.4f},sep={stats['separation']:.4f}"
                        )
                        for idx, stats in enumerate(block_stats)
                    ]
                )

                if not should_eval:
                    print(
                        f"[ff-ar step {step}] loss={total_loss.item():.4f} temp={self.temperature:.3f} "
                        f"cand={candidate_loss_total.item():.4f} "
                        f"cand_unw={candidate_loss_unweighted_total.item():.4f} "
                        f"cand_final={candidate_loss_final_block.item():.4f} "
                        f"good_aux={goodness_aux_loss_total.item():.4f} "
                        f"steps_per_sec={steps_per_sec:.2f} "
                        f"(eval skipped) {block_msg}"
                    )
                else:
                    train_eval = self._sample_eval_subset(self.train_problems, self.eval_train_max_samples)
                    test_eval = self._sample_eval_subset(self.test_problems, self.eval_test_max_samples)
                    train_logits = self.evaluate_logits(train_eval)
                    test_logits = self.evaluate_logits(test_eval)
                    if self.enable_goodness_eval:
                        train_good = self.evaluate_goodness(train_eval)
                        test_good = self.evaluate_goodness(test_eval)
                    else:
                        train_good = FFGoodnessEvalResult(
                            token_accuracy=float("nan"),
                            sequence_exact_match=float("nan"),
                            candidate_ranking_accuracy=float("nan"),
                            block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                            predictions=[],
                            targets=[],
                        )
                        test_good = FFGoodnessEvalResult(
                            token_accuracy=float("nan"),
                            sequence_exact_match=float("nan"),
                            candidate_ranking_accuracy=float("nan"),
                            block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                            predictions=[],
                            targets=[],
                        )

                    self.history["step"].append(step)
                    self.history["loss"].append(float(total_loss.item()))
                    self.history["candidate_loss"].append(float(candidate_loss_total.item()))
                    self.history["candidate_loss_unweighted"].append(float(candidate_loss_unweighted_total.item()))
                    self.history["candidate_loss_final_block"].append(float(candidate_loss_final_block.item()))
                    self.history["candidate_loss_nonfinal_mean"].append(float(candidate_loss_nonfinal_mean.item()))
                    self.history["goodness_aux_loss"].append(float(goodness_aux_loss_total.item()))
                    self.history["temperature"].append(float(self.temperature))
                    self.history["train_token_accuracy"].append(train_logits.token_accuracy)
                    self.history["test_token_accuracy"].append(test_logits.token_accuracy)
                    self.history["train_sequence_exact_match"].append(train_logits.sequence_exact_match)
                    self.history["test_sequence_exact_match"].append(test_logits.sequence_exact_match)
                    self.history["train_goodness_candidate_accuracy"].append(train_good.candidate_ranking_accuracy)
                    self.history["test_goodness_candidate_accuracy"].append(test_good.candidate_ranking_accuracy)
                    self.history["eval_train_size"].append(len(train_eval))
                    self.history["eval_test_size"].append(len(test_eval))

                    print(
                        f"[ff-ar step {step}] loss={total_loss.item():.4f} temp={self.temperature:.3f} "
                        f"cand={candidate_loss_total.item():.4f} "
                        f"cand_unw={candidate_loss_unweighted_total.item():.4f} "
                        f"cand_final={candidate_loss_final_block.item():.4f} "
                        f"good_aux={goodness_aux_loss_total.item():.4f} "
                        f"steps_per_sec={steps_per_sec:.2f} "
                        f"train_exact={train_logits.sequence_exact_match:.3f} "
                        f"test_exact={test_logits.sequence_exact_match:.3f} "
                        f"train_good_rank={train_good.candidate_ranking_accuracy:.3f} "
                        f"test_good_rank={test_good.candidate_ranking_accuracy:.3f} "
                        f"eval_train={len(train_eval)} eval_test={len(test_eval)} {block_msg}"
                    )

            if step % self.checkpoint_every == 0:
                path = self._save_checkpoint(step)
                print(f"Saved checkpoint: {path}")

        final_path = self._save_checkpoint(self.num_steps)
        print(f"Saved final checkpoint: {final_path}")

        final_train_eval = self._sample_eval_subset(self.train_problems, self.eval_train_max_samples)
        final_test_eval = self._sample_eval_subset(self.test_problems, self.eval_test_max_samples)
        final_train_logits = self.evaluate_logits(final_train_eval)
        final_test_logits = self.evaluate_logits(final_test_eval)
        if self.enable_goodness_eval:
            final_train_good = self.evaluate_goodness(final_train_eval)
            final_test_good = self.evaluate_goodness(final_test_eval)
        else:
            final_train_good = FFGoodnessEvalResult(
                token_accuracy=float("nan"),
                sequence_exact_match=float("nan"),
                candidate_ranking_accuracy=float("nan"),
                block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                predictions=[],
                targets=[],
            )
            final_test_good = FFGoodnessEvalResult(
                token_accuracy=float("nan"),
                sequence_exact_match=float("nan"),
                candidate_ranking_accuracy=float("nan"),
                block_candidate_accuracy=[float("nan")] * len(self.model.blocks),
                predictions=[],
                targets=[],
            )

        return {
            "train_logits": final_train_logits,
            "test_logits": final_test_logits,
            "train_goodness": final_train_good,
            "test_goodness": final_test_good,
            "history": self.history,
            "checkpoint": str(final_path),
            "temperature": self.temperature,
        }
