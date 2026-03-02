from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from .data import (
    ANSWER_MAX,
    Problem,
    Vocab,
    answer_to_token_ids_variable,
    build_problem_tensor,
    generate_all_problems,
    parse_answer_tokens_variable,
    train_test_split,
)
from .model import BaselineTransformer
from .utils import capture_rng_states, save_checkpoint


@dataclass
class BaselineEvalResult:
    token_accuracy: float
    sequence_exact_match: float
    predictions: list[int]
    targets: list[int]


class BackpropTrainer:
    def __init__(
        self,
        model: BaselineTransformer,
        vocab: Vocab,
        train_problems: list[Problem],
        test_problems: list[Problem],
        lr: float = 1e-3,
        batch_size: int = 64,
        num_steps: int = 5000,
        checkpoint_every: int = 1000,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        sequence_length: int | None = None,
        max_answer_tokens: int | None = None,
        max_answer_value: int | None = None,
        eval_every: int | None = None,
        eval_train_max_samples: int | None = None,
        eval_test_max_samples: int | None = None,
        eval_at_step_one: bool = True,
        eval_seed: int = 12345,
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
        self.device = torch.device(device)
        self.sequence_length = int(sequence_length) if sequence_length is not None else int(self.model.config.max_seq_len)

        if self.sequence_length > self.model.config.max_seq_len:
            raise ValueError(
                f"sequence_length={self.sequence_length} exceeds model max_seq_len={self.model.config.max_seq_len}"
            )

        all_problems = self.train_problems + self.test_problems
        inferred_max_answer_tokens = max(len(str(problem.answer)) for problem in all_problems) if all_problems else 2
        self.max_answer_tokens = int(max_answer_tokens) if max_answer_tokens is not None else inferred_max_answer_tokens
        self.max_answer_value = (
            int(max_answer_value) if max_answer_value is not None else max((problem.answer for problem in all_problems), default=ANSWER_MAX)
        )
        self.eval_every = int(eval_every) if eval_every is not None else None
        self.eval_train_max_samples = int(eval_train_max_samples) if eval_train_max_samples is not None else None
        self.eval_test_max_samples = int(eval_test_max_samples) if eval_test_max_samples is not None else None
        self.eval_at_step_one = bool(eval_at_step_one)
        self.eval_rng = random.Random(eval_seed)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_tensor = build_problem_tensor(self.train_problems, self.vocab, max_len=self.sequence_length).to(self.device)

        self.history: dict[str, list[float | int]] = {
            "step": [],
            "loss": [],
            "train_token_accuracy": [],
            "test_token_accuracy": [],
            "train_sequence_exact_match": [],
            "test_sequence_exact_match": [],
            "train_eval_size": [],
            "test_eval_size": [],
        }

    def _sample_batch(self) -> torch.Tensor:
        idx = torch.randint(0, self.train_tensor.shape[0], (self.batch_size,), device=self.device)
        return self.train_tensor[idx]

    def _prompt_token_ids(self, a: int, b: int) -> list[int]:
        prompt_tokens = [*list(str(a)), "+", *list(str(b)), "="]
        return self.vocab.encode_tokens(prompt_tokens)

    def _should_eval(self, step: int, log_every: int) -> bool:
        interval = self.eval_every if self.eval_every is not None else log_every
        if step == self.num_steps:
            return True
        if self.eval_at_step_one and step == 1:
            return True
        return interval > 0 and (step % interval == 0)

    def _maybe_subsample(self, problems: list[Problem], max_samples: int | None) -> list[Problem]:
        if max_samples is None or max_samples <= 0 or len(problems) <= max_samples:
            return list(problems)
        return self.eval_rng.sample(problems, max_samples)

    @torch.no_grad()
    def predict_with_logits(self, a: int, b: int) -> tuple[int | None, list[int]]:
        self.model.eval()

        context = self._prompt_token_ids(a, b)
        generated: list[int] = []

        for _ in range(self.max_answer_tokens):
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            pad_mask = input_ids != self.vocab.pad_id
            logits, _, _ = self.model(input_ids, pad_mask=pad_mask, causal=True)
            next_token = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_token)
            context.append(next_token)
            if next_token == self.vocab.pad_id:
                break

        if len(generated) < self.max_answer_tokens:
            generated.extend([self.vocab.pad_id] * (self.max_answer_tokens - len(generated)))

        pred_answer, normalized = parse_answer_tokens_variable(
            generated[: self.max_answer_tokens],
            self.vocab,
            expected_length=self.max_answer_tokens,
            max_answer_value=self.max_answer_value,
        )
        return pred_answer, normalized

    @torch.no_grad()
    def evaluate(self, problems: list[Problem]) -> BaselineEvalResult:
        token_correct = 0
        token_total = 0
        sequence_correct = 0
        predictions: list[int] = []
        targets: list[int] = []

        for problem in problems:
            pred_answer, pred_tokens = self.predict_with_logits(problem.a, problem.b)
            target_tokens = answer_to_token_ids_variable(
                problem.answer,
                self.vocab,
                total_answer_tokens=self.max_answer_tokens,
                max_answer_value=self.max_answer_value,
            )

            token_correct += sum(int(p == t) for p, t in zip(pred_tokens, target_tokens))
            token_total += len(target_tokens)
            sequence_correct += int(pred_tokens == target_tokens)

            predictions.append(int(pred_answer) if pred_answer is not None else -1)
            targets.append(problem.answer)

        return BaselineEvalResult(
            token_accuracy=float(token_correct / max(token_total, 1)),
            sequence_exact_match=float(sequence_correct / max(len(problems), 1)),
            predictions=predictions,
            targets=targets,
        )

    def _save_checkpoint(self, step: int) -> Path:
        ckpt_path = self.checkpoint_dir / f"baseline_step{step}.pt"
        state: dict[str, Any] = {
            "mode": "baseline",
            "step": step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.model.config.__dict__,
            "sequence_length": self.sequence_length,
            "max_answer_tokens": self.max_answer_tokens,
            "max_answer_value": self.max_answer_value,
            "eval_every": self.eval_every,
            "eval_train_max_samples": self.eval_train_max_samples,
            "eval_test_max_samples": self.eval_test_max_samples,
            "eval_at_step_one": self.eval_at_step_one,
            "rng_states": capture_rng_states(),
        }
        return save_checkpoint(ckpt_path, state)

    def train(self, log_every: int = 100) -> dict[str, Any]:
        self.model.train()

        for step in range(1, self.num_steps + 1):
            batch = self._sample_batch()
            x = batch[:, :-1]
            y = batch[:, 1:]
            pad_mask = x != self.vocab.pad_id

            logits, _, _ = self.model(x, pad_mask=pad_mask, causal=True)
            loss = F.cross_entropy(logits.reshape(-1, self.vocab.size), y.reshape(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if step % log_every == 0:
                print(f"[baseline step {step}] loss={loss.item():.4f}")

            if self._should_eval(step=step, log_every=log_every):
                eval_train_set = self._maybe_subsample(self.train_problems, self.eval_train_max_samples)
                eval_test_set = self._maybe_subsample(self.test_problems, self.eval_test_max_samples)

                train_eval = self.evaluate(eval_train_set)
                test_eval = self.evaluate(eval_test_set)

                self.history["step"].append(step)
                self.history["loss"].append(float(loss.item()))
                self.history["train_token_accuracy"].append(train_eval.token_accuracy)
                self.history["test_token_accuracy"].append(test_eval.token_accuracy)
                self.history["train_sequence_exact_match"].append(train_eval.sequence_exact_match)
                self.history["test_sequence_exact_match"].append(test_eval.sequence_exact_match)
                self.history["train_eval_size"].append(len(eval_train_set))
                self.history["test_eval_size"].append(len(eval_test_set))

                print(
                    f"[baseline step {step}] "
                    f"loss={loss.item():.4f} "
                    f"train_exact={train_eval.sequence_exact_match:.3f} "
                    f"test_exact={test_eval.sequence_exact_match:.3f} "
                    f"(eval_sizes train={len(eval_train_set)} test={len(eval_test_set)})"
                )

            if step % self.checkpoint_every == 0:
                path = self._save_checkpoint(step)
                print(f"Saved checkpoint: {path}")

        final_path = self._save_checkpoint(self.num_steps)
        print(f"Saved final checkpoint: {final_path}")

        final_train = self.evaluate(self.train_problems)
        final_test = self.evaluate(self.test_problems)
        return {
            "train": final_train,
            "test": final_test,
            "history": self.history,
            "checkpoint": str(final_path),
        }


@torch.no_grad()
def quick_baseline_eval(model: BaselineTransformer, vocab: Vocab) -> dict[str, float]:
    problems = generate_all_problems()
    train, test = train_test_split(problems)
    trainer = BackpropTrainer(
        model=model,
        vocab=vocab,
        train_problems=train,
        test_problems=test,
        num_steps=0,
    )
    train_eval = trainer.evaluate(train)
    test_eval = trainer.evaluate(test)
    return {
        "train_exact": train_eval.sequence_exact_match,
        "test_exact": test_eval.sequence_exact_match,
        "train_token": train_eval.token_accuracy,
        "test_token": test_eval.token_accuracy,
    }
