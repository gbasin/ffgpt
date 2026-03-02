from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .data import (
    ANSWER_MAX,
    Problem,
    Vocab,
    answer_to_token_ids,
    build_problem_tensor,
    generate_all_problems,
    parse_answer_tokens,
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

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.train_tensor = build_problem_tensor(self.train_problems, self.vocab).to(self.device)

        self.history: dict[str, list[float | int]] = {
            "step": [],
            "loss": [],
            "train_token_accuracy": [],
            "test_token_accuracy": [],
            "train_sequence_exact_match": [],
            "test_sequence_exact_match": [],
        }

    def _sample_batch(self) -> torch.Tensor:
        idx = torch.randint(0, self.train_tensor.shape[0], (self.batch_size,), device=self.device)
        return self.train_tensor[idx]

    @torch.no_grad()
    def predict_with_logits(self, a: int, b: int) -> tuple[int | None, list[int]]:
        self.model.eval()

        context = [self.vocab.token_to_id[str(a)], self.vocab.plus_id, self.vocab.token_to_id[str(b)], self.vocab.equals_id]
        generated: list[int] = []

        for _ in range(2):
            input_ids = torch.tensor([context], dtype=torch.long, device=self.device)
            pad_mask = input_ids != self.vocab.pad_id
            logits, _, _ = self.model(input_ids, pad_mask=pad_mask, causal=True)
            next_token = int(torch.argmax(logits[0, -1]).item())
            generated.append(next_token)
            context.append(next_token)
            if next_token == self.vocab.pad_id:
                break

        if len(generated) < 2:
            generated.extend([self.vocab.pad_id] * (2 - len(generated)))

        pred_answer, normalized = parse_answer_tokens(generated[:2], self.vocab)
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
            target_tokens = answer_to_token_ids(problem.answer, self.vocab)

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

            if step % log_every == 0 or step == 1 or step == self.num_steps:
                train_eval = self.evaluate(self.train_problems)
                test_eval = self.evaluate(self.test_problems)

                self.history["step"].append(step)
                self.history["loss"].append(float(loss.item()))
                self.history["train_token_accuracy"].append(train_eval.token_accuracy)
                self.history["test_token_accuracy"].append(test_eval.token_accuracy)
                self.history["train_sequence_exact_match"].append(train_eval.sequence_exact_match)
                self.history["test_sequence_exact_match"].append(test_eval.sequence_exact_match)

                print(
                    f"[baseline step {step}] "
                    f"loss={loss.item():.4f} "
                    f"train_exact={train_eval.sequence_exact_match:.3f} "
                    f"test_exact={test_eval.sequence_exact_match:.3f}"
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
