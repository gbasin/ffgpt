from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Iterable, Sequence

import torch
from torch.utils.data import Dataset

DIGIT_TOKENS: tuple[str, ...] = tuple(str(i) for i in range(10))
PLUS_TOKEN = "+"
EQUALS_TOKEN = "="
PAD_TOKEN = "<PAD>"
VOCAB_TOKENS: tuple[str, ...] = DIGIT_TOKENS + (PLUS_TOKEN, EQUALS_TOKEN, PAD_TOKEN)
MAX_SEQ_LEN = 6
ANSWER_MIN = 0
ANSWER_MAX = 18


@dataclass(frozen=True)
class Problem:
    a: int
    b: int
    answer: int


class Vocab:
    def __init__(self, tokens: Sequence[str] = VOCAB_TOKENS) -> None:
        self.tokens = list(tokens)
        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.tokens)}

        required = set(VOCAB_TOKENS)
        missing = required.difference(self.token_to_id)
        if missing:
            raise ValueError(f"Missing required vocab tokens: {sorted(missing)}")

        self.pad_id = self.token_to_id[PAD_TOKEN]
        self.plus_id = self.token_to_id[PLUS_TOKEN]
        self.equals_id = self.token_to_id[EQUALS_TOKEN]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode_tokens(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]

    def decode_ids(self, token_ids: Sequence[int]) -> list[str]:
        return [self.id_to_token[int(token_id)] for token_id in token_ids]

    def encode_equation(self, a: int, b: int, answer: int, max_len: int = MAX_SEQ_LEN) -> list[int]:
        tokens = format_equation_tokens(a, b, answer)
        if len(tokens) > max_len:
            raise ValueError(f"Equation tokens exceed max_len={max_len}: {tokens}")
        padded = tokens + [PAD_TOKEN] * (max_len - len(tokens))
        return self.encode_tokens(padded)

    def decode_equation(self, token_ids: Sequence[int]) -> str:
        tokens = self.decode_ids(token_ids)
        no_pad = [t for t in tokens if t != PAD_TOKEN]
        return "".join(no_pad)


def format_equation_tokens(a: int, b: int, answer: int) -> list[str]:
    if not (0 <= a <= 9 and 0 <= b <= 9 and ANSWER_MIN <= answer <= ANSWER_MAX):
        raise ValueError(f"Invalid equation values: {a}+{b}={answer}")
    return [str(a), PLUS_TOKEN, str(b), EQUALS_TOKEN, *list(str(answer))]


def parse_answer_tokens(token_ids: Sequence[int], vocab: Vocab) -> tuple[int | None, list[int]]:
    """Parse up to two answer tokens into an int in [0, 18].

    Returns `(parsed_answer_or_none, normalized_token_ids_len_2)`.
    """
    ids = [int(x) for x in token_ids[:2]]
    if len(ids) < 2:
        ids.extend([vocab.pad_id] * (2 - len(ids)))

    toks = vocab.decode_ids(ids)
    digits: list[str] = []
    for tok in toks:
        if tok == PAD_TOKEN:
            break
        if tok not in DIGIT_TOKENS:
            return None, ids
        digits.append(tok)

    if not digits:
        return None, ids

    val = int("".join(digits))
    if val < ANSWER_MIN or val > ANSWER_MAX:
        return None, ids

    expected_ids = answer_to_token_ids(val, vocab)
    return val, expected_ids


def answer_to_token_ids(answer: int, vocab: Vocab) -> list[int]:
    if answer < ANSWER_MIN or answer > ANSWER_MAX:
        raise ValueError(f"Answer out of range: {answer}")
    digits = list(str(answer))
    if len(digits) == 1:
        digits.append(PAD_TOKEN)
    return vocab.encode_tokens(digits)


def tokenize_problem(problem: Problem, vocab: Vocab, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    return torch.tensor(vocab.encode_equation(problem.a, problem.b, problem.answer, max_len=max_len), dtype=torch.long)


def generate_all_problems() -> list[Problem]:
    return [Problem(a=a, b=b, answer=a + b) for a in range(10) for b in range(10)]


def default_holdout_fn(problem: Problem) -> bool:
    return (problem.a + problem.b) % 5 == 0


def train_test_split(
    problems: Sequence[Problem], holdout_fn: Callable[[Problem], bool] = default_holdout_fn
) -> tuple[list[Problem], list[Problem]]:
    test = [p for p in problems if holdout_fn(p)]
    train = [p for p in problems if not holdout_fn(p)]
    return train, test


def sample_negative_answer(
    correct_answer: int,
    strategy: str = "random",
    near_miss_offsets: Sequence[int] = (1,),
    rng: random.Random | None = None,
) -> int:
    rng = rng or random
    strategy = strategy.lower()

    if strategy == "near_miss":
        candidates: list[int] = []
        for offset in near_miss_offsets:
            for sign in (-1, 1):
                candidate = correct_answer + sign * int(offset)
                if ANSWER_MIN <= candidate <= ANSWER_MAX and candidate != correct_answer:
                    candidates.append(candidate)
        if candidates:
            return int(rng.choice(candidates))

    all_candidates = [x for x in range(ANSWER_MIN, ANSWER_MAX + 1) if x != correct_answer]
    return int(rng.choice(all_candidates))


def sample_negatives(
    correct_token: int,
    vocab_size: int,
    k: int,
    allowed_tokens: Sequence[int] | None = None,
    rng: random.Random | None = None,
) -> list[int]:
    rng = rng or random
    pool = list(range(vocab_size)) if allowed_tokens is None else [int(x) for x in allowed_tokens]
    pool = [token for token in pool if token != int(correct_token)]
    if k <= 0:
        return []
    if k >= len(pool):
        return pool
    return rng.sample(pool, k)


class DiscriminativeDataset(Dataset[dict[str, torch.Tensor | int]]):
    def __init__(
        self,
        problems: Sequence[Problem],
        vocab: Vocab,
        negative_strategy: str = "random",
        near_miss_offsets: Sequence[int] = (1,),
        seed: int | None = None,
    ) -> None:
        self.problems = list(problems)
        self.vocab = vocab
        self.negative_strategy = negative_strategy
        self.near_miss_offsets = tuple(int(x) for x in near_miss_offsets)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.problems)

    def set_negative_strategy(self, strategy: str, near_miss_offsets: Sequence[int] | None = None) -> None:
        self.negative_strategy = strategy
        if near_miss_offsets is not None:
            self.near_miss_offsets = tuple(int(x) for x in near_miss_offsets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        problem = self.problems[idx]
        negative_answer = sample_negative_answer(
            problem.answer,
            strategy=self.negative_strategy,
            near_miss_offsets=self.near_miss_offsets,
            rng=self.rng,
        )
        pos_tokens = tokenize_problem(problem, self.vocab)
        neg_tokens = torch.tensor(
            self.vocab.encode_equation(problem.a, problem.b, negative_answer, max_len=MAX_SEQ_LEN), dtype=torch.long
        )

        return {
            "pos_tokens": pos_tokens,
            "neg_tokens": neg_tokens,
            "a": problem.a,
            "b": problem.b,
            "answer": problem.answer,
            "negative_answer": negative_answer,
        }


class AutoregressiveDataset(Dataset[dict[str, torch.Tensor | int]]):
    def __init__(self, problems: Sequence[Problem], vocab: Vocab) -> None:
        self.problems = list(problems)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        problem = self.problems[idx]
        tokens = tokenize_problem(problem, self.vocab)
        return {
            "tokens": tokens,
            "a": problem.a,
            "b": problem.b,
            "answer": problem.answer,
        }


def build_problem_tensor(problems: Sequence[Problem], vocab: Vocab) -> torch.Tensor:
    return torch.stack([tokenize_problem(problem, vocab) for problem in problems], dim=0)


def run_roundtrip_tests() -> None:
    vocab = Vocab()

    # Tokenization round-trip for all equations.
    for problem in generate_all_problems():
        encoded = vocab.encode_equation(problem.a, problem.b, problem.answer)
        decoded = vocab.decode_equation(encoded)
        expected = f"{problem.a}+{problem.b}={problem.answer}"
        if decoded != expected:
            raise AssertionError(f"Round-trip mismatch: expected={expected} got={decoded}")

    problems = generate_all_problems()
    train, test = train_test_split(problems)
    if len(train) != 80 or len(test) != 20:
        raise AssertionError(f"Expected 80/20 split, got {len(train)}/{len(test)}")


if __name__ == "__main__":
    run_roundtrip_tests()
    print("data.py checks passed")
