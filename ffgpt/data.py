from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence

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

    def encode_equation(
        self,
        a: int,
        b: int,
        answer: int,
        max_len: int = MAX_SEQ_LEN,
        validate_sum: bool = False,
    ) -> list[int]:
        tokens = format_equation_tokens(a, b, answer, validate_sum=validate_sum)
        if len(tokens) > max_len:
            raise ValueError(f"Equation tokens exceed max_len={max_len}: {tokens}")
        padded = tokens + [PAD_TOKEN] * (max_len - len(tokens))
        return self.encode_tokens(padded)

    def decode_equation(self, token_ids: Sequence[int]) -> str:
        tokens = self.decode_ids(token_ids)
        no_pad = [t for t in tokens if t != PAD_TOKEN]
        return "".join(no_pad)


def format_equation_tokens(a: int, b: int, answer: int, validate_sum: bool = True) -> list[str]:
    if a < 0 or b < 0 or answer < 0:
        raise ValueError(f"Only non-negative integers are supported, got {a}+{b}={answer}")
    if validate_sum and a + b != answer:
        raise ValueError(f"Answer does not match operands: {a}+{b}!={answer}")
    return [*list(str(a)), PLUS_TOKEN, *list(str(b)), EQUALS_TOKEN, *list(str(answer))]


def max_sum_for_operand_digits(operand_digits: int) -> int:
    if operand_digits <= 0:
        raise ValueError(f"operand_digits must be >= 1, got {operand_digits}")
    max_operand = (10**operand_digits) - 1
    return max_operand * 2


def max_seq_len_for_operand_digits(operand_digits: int) -> int:
    # [a_digits] + '+' + [b_digits] + '=' + [answer_digits<=operand_digits+1]
    return (2 * operand_digits) + (operand_digits + 1) + 2


def generate_problems_for_operand_digits(
    operand_digits: int,
    num_samples: int | None = None,
    seed: int = 42,
    exhaustive_limit: int = 100_000,
) -> list[Problem]:
    if operand_digits <= 0:
        raise ValueError(f"operand_digits must be >= 1, got {operand_digits}")

    max_operand = (10**operand_digits) - 1
    domain_size = (max_operand + 1) ** 2

    if num_samples is None and domain_size <= exhaustive_limit:
        return [Problem(a=a, b=b, answer=a + b) for a in range(max_operand + 1) for b in range(max_operand + 1)]

    if num_samples is None:
        num_samples = min(exhaustive_limit, domain_size)
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0 when provided, got {num_samples}")

    rng = random.Random(seed)
    problems: list[Problem] = []
    seen: set[tuple[int, int]] = set()

    while len(problems) < num_samples:
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        problems.append(Problem(a=a, b=b, answer=a + b))

    return problems


def parse_answer_tokens(token_ids: Sequence[int], vocab: Vocab) -> tuple[int | None, list[int]]:
    return parse_answer_tokens_variable(
        token_ids=token_ids[:2],
        vocab=vocab,
        expected_length=2,
        max_answer_value=ANSWER_MAX,
    )


def answer_to_token_ids(answer: int, vocab: Vocab) -> list[int]:
    return answer_to_token_ids_variable(answer=answer, vocab=vocab, total_answer_tokens=2, max_answer_value=ANSWER_MAX)


def answer_to_token_ids_variable(
    answer: int,
    vocab: Vocab,
    total_answer_tokens: int,
    max_answer_value: int | None = None,
) -> list[int]:
    if answer < 0:
        raise ValueError(f"Answer must be non-negative, got {answer}")
    if max_answer_value is not None and answer > max_answer_value:
        raise ValueError(f"Answer out of range: {answer} > {max_answer_value}")

    digits = list(str(answer))
    if len(digits) > total_answer_tokens:
        raise ValueError(
            f"Answer {answer} has {len(digits)} digits, exceeds total_answer_tokens={total_answer_tokens}"
        )
    padded = digits + [PAD_TOKEN] * (total_answer_tokens - len(digits))
    return vocab.encode_tokens(padded)


def parse_answer_tokens_variable(
    token_ids: Sequence[int],
    vocab: Vocab,
    expected_length: int | None = None,
    max_answer_value: int | None = None,
) -> tuple[int | None, list[int]]:
    ids = [int(x) for x in token_ids]
    if expected_length is not None:
        if len(ids) > expected_length:
            ids = ids[:expected_length]
        elif len(ids) < expected_length:
            ids.extend([vocab.pad_id] * (expected_length - len(ids)))

    tokens = vocab.decode_ids(ids)
    digits: list[str] = []
    for token in tokens:
        if token == PAD_TOKEN:
            break
        if token not in DIGIT_TOKENS:
            return None, ids
        digits.append(token)

    if not digits:
        return None, ids

    value = int("".join(digits))
    if max_answer_value is not None and value > max_answer_value:
        return None, ids

    normalized = answer_to_token_ids_variable(
        answer=value,
        vocab=vocab,
        total_answer_tokens=len(ids),
        max_answer_value=max_answer_value,
    )
    return value, normalized


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
        max_len: int = MAX_SEQ_LEN,
        seed: int | None = None,
    ) -> None:
        self.problems = list(problems)
        self.vocab = vocab
        self.negative_strategy = negative_strategy
        self.near_miss_offsets = tuple(int(x) for x in near_miss_offsets)
        self.max_len = int(max_len)
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
        pos_tokens = tokenize_problem(problem, self.vocab, max_len=self.max_len)
        neg_tokens = torch.tensor(
            self.vocab.encode_equation(problem.a, problem.b, negative_answer, max_len=self.max_len), dtype=torch.long
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
    def __init__(self, problems: Sequence[Problem], vocab: Vocab, max_len: int = MAX_SEQ_LEN) -> None:
        self.problems = list(problems)
        self.vocab = vocab
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        problem = self.problems[idx]
        tokens = tokenize_problem(problem, self.vocab, max_len=self.max_len)
        return {
            "tokens": tokens,
            "a": problem.a,
            "b": problem.b,
            "answer": problem.answer,
        }


def build_problem_tensor(problems: Sequence[Problem], vocab: Vocab, max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
    return torch.stack([tokenize_problem(problem, vocab, max_len=max_len) for problem in problems], dim=0)


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
