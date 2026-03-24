"""Human-in-the-Loop Interviewer pattern (Section 6)."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from collections import deque


class QuestionType(str, Enum):
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    FREEFORM = "freeform"
    CONFIRMATION = "confirmation"


class AnswerValue(str, Enum):
    YES = "yes"
    NO = "no"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class Option:
    key: str
    label: str


@dataclass
class Question:
    text: str
    type: QuestionType = QuestionType.MULTIPLE_CHOICE
    options: list[Option] = field(default_factory=list)
    default: "Answer | None" = None
    timeout_seconds: float | None = None
    stage: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Answer:
    value: str | AnswerValue = ""
    selected_option: Option | None = None
    text: str = ""


class Interviewer:
    def ask(self, question: Question) -> Answer:
        raise NotImplementedError

    def ask_multiple(self, questions: list[Question]) -> list[Answer]:
        return [self.ask(q) for q in questions]

    def inform(self, message: str, stage: str = "") -> None:
        pass


class AutoApproveInterviewer(Interviewer):
    """Always approves / selects first option. For CI/testing."""

    def ask(self, question: Question) -> Answer:
        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            return Answer(value=AnswerValue.YES)
        if question.type == QuestionType.MULTIPLE_CHOICE and question.options:
            opt = question.options[0]
            return Answer(value=opt.key, selected_option=opt)
        return Answer(value="auto-approved", text="auto-approved")


class ConsoleInterviewer(Interviewer):
    """Reads from stdin, presents formatted prompts."""

    def ask(self, question: Question) -> Answer:
        print(f"\n[?] {question.text}")
        if question.stage:
            print(f"    Stage: {question.stage}")

        if question.type == QuestionType.MULTIPLE_CHOICE:
            for opt in question.options:
                print(f"  [{opt.key}] {opt.label}")
            try:
                response = input("Select: ").strip()
            except (EOFError, KeyboardInterrupt):
                return Answer(value=AnswerValue.SKIPPED)
            return _find_matching_option(response, question.options)

        if question.type in (QuestionType.YES_NO, QuestionType.CONFIRMATION):
            try:
                response = input("[Y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return Answer(value=AnswerValue.SKIPPED)
            if response in ("y", "yes"):
                return Answer(value=AnswerValue.YES)
            return Answer(value=AnswerValue.NO)

        if question.type == QuestionType.FREEFORM:
            try:
                response = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                return Answer(value=AnswerValue.SKIPPED)
            return Answer(text=response, value=response)

        return Answer(value=AnswerValue.SKIPPED)

    def inform(self, message: str, stage: str = "") -> None:
        prefix = f"[{stage}] " if stage else ""
        print(f"{prefix}{message}")


class CallbackInterviewer(Interviewer):
    """Delegates to a provided callback."""

    def __init__(self, callback: Callable[[Question], Answer]):
        self._callback = callback

    def ask(self, question: Question) -> Answer:
        return self._callback(question)


class QueueInterviewer(Interviewer):
    """Reads from a pre-filled answer queue. For testing/replay."""

    def __init__(self, answers: list[Answer] | None = None):
        self._queue: deque[Answer] = deque(answers or [])

    def ask(self, question: Question) -> Answer:
        if self._queue:
            return self._queue.popleft()
        return Answer(value=AnswerValue.SKIPPED)

    def enqueue(self, answer: Answer) -> None:
        self._queue.append(answer)


class RecordingInterviewer(Interviewer):
    """Wraps another interviewer and records all interactions."""

    def __init__(self, inner: Interviewer):
        self._inner = inner
        self.recordings: list[tuple[Question, Answer]] = []

    def ask(self, question: Question) -> Answer:
        answer = self._inner.ask(question)
        self.recordings.append((question, answer))
        return answer

    def inform(self, message: str, stage: str = "") -> None:
        self._inner.inform(message, stage)


def parse_accelerator_key(label: str) -> str:
    """Extract shortcut key from edge label."""
    import re
    # [K] Label
    m = re.match(r"^\[([A-Za-z0-9])\]\s*", label)
    if m:
        return m.group(1).upper()
    # K) Label
    m = re.match(r"^([A-Za-z0-9])\)\s*", label)
    if m:
        return m.group(1).upper()
    # K - Label
    m = re.match(r"^([A-Za-z0-9])\s+-\s+", label)
    if m:
        return m.group(1).upper()
    # First character
    return label[0].upper() if label else "?"


def normalize_label(label: str) -> str:
    """Normalize edge label for matching."""
    import re
    label = label.lower().strip()
    # Strip accelerator prefixes
    label = re.sub(r"^\[[a-z0-9]\]\s*", "", label)
    label = re.sub(r"^[a-z0-9]\)\s*", "", label)
    label = re.sub(r"^[a-z0-9]\s+-\s+", "", label)
    return label.strip()


def _find_matching_option(response: str, options: list[Option]) -> Answer:
    response_upper = response.upper()
    for opt in options:
        if opt.key.upper() == response_upper:
            return Answer(value=opt.key, selected_option=opt)
    # Try matching by label start
    response_lower = response.lower()
    for opt in options:
        if opt.label.lower().startswith(response_lower):
            return Answer(value=opt.key, selected_option=opt)
    # Fallback to first
    if options:
        return Answer(value=options[0].key, selected_option=options[0])
    return Answer(value=AnswerValue.SKIPPED)
