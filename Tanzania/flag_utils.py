"""Build a task flag from ordered retrieval results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON file.

    :param path: File to read.
    :return: Parsed JSON value.
    """
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def load_verification_codes(corpus_path: Path) -> dict[str, str]:
    """Map every corpus document ID to its verification character.

    :param corpus_path: JSON Lines corpus supplied with the task.
    :return: Verification characters indexed by document ID.
    """
    codes: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            try:
                document = json.loads(line)
                document_id = document["document_id"]
                code = document["verification_code"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid corpus row at line {line_number}."
                ) from error
            if document_id in codes:
                raise ValueError(f"Duplicate document ID: {document_id}")
            if not isinstance(code, str) or len(code) != 1:
                raise ValueError(f"Invalid verification code for {document_id}.")
            codes[document_id] = code
    return codes


def derive_flag(
    results: list[dict[str, str]],
    questions: list[dict[str, str]],
    verification_codes: dict[str, str],
) -> str:
    """Hash verification characters in the official question order.

    :param results: One top document ID for every question.
    :param questions: Ordered question definitions supplied with the task.
    :param verification_codes: Character indexed by corpus document ID.
    :return: Flag derived from the complete ordered result set.
    """
    result_by_question: dict[str, str] = {}
    for result in results:
        try:
            question_id = result["question_id"]
            document_id = result["document_id"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Each result needs question_id and document_id strings."
            ) from error
        if not isinstance(question_id, str) or not isinstance(document_id, str):
            raise ValueError("Result IDs must be strings.")
        if question_id in result_by_question:
            raise ValueError(f"Duplicate result for {question_id}.")
        result_by_question[question_id] = document_id

    expected_question_ids = [question["question_id"] for question in questions]
    if set(result_by_question) != set(expected_question_ids):
        missing = sorted(set(expected_question_ids) - set(result_by_question))
        unexpected = sorted(set(result_by_question) - set(expected_question_ids))
        raise ValueError(
            f"Question IDs do not match. Missing: {missing}; unexpected: {unexpected}."
        )

    sequence: list[str] = []
    for question_id in expected_question_ids:
        document_id = result_by_question[question_id]
        try:
            sequence.append(verification_codes[document_id])
        except KeyError as error:
            raise ValueError(f"Unknown document ID: {document_id}") from error

    digest = hashlib.sha256("".join(sequence).encode("utf-8")).hexdigest()
    return f"SIGMOID_{digest[:20].upper()}"


def derive_flag_from_files(
    results_path: Path,
    questions_path: Path,
    corpus_path: Path,
) -> str:
    """Derive a flag from participant result and task data files.

    :param results_path: JSON list produced by the retrieval pipeline.
    :param questions_path: Ordered task questions.
    :param corpus_path: Task corpus in JSON Lines format.
    :return: Derived flag string.
    """
    results = load_json(results_path)
    questions = load_json(questions_path)
    if not isinstance(results, list) or not isinstance(questions, list):
        raise ValueError("Questions and results must both be JSON lists.")
    return derive_flag(results, questions, load_verification_codes(corpus_path))
