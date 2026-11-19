#!/usr/bin/env python3
"""Run the golden set and red-team prompts against a running chatbot.

    scripts/run-evals.py [--url http://localhost:8000] [--user NAME:PASSWORD]
                         [--canary STRING ...] [--min-recall 0.8]

Asks each question in evals/golden.json and each prompt in
evals/redteam.json through POST /api/chat (as temporary chats, so nothing
is saved), then prints recall@k for questions with expected_doc_ids, the
answer checks, and any red-team leaks. Exits 1 if a red-team answer leaks
or a golden check fails or recall is under --min-recall.

Canaries are unique strings planted in places that must never be indexed
(an admin wiki page, a private Slack channel, secrets.yaml, an excluded
ESPHome device); pass each with --canary.
"""

import argparse
import calendar
import json
import sys
from datetime import date, timedelta
from pathlib import Path

# Any: chat responses are JSON.
from typing import Any

import httpx

EVALS = Path(__file__).resolve().parent.parent / "evals"


def next_last_saturday(today: date) -> date:
    """The last Saturday of this month, or of next month if it has passed."""
    year, month = today.year, today.month
    for _ in range(2):
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        candidate = last_day - timedelta(days=(last_day.weekday() - calendar.SATURDAY) % 7)
        if candidate >= today:
            return candidate
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    raise AssertionError("unreachable")


def doc_ids(response: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for chunk in response.get("context", []):
        metadata = chunk.get("metadata", {})
        if "doc_id" in metadata:
            ids.append(f"{metadata.get('source', chunk.get('collection'))}:{metadata['doc_id']}")
    return list(dict.fromkeys(ids))


def ask(client: httpx.Client, question: str) -> dict[str, Any]:
    response = client.post("/api/chat", json={"question": question, "temporary_chat": True})
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    return data


def golden_failures(item: dict[str, Any], response: dict[str, Any], today: date) -> list[str]:
    answer = response.get("answer", "").lower()
    failures = [
        f"doesn't mention {word!r}"
        for word in item.get("must_mention_all", [])
        if word.lower() not in answer
    ]
    any_words = item.get("must_mention_any", [])
    if any_words and not any(word.lower() in answer for word in any_words):
        failures.append(f"mentions none of {any_words}")
    if "expect_answered" in item and response.get("answered") != item["expect_answered"]:
        failures.append(f"answered={response.get('answered')}")
    if item.get("expect_next_last_saturday"):
        day = next_last_saturday(today)
        forms = {day.isoformat(), f"{calendar.month_name[day.month]} {day.day}".lower()}
        if not any(form.lower() in answer for form in forms):
            failures.append(f"doesn't give {day.isoformat()}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--user", help="NAME:PASSWORD of the local login account")
    parser.add_argument("--canary", action="append", default=[])
    parser.add_argument("--min-recall", type=float, default=0.0)
    args = parser.parse_args()

    golden = json.loads((EVALS / "golden.json").read_text())
    redteam = json.loads((EVALS / "redteam.json").read_text())
    today = date.today()  # noqa: DTZ011 - the open house date is local.
    failed = False

    with httpx.Client(base_url=args.url, timeout=300) as client:
        if args.user:
            name, _, password = args.user.partition(":")
            login = client.post("/login", data={"username": name, "password": password})
            if login.status_code != 303:
                print(f"login as {name!r} failed: HTTP {login.status_code}")
                return 1
        recalls: list[float] = []
        for item in golden["questions"]:
            response = ask(client, item["question"])
            expected = set(item.get("expected_doc_ids", []))
            if expected:
                recalls.append(len(expected & set(doc_ids(response))) / len(expected))
            failures = golden_failures(item, response, today)
            failed |= bool(failures)
            print(f"{'FAIL' if failures else 'ok  '} {item['question']}")
            for failure in failures:
                print(f"       {failure}")
        if recalls:
            recall = sum(recalls) / len(recalls)
            print(f"recall@k over {len(recalls)} questions: {recall:.2f}")
            failed |= recall < args.min_recall

        banned = [*redteam["must_not_mention"], *args.canary]
        for prompt in redteam["prompts"]:
            answer = ask(client, prompt).get("answer", "")
            leaks = [text for text in banned if text.lower() in answer.lower()]
            failed |= bool(leaks)
            print(f"{'LEAK' if leaks else 'ok  '} {prompt}")
            for leak in leaks:
                print(f"       contains {leak!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
