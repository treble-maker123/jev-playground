"""Minimal client for the TypeSafe System One API (the Jev model).

One endpoint: POST https://api.typesafe.ai/v1/systemone with a `state` (the
content to evaluate), a `model`, and a map of typed `questions`. Answers come
back under the same keys you chose, each with a probability distribution.

    from jev import ask, choice, noul, score

    answers = ask(
        "Help! My payouts have been failing for 3 days.",
        {
            "department": choice(
                "Which team should handle this?",
                {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                },
            ),
            "frustration": score(
                "How frustrated is the customer?",
                ["Calm", "Frustrated", "Very angry"],
            ),
            "is_urgent": noul("Does this convey urgency?"),
        },
    )
    answers["department"]["choice"]        # "billing"
    answers["department"]["probabilities"] # {"billing": 0.88, ...}
    answers["is_urgent"]["noul"]           # 0.95  (0 = no, 1 = yes)

Docs: https://docs.typesafe.ai/api
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 30.0

# The API asks us to back off rather than retry immediately on these.
RETRY_STATUS = {429, 529}
MAX_RETRIES = 4

# `instructions` and `criteria` values may be a string or structured data
# (an object/array holding the question in one field and the data it refers
# to in others).
Prompt = str | dict[str, Any] | list[Any]


class JevError(RuntimeError):
    """An error response from the TypeSafe API."""

    def __init__(self, status: int, body: str):
        super().__init__(f"TypeSafe API returned {status}: {body}")
        self.status = status
        self.body = body


def noul(instructions: Prompt, *, true: Prompt | None = None, false: Prompt | None = None) -> dict:
    """A yes/no question. The answer is a probability from 0 (no) to 1 (yes)."""
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    criteria = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    if criteria:
        question["criteria"] = criteria
    return question


def choice(instructions: Prompt, criteria: dict[str, Prompt | None]) -> dict:
    """Pick one of `criteria` (option -> description, max 255 options)."""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: Prompt, criteria: list[Prompt]) -> dict:
    """Rate along an ordered rubric, lowest level first (2-10 levels)."""
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def system_one(
    state: Prompt,
    questions: dict[str, dict],
    *,
    model: str = DEFAULT_MODEL,
    api_token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    client: httpx.Client | None = None,
) -> dict:
    """Evaluate `state` against `questions`, returning the full response body.

    The body has `model`, `answers` (keyed by your question ids) and `usage`.
    """
    token = api_token or os.environ.get("JEV_API_TOKEN")
    if not token:
        raise JevError(401, "JEV_API_TOKEN is not set (put it in .env)")

    payload = {"state": state, "model": model, "questions": questions}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    owned = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        for attempt in range(MAX_RETRIES + 1):
            response = http.post(ENDPOINT, json=payload, headers=headers)
            if response.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                time.sleep(2**attempt + random.random())
                continue
            if response.status_code >= 400:
                raise JevError(response.status_code, response.text)
            return response.json()
    finally:
        if owned:
            http.close()


def ask(state: Prompt, questions: dict[str, dict], **kwargs) -> dict[str, dict]:
    """`system_one`, but returning just the `answers` map."""
    return system_one(state, questions, **kwargs)["answers"]


if __name__ == "__main__":
    ticket = (
        "Hi, I've been trying to connect my Stripe account for 3 days and the "
        "integration keeps failing. I'm losing sales. Please help ASAP."
    )
    answers = ask(
        ticket,
        {
            "department": choice(
                "Which team should handle this?",
                {
                    "billing": "Payment or subscription issues",
                    "technical": "Bugs or integration problems",
                    "sales": "Pricing or account questions",
                },
            ),
            "frustration": score(
                "How frustrated the customer appears",
                ["Calm, just stating facts", "Frustrated but civil", "Very angry"],
            ),
            "is_urgent": noul("The message conveys urgency or time-sensitivity"),
        },
    )
    for name, answer in answers.items():
        print(name, answer)
