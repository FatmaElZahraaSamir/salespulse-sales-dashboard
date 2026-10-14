"""Command line for the AI analyst.

    python -m salespulse examples                         # the example questions, no API key needed
    python -m salespulse ask "Which product lost the most money in 2019?" --provider gemini

The key is read from GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY (Ollama needs none).
"""

from __future__ import annotations

import argparse
import sys

from .analyst import EXAMPLES, Analyst, AnalystAnswer, run_example
from .data import load_sales
from .llm import PROVIDERS, LLMClient, LLMError
from .sql import SalesDB


def show(answer: AnalystAnswer) -> None:
    print(f"\nQ: {answer.question}\nA: {answer.text}")
    check = answer.check
    if check.matched or check.unmatched:
        status = "all match the query results" if check.ok else f"not found in the results: {', '.join(check.unmatched)}"
        print(f"   Numbers checked: {len(check.matched) + len(check.unmatched)} ({status})")
    for i, step in enumerate(answer.steps, 1):
        outcome = f"{step.rows} rows" if step.ok else f"error: {step.error}"
        print(f"\n   Query {i} · {step.title} · {outcome}\n" + "\n".join("      " + line for line in step.sql.splitlines()))
    if answer.final:
        print("\n" + answer.final.result.frame.head(20).to_string(index=False))
    print(f"\n   {answer.engine} · {answer.seconds:.1f}s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m salespulse", description="Ask the sales data a question.")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer a question with an LLM")
    ask.add_argument("question")
    ask.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    ask.add_argument("--model", default=None)
    sub.add_parser("examples", help="run the example questions (hand-written SQL, no key needed)")
    args = parser.parse_args(argv)

    db = SalesDB(load_sales())
    if args.command == "examples":
        for example in EXAMPLES:
            show(run_example(db, example))
        return 0
    try:
        llm = LLMClient(args.provider, model=args.model)
        show(Analyst(llm, db).ask(args.question, on_step=lambda s: print(f"   … {s.title}", file=sys.stderr)))
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
