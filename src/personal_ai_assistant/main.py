#!/usr/bin/env python
"""Entry points for the Student Study & Productivity Assistant.

`run()` is an interactive multi-turn chat loop (used by `crewai run`).
Pass `--file <path>` to make a document available to the assistant for the session.
"""

import json
import sys

from personal_ai_assistant.crew import PersonalAiAssistant

DEFAULT_REQUEST = "What is my name and what am I currently learning?"
MAX_TURNS_IN_CONTEXT = 6


def _format_history(history: list[tuple[str, str]]) -> str:
    if not history:
        return "(no previous messages)"
    recent = history[-MAX_TURNS_IN_CONTEXT:]
    return "\n".join(f"{role}: {text}" for role, text in recent)


def _parse_file_arg(argv: list[str]) -> str | None:
    if "--file" in argv:
        idx = argv.index("--file")
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _run_once(user_request: str, history: list[tuple[str, str]], attached_file: str | None) -> str:
    request = user_request
    if attached_file:
        request = f"{user_request}\n\n[Attached file for this session: {attached_file}]"
    inputs = {
        "user_request": request,
        "conversation_history": _format_history(history),
    }
    result = PersonalAiAssistant().crew().kickoff(inputs=inputs)
    return str(result)


def run():
    """Interactive multi-turn assistant."""
    attached_file = _parse_file_arg(sys.argv)
    history: list[tuple[str, str]] = []

    print("\nStudy Assistant ready. Type your question, or 'exit' to quit.")
    if attached_file:
        print(f"Attached file: {attached_file}")

    while True:
        try:
            user_request = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not user_request:
            continue
        if user_request.lower() in {"exit", "quit", "bye"}:
            print("Bye!")
            return

        try:
            answer = _run_once(user_request, history, attached_file)
        except Exception as exc:  # noqa: BLE001
            print(f"\nAn error occurred: {exc}")
            continue

        print("\n" + "=" * 60)
        print("ASSISTANT")
        print("=" * 60)
        print(answer)

        history.append(("User", user_request))
        history.append(("Assistant", answer))


def train():
    """Train the crew for a given number of iterations."""
    inputs = {"user_request": DEFAULT_REQUEST, "conversation_history": "(no previous messages)"}
    try:
        PersonalAiAssistant().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        PersonalAiAssistant().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and return the results."""
    inputs = {"user_request": DEFAULT_REQUEST, "conversation_history": "(no previous messages)"}
    try:
        PersonalAiAssistant().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs,
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
    """Run once with a JSON payload passed as the first CLI argument."""
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    inputs = {
        "user_request": payload.get("user_request", DEFAULT_REQUEST),
        "conversation_history": payload.get("conversation_history", "(no previous messages)"),
    }
    return PersonalAiAssistant().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    run()
