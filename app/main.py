from .graph import build_graph
from tracking.metrics import RunMetrics, set_current_run
from tracking.db import DB_PATH


def run_cli():
    print("=" * 55)
    print("Welcome to the LinkedIn Post Generator")
    print("=" * 55)
    print("\nThis tool will draft a LinkedIn post for you, review it")
    print("itself, and iterate until it's publish-ready.")
    print("=" * 55)

    topic = input("\nWhat topic do you want a LinkedIn post about?\n> ").strip()

    if not topic:
        print("\nNo topic given. Exiting.")
        return

    print("\nStarting generation...\n")

    app = build_graph()

    run_metrics = RunMetrics()
    set_current_run(run_metrics)

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)
    print(final_state["draft"])
    print("=" * 55)
    print(f"Total attempts: {final_state['attempt']}")
    print(f"Approved: {final_state['is_approved']}")

    run_metrics.save(topic, final_state["attempt"], final_state["is_approved"])
    run_metrics.print_summary(topic, final_state["attempt"], final_state["is_approved"])

    print(f"\nAll metrics stored in SQLite at: {DB_PATH}")
    print('Query it any time, e.g.: sqlite3 data/metrics.db "SELECT * FROM runs;"')


if __name__ == "__main__":
    run_cli()
