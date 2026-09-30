"""Sanity-check an eval file BEFORE running the eval.

For every in-scope question it confirms that:
  - the expected_source file exists in the corpus folder, and
  - the expected_text really appears in that document's extracted text.
If a phrase is missing, the question (or PDF extraction) is wrong, and any "miss" in the
eval would be a false alarm. Fix these first.

Usage:  python -m scripts.check_questions eval/rbi_questions.json data/rbi
"""
import json
import sys
from pathlib import Path

from app.ingest import load_file
from eval.run_eval import _norm


def main():
    questions_path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/rbi_questions.json")
    corpus = Path(sys.argv[2] if len(sys.argv) > 2 else "data/rbi")
    items = json.loads(questions_path.read_text())
    texts: dict[str, str] = {}
    problems = 0
    for q in (x for x in items if x["in_scope"]):
        src = q["expected_source"]
        if src not in texts:
            path = corpus / src
            if not path.exists():
                print(f"MISSING FILE  {src}")
                texts[src] = None
            else:
                texts[src] = _norm(" ".join(d.page_content for d in load_file(path)))
        if texts[src] is None:
            problems += 1
            continue
        expected = q.get("expected_text")
        if expected and _norm(expected) not in texts[src]:
            problems += 1
            print(f"NOT FOUND     [{src}] '{expected}'  <- {q['question']}")
    total = sum(1 for x in items if x["in_scope"])
    print(f"\n{total - problems}/{total} in-scope questions verified against the documents.")


if __name__ == "__main__":
    main()
