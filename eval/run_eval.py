"""Evaluate the pipeline on eval/questions.json.

1) Retrieval (no LLM needed, fast) - compares configurations side by side:
     Hit Rate@k   share of in-scope questions whose expected document is in the top k
     MRR          mean of 1/rank of the expected document (1.0 = always ranked first)
     Gate         share of out-of-scope questions where retrieval correctly returned nothing
2) End-to-end with the agent (--with-llm):
     correct refusals on out-of-scope questions, false refusals on in-scope ones,
     citation rate (answers containing [n]), faithfulness via LLM-as-judge.
Results are also saved to eval/results/<timestamp>.json so runs can be compared over time.

Usage:
    python -m eval.run_eval                  # retrieval: vector vs hybrid
    python -m eval.run_eval --rerank         # also hybrid + cross-encoder re-ranker
    python -m eval.run_eval --with-llm       # + end-to-end agent evaluation (slow on Ollama)
    python -m eval.run_eval --questions eval/rbi_questions.json --split dev
"""
import argparse
import json
import re
import time
from pathlib import Path

from app.agent import build_agent, run_turn
from app.config import get_settings
from app.providers import CrossEncoderReranker, get_llm, get_vectorstore
from app.rag import NOT_FOUND, format_context
from app.retrieval import HybridRetriever
from eval.judge import judge_faithfulness


def _norm(text: str) -> str:
    """For phrase matching only: lower-case and drop ALL whitespace. pypdf inserts stray
    spaces ("low -risk", "medi um", "per son"), so comparing without whitespace makes
    matching robust to this layout noise."""
    return re.sub(r"\s+", "", text.lower())

def is_hit(doc, q: dict) -> bool:
    """A retrieved chunk counts only if it comes from the expected document AND, when the
    question specifies expected_text, actually contains that passage. Matching the text,
    not just the file, matters for long documents: finding *some* KYC chunk is easy,
    finding the chunk that holds the answer is the real test."""
    if doc.metadata.get("source") != q["expected_source"]:
        return False
    expected = q.get("expected_text")
    return expected is None or _norm(expected) in _norm(doc.page_content)


def retrieval_metrics(retriever, items, k) -> dict:
    in_scope = [q for q in items if q["in_scope"]]
    out_scope = [q for q in items if not q["in_scope"]]
    hits, rr, misses = 0, 0.0, []
    for q in in_scope:
        docs = retriever.search(q["question"], k)
        rank = next((i for i, d in enumerate(docs, start=1) if is_hit(d, q)), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(q["question"])
    leaks = [q["question"] for q in out_scope if retriever.search(q["question"], k)]
    return {"hit_rate": hits / len(in_scope) if in_scope else None,
            "mrr": rr / len(in_scope) if in_scope else None,
            "gate": (len(out_scope) - len(leaks)) / len(out_scope) if out_scope else None,
            "misses": misses, "leaks": leaks}


def end_to_end(retriever, llm, s, items) -> dict:
    agent = build_agent(retriever, llm, s)
    rows, correct_refusals, false_refusals, cited = [], 0, 0, 0
    verdicts = []
    for i, q in enumerate(items):
        out = run_turn(agent, f"eval-{i}", q["question"])          # fresh session per question
        refused = NOT_FOUND.lower() in out["answer"].lower()
        row = {"question": q["question"], "answer": out["answer"], "route": out["route"], "refused": refused}
        if not q["in_scope"]:
            correct_refusals += refused
        elif refused:
            false_refusals += 1
        else:
            cited += bool(re.search(r"\[\d+\]", out["answer"]))
            docs = retriever.search(out["search_query"], s.top_k)
            verdict, reason = judge_faithfulness(llm, format_context(docs), q["question"], out["answer"])
            row.update(faithful=verdict, judge_reason=reason)
            verdicts.append(verdict)
        rows.append(row)
        print(f"  [{i + 1}/{len(items)}] {q['question'][:60]}")
    answered = sum(1 for q in items if q["in_scope"]) - false_refusals
    judged = [v for v in verdicts if v is not None]
    return {
        "correct_refusals": f"{correct_refusals}/{sum(1 for q in items if not q['in_scope'])}",
        "false_refusals": f"{false_refusals}/{sum(1 for q in items if q['in_scope'])}",
        "citation_rate": cited / answered if answered else None,
        "faithfulness": sum(judged) / len(judged) if judged else None,
        "unparseable_judgements": len(verdicts) - len(judged),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all",
                        help="tune on dev; run test ONCE at the end and report that number")
    args = parser.parse_args()

    s = get_settings()
    k = args.k or s.top_k
    vs = get_vectorstore(s)
    items = json.loads(Path(args.questions).read_text())
    if args.split != "all":
        items = [q for q in items if q.get("split") == args.split]

    configs = {
        "vector": HybridRetriever(vs, s.model_copy(update={"retrieval_mode": "vector"})),
        "hybrid": HybridRetriever(vs, s.model_copy(update={"retrieval_mode": "hybrid"})),
    }
    if args.rerank:
        configs["hybrid+rerank"] = HybridRetriever(
            vs, s.model_copy(update={"retrieval_mode": "hybrid"}), CrossEncoderReranker(s.reranker_model))

    report = {"k": k, "questions": args.questions, "split": args.split,
              "n_questions": len(items), "retrieval": {}}
    print(f"\nRetrieval  (k={k}, {sum(q['in_scope'] for q in items)} in-scope, "
          f"{sum(not q['in_scope'] for q in items)} out-of-scope)")
    print(f"{'config':<16}{'HitRate@k':>10}{'MRR':>8}{'Gate':>8}")
    for name, retriever in configs.items():
        m = retrieval_metrics(retriever, items, k)
        report["retrieval"][name] = m
        gate = f"{m['gate']:.0%}" if m["gate"] is not None else "-"
        print(f"{name:<16}{m['hit_rate']:>10.0%}{m['mrr']:>8.3f}{gate:>8}")
        for miss in m["misses"]:
            print(f"    miss: {miss}")
        for leak in m["leaks"]:
            print(f"    leak: {leak}")

    if args.with_llm:
        best = "hybrid+rerank" if args.rerank else s.retrieval_mode
        print(f"\nEnd-to-end agent evaluation with '{best}' retrieval and {s.llm_provider}:")
        e2e = end_to_end(configs[best], get_llm(s), s, items)
        report["end_to_end"] = e2e
        for key in ("correct_refusals", "false_refusals", "citation_rate", "faithfulness", "unparseable_judgements"):
            val = e2e[key]
            print(f"  {key:<24}{val:.0%}" if isinstance(val, float) else f"  {key:<24}{val}")

    out_dir = Path("eval/results")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
