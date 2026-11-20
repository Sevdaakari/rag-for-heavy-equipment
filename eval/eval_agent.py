import json
from chunking import load_documents, chunk_text_naive, chunk_text_improved
from embedding import embed_text
from vectorstore import search
from generate import build_prompt
from agent import is_context_sufficient, rewrite_query
from embeddings_cache import load_cache
from eval_questions import EVAL_QUESTIONS
import ollama

DOCS_FOLDER = "docs"
MAX_ATTEMPTS = 2

# Must match the tags/paths used in eval_retrieval.py so we reuse the
# already-embedded, already-cached chunk sets instead of re-embedding.
STRATEGIES = {
    "naive": ("eval-naive-v1", "eval_cache_naive.json"),
    "improved": ("eval-improved-v1", "eval_cache_improved.json"),
}


def run_agent_on_question(question, all_chunks):
    """
    Same retrieve -> judge -> rewrite loop as main.py's ask(), but returns
    the per-attempt trace instead of just the final answer, so we can
    measure how often retry changes the outcome.
    """
    current_query = question
    retrieved = None
    attempt_results = []  # list of bool, sufficient per attempt

    for attempt in range(MAX_ATTEMPTS):
        retrieved = search(current_query, all_chunks, top_k=3)
        sufficient = is_context_sufficient(current_query, retrieved)
        attempt_results.append(sufficient)

        if sufficient:
            break

        if attempt < MAX_ATTEMPTS - 1:
            current_query = rewrite_query(current_query, retrieved)

    augmented_prompt = build_prompt(question, retrieved)
    response = ollama.chat(model='qwen2:7b', messages=[
        {'role': 'user', 'content': augmented_prompt}
    ])
    answer = response['message']['content']

    return {
        "attempts_used": len(attempt_results),
        "attempt_results": attempt_results,   # e.g. [False, True]
        "final_sufficient": attempt_results[-1],
        "answer": answer,
    }


def evaluate_strategy(name, tag, cache_path):
    all_chunks = load_cache(DOCS_FOLDER, extra_tag=tag, cache_path=cache_path)
    if all_chunks is None:
        raise RuntimeError(
            f"No cache found for '{name}' at {cache_path}. "
            f"Run eval_retrieval.py first to build it."
        )

    results = []
    for i, item in enumerate(EVAL_QUESTIONS):
        print(f"[{name}] {i + 1}/{len(EVAL_QUESTIONS)}: {item['question'][:60]}...")
        trace = run_agent_on_question(item["question"], all_chunks)
        trace["question"] = item["question"]
        trace["expected_source"] = item["expected_source"]
        results.append(trace)
        print(f"    attempts={trace['attempts_used']} final_sufficient={trace['final_sufficient']}")

    return results


def summarize(results):
    total = len(results)
    final_sufficient_count = sum(1 for r in results if r["final_sufficient"])
    rescued = sum(
        1 for r in results
        if r["attempts_used"] == 2 and r["attempt_results"][0] is False and r["attempt_results"][1] is True
    )
    stayed_insufficient = sum(
        1 for r in results
        if r["attempts_used"] == 2 and r["attempt_results"][1] is False
    )
    avg_attempts = sum(r["attempts_used"] for r in results) / total

    return {
        "total_questions": total,
        "final_sufficient_rate": final_sufficient_count / total,
        "avg_attempts": avg_attempts,
        "rescued_by_retry": rescued,         # attempt 1 failed, attempt 2 succeeded
        "stayed_insufficient_after_retry": stayed_insufficient,  # both attempts failed
    }


if __name__ == "__main__":
    all_results = {}
    all_summaries = {}

    for name, (tag, cache_path) in STRATEGIES.items():
        print(f"\n=== {name} chunking ===")
        results = evaluate_strategy(name, tag, cache_path)
        all_results[name] = results
        all_summaries[name] = summarize(results)

    with open("eval_results_agent.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(all_summaries, indent=2))