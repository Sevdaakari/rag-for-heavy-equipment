import json
import os
from chunking import load_documents, chunk_text_naive, chunk_text_improved
from embedding import embed_text
from vectorstore import search
from embeddings_cache import save_cache, load_cache
from eval_questions import EVAL_QUESTIONS

DOCS_FOLDER = "docs"
TOP_K = 5

STRATEGIES = {
    "naive": (chunk_text_naive, "eval-naive-v1", "eval_cache_naive.json"),
    "improved": (chunk_text_improved, "eval-improved-v1", "eval_cache_improved.json"),
}


def build_or_load_chunks(chunk_fn, tag, cache_path):
    cached = load_cache(DOCS_FOLDER, extra_tag=tag, cache_path=cache_path)
    if cached is not None:
        print(f"  loaded {len(cached)} cached chunks")
        return cached

    docs = load_documents(DOCS_FOLDER)
    all_chunks = []
    for filename, text in docs.items():
        for chunk_text_piece in chunk_fn(text):
            all_chunks.append({"source": filename, "text": chunk_text_piece})

    print(f"  embedding {len(all_chunks)} chunks (one-time, slow)...")
    for i, chunk in enumerate(all_chunks):
        chunk["embedding"] = embed_text(chunk["text"])
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(all_chunks)}")

    save_cache(all_chunks, DOCS_FOLDER, extra_tag=tag, cache_path=cache_path)
    return all_chunks


def evaluate(all_chunks, strategy_name):
    hits = 0
    reciprocal_ranks = []

    for item in EVAL_QUESTIONS:
        results = search(item["question"], all_chunks, top_k=TOP_K)
        sources = [r["source"] for r in results]

        if item["expected_source"] in sources:
            hits += 1
            rank = sources.index(item["expected_source"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    hit_rate = hits / len(EVAL_QUESTIONS)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    print(f"\n[{strategy_name}] hit rate@{TOP_K}: {hit_rate:.2%}  |  MRR: {mrr:.3f}")
    return hit_rate, mrr


if __name__ == "__main__":
    results_summary = {}
    for name, (chunk_fn, tag, cache_path) in STRATEGIES.items():
        print(f"\n=== {name} chunking ===")
        chunks = build_or_load_chunks(chunk_fn, tag, cache_path)
        hit_rate, mrr = evaluate(chunks, name)
        results_summary[name] = {"hit_rate": hit_rate, "mrr": mrr, "num_chunks": len(chunks)}

    print("\n=== SUMMARY ===")
    print(json.dumps(results_summary, indent=2))