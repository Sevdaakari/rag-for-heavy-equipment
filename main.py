from chunking import load_documents, chunk_text_improved
from embedding import embed_text
from vectorstore import search
from generate import build_prompt
from agent import is_context_sufficient, rewrite_query
from embeddings_cache import load_cache, save_cache
from history import load_history, save_turn
from docs_validator import process_upload_folder
import ollama

CHUNKING_VERSION = "improved-v1-tokens200-overlap40"


def load_knowledge_base(folder_path):
    """
    Load the knowledge base, using a cached, pre-embedded version from disk
    if one exists, the docs/ folder hasn't changed, AND the chunking
    strategy hasn't changed since the cache was built.
    """
    cached_chunks = load_cache(folder_path, extra_tag=CHUNKING_VERSION)
    if cached_chunks is not None:
        print("Loaded knowledge base from cache (docs/ and chunking strategy unchanged).")
        return cached_chunks

    print("No valid cache found — embedding all documents from scratch...")
    docs = load_documents(folder_path)

    all_chunks = []
    for filename, text in docs.items():
        for chunk in chunk_text_improved(text):
            all_chunks.append({"source": filename, "text": chunk})

    for chunk in all_chunks:
        chunk['embedding'] = embed_text(chunk['text'])

    save_cache(all_chunks, folder_path, extra_tag=CHUNKING_VERSION)
    print(f"Cached {len(all_chunks)} chunks to disk for next time.")

    return all_chunks


def ask(query, all_chunks, conversation_history, max_attempts=2):
    """
    Run the agentic RAG loop: retrieve, judge, rewrite-and-retry if needed.
    """
    current_query = query
    retrieved = None

    for attempt in range(max_attempts):
        retrieved = search(current_query, all_chunks, top_k=3)
        sufficient = is_context_sufficient(current_query, retrieved)

        print(f"  [attempt {attempt + 1}] query: '{current_query}' -> sufficient: {sufficient}")

        if sufficient:
            break

        if attempt < max_attempts - 1:
            current_query = rewrite_query(current_query, retrieved)

    # Generate the final answer using whatever was retrieved on the last
    # attempt, using the ORIGINAL query (not the rewritten one) so the
    # answer stays on-topic
    augmented_prompt = build_prompt(query, retrieved)
    messages = conversation_history + [{'role': 'user', 'content': augmented_prompt}]

    response = ollama.chat(model='qwen2:7b', messages=messages)
    answer = response['message']['content']

    conversation_history.append({'role': 'user', 'content': query})
    conversation_history.append({'role': 'assistant', 'content': answer})

    return answer


if __name__ == "__main__":
    process_upload_folder()  
    print("Loading knowledge base...")
    all_chunks = load_knowledge_base("docs")
    print(f"Ready — {len(all_chunks)} chunks loaded.\n")

    conversation_history = load_history()
    if conversation_history:
        print(f"Resumed previous conversation ({len(conversation_history)} messages).\n")

    while True:
        query = input("Ask a question (or 'quit'): ")
        if query.lower() == "quit":
            break

        answer = ask(query, all_chunks, conversation_history)
        print(f"\n{answer}\n")

        save_turn(query, answer)