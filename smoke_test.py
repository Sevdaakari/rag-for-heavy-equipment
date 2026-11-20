from main import load_knowledge_base, ask

def test_app_answers_a_known_question():
    all_chunks = load_knowledge_base("docs")
    conversation_history = []

    question = "How many serious injuries involving haulage trucks at dump sites were identified?"
    answer = ask(question, all_chunks, conversation_history)

    assert len(answer.strip()) > 0
    assert "i don't know" not in answer.lower()