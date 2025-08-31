from typing import List, Dict


def build_system_prompt(user_question: str, retrieved_docs: List[Dict]) -> str:
    """
    Build system prompt for RAG answer generation.

    Injects retrieved context (question + answer) into the prompt.

    Args:
        user_question: The question to answer
        retrieved_docs: List of dicts containing retrieved records (e.g., question, answer)

    Returns:
        Formatted prompt string for LLM
    """
    context = ""
    for idx, doc in enumerate(retrieved_docs, start=1):
        a = "Question: " + doc.get("question", "").strip() + " -> Answer: " + doc.get("answer", "").strip()
        context += f"{a}\n"

    prompt = (
        "You are a helpful medical assistant. "
        "Use the following retrieved context to answer the user's query. "
        "If the retrieved context is NOT relevant to the user's query, do NOT use your general knowledge to answer. \n\n"
        "CONTEXT: \n\n"
        f"{context}"
        "\n\n"
        f"USER QUERY: {user_question}\n"
        "Answer:"
    )
    
    # prompt = f"""
    # SYSTEM PROMPT:

    # You are a highly accurate and context-aware assistant. Your task is to synthesize and generate a new answer to the user questions strictly based on the provided context. 

    # INSTRUCTIONS:

    # - Read the user's question carefully.
    # - Examine all the retrieved FAQ context below to synthesize new answer.
    # - Generate a concise, accurate answer only using the information present in the context.
    # - Do not invent facts or use general world knowledge outside of the given context.
    # - Provide your answer in clear, complete sentences.
    # - Do NOT reference the context explicitly in your answer.
    # - Synthesize information from multiple answers if present.
    # - You are required to just return the answer.

    # Use natural language, not direct quotations.

    # USER QUESTION: {user_question}

    # RETRIEVED FAQ CONTEXT:
    # {context}

    # YOUR ANSWER:
    # """

    return prompt
