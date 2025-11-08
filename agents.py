# agents.py - COMPLETE AND OPTIMIZED

import asyncio
from typing import List, Dict, Any

from retriever import retrieve
from config import CHAT_MODEL, get_genai_client

# System prompts define the persona for each agent.
# CRITICAL: Added explicit instructions to enforce evidence adherence.
AGENT_SYSTEM_PROMPTS = {
    "CEO": (
        "You are the CEO. Provide strategic, high-level guidance. Your response MUST be formatted using Markdown. "
        "Base your answer ONLY on the provided evidence digest. If the evidence is insufficient, state that the specific "
        "data is unavailable and answer based only on your general expertise. DO NOT fabricate information."
    ),
    "CTO": (
        "You are the CTO. Focus on technical feasibility, innovation, and architecture. Your response MUST be formatted "
        "using Markdown. Base your answer ONLY on the provided evidence digest. If the evidence is insufficient, state "
        "that the specific data is unavailable and answer based only on your general expertise. DO NOT fabricate information."
    ),
    "CFO": (
        "You are the CFO. Be conservative; focus on financial viability, profitability, and risk. Your response MUST be "
        "formatted using Markdown. Base your answer ONLY on the provided evidence digest. If the evidence is insufficient, "
        "state that the specific data is unavailable and answer based only on your general expertise. DO NOT fabricate information."
    ),
    "CMO": (
        "You are the CMO. Discuss market strategy and customer acquisition, using evidence where available. Your response "
        "MUST be formatted using Markdown. Base your answer ONLY on the provided evidence digest. If the evidence is insufficient, "
        "state that the specific data is unavailable and answer based only on your general expertise. DO NOT fabricate information."
    ),
}

async def _create_standalone_query(query: str, history: List[Dict]) -> str:
    """
    Uses the LLM to rephrase the user's query into a standalone question
    based on the conversation history. This improves retrieval accuracy.
    """
    if not history:
        return query

    genai = get_genai_client()
    model = genai.GenerativeModel(CHAT_MODEL)

    # Prepare a concise history for the prompt
    history_prompt = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
        for msg in history[-4:]  # Use last 4 messages for brevity
    )

    prompt = f"""
Given the following conversation history and a follow-up question, rephrase the
follow-up question to be a standalone question. If the follow-up question is
already standalone, just return it as is.

Conversation History:
{history_prompt}

Follow-up Question: {query}

Standalone Question:"""

    try:
        # Asynchronously generate the standalone query
        response = await model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error creating standalone query: {e}. Falling back to original query.")
        return query

async def run_agent_conversational(
    role: str, query: str, history: List[Dict]
) -> Dict[str, Any]:
    """
    Runs a generative AI agent that is aware of the conversation history,
    with RAG optimizations implemented.
    """
    print(f"Original query for {role}: '{query}'")
    # 1. Create a standalone query for better retrieval
    standalone_query = await _create_standalone_query(query, history)
    print(f"Standalone query for retrieval: '{standalone_query}'")

    # 2. Retrieve documents using the standalone query
    # This now returns items with a score/distance from Pinecone.
    docs, digest = await retrieve(standalone_query, agent_name=role, k=5)
    
    # 3. Apply the Score Filter (NEW OPTIMIZATION)
    # Filter for documents above a quality threshold (distance score below 0.6)
    # Note: 0.6 is an illustrative threshold; a better one might be found experimentally.
    HIGH_QUALITY_DOCS = [d for d in docs if d.get("score", 1.0) < 0.6]
    QUALITY_DIGEST = None

    if HIGH_QUALITY_DOCS:
        # Rebuild the digest using only the high-quality documents
        digest_lines = [
            f"- Snippet from {item['metadata'].get('source_file', 'N/A')}: "
            f"{item['preview'][:240].replace('/n', ' ')} (score={item.get('score', 0):.4f})"
            for item in HIGH_QUALITY_DOCS
        ]
        QUALITY_DIGEST = "\n".join(digest_lines)


    # 4. Get the system prompt for the selected agent role
    system_prompt = AGENT_SYSTEM_PROMPTS.get(
        role.upper(), AGENT_SYSTEM_PROMPTS["CEO"] # Fallback to a strict, standard CEO role
    )

    # 5. Build the instructional prompt for the generation step
    if QUALITY_DIGEST:
        instructional_prompt = f"{system_prompt}\n\nUse the following evidence (which is highly relevant) to inform your answer:\n--- Evidence ---\n{QUALITY_DIGEST}\n------------------"
    else:
        # Give the agent a fallback to general expertise if the retrieved evidence is poor quality
        instructional_prompt = f"{system_prompt}\n\nNo specific, high-quality evidence was found. Please answer based on your general expertise, remaining honest about the source of your knowledge."

    # 6. Format the conversation history for the Gemini model
    chat_history_for_model = [
        # Start with the instructional prompt
        {'role': 'user', 'parts': [instructional_prompt]},
        {'role': 'model', 'parts': ["Understood. I will use the provided context, if available, and my assigned role to answer."]},
    ]
    
    # Add the actual conversation history - NEW OPTIMIZATION: TRIM HISTORY TO SAVE TOKENS
    # Use only the last 6 messages (3 user + 3 AI turns) for conversational flow
    history_slice = history[-6:] 
    for message in history_slice:
        model_role = "user" if message["role"] == "user" else "model"
        chat_history_for_model.append({'role': model_role, 'parts': [message["content"]]})

    # Finally, add the latest user query
    chat_history_for_model.append({'role': 'user', 'parts': [query]})

    # 7. Generate the response asynchronously
    genai = get_genai_client()
    model = genai.GenerativeModel(CHAT_MODEL)
    response = await model.generate_content_async(chat_history_for_model)

    return {
        "role": role,
        "answer": response.text,
        "evidence_used": HIGH_QUALITY_DOCS or "No high-quality evidence found, fallback to general knowledge.",
    }