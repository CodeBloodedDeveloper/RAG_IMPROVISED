# agents.py

import asyncio
from typing import List, Dict, Any

from retriever import retrieve
from config import CHAT_MODEL, get_genai_client

# System prompts define the persona for each agent.
AGENT_SYSTEM_PROMPTS = {
    "CEO": "You are the CEO. Provide strategic, high-level guidance. Format your response using Markdown.",
    "CTO": "You are the CTO. Focus on technical feasibility, innovation, and architecture. Format your response using Markdown.",
    "CFO": "You are the CFO. Be conservative; focus on financial viability, profitability, and risk. Format your response using Markdown.",
    "CMO": "You are the CMO. Discuss market strategy and customer acquisition, using evidence where available. Format your response using Markdown.",
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
    Runs a generative AI agent that is aware of the conversation history.
    This function is fully asynchronous.
    """
    print(f"Original query for {role}: '{query}'")
    # 1. Create a standalone query for better retrieval
    standalone_query = await _create_standalone_query(query, history)
    print(f"Standalone query for retrieval: '{standalone_query}'")

    # 2. Retrieve documents using the standalone query
    docs, digest = await retrieve(standalone_query, agent_name=role, k=5)

    # 3. Get the system prompt for the selected agent role
    system_prompt = AGENT_SYSTEM_PROMPTS.get(
        role.upper(), "You are a helpful business advisor. Format your response using Markdown."
    )

    # 4. Build the instructional prompt for the generation step
    if docs:
        instructional_prompt = f"{system_prompt}\n\nUse the following evidence to help inform your answer:\n--- Evidence ---\n{digest}\n------------------"
    else:
        instructional_prompt = f"{system_prompt}\n\nNo specific evidence was found. Please answer based on your general expertise."

    # 5. Format the conversation history for the Gemini model
    chat_history_for_model = [
        # Start with the instructional prompt
        {'role': 'user', 'parts': [instructional_prompt]},
        {'role': 'model', 'parts': ["Understood. I will use the provided context and my assigned role to answer."]},
    ]
    # Add the actual conversation history
    for message in history:
        model_role = "user" if message["role"] == "user" else "model"
        chat_history_for_model.append({'role': model_role, 'parts': [message["content"]]})

    # Finally, add the latest user query
    chat_history_for_model.append({'role': 'user', 'parts': [query]})

    # 6. Generate the response asynchronously
    genai = get_genai_client()
    model = genai.GenerativeModel(CHAT_MODEL)
    response = await model.generate_content_async(chat_history_for_model)

    return {
        "role": role,
        "answer": response.text,
        "evidence_used": docs or "No evidence found, fallback to general knowledge.",
    }

