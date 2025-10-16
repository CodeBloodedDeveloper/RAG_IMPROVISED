# CODE.zip/agents.py

from retriever import retrieve
from config import CHAT_MODEL, get_genai_client
from typing import List, Dict



# Add "Format your response using Markdown" to each prompt, which aligns with the frontend renderer.
AGENT_SYSTEM_PROMPTS = {
    "CEO": "You are the CEO agent. Provide strategic guidance grounded in evidence when possible. Format your response using Markdown.",
    "CTO": "You are the CTO agent. Focus on tech feasibility, innovation, and architecture. Format your response using Markdown.",
    "CFO": "You are the CFO agent. Be conservative and focus on financial viability, profitability, and risk. Format your response using Markdown.",
    "CMO": "You are the CMO agent, a seasoned and supportive marketing leader. Use evidence where available to discuss market strategy and customer acquisition. Format your response using Markdown.",
}

# This is a new helper function to create a better query for retrieval
def _create_standalone_query(query: str, history: List[Dict]):
    """
    Uses the LLM to rephrase the user's query into a standalone question
    based on the conversation history.
    """
    if not history:
        return query

    genai = get_genai_client()
    model = genai.GenerativeModel(CHAT_MODEL)
    
    # Prepare a concise history
    history_prompt = ""
    for msg in history[-4:]: # Use last 4 messages for brevity
        role = "User" if msg['role'] == 'user' else "AI"
        history_prompt += f"{role}: {msg['content']}\n"

    prompt = f"""
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.
If the follow-up question is already standalone, just return it as is.

Conversation History:
{history_prompt}
Follow-up Question: {query}

Standalone Question:"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Error creating standalone query: {e}. Falling back to original query.")
        return query


# The main run_agent function is now `run_agent_conversational`
def run_agent_conversational(role: str, query: str, history: List[Dict]):
    """
    Runs a generative AI agent that is aware of the conversation history.
    """
    genai = get_genai_client()
    
    # 1. Create a standalone query for better retrieval
    print(f"Original query: '{query}'")
    standalone_query = _create_standalone_query(query, history)
    print(f"Standalone query for retrieval: '{standalone_query}'")
    
    # 2. Retrieve documents using the standalone query
    docs, digest = retrieve(standalone_query, agent_name=role, k=5, return_digest=True)

    system_prompt = AGENT_SYSTEM_PROMPTS.get(role.upper(), "You are a helpful business advisor. Format your response using Markdown.")

    # 3. Build the prompt for the final generation step
    # Start with the system prompt
    final_prompt_parts = [system_prompt]
    
    # Add evidence if found
    if docs:
        final_prompt_parts.append(f"\nUse the following evidence to help inform your answer:\n--- Evidence ---\n{digest}\n------------------")
    else:
        final_prompt_parts.append("\nNo specific evidence was found. Please answer based on your general expertise.")
        
    # Combine system prompt and evidence
    instructional_prompt = "\n".join(final_prompt_parts)

    # 4. Format the conversation history for the Gemini model
    chat_history_for_model = []
    # Add the instructional part first
    chat_history_for_model.append({'role': 'user', 'parts': [instructional_prompt]})
    chat_history_for_model.append({'role': 'model', 'parts': ["Understood. I will use the provided context and my assigned role to answer the user's questions."]})

    # Add the actual conversation history
    for message in history:
        model_role = "user" if message["role"] == "user" else "model"
        chat_history_for_model.append({'role': model_role, 'parts': [message["content"]]})
        
    # Finally, add the latest user query
    chat_history_for_model.append({'role': 'user', 'parts': [query]})

    # 5. Generate the response
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(chat_history_for_model)
    
    return {
        "role": role,
        "answer": response.text,
        "evidence_used": docs if docs else "❌ No evidence, fallback to freeform Gemini"
    }