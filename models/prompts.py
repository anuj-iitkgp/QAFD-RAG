"""
Prompt templates for answer generation, extraction, and evaluation.
Enforces strict grounding on retrieved multi-hop context.
"""

# Prompt for answer generation conditioned on retrieved multi-hop context
QA_SYSTEM_PROMPT = """You are an accurate, strict Question-Answering assistant.
Your task is to answer the user's question using ONLY the provided multi-hop context.
Follow these strict rules:
1. Ground your answer strictly in the facts stated in the retrieved context.
2. Do NOT extrapolate or introduce external knowledge.
3. Be concise and direct: provide the exact entity, name, date, or fact requested whenever possible.
4. If the question cannot be answered from the retrieved context, respond with "Insufficient context to answer"."""

QA_USER_PROMPT_TEMPLATE = """Retrieved Context:
{context}

Question: {question}

Answer:"""


# OpenIE extraction prompt for LLM-based triple extraction
OPENIE_SYSTEM_PROMPT = """You are an information extraction assistant.
Given a text passage, extract factual triples in the JSON format:
[
  {{"subject": "Entity A", "relation": "relationship name", "object": "Entity B"}}
]
Keep entities normalized and relations concise (e.g., 'born in', 'capital of', 'directed', 'spouse')."""

OPENIE_USER_PROMPT_TEMPLATE = """Passage Title: {title}
Passage Content:
{text}

JSON Triples:"""
