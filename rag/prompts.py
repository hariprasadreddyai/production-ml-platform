"""Prompt templates for different query types."""

QA_PROMPT = """Answer the question using only the context below. If the context
does not contain the answer, say so clearly instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

SUMMARIZE_PROMPT = """Summarize the following documents in three to five bullet points,
keeping the key facts and numbers.

Documents:
{context}

Summary:"""

COMPARE_PROMPT = """Compare the items described in the context below. List the
similarities first, then the differences, then a short recommendation.

Context:
{context}

Comparison:"""

VALIDATE_PROMPT = """Check the draft answer against the context. Reply with PASS if
every claim in the answer is supported by the context, or FAIL followed by
the unsupported claims.

Context:
{context}

Draft answer:
{answer}

Verdict:"""

PROMPTS = {
    "qa": QA_PROMPT,
    "summarize": SUMMARIZE_PROMPT,
    "compare": COMPARE_PROMPT,
    "validate": VALIDATE_PROMPT,
}


def get_prompt(query_type: str) -> str:
    """Return the template for a query type, defaulting to QA."""
    return PROMPTS.get(query_type, QA_PROMPT)


def render_prompt(query_type: str, **kwargs) -> str:
    return get_prompt(query_type).format(**kwargs)
