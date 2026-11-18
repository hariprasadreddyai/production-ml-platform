"""Multi-agent orchestration with LangGraph: Planner, Researcher, Validator.

The planner breaks a question into search steps, the researcher retrieves
and reranks evidence for each step, and the validator checks that the final
answer is grounded in the retrieved context before it is returned.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    question: str
    plan: list[str]
    evidence: list[dict]
    draft_answer: str
    verdict: str
    answer: str


def build_agent_graph(store, reranker, answer_fn=None):
    """Build the planner, researcher, validator graph.

    answer_fn(context, question) produces the draft answer from evidence.
    The default implementation concatenates the top evidence passages.
    """

    def plan_node(state: AgentState) -> dict:
        question = state["question"]
        steps = [question]
        lowered = question.lower()
        if " and " in lowered or "compare" in lowered:
            parts = [p.strip() for p in lowered.replace("compare", "").split(" and ")]
            steps = [p for p in parts if p] or [question]
        return {"plan": steps}

    def research_node(state: AgentState) -> dict:
        evidence: list[dict] = []
        for step in state["plan"]:
            candidates = store.search(step, k=5)
            reranked = reranker.rerank(step, candidates, top_k=3)
            evidence.extend(reranked)
        seen = set()
        deduped = []
        for doc in evidence:
            if doc["text"] not in seen:
                seen.add(doc["text"])
                deduped.append(doc)
        return {"evidence": deduped}

    def draft_node(state: AgentState) -> dict:
        context = "\n\n".join(doc["text"] for doc in state["evidence"][:3])
        if answer_fn is not None:
            draft = answer_fn(context, state["question"])
        elif context:
            draft = (
                f"Based on the retrieved documents: {context[:800]}"
            )
        else:
            draft = "I do not have enough information in the documents to answer."
        return {"draft_answer": draft}

    def validate_node(state: AgentState) -> dict:
        verdict = "PASS" if state["evidence"] else "FAIL"
        if verdict == "PASS" and state["draft_answer"].startswith(
            "I do not have enough"
        ):
            verdict = "FAIL"
        answer = state["draft_answer"] if verdict == "PASS" else (
            "I could not verify an answer from the available documents."
        )
        return {"verdict": verdict, "answer": answer}

    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_node)
    graph.add_node("researcher", research_node)
    graph.add_node("drafter", draft_node)
    graph.add_node("validator", validate_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "drafter")
    graph.add_edge("drafter", "validator")
    graph.add_edge("validator", END)
    return graph.compile()
