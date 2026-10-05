from typing import Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_chunk_model


class CompanyEventCheck(BaseModel):
    contains_company_event: bool = Field(
        description="Whether the text chunk contains significant company events"
    )


class ChunkResult(BaseModel):
    content: str
    contains_company_event: bool = Field(
        description="Whether the text chunk contains company events"
    )


class ChunkState(TypedDict):
    text: str
    chunks: List[str]
    results: Dict[str, ChunkResult]


def split_text(state: ChunkState) -> ChunkState:
    """Split text into smaller chunks."""
    text = state["text"]
    chunk_size = 2000
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    return {"chunks": chunks}


def check_chunk_for_events(state: ChunkState, config) -> ChunkState:
    """Check each chunk for company events using structured output."""
    model = create_llm_chunk_model(config, CompanyEventCheck)
    results = {}

    for i, chunk in enumerate(state["chunks"]):
        prompt = f"""
        Analyze this text chunk and determine if it contains SPECIFIC company events.

        ONLY mark as true if the chunk contains concrete information such as:
        - Company founding or establishment
        - Major product or service launch
        - Acquisition or merger
        - Funding round or major financial milestone
        - IPO or stock-market event
        - Founder, CEO, or executive appointment/change
        - Major partnership or strategic agreement
        - Expansion into a new country or major market
        - Major business announcement
        - Important technology or business milestone

        DO NOT mark as true for:
        - General descriptions of the company
        - Generic descriptions of products or services
        - General industry information
        - Marketing language without a concrete event
        - Opinions or vague statements
        - Information unrelated to the company

        The event must be specific and concrete, not general background information.

        Text chunk: "{chunk}"
        """

        result = model.invoke(prompt)
        results[f"chunk_{i}"] = ChunkResult(
            content=chunk, contains_company_event=result.contains_company_event
        )

    return {"results": results}


def create_company_event_graph() -> CompiledStateGraph:
    """Create and return the company event detection graph."""
    graph = StateGraph(ChunkState, config_schema=Configuration)

    graph.add_node("split_text", split_text)
    graph.add_node("check_events", check_chunk_for_events)

    graph.add_edge(START, "split_text")
    graph.add_edge("split_text", "check_events")
    graph.add_edge("check_events", END)

    return graph.compile()


graph = create_company_event_graph()
