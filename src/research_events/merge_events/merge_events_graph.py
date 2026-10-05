from typing import List, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import Command, RunnableConfig
from langgraph.pregel.main import asyncio
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_with_tools
from src.research_events.chunk_graph import create_company_event_graph
from src.research_events.merge_events.prompts import (
    EXTRACT_AND_CATEGORIZE_PROMPT,
    MERGE_EVENTS_TEMPLATE,
)
from src.research_events.merge_events.utils import ensure_categories_with_events
from src.services.event_service import EventService
from src.state import CategoriesWithEvents
from src.url_crawler.utils import chunk_text_by_tokens
from src.utils import get_langfuse_handler


class RelevantEventsCategorized(BaseModel):
    """The chunk contains relevant company events that have been categorized."""

    company_profile: str = Field(
        description="Bullet points about company history, founding, headquarters, industry, mission, and background"
    )
    products_services: str = Field(
        description="Bullet points about major products, services, technologies, platforms, and product launches"
    )
    leadership: str = Field(
        description="Bullet points about founders, executives, leadership changes, appointments, and organizational changes"
    )
    business_milestones: str = Field(
        description="Bullet points about funding, acquisitions, partnerships, IPOs, expansions, major announcements, and other significant business milestones"
    )


class IrrelevantChunk(BaseModel):
    """The chunk contains NO company events relevant to the research question."""


class InputMergeEventsState(TypedDict):
    """The complete state for the enhanced event merging sub-graph."""

    existing_events: CategoriesWithEvents
    extracted_events: str
    research_question: str


class MergeEventsState(InputMergeEventsState):
    text_chunks: List[str]  # token-based chunks
    categorized_chunks: List[CategoriesWithEvents]  # results per chunk
    extracted_events_categorized: CategoriesWithEvents


class OutputMergeEventsState(TypedDict):
    existing_events: CategoriesWithEvents  # includes the existing events + the events from the new events


async def split_events(
    state: MergeEventsState,
) -> Command[Literal["filter_chunks", "__end__"]]:
    """Use token-based chunking from the URL crawler."""
    extracted_events = state.get("extracted_events", "")

    if not extracted_events.strip():
        # No content to process
        return Command(
            goto="__end__",
            update={"text_chunks": [], "categorized_chunks": []},
        )

    chunks = await chunk_text_by_tokens(extracted_events)

    return Command(
        goto="filter_chunks",
        update={"text_chunks": chunks[0:20], "categorized_chunks": []},
    )


async def filter_chunks(
    state: MergeEventsState, config: RunnableConfig
) -> Command[Literal["extract_and_categorize_chunk", "__end__"]]:
    """Filter chunks to only process those containing significant company events."""
    chunks = state.get("text_chunks", [])

    if not chunks:
        return Command(
            goto="__end__",
        )

    # Use chunk graph to filter for company events
    chunk_graph = create_company_event_graph()

    configurable = Configuration.from_runnable_config(config)
    if len(chunks) > configurable.max_chunks:
        # To avoid recursion issues, set max chunks
        chunks = chunks[: configurable.max_chunks]

    # Process each chunk through the company event detection graph
    relevant_chunks = []
    for chunk in chunks:
        chunk_result = await chunk_graph.ainvoke({"text": chunk}, config)

        # Check if any chunk contains company events
        has_events = any(
            result.contains_company_event
            for result in chunk_result["results"].values()
        )
        print(f"contains_company_event: {has_events}")

        if has_events:
            relevant_chunks.append(chunk)

    if not relevant_chunks:
        # No relevant chunks found
        return Command(goto="__end__")

    return Command(
        goto="extract_and_categorize_chunk",
        update={"text_chunks": chunks, "categorized_chunks": []},
    )


async def extract_and_categorize_chunk(
    state: MergeEventsState, config: RunnableConfig
) -> Command[Literal["extract_and_categorize_chunk", "merge_categorizations"]]:
    """Combined extraction and categorization"""
    chunks = state.get("text_chunks", [])
    categorized_chunks = state.get("categorized_chunks", [])

    if len(categorized_chunks) >= len(chunks):
        # all categorized_chunks done → move to merge
        return Command(goto="merge_categorizations")

    # take next chunk
    chunk = chunks[len(categorized_chunks)]
    research_question = state.get("research_question", "")

    prompt = EXTRACT_AND_CATEGORIZE_PROMPT.format(
        # research_question=research_question,
        text_chunk=chunk
    )

    tools = [tool(RelevantEventsCategorized), tool(IrrelevantChunk)]
    model = create_llm_with_tools(tools=tools, config=config)
    response = await model.ainvoke(prompt)

    # Parse response
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == "RelevantEventsCategorized"
    ):
        categorized_data = response.tool_calls[0]["args"]
        # Convert any list values to strings
        categorized_data = {
            k: "\n".join(v) if isinstance(v, list) else v
            for k, v in categorized_data.items()
        }
        categorized = CategoriesWithEvents(**categorized_data)
    else:
        categorized = CategoriesWithEvents(
        company_profile="",
        products_services="",
        leadership="",
        business_milestones="",
    )

    return Command(
        goto="extract_and_categorize_chunk",  # loop until all chunks processed
        update={"categorized_chunks": categorized_chunks + [categorized]},
    )


async def merge_categorizations(
    state: MergeEventsState,
) -> Command[Literal["combine_new_and_original_events"]]:
    """Merge all categorized chunks into a single CategoriesWithEvents"""
    results = state.get("categorized_chunks", [])

    merged = EventService.merge_categorized_events(results)

    return Command(
        goto="combine_new_and_original_events",
        update={"extracted_events_categorized": merged},
    )


async def combine_new_and_original_events(
    state: MergeEventsState, config: RunnableConfig
) -> Command:
    """Merge original and new events for each category using an LLM."""
    print("Combining new and original events...")

    existing_events_raw = state.get(
    "existing_events",
    CategoriesWithEvents(
        company_profile="",
        products_services="",
        leadership="",
        business_milestones="",
    ),
)
    new_events_raw = state.get(
        "extracted_events_categorized",
        CategoriesWithEvents(
            company_profile="",
            products_services="",
            leadership="",
            business_milestones="",
        ),
    )

    # Convert to proper Pydantic models if they're dicts
    existing_events = ensure_categories_with_events(existing_events_raw)
    new_events = ensure_categories_with_events(new_events_raw)

    if not new_events or not any(
        getattr(new_events, cat, "").strip()
        for cat in CategoriesWithEvents.model_fields.keys()
    ):
        print("No new events found. Keeping existing events.")
        return Command(goto="__end__", update={"existing_events": existing_events})

    merge_tasks = []
    categories = CategoriesWithEvents.model_fields.keys()

    for category in categories:
        # Now you can safely use getattr since they're guaranteed to be Pydantic models
        existing_text = getattr(existing_events, category, "").strip()
        new_text = getattr(new_events, category, "").strip()

        if not (existing_text or new_text):
            continue  # nothing to merge in this category

        existing_display = existing_text if existing_text else "No events"
        new_display = new_text if new_text else "No events"

        prompt = MERGE_EVENTS_TEMPLATE.format(
            original=existing_display, new=new_display
        )

        # Use regular structured model for merging (not tools model)
        from src.llm_service import create_llm_structured_model

        merge_tasks.append(
            (category, create_llm_structured_model(config=config).ainvoke(prompt))
        )

    final_merged_dict = {}
    if merge_tasks:
        categories, tasks = zip(*merge_tasks)
        responses = await asyncio.gather(*tasks)
        final_merged_dict = {
            cat: resp.content for cat, resp in zip(categories, responses)
        }

    # Ensure all categories are included
    for category in CategoriesWithEvents.model_fields.keys():
        if category not in final_merged_dict:
            final_merged_dict[category] = getattr(existing_events, category, "")

    final_merged_output = CategoriesWithEvents(**final_merged_dict)
    return Command(goto="__end__", update={"existing_events": final_merged_output})


merge_events_graph_builder = StateGraph(
    MergeEventsState, input_schema=InputMergeEventsState, config_schema=Configuration
)

merge_events_graph_builder.add_node("split_events", split_events)
merge_events_graph_builder.add_node("filter_chunks", filter_chunks)
merge_events_graph_builder.add_node(
    "extract_and_categorize_chunk", extract_and_categorize_chunk
)
merge_events_graph_builder.add_node("merge_categorizations", merge_categorizations)
merge_events_graph_builder.add_node(
    "combine_new_and_original_events", combine_new_and_original_events
)

merge_events_graph_builder.add_edge(START, "split_events")


merge_events_app = merge_events_graph_builder.compile().with_config(
    {
        "callbacks": [get_langfuse_handler()],
        "recursionLimit": 200,
    },
)
