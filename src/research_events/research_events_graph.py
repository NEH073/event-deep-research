from typing import Literal, TypedDict


from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_structured_model
from src.research_events.merge_events.merge_events_graph import merge_events_app
from src.services.url_service import URLService
from src.state import CategoriesWithEvents
from src.url_crawler.url_krawler_graph import url_crawler_app
from src.utils import get_langfuse_handler


class InputResearchEventsState(TypedDict):
    research_question: str
    existing_events: CategoriesWithEvents
    used_domains: list[str]


class ResearchEventsState(InputResearchEventsState):
    urls: list[str]
    # Add this temporary field
    extracted_events: str


class OutputResearchEventsState(TypedDict):
    existing_events: CategoriesWithEvents
    used_domains: list[str]


class BestUrls(BaseModel):
    selected_urls: list[str] = Field(description="A list of the two best URLs.")


def url_finder(
    state: ResearchEventsState,
    config: RunnableConfig,
) -> Command[Literal["should_process_url_router"]]:
    """Find the urls for the research_question"""
    research_question = state.get("research_question", "")
    used_domains = state.get("used_domains", [])

    if not research_question:
        raise ValueError("research_question is required")

    tool = TavilySearch(
        max_results=6,
        topic="general",
        include_raw_content=False,
        include_answer=False,
        exclude_domains=used_domains,
    )

    result = tool.invoke({"query": research_question})
    

    urls = [item["url"] for item in result["results"]]

    prompt = """
        From the results below, select the two URLs that will provide the most
        reliable and useful information about the company in relation to the
        research question.

        Prioritize sources containing:
        - Company history and founding
        - Products and services
        - Leadership and executives
        - Acquisitions and partnerships
        - Funding or financial milestones
        - Major product launches
        - Business expansion
        - Important company announcements
        - Significant business milestones

        Prefer authoritative sources such as:
        - The company's official website
        - Official company announcements
        - Government or regulatory sources
        - Reputable business and technology publications

        <Results>
        {results}
        </Results>

        <Research Question>
        {research_question}
        </Research Question>
    """

    prompt = prompt.format(results=urls, research_question=research_question)

    structured_llm = create_llm_structured_model(config=config, class_name=BestUrls)

    structured_result = structured_llm.invoke(prompt)

    return Command(
        goto="should_process_url_router",
        update={"urls": structured_result.selected_urls},
    )


def updateUrlList(
    state: ResearchEventsState,
) -> tuple[list[str], list[str]]:
    urls = state.get("urls", [])
    used_domains = state.get("used_domains", [])

    return URLService.update_url_list(urls, used_domains)


def should_process_url_router(
    state: ResearchEventsState,
) -> Command[Literal["crawl_url", "__end__"]]:
    urls = state.get("urls", [])
    used_domains = state.get("used_domains", [])

    if urls and len(urls) > 0:
        domain = URLService.extract_domain(urls[0])
        if domain in used_domains:
            # remove first url
            remaining_urls = urls[1:]
            return Command(
                goto="should_process_url_router",
                update={"urls": remaining_urls, "used_domains": used_domains},
            )

        print(f"URLs remaining: {len(state['urls'])}. Routing to crawl.")
        return Command(goto="crawl_url")
    else:
        print("No URLs remaining. Routing to __end__.")
        # Otherwise, end the graph execution
        return Command(
            goto=END,
        )


async def crawl_url(
    state: ResearchEventsState,
) -> Command[Literal["merge_events_and_update"]]:
    """Crawls the next URL and updates the temporary state with new events."""
    urls = state["urls"]
    url_to_process = urls[0]  # Always process the first one
    research_question = state.get("research_question", "")

    if not research_question:
        raise ValueError("research_question is required for url crawling")

    # Invoke the crawler subgraph
    result = await url_crawler_app.ainvoke(
        {"url": url_to_process, "research_question": research_question}
    )
    extracted_events = result["extracted_events"]
    # Go to the merge node, updating the state with the extracted events
    return Command(
        goto="merge_events_and_update",
        update={"extracted_events": extracted_events},
    )


async def merge_events_and_update(
    state: ResearchEventsState,
) -> Command[Literal["should_process_url_router"]]:
    """Merges new events, removes the processed URL, and loops back to the router."""
    existing_events = state.get("existing_events", CategoriesWithEvents())
    extracted_events = state.get("extracted_events", "")
    research_question = state.get("research_question", "")

    # Invoke the merge subgraph
    result = await merge_events_app.ainvoke(
        {
            "existing_events": existing_events,
            "extracted_events": extracted_events,
            "research_question": research_question,
        }
    )

    remaining_urls, used_domains = updateUrlList(state)

    # Remaining URLs after removal
    return Command(
        goto="should_process_url_router",
        update={
            "existing_events": result["existing_events"],
            "urls": remaining_urls,
            "used_domains": used_domains,
            # "extracted_events": "",  # Clear the temporary state
        },
    )


research_events_builder = StateGraph(
    ResearchEventsState,
    input_schema=InputResearchEventsState,
    output_schema=OutputResearchEventsState,
    config_schema=Configuration,
)

# Add all the nodes to the graph
research_events_builder.add_node("url_finder", url_finder)
research_events_builder.add_node("should_process_url_router", should_process_url_router)
research_events_builder.add_node("crawl_url", crawl_url)
research_events_builder.add_node("merge_events_and_update", merge_events_and_update)

# Set the entry point
research_events_builder.add_edge(START, "url_finder")


research_events_app = research_events_builder.compile().with_config(
    {"callbacks": [get_langfuse_handler()]}
)
