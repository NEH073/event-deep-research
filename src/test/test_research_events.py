# tests/research_events/test_research_events.py

"""Tests for the research_events_graph."""

from unittest.mock import AsyncMock, patch

import pytest
from src.state import CategoriesWithEvents

# Imports are relative to the src directory (configured in pyproject.toml pythonpath)
from research_events.research_events_graph import research_events_app


@pytest.fixture
def sample_input_state() -> dict:
    """Provide a sample input state for the research_events_app graph."""
    return {
        "research_question": "Research the company NVIDIA",
        "existing_events": CategoriesWithEvents(
            company_profile="Founded in 1993.",
            products_services="Developed graphics processors.",
            leadership="Jensen Huang is the CEO.",
            business_milestones="Became a major technology company.",
        ),
        "used_domains": [],
    }


class MockResponse:
    """Mock response class for LLM responses."""

    def __init__(self, content):
        """Initialize mock response with content."""
        self.content = content


class MockToolCall:
    """Mock tool call for structured LLM responses."""

    def __init__(self, name, args):
        """Initialize mock tool call with name and args."""
        self.name = name
        self.args = args


class MockToolResponse:
    """Mock tool response for structured LLM responses."""

    def __init__(self, tool_calls=None):
        """Initialize mock tool response with tool calls."""
        self.tool_calls = tool_calls or []


@pytest.fixture
def mock_url_crawler():
    """Provide a reusable mock for url_crawler_app with configurable responses."""

    def create_mock_crawler(extracted_events):
        """Create a configured mock crawler."""
        mock_crawler = AsyncMock()
        mock_crawler.ainvoke.return_value = {
            "extracted_events": extracted_events,
            "raw_scraped_content": "Mock scraped content",
        }
        return mock_crawler

    return create_mock_crawler


@pytest.fixture
def mock_merge_events():
    """Provide a reusable mock for merge_events_app with configurable responses."""

    def create_mock_merger(existing_events):
        """Create a configured mock merger."""
        mock_merger = AsyncMock()
        mock_merger.ainvoke.return_value = {"existing_events": existing_events}
        return mock_merger

    return create_mock_merger


# @pytest.mark.skip(reason="Skip mocked LLM test for now")
@pytest.mark.asyncio
async def test_research_events_with_mocked_llm(
    sample_input_state: dict, mock_url_crawler, mock_merge_events
):
    """Unit test for the research events graph with mocked dependencies."""
    # --- Arrange: Mock Data Setup ---
    mock_extracted_events = "NVIDIA was founded in 1993. The company developed graphics processors and expanded its business significantly."
    mock_existing_events = CategoriesWithEvents(
        company_profile="NVIDIA was founded in 1993.",
        products_services="The company developed graphics processors.",
        leadership="Jensen Huang is the CEO.",
        business_milestones="NVIDIA expanded its business significantly.",
    )

    # --- Act: Execute the graph with patched dependencies ---
    with (
        patch(
            "research_events.research_events_graph.url_crawler_app"
        ) as mock_crawler_patch,
        patch(
            "research_events.research_events_graph.merge_events_app"
        ) as mock_merger_patch,
        patch("research_events.research_events_graph.TavilySearch") as mock_tavily,
        patch("research_events.research_events_graph.create_llm_structured_model") as mock_llm,
    ):
        # Configure the mocks
        mock_crawler_patch.ainvoke = mock_url_crawler(mock_extracted_events).ainvoke
        mock_merger_patch.ainvoke = mock_merge_events(mock_existing_events).ainvoke

        # Mock TavilySearch to return empty results (no URLs found)
        from unittest.mock import Mock

        mock_tavily_instance = Mock()
        mock_tavily_instance.invoke.return_value = {"results": []}
        mock_tavily.return_value = mock_tavily_instance

        # Mock the structured LLM to return a test URL
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(selected_urls=["https://example.com/test"])
        mock_llm.return_value = mock_llm_instance

        result = await research_events_app.ainvoke(sample_input_state)

    # --- Assert: Verify the output ---
    # The result structure should have existing_events and used_domains
    assert "existing_events" in result
    assert "used_domains" in result

    existing_events = result["existing_events"]
    used_domains = result["used_domains"]

    assert isinstance(existing_events, CategoriesWithEvents)
    assert (
        existing_events.company_profile
        == "NVIDIA was founded in 1993."
    )
    assert existing_events.products_services == "The company developed graphics processors."
    assert existing_events.leadership == "Jensen Huang is the CEO."
    assert existing_events.business_milestones == "NVIDIA expanded its business significantly."

    # Verify that domains were tracked
    assert isinstance(used_domains, list)


# @pytest.mark.skip(reason="Skip real LLM test for now")
@pytest.mark.llm
@pytest.mark.asyncio
async def test_research_events_with_real_llm(sample_input_state: dict):
    """Integration test for the research events graph with real LLM calls."""
    # --- Act ---
    result = await research_events_app.ainvoke(sample_input_state)

    # --- Assert ---
    # The result structure should have existing_events and used_domains
    assert "existing_events" in result
    assert "used_domains" in result

    existing_events = result["existing_events"]
    used_domains = result["used_domains"]
    assert isinstance(existing_events, CategoriesWithEvents)

    # Check that key information is present somewhere in the final events
    all_merged_text = " ".join(vars(existing_events).values())

    # Verify that some content was extracted and merged
    assert len(all_merged_text) > 0
    # The final events should contain some information from the existing events
    assert (
        "1993" in all_merged_text
        or "graphics processors" in all_merged_text
        or "NVIDIA expanded its business" in all_merged_text
    )

    # Verify that domains were tracked
    assert isinstance(used_domains, list)
