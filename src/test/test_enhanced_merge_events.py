"""Tests for the enhanced merge events graph."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.state import CategoriesWithEvents
from src.research_events.merge_events.merge_events_graph import (
    extract_and_categorize_chunk,
    merge_categorizations,
    combine_new_and_original_events,
)


@pytest.fixture
def sample_merge_input_state() -> dict:
    """Provide a sample input state for the enhanced merge events graph."""
    return {
        "existing_events": CategoriesWithEvents(
            company_profile="Founded in 1993.",
            products_services="Developed graphics processors.",
            leadership="Jensen Huang is the CEO.",
            business_milestones="Became a major public technology company.",
        ),
        "extracted_events": (
            "NVIDIA was founded in 1993. The company developed graphics "
            "processors and expanded its business significantly."
        ),
        "research_question": "Research the company NVIDIA",
    }


class MockToolCall:
    """Mock tool call for structured LLM responses."""

    def __init__(self, name, args):
        """Initialize mock tool call with name and args."""
        self.name = name
        self.args = args

    def __getitem__(self, key):
        """Make the mock tool call subscriptable."""
        if key == "name":
            return self.name
        elif key == "args":
            return self.args
        raise KeyError(f"Key {key} not found in MockToolCall")


class MockToolResponse:
    """Mock tool response for structured LLM responses."""

    def __init__(self, tool_calls=None):
        """Initialize mock tool response with tool calls."""
        self.tool_calls = tool_calls or []


@pytest.mark.asyncio
async def test_enhanced_merge_events_with_mocked_llm(
    sample_merge_input_state: dict,
):
    """Test company event extraction and categorization with a mocked LLM."""

    state = {
        "existing_events": sample_merge_input_state["existing_events"],
        "extracted_events": sample_merge_input_state["extracted_events"],
        "research_question": sample_merge_input_state["research_question"],
        "text_chunks": [
            sample_merge_input_state["extracted_events"]
        ],
        "categorized_chunks": [],
    }

    mock_response = MockToolResponse(
        [
            MockToolCall(
                "RelevantEventsCategorized",
                {
                    "company_profile": "- NVIDIA was founded in 1993",
                    "products_services": "- Developed graphics processors",
                    "leadership": "- Jensen Huang is the CEO",
                    "business_milestones": (
                        "- NVIDIA became a major technology company"
                    ),
                },
            )
        ]
    )

    mock_model = AsyncMock()
    mock_model.ainvoke.return_value = mock_response

    with patch(
        "src.research_events.merge_events.merge_events_graph.create_llm_with_tools",
        return_value=mock_model,
    ):
        result = await extract_and_categorize_chunk(state, Mock())

    categorized_chunks = result.update["categorized_chunks"]

    assert len(categorized_chunks) == 1

    categorized = categorized_chunks[0]

    assert isinstance(categorized, CategoriesWithEvents)
    assert "NVIDIA was founded in 1993" in categorized.company_profile
    assert "graphics processors" in categorized.products_services
    assert "Jensen Huang" in categorized.leadership
    assert "major technology company" in categorized.business_milestones

    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_merge_categorizations():
    """Test merging categorized company events."""

    categorized_events = [
        CategoriesWithEvents(
            company_profile="- NVIDIA was founded in 1993",
            products_services="- Developed graphics processors",
            leadership="- Jensen Huang is the CEO",
            business_milestones="- NVIDIA became a major technology company",
        ),
        CategoriesWithEvents(
            company_profile="- NVIDIA is headquartered in Santa Clara",
            products_services="- NVIDIA develops AI computing platforms",
            leadership="- Jensen Huang remains CEO",
            business_milestones="- NVIDIA expanded into data centers",
        ),
    ]

    state = {
        "categorized_chunks": categorized_events,
    }

    result = await merge_categorizations(state)

    merged = result.update["extracted_events_categorized"]

    assert isinstance(merged, CategoriesWithEvents)
    assert "NVIDIA was founded in 1993" in merged.company_profile
    assert "Santa Clara" in merged.company_profile
    assert "graphics processors" in merged.products_services
    assert "AI computing platforms" in merged.products_services
    assert "Jensen Huang" in merged.leadership
    assert "data centers" in merged.business_milestones


@pytest.mark.asyncio
async def test_enhanced_merge_events_with_empty_content():
    """Test enhanced merge events with empty extracted content."""

    input_state = {
        "existing_events": CategoriesWithEvents(
            company_profile="Founded in 1993.",
            products_services="Developed graphics processors.",
            leadership="Jensen Huang is the CEO.",
            business_milestones="Became a major technology company.",
        ),
        "extracted_events": "",
        "research_question": "Test question",
    }

    from src.research_events.merge_events.merge_events_graph import merge_events_app

    result = await merge_events_app.ainvoke(input_state)

    assert "existing_events" in result

    existing_events = result["existing_events"]

    assert existing_events.company_profile == "Founded in 1993."
    assert existing_events.products_services == "Developed graphics processors."
    assert existing_events.leadership == "Jensen Huang is the CEO."
    assert (
        existing_events.business_milestones
        == "Became a major technology company."
    )