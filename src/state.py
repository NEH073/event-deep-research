"""Defines the Pydantic models and TypedDicts for the research agent graph.
This file serves as the schema for data structures, agent tools, and state management.
"""

import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field

################################################################################
# Section 1: Core Data Models
# - Defines the structure of the primary research output: the chronological timeline.
################################################################################


class ChronologyDate(BaseModel):
    """A structured representation of a date for a chronological event."""

    year: int | None = Field(None, description="The year of the event.")
    note: str | None = Field(
        None, description="Adds extra information to the date (month, day, range...)."
    )


class ChronologyEventInput(BaseModel):
    """Represents a single event, typically used for initial data extraction before an ID is assigned."""

    name: str = Field(description="A short, title-like name for the event.")
    description: str = Field(description="A concise description of the event.")
    date: ChronologyDate = Field(..., description="The structured date of the event.")
    location: str | None = Field(
        None, description="The geographical location where the event occurred."
    )


class ChronologyEvent(ChronologyEventInput):
    """The final, canonical event model with a unique identifier."""

    id: str = Field(
        description="The id of the event in lowercase and underscores. Ex: 'word1_word2'"
    )


class ChronologyInput(BaseModel):
    """A list of newly extracted events from a research source."""

    events: list[ChronologyEventInput]


class Chronology(BaseModel):
    """A complete chronological timeline with finalized (ID'd) events."""

    events: list[ChronologyEvent]


class CategoriesWithEvents(BaseModel):
    company_profile: str = Field(
        default="",
        description="Company overview, founding, headquarters, industry, mission, and major background information."
    )
    products_services: str = Field(
        default="",
        description="Major products, services, platforms, technologies, and important product launches."
    )
    leadership: str = Field(
        default="",
        description="Key executives, founders, leadership changes, appointments, and organizational changes."
    )
    business_milestones: str = Field(
        default="",
        description="Major business events such as funding, acquisitions, partnerships, IPO, expansions, major announcements, and other significant milestones."
    )


################################################################################
# Section 2: Agent Tools
# - Pydantic models that define the tools available to the LLM agents.
################################################################################


class ResearchEventsTool(BaseModel):
    """The research question to find important information about a company."""
    research_question: str
    pass


class FinishResearchTool(BaseModel):
    """Concludes the company research process.
    Call this tool ONLY when you have comprehensive company information,
    including major business milestones, products or services, leadership,
    and other significant company events, with no major gaps remaining.
    """
    pass


################################################################################
# Section 3: Graph State Definitions
# - TypedDicts and models that define the "memory" for the agent graphs.
################################################################################


def override_reducer(current_value, new_value):
    """Reducer function that allows a new value to completely replace the old one."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


# --- Main Supervisor Graph State ---


class SupervisorStateInput(TypedDict):
    """The initial input to start the main research graph."""
    company_to_research: str
    existing_events: CategoriesWithEvents = Field(
        default=CategoriesWithEvents(
            company_profile="",
            products_services="",
            leadership="",
            business_milestones="",
        ),
        description="Covers company intelligence events.",
    )
    used_domains: list[str] = Field(
        default=[],
        description="The domains that have been used to extract events.",
    )
    events_summary: str = Field(
        default="",
        description="A summary of the events.",
    )


class SupervisorState(SupervisorStateInput):
    """The complete state for the main supervisor graph."""

    final_events: List[ChronologyEvent]
    conversation_history: Annotated[list[MessageLikeRepresentation], override_reducer]
    iteration_count: int = 0
    structured_events: list[ChronologyEvent] | None
