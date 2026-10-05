from typing import Literal

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from src.configuration import Configuration
from src.llm_service import (
    create_llm_structured_model,
    create_llm_with_tools,
)
from src.prompts import (
    events_summarizer_prompt,
    lead_researcher_prompt,
    structure_events_prompt,
)
from src.research_events.research_events_graph import research_events_app
from src.state import (
    CategoriesWithEvents,
    Chronology,
    FinishResearchTool,
    ResearchEventsTool,
    SupervisorState,
    SupervisorStateInput,
)
from src.utils import get_buffer_string_with_tools, get_langfuse_handler, think_tool

config = Configuration()
MAX_TOOL_CALL_ITERATIONS = config.max_tool_iterations


# Verify connection
# if langfuse.auth_check():
#     print("Langfuse client is authenticated and ready!")
# else:
#     print("Authentication failed. Please check your credentials and host.")


async def supervisor_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor_tools"]]:
    """The 'brain' of the agent. It decides the next action."""
    tools = [
        ResearchEventsTool,
        FinishResearchTool,
        think_tool,
    ]

    tools_model = create_llm_with_tools(tools=tools, config=config)
    messages = state.get("conversation_history", "")
    messages_summary = get_buffer_string_with_tools(messages)
    last_message = ""
    if len(messages_summary) > 0:
        last_message = messages_summary
    system_message = SystemMessage(
        content=lead_researcher_prompt.format(
            company_to_research=state["company_to_research"],
            events_summary=state.get("events_summary", "Everything is missing"),
            last_message=last_message,
            max_iterations=5,
        )
    )

    human_message = HumanMessage(content="Start the research process.")
    prompt = [system_message, human_message]

    response = await tools_model.ainvoke(prompt)

    # The output is an AIMessage with tool_calls, which we add to the history
    return Command(
        goto="supervisor_tools",
        update={
            "conversation_history": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        },
    )


async def supervisor_tools_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "structure_events"]]:
    """The 'hands' of the agent. Executes tools and returns a Command for routing."""
    existing_events = state.get(
    "existing_events",
    CategoriesWithEvents(
        company_profile="",
        products_services="",
        leadership="",
        business_milestones="",
    ),
)
    events_summary = state.get("events_summary", "")
    used_domains = state.get("used_domains", [])
    last_message = state["conversation_history"][-1]
    iteration_count = state.get("iteration_count", 0)
    exceeded_allowed_iterations = iteration_count >= MAX_TOOL_CALL_ITERATIONS

    # If the LLM made no tool calls, we finish.
    if not last_message.tool_calls or exceeded_allowed_iterations:
        return Command(goto="structure_events")

    # This is the core logic for executing tools and updating state.
    all_tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "FinishResearchTool":
            return Command(goto="structure_events")

        elif tool_name == "think_tool":
            # The 'think' tool is special: it just records a reflection.
            # The reflection will be in the message history for the *next* supervisor turn.
            response_content = tool_args["reflection"]
            all_tool_messages.append(
                ToolMessage(
                    content=response_content,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        elif tool_name == "ResearchEventsTool":
            research_question = tool_args["research_question"]
            result = await research_events_app.ainvoke(
                {
                    "research_question": research_question,
                    "existing_events": existing_events,
                    "used_domains": used_domains,
                }
            )
            existing_events = result["existing_events"]
            used_domains = result["used_domains"]

            summarizer_prompt = events_summarizer_prompt.format(
                existing_events=existing_events
            )
            response = await create_llm_structured_model(config=config).ainvoke(
                summarizer_prompt
            )

            existing_events = existing_events
            events_summary = response.content
            all_tool_messages.append(
                ToolMessage(
                    content="Called ResearchEventsTool and returned multiple events",
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

    # The Command helper tells the graph where to go next and what state to update.
    return Command(
        goto="supervisor",
        update={
            "existing_events": existing_events,
            "conversation_history": all_tool_messages,
            "used_domains": used_domains,
            "events_summary": events_summary,
        },
    )


async def structure_events(
    state: SupervisorState, config: RunnableConfig
) -> dict:
    """Step 2: Structures the cleaned events into JSON format.

    Args:
        state: Current researcher state with cleaned events text.
        config: Runtime configuration with model settings.

    Returns:
        Dictionary containing a list of structured chronology events.
    """
    print("--- Step 2: Structuring Events into JSON ---")

    # Get the cleaned events from the previous step
    existing_events = state.get(
       "existing_events",
       CategoriesWithEvents(
           company_profile="",
           products_services="",
           leadership="",
           business_milestones="",
    ),
)

    if not existing_events:
        print("Warning: No company events found in state")
        return {"structured_events": []}

    structured_llm = create_llm_structured_model(config=config, class_name=Chronology)

    company_profile_prompt = structure_events_prompt.format(
    existing_events=existing_events.company_profile
   )
    products_services_prompt = structure_events_prompt.format(
    existing_events=existing_events.products_services
    )
    leadership_prompt = structure_events_prompt.format(
    existing_events=existing_events.leadership
    )
    business_milestones_prompt = structure_events_prompt.format(
        existing_events=existing_events.business_milestones
    )

    company_profile_response = await structured_llm.ainvoke(company_profile_prompt)
    products_services_response = await structured_llm.ainvoke(products_services_prompt)
    leadership_response = await structured_llm.ainvoke(leadership_prompt)
    business_milestones_response = await structured_llm.ainvoke(business_milestones_prompt)
    # Invoke the second model to get the final structured output

    all_events = (
    company_profile_response.events
    + products_services_response.events
    + leadership_response.events
    + business_milestones_response.events
)

    return {
        "structured_events": all_events,
    }


workflow = StateGraph(SupervisorState, input_schema=SupervisorStateInput)

# Add the two core nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("supervisor_tools", supervisor_tools_node)
workflow.add_node("structure_events", structure_events)

workflow.add_edge(START, "supervisor")

graph = workflow.compile().with_config({"callbacks": [get_langfuse_handler()]})
