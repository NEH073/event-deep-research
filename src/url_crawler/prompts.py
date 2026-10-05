# --- Prompt 1: For extracting events from a small text chunk ---

# --- Prompt 1: For extracting events from a small text chunk ---
EXTRACT_EVENTS_PROMPT = """
You are a Biographical Event Extractor. Your single focus is to find events that directly answer the research question: **"{research_question}"**


<Available Tools>
- `RelevantChunk` (use this if the text is almost entirely relevant (>80%))
- `PartialChunk` (use this if the text is a mix of relevant and irrelevant content)
- `IrrelevantChunk` (use this if the text contains no events that are relevant to the biography of the person in the researc question)
</Available Tools>


**EXTRACTION RULE for `PartialChunk`**: You *must* extract the complete relevant sentences, including all details like dates, names, locations, and context. Do not summarize.

<Text to Analyze>
{text_chunk}
</Text to Analyze>

You must call exactly one of the provided tools. Do not respond with plain text.
Choose only the tool call and the tool call arguments.
"""
# src/url_crawler/prompts.py

create_event_list_prompt = """You are a biographical assistant. Your task is to convert blocks of text that contains events of a person into single events where the date, description of the event, location of the event are included for {research_question}.

**Instructions**:
- Analyze the "New Extracted Events" and convert them into single events where the date, description of the event, location of the event are included.
- **MAINTAIN** a chronological order.

**Output Format**:
- A single, comprehensive, and chronological list in bullet points.

<Input>
New Extracted Events:
----
{newly_extracted_events}

</Input>

<Output>
Provide the single, consolidated, and chronological list of biographical events.
</Output>
"""
