categorize_events_prompt = """
You are a helpful assistant that will categorize company events into 4 categories.

<Events>
{events}
</Events>

<Categories>
company_profile: Covers company founding, history, headquarters, industry, mission, background, and major company development.

products_services: Covers major products, services, technologies, platforms, product launches, and important product developments.

leadership: Covers founders, CEOs, executives, leadership appointments, departures, and major organizational changes.

business_milestones: Covers funding, acquisitions, mergers, partnerships, IPOs, market expansion, major announcements, and other significant business milestones.
</Categories>

<Rules>
INCLUDE ALL THE INFORMATION FROM THE EVENTS. Do not abbreviate or omit any important information.
Categorize each event into the most appropriate category.
</Rules>
"""

EXTRACT_AND_CATEGORIZE_PROMPT = """
You are a Company Intelligence Event Extractor and Categorizer.

Your task is to analyze the text chunk and identify significant, concrete events
related to the company being researched.

<Available Tools>
- `IrrelevantChunk`: Use if the text contains NO significant company events.
- `RelevantEventsCategorized`: Use if the text contains relevant company events.
</Available Tools>

<Categories>
company_profile:
Covers company founding, history, headquarters, industry, mission, background,
and major company development.

products_services:
Covers major products, services, technologies, platforms, product launches,
and important product developments.

leadership:
Covers founders, CEOs, executives, leadership appointments, departures,
and major organizational changes.

business_milestones:
Covers funding, acquisitions, mergers, partnerships, IPOs, market expansion,
major announcements, and other significant business milestones.
</Categories>

<EXTRACTION RULES>
- Extract complete sentences with all important available details.
- Include dates, names, locations, numbers, and relevant context when available.
- Include cause-and-effect relationships when explicitly stated.
- Include only information directly relevant to the company.
- Do not invent facts, dates, locations, or relationships.
- Maintain chronological order within each category.
- Do not include generic marketing statements unless they describe a significant event.
- Format each category as a SINGLE string containing bullet points.
- Do not return category values as lists.

<Text to Analyze>
{text_chunk}
</Text to Analyze>

You must call exactly one of the provided tools.
Do not respond with plain text.
"""

MERGE_EVENTS_TEMPLATE = """You are a helpful assistant that merges two sets of company intelligence events.

The original events must always be preserved. New events may contain additional
details or events that are not already present.

<Rules>
- Always preserve all important information from the original events.
- Add new events only when they contain genuinely new information.
- If two events describe the same event, combine their details into one event.
- Do not invent or change facts.
- Remove duplicate information.
- Keep the events in chronological order when dates are available.
- Format the final result as bullet points, one event per line.
- Do not include commentary or explanations.

<Events>
Original events:
{original}

New events:
{new}
</Events>

<Output>
Return only the merged list of events as bullet points.
</Output>
"""
