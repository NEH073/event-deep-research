# Event Deep Research

An AI-powered event research system that discovers relevant information about companies from web sources, extracts important events, merges and deduplicates the results, and produces a structured event timeline.

The project uses **LangGraph** to orchestrate multiple specialized workflows for research, web crawling, event extraction, processing, and event merging.

## Features

* Supervisor-based LangGraph workflow
* Web research using Tavily
* Web page extraction using Firecrawl
* LLM-based event extraction
* Structured event output
* Event merging and deduplication
* Chunk-based processing
* Async Python workflows
* Configurable LLM provider and model
* LangGraph Studio support
* Automated tests with Pytest

## Architecture

The system is organized into multiple workflows:

1. **Supervisor**

   * Coordinates the overall research process
   * Decides which tools/workflows should be executed

2. **Research Events**

   * Searches the web for relevant company information
   * Uses Tavily for web research
   * Identifies useful sources and event information

3. **URL Crawler**

   * Fetches and extracts content from web pages
   * Uses Firecrawl for web page extraction

4. **Chunk Processing**

   * Processes large amounts of extracted content in smaller chunks
   * Sends relevant chunks for event extraction

5. **Merge Events**

   * Combines events collected from multiple sources
   * Removes duplicate or overlapping events
   * Produces a cleaner event dataset

6. **Structured Output**

   * Converts the processed information into a structured chronology

## Tech Stack

* **Python**
* **LangGraph**
* **LangChain**
* **Tavily**
* **Firecrawl**
* **LLMs**
* **Pydantic**
* **Pytest**
* **LangGraph Studio**

## Project Structure

```text
event-deep-research/
│
├── src/
│   ├── graph.py
│   ├── configuration.py
│   │
│   ├── research_events/
│   │   ├── research_events_graph.py
│   │   ├── chunk_graph.py
│   │   │
│   │   └── merge_events/
│   │       └── merge_events_graph.py
│   │
│   └── url_crawler/
│       └── url_krawler_graph.py
│
├── tests/
│
├── langgraph.json
├── pyproject.toml
├── .env.example
└── README.md
```

## Prerequisites

* Python 3.12+
* Git
* Tavily API key
* Firecrawl API key
* LLM API key

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project:

```powershell
pip install -e .
```

Create your environment file:

```powershell
Copy-Item .env.example .env
```

Add the required API keys to `.env`.

The LLM provider and model can be configured through:

```text
src/configuration.py
```

## Running the Project

Start the LangGraph development server:

```powershell
python -m langgraph_cli dev
```

The server will provide access to:

* LangGraph API
* LangGraph Studio
* API documentation

Open LangGraph Studio and select the **supervisor** graph to run the complete workflow.

## Example Input

```json
{
  "company_to_research": "Microsoft",
  "existing_events": {},
  "used_domains": [],
  "events_summary": ""
}
```

The workflow researches the company, collects information from web sources, extracts relevant events, merges duplicate information, and produces a structured event chronology.

## Testing

Run the test suite with:

```powershell
python -m pytest -q
```

The project includes tests for the research, crawling, chunk processing, and event-merging workflows.

## Configuration

The main configuration is available in:

```text
src/configuration.py
```

This controls the default model and other runtime configuration used by the LangGraph workflows.

API keys and environment-specific values should be stored in:

```text
.env
```

The `.env` file should never be committed to the repository.

## What This Project Demonstrates

This project demonstrates practical experience with:

* LangGraph workflow orchestration
* Multi-step AI workflows
* LLM tool calling
* Stateful graph execution
* Structured LLM output
* Web search and web data extraction
* Event information extraction
* Data processing and deduplication
* Async Python
* API integrations
* Pydantic models
* Automated testing
* Configurable AI pipelines

## Future Improvements

Possible future improvements include:

* Better event ranking
* Improved source validation
* More advanced deduplication

