# Event RAG Agent

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
  - [Core Components](#core-components)
  - [Detailed Agent Workflow](#detailed-agent-workflow)
- [LLM-as-a-Judge Evaluation System](#llm-as-a-judge-evaluation-system)
  - [Evaluation Architecture](#evaluation-architecture)
  - [Custom "Factual Correctness" Judge](#custom-factual-correctness-judge)
  - [Standard DeepEval Judges](#standard-deepeval-judges)
  - [Synthetic Dataset Generation](#synthetic-dataset-generation)
  - [Example Queries](#example-queries)
- [Technical Implementation Details](#technical-implementation-details)
  - [Agent State Management Architecture](#agent-state-management-architecture)
- [Vector Database Creation Process](#vector-database-creation-process)
- [Main Obstacles Encountered](#main-obstacles-encountered)
- [Planned Improvements](#planned-improvments)

An Agentic RAG system for finding and recommending events.<br>
Though, due to some implementation decisions, I am no longer sure whether it is actually an 'agent' or 
a agentic-workflow, as defined by [Anthropic](https://www.anthropic.com/engineering/building-effective-agents).<br>
In any case, for ease of use, I will be referring to it as a agent.<br>
The agent uses multi-step reasoning, state management, and vector search to process natural language queries in Polish and return relevant events scraped from Warsaw's official event calendar.

For now the agent is focused on events in Warsaw, Poland, but the architecture is designed to be easily scaled to potentially any city in Poland.<br>
## Overview
Firstly, I scraped the event data from Warsaw's official local government website (`um.warszawa.pl`).<br>
Since LLM's prefer (or rather, perform better with) .md files, I converted each file, parsing and condensing them ever so slightly, removing unnecessary fluff such as links, images etc.

Then, I built a vector databaase using ChromaDB, where I stored the semantic summaries of the events (details in the latter section of the REEADME).<br>

The agent works as follows:
1. Parses the user query, extracting key information and inferring information that the user might not have explicitly mentioned, such as the date or some keywords.
2. Based on the information extracted from the query, it performs a vector search in the database.
3. Selects the best event file.
4. Reads the event file (extracting relevant information) and evaluates it against the user query.
5. Repeats steps 3-4 until a satisfactory match is found (or multiple matching events!) and returns the final answer.

The agent has the 'freedom' to choose when to stop searching for events, based on the diminishing returns analysis (more details follow).

In order to evaluate the agent's performance, I needed some ground-truth data.<br>
I generated synthetic quereis, mimicking a potential user, based on the event files. <br>
I then tried evaluating the agent using a LLM-as-a-Judge framework, where based on a custom metric and some predefined GEval metrics, the agent's performance is evaluated against the expected output.<br>
Turns out, creating realistic queries that would result in the expected output was harder than I thought (more on that in the last section) and the agent performance is not as good as it could've been.<br>
I believe that the key to properly evaluating the agent, is generating and adequate set of queries, but it seems extremely hard to emulate user behaviour.

## System Architecture

### Core Components

1. **Scraping Pipeline** (`scrape_events.py`)
2. **Vector Database Pipeline** (`build_db.py`, `setup_db.py`)
3. **Multi-Step Agent** (`agent.py`)


### Detailed Agent Workflow
```
1. User Query → ParseUserQueryTool
   ├─ Date Logic (current_date, relative dates, ranges)
   ├─ Location Extraction (districts, venues)  
   ├─ Semantic Query Synthesis
   └─ Intent Structuring 

2. Structured Intent → SearchEventPageTitlesTool
   ├─ Vector Query (ChromaDB with n_results=20)
   ├─ Distance Sorting
   └─ Event Title Results

3. Search Results → SelectEventFileTool
   ├─ Strategic Decision Making
   ├─ Diminishing Returns Analysis  
   ├─ Best Unread Event Selection
   └─ Continue/Stop Decision

4. Selected Event → ReadAndEvaluateEventTool
   ├─ Phase 1: Comprehensive Data Extraction
   │  ├─ Date/Time Parsing
   │  ├─ Location/District Detection
   │  ├─ Event Type Classification
   │  └─ Content Summarization
   └─ Phase 2: Multi-Criteria Evaluation
      ├─ Theme Proximity (perfect/closely_related/broadly_related/unrelated)
      ├─ Date Proximity (perfect/within_a_week/within_a_month/outside_timespan)  
      ├─ Location Proximity (perfect/adjacent_district/different_district/different_city)
      └─ Confidence Scoring (0.0-1.0)

5. Evaluation Results → Continue Loop or FinalAction
   ├─ Match Found → Final Response Generation
   ├─ No Match → Next Event Selection
   └─ Exhausted Results → Alternative Suggestions

6. Final Response → FinalActionTool
   ├─ Hierarchical Response Structure
   │  ├─ Perfect Matches
   │  ├─ Close Alternatives
   │  ├─ Theme Matches
   │  └─ Best Overall Attempt
```
1. **Parsing User Query**: The agent starts by parsing the user query to extract key information. It attempts to infer details not explicitly mentioned by the user, e.g. 'next weekend' is interpreted as the upcoming Saturday and Sunday.
Structures these details into specific concepts, creating a semantic representation of the user's intent, in a form of search query.
2. **Searching Event Titles**: Using the semantic search query from step 1, the agent searches the vector database for events most closely matching the query. It retrieves the top 20 results.
3. **Selecting Event File**: The events are evaluated based on the calculated distance measures.<br>
This is the key decision point for the agent. At this stage, the agent decides whether to continue searching for more vents or stop processing and return the best match found so far. It 'looks ahead' at the next 5 events, compares their distance scores against the best 2 matches found so far and makes a decision.<br>
If the difference between the best match and the next event is small (implying that the next event might be of interest to the user), the agent continues to read and evaluate said event. If, however, the difference is large, the agent stops processing and returns the best match(es) found so far.<br>

   1. If this is the first iteration, the agent selects the first 'closest' event.
   2. In latter iterations, the agent 'looks ahead' and the next 5 events' distance scores and decides whether to continue processing or stop.
4. **Reading and Evaluating Event**: The agent reads the selected event file, extracting relevant information such as date, time, location, and event type. It then evaluates the event against the user query using a multi-criteria matching system.<br> 
   + **Phase 1: Reading The Event**<br>
   Frist step is to extract the obvious event information such as date, time and location, then come the more complex details such as the event type, the underlying theme, and the purpose of the event. The key is to _understand_ what the event is about, not just to extract the details.

   + **Phase 2: Evaluating The Event**<br> 
The evaluation phase is quite an intricate process, it compares the details extracted in Phase 1 with the user requirements.<br>
The agent must decide and understand how well the event theme/location/date/purpose matches the user's query.<br>
The events are scored on multiple criteria such as:
     + **Theme Proximity**: How closely the event's theme matches the user's query (perfect, closely related, broadly related, unrelated).
     - **Date Proximity**: How well the event's date aligns with the user's requirements (perfect, within a week, within a month, outside timespan).
     - **Location Proximity**: How close the event's location is to the user's specified area (perfect, adjacent district, different district, different city).
     - **Confidence Scoring**: A numerical score from 0.0 to 1.0 indicating how well the event matches the user's query.

5. **Final Response Generation**: steps 3 and 4 are repeated until a satisfactory match is found or all events have been processed. If a perfect match is found, the agent generates a response containing the event details. If no perfect match is found, it provides close alternatives or suggests the best overall event based on the evaluation criteria.



## LLM-as-a-Judge Evaluation System

The project implements a LLM-as-a-Judge evaluation framework using DeepEval, where LLMs serve as the evaluators to assess agent performance across multiple dimensions.

### Evaluation Architecture

**Performance Metrics:**
1. **Custom GEval Metric** - Domain-specific factual correctness evaluation  
2. **DeepEval Built-ins** - Answer Relevancy and Faithfulness metrics

### Custom "Factual Correctness" Judge

The core evaluation metric uses a detailed 3-step process implemented in DeepEval's GEval framework:

```python
custom_metric = GEval(
    name="Factual Correctness", 
    model="gpt-4.1-mini",
    evaluation_steps=[
        "Step 1: Scan the 'actual_output', which may contain a list of several events, and determine if any event in the list has a title that EXACTLY or VERY CLOSELY matches the title in the 'expected_output'.",
        "Step 2: Verify that the date(s) of that specific event in the 'actual_output' align with the date(s) in the 'expected_output'. Minor formatting differences are acceptable as long as the core date is correct.", 
        "Step 3: Verify that the location in the 'actual_output' aligns with the location in the 'expected_output'. Minor variations (e.g. missing postal code, or truncated location names) are acceptable."
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5
)
```

**Custom Metric Details:**
- **Multi-Event Handling**: Specifically designed to handle agent responses containing multiple events
- **Fuzzy Matching**: Allows "EXACTLY or VERY CLOSELY" matching titles, not strict string equality
- **Date Flexibility**: Accepts "minor formatting differences" while requiring "core date" accuracy
- **Location Tolerance**: Permits missing postal codes and truncated venue names
- **Sequential Validation**: Must pass all 3 steps to be considered a match

### Standard DeepEval Judges

**Answer Relevancy Judge:**
- Evaluates how well the response addresses the original user query
- Uses semantic similarity and content analysis
- Threshold: 0.7

**Faithfulness Judge:**  
- Measures consistency between retrieval context and generated response
- Detects hallucinations and information distortions
- Compares event names, dates, locations against source material
- Threshold: 0.6 

### Synthetic Dataset Generation

**Synthetic Question Generation:**
- Generates realistic queries from event files
- Creates questions using general terms (not technical jargon)
- Ensures entity-specific queries mention key details

**Quality Control:**
- Date-aware query generation (specific timeframes for past events)
- Natural language patterns matching real user behavior


### Example Queries
- "Bezpłatne koncerty w Warszawie w weekend"
- "Warsztaty dla dzieci na Mokotowie w sierpniu"
- "Wystawy sztuki współczesnej w centrum miasta"
- "Wydarzenia sportowe na Bemowie w przyszłym tygodniu"

## Technical Implementation Details

### Agent State Management Architecture
- **StateManager**: Centralized conversation state with 8 tracked properties
  - `original_query`: User's initial request
  - `user_intent`: Parsed structured intent (UserIntent model)
  - `search_title_results`: Vector search results (List[EventTitleResult])
  - `selected_page_id`: Currently processing event ID
  - `read_event_page_ids`: Set of processed events (deduplication)
  - `read_event_pages_content_dict`: Full event details cache
  - `evaluated_page_ids`: Evaluation tracking for decision logic
  - `evaluation_history`: Complete match history for pattern analysis

- **DependencyManager**: Shared resources and configuration
  - Multiple model configurations for different tasks
  - ChromaDB collection management

## Vector Database Creation Process

The raw scraped event files need to be transformed into something searchable.<br>


### Step 1: Making Sense of Raw Event Data

Using another model, I extract the most important pieces of information into (title, time and location) and distil the files into a (hopefully) semantically complete summary of the event.<br>
```python
class SemanticExtraction(BaseModel):
    title: str = Field(description="The official, clean title of the event.")
    summary: str = Field(description="1-3 sentence summary of theme and purpose, NO logistics")
    time_summary: str = Field(description="Clean, human-readable timeframe",
                             examples=['odbywa się w lipcu 2025', 'w każdy weekend czerwca'])
    location_summary: str = Field(description="Clean location summary",
                                 examples=['Ursynów, Warszawa', 'Park Kultury w Powsinie'])
```
### Stage 2: Document Preparation for Embedding

The extracted information and the summary are then formatted int oa structured doctument ready for embedding, and converted into vector representations.<br>
```python
document_for_embedding = (
    f"Tytuł: {summary.title}\n"
    f"Opis: {summary.summary}\n" 
    f"Kiedy: {summary.time_summary}\n"
    f"Gdzie: {summary.location_summary}"
)
```



## Main Obstacles Encountered
The main challenges I faced during development were:
- Generating synthetic queries that accurately reflect real user behavior.
  - It proved harder than expected to have a model generate realistic queries without relying too much on the actual contents of the event files while simultaneously ensuring the query would yield the described event.<br>
  The main challenge here was that for a given synthetic query, there might've been events better suited than the one on which the query was based.
  The model would often use highly-specilised lingo, very distinctive keywords taken directly from the file, details the user might not know about, or even the exact title of the event.

- Ensuring that the 'main' agent would continue searching through the files until it found a match, rather than stopping at the first event that matched the query.
  - This was solved by implementing a 'diminishing returns' analysis, which would determine whether the next event was worth processing based on how similar it was to the best match so far. However, despite that the model would often still stop before indentifying the file based on which the query was generated. (coming back to the challenge of generating **valid** syntheitic queries).

- Ensuring the agent selects the correct tool for the task at hand.
  - The agent would often fall into endless loops of selecting one or two tools, over and over again, until it hit max steps.
    This might've been linked to the model getting confused by the information it gathered from the previous steps.
    I tried solving it by 'nudging' the model towards selecting the correct tool, based on the results of the current step.
    The `summary()` method would return and append a summary of the current state alongside a suggestion of what the next best action would be.
## Planned Improvments

- **Real-time Scraping**: Automated daily updates
- **Multi-city Support**: Expansion beyond Warsaw
- **Websearch Integration**: Live web search for real-time events via Brave API

---

