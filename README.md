# Event RAG Agent

An Agentic RAG system for finding and recommending events. The agent uses multi-step reasoning, state management, and vector search to process natural language queries in Polish and return relevant events scraped from Warsaw's official event calendar.

For now the agent is focused on events in Warsaw, Poland, but the architecture is designed to be expandable to potentially any city in Poland.
## Overview

This project implements a RAG pipeline with multi-step reasoning that:
- **Scrapes** event data from Warsaw's official city website (`um.warszawa.pl`)
- **Processes** and indexes events using semantic embeddings via ChromaDB
- **Understands** user queries through structured intent parsing with detailed date/location logic
- **Searches** using vector similarity with diminishing returns optimization
- **Evaluates** events through two-phase extraction and matching against multiple criteria
- **Decides** when to continue searching vs. provide results
- **Responds** with event information in Polish with multiple fallback strategies

## System Architecture

### Core Components

1. **Scraping Pipeline** (`scrape_events.py`)


2. **Vector Database Pipeline** (`build_db.py`, `setup_db.py`)
   - **Semantic Extraction**: GPT-4.1-mini processes raw event files into structured summaries
   - **Structured Models**: Pydantic validation ensures consistent data format
   - **Embedding Generation**: OpenAI text-embedding-3-small creates vector representations
   - **ChromaDB Storage**: Persistent client with cosine similarity indexing

3. **Multi-Step Agent** (`agent.py`)
   - **StateManager**: Tracks conversation, search results, evaluations, and extracted data
   - **DependencyManager**: Manages LLM clients, models, ChromaDB collection, and retry logic
   - **5 Structured Tools**:
     - `ParseUserQueryTool`: Intent extraction with date logic and semantic query synthesis
     - `SearchEventPageTitlesTool`: Vector similarity search with result deduplication
     - `SelectEventFileTool`: Decision-making with diminishing returns analysis
     - `ReadAndEvaluateEventTool`: Two-phase extraction and multi-criteria evaluation
     - `FinalAction`: Response generation with multiple fallback strategies
   - **Conversation Management**: System prompts, action history, and error recovery
   - **Logfire Integration**: Monitoring and debugging

4. **Evaluation Framework**
   - **Question Generation** (`question_generation.ipynb`): Synthetic dataset creation with GPT models
   - **Automated Testing** (`eval.ipynb`): DeepEval integration with custom metrics
   - **Performance Analysis** (`deepeval_visualization.ipynb`): Statistical analysis and comparison tools
   - **Multi-run Comparison**: Side-by-side performance tracking with improvement detection

### Detailed Agent Workflow

```
1. User Query → ParseUserQueryTool
   ├─ Date Logic (current_date, relative dates, ranges)
   ├─ Location Extraction (districts, venues)  
   ├─ Semantic Query Synthesis
   └─ Intent Structuring (UserIntent model)

2. Structured Intent → SearchEventPageTitlesTool
   ├─ Vector Query (ChromaDB with n_results=20)
   ├─ Result Deduplication
   ├─ Distance Sorting
   └─ Event Title Results

3. Search Results → SelectEventFileTool
   ├─ Strategic Decision Making
   ├─ Diminishing Returns Analysis  
   ├─ Best Unread Event Selection
   └─ Continue/Stop Decision

4. Selected Event → ReadAndEvaluateEventTool
   ├─ Phase 1: Comprehensive Data Extraction
   │  ├─ Date/Time Parsing (ISO 8601)
   │  ├─ Location/District Detection
   │  ├─ Event Type Classification
   │  └─ Content Summarization (Polish)
   └─ Phase 2: Multi-Criteria Evaluation
      ├─ Theme Proximity (perfect/closely_related/broadly_related/unrelated)
      ├─ Date Proximity (perfect/within_a_week/within_a_month/outside_timespan)  
      ├─ Location Proximity (perfect/adjacent_district/different_district/different_city)
      └─ Confidence Scoring (0.0-1.0)

5. Evaluation Results → Continue Loop or FinalAction
   ├─ Match Found → Final Response Generation
   ├─ No Match → Next Event Selection
   └─ Exhausted Results → Alternative Suggestions
```

## Key Features

### Query Processing
- **Date Parsing**: Handles Polish relative dates ("w ten weekend", "w przyszłym miesiącu")
- **District Recognition**: Understands Warsaw districts and adjacent relationships  
- **Semantic Enhancement**: Transforms casual queries into detailed search terms

### Search Optimization
- **Diminishing Returns**: Analyzes search distance patterns to optimize stopping points
- **Confidence Thresholding**: Uses 0.7+ confidence for high-quality matches
- **Title Similarity**: Prevents processing duplicate events across pages
- **Distance Analysis**: Compares new results against best found matches

### Event Evaluation  
- **Two-Phase Processing**: Extraction then evaluation in separate phases
- **Multi-Criteria Matching**: Boolean logic combining theme, date, and location proximity
- **Confidence Scoring**: 0.0-1.0 scoring based on proximity classifications  
- **Fallback Strategies**: Close alternatives, theme-only matches, best overall when no perfect match

### Response Generation
- **Hierarchical Results**: Perfect matches → close alternatives → theme matches → best attempt
- **Polish Formatting**: Native language dates, descriptions, and reasoning
- **Recurring Events**: Smart date filtering for event series

## LLM-as-a-Judge Evaluation System

The project implements a comprehensive LLM-as-a-Judge evaluation framework using DeepEval, where GPT-4.1-mini serves as the evaluator to assess agent performance across multiple dimensions.

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
- Penalizes irrelevant details (fees, registration) when user asks for basic info (when/where)
- Uses semantic similarity and content analysis
- Threshold: 0.7 (success rate: 72-90%)

**Faithfulness Judge:**  
- Measures consistency between retrieval context and generated response
- Detects hallucinations and information distortions
- Compares event names, dates, locations against source material
- Threshold: 0.6 (success rate: 70-80%)

### Synthetic Dataset Generation

**GPT-4o Question Generation Pipeline:**
- Generates realistic Polish queries from event files
- Creates questions using general terms (not technical jargon)
- Ensures entity-specific queries mention key details
- Produces 10-30 question-answer pairs per evaluation run

**Quality Control:**
- Date-aware query generation (specific timeframes for past events)
- Natural language patterns matching real user behavior



## Technical Implementation Details

### State Management Architecture
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

The raw scraped event files need to be transformed into something searchable.

### Step 1: Making Sense of Raw Event Data

Each scraped markdown file gets fed through GPT-4.1-mini to extract the important bits:

```python
class SemanticExtraction(BaseModel):
    title: str = Field(description="The official, clean title of the event.")
    summary: str = Field(description="1-3 sentence summary of theme and purpose, NO logistics")
    time_summary: str = Field(description="Clean, human-readable timeframe",
                             examples=['odbywa się w lipcu 2025', 'w każdy weekend czerwca'])
    location_summary: str = Field(description="Clean location summary",
                                 examples=['Ursynów, Warszawa', 'Park Kultury w Powsinie'])
```

The extraction runs 5 files at once to speed things up. GPT gets strict instructions to focus on what the event is actually about, capturing the underlying theme of the event.

### Stage 2: Document Preparation for Embedding

Each extracted summary is formatted into a searchable document:

```python
document_for_embedding = (
    f"Tytuł: {summary.title}\n"
    f"Opis: {summary.summary}\n" 
    f"Kiedy: {summary.time_summary}\n"
    f"Gdzie: {summary.location_summary}"
)
```

**Document Structure:**
- **Polish Headers**: "Tytuł", "Opis", "Kiedy", "Gdzie" for semantic consistency
- **Structured Format**: Consistent layout enables better vector similarity
- **Metadata Storage**: Page ID and title stored separately for retrieval

### Stage 3: Vector Generation and Storage

**ChromaDB Configuration:**
```python
collection = db_client.create_collection(
    name="event_semantic_summaries",
    embedding_function=OpenAIEmbeddingFunction(model="text-embedding-3-small"),
    metadata={"hnsw:space": "cosine"}
)
```

**Embedding Process:**
- **OpenAI Model**: text-embedding-3-small
- **Persistent Storage**: ChromaDB saves to `.chroma_db/` directory

### Stage 4: Database Indexing and Optimization

### Search Query Processing

When the agent performs vector search:

1. **Query Vectorization**: User query converted 
2. **Similarity Calculation**: Cosine similarity computed against all stored vectors  
3. **Result Ranking**: Top 20 most similar events returned with distance scores
4. **Metadata Retrieval**: Page IDs and titles extracted for further processing

This creates a semantically aware search system where queries like "zajęcia jogi na Ursynowie" automatically find yoga-related events in the Ursynów district, even if the exact words don't appear in the event descriptions.


### Example Queries
- "Bezpłatne koncerty w Warszawie w weekend"
- "Warsztaty dla dzieci na Mokotowie w sierpniu"
- "Wystawy sztuki współczesnej w centrum miasta"
- "Wydarzenia sportowe na Bemowie w przyszłym tygodniu"

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

