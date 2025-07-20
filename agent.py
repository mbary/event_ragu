from __future__ import annotations
import os
import json
from datetime import datetime, date, timedelta
from typing import Union, List, Literal, Optional, Dict, Any, Set
from pathlib import Path

import chromadb
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
from setup_db import init_collection


EVENT_DIR = Path("data/events")


###################################################################
################### TOOL AND OUTPUT DEFINITIONS ###################
###################################################################

###########################################################
######### STATE MANAGER AND COLLECTION DEFINITION #########
###########################################################

class StateManager(BaseModel):
    """State manager to keep track of the conversation, extracted values and actions taken."""

    original_query: Optional[str] = Field(description="The original user query that initiated the conversation",
                                          default=None)

    user_intent: Optional[UserIntent] = Field(description="The user's intent extracted from the original query",
                                              default=None)
    
    search_title_results: List = Field(description="List of event pages found using title embedding similarity search",
                                       default_factory=list)                                       
    selected_page_id: Optional[str] = Field(description="The current page_id being processed",
                                            default=None)

    read_event_page_ids: Set = Field(description="List of event page_ids that have been read", 
                                     default_factory=set)
    read_event_pages_content_dict: Dict = Field(description="Dictionary of event pages that have been read with their contents",
                                                default_factory=dict)

    event_details: Optional[Dict[str, Any]] = Field(description="Structured event information extracted from the file",
                                                    default=None)
    evaluated_page_ids: List = Field(description="List of event pages that have been evaluated for relevance against the user intent",
                                        default_factory=list)
    last_evaluation: EventEvaluation = Field(description="The last evaluation result of the event against user intent",
                                             default=None)
    evaluation_history: List[Dict[str, Any]] = Field(default_factory=list,
                                                     description="All evaluated events with details and confidence scores")

class DependencyManager(BaseModel):
    """A class to manage shared dependencies and configurations for the agent."""
    client: Any = Field(description="The instructor client used for LLM interactions")
    collection: chromadb.Collection = Field(description="ChromaDB collection for storing event pages and their embeddings")
    main_model: str = Field(description="The LLM model to use for interactions")
    event_dir: Path = Field(description="Directory where event files are stored")
    max_retries: int = Field(default=5, description="Maximum number of retries for LLM calls")
    parsing_model: str = Field(description="The model to use for parsing user queries")
    reading_model: str = Field(description="The model to use for reading event file contents")
    evaluation_model: str = Field(description="The model to use for evaluating event details against user intent")

#########################
###### User Intent ######
#########################

class UserIntentDateTime(BaseModel):
    """Class represents the datetime information extracted from the user query."""
    start_date: date = Field(
            description="The start date extracted from the user query in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).")
    end_date: Optional[date] = Field(
            description="The end date extracted from the user query in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).")
    confidence: float = Field(ge=0, le=1,
                                description="Confidence score of the datetime extraction (0-1).")
    
class UserIntentKeyWord(BaseModel):
    """Class represents a keyword and associated confidence extracted from the user query."""
    keyword: str = Field(description="A keyword identifying an event type or activity.")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the keyword extraction (0-1).")
    
class UserIntent(BaseModel):
    """Represents the user's intent for the event search."""
    think: str = Field(
        description="A thought process or reasoning behind the user's intent extraction.")
    
    query_refined: str = Field(description="A refined user query.")

    timeframe: UserIntentDateTime = Field(description="The desired start and end date for the event in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)")

    city: str = Field(description="The city where the user wants to find events.")

    location: Optional[str] = Field(description="The location where the user wants to find events.", 
                                    example=["Ursus", "Stadion Narodowy", 
                                            "Centrum Nauki Kopernik",
                                            "Park","Teatr","Opera"])

    keywords: List[UserIntentKeyWord] = Field(description="Keywords that identify the type of event or activity being sought.",
                                              max_length=5, 
                                              min_length=2)

    expanded_keywords: Optional[List[str]] = Field(description="A list of 2-3 highly relevant synonyms or related concepts to broaden the search.",
                                                    default=None)                                                    

    semantic_search_query: str = Field(description="A detailed, descriptive query synthesized from all extracted user intent components, optimized for vector search.")

class ParseUserQueryTool(BaseModel):
    """
    Parses the user's initial text query to extract structured intent (keywords, location, dates).
    Use this ONLY as the FIRST step for a new user request.
    Do NOT use this tool if you have already parsed the query and are in the process of evaluating search results.
    """
    think: str = Field(description="Why is this extraction needed and what information is sought")
    action_type: Literal["parse_user_query"] = "parse_user_query"

    def execute(self, state: StateManager, deps: DependencyManager) -> UserIntent:
        """Extract user intent from the user query."""

        SYSTEM_PROMPT_INTENT_EXTRACTION = f"""You are a world-class expert at extracting user intent from the user query in a form of unstructured text.

        The current_date is {date.today().isoformat()}.
        **Returns**:
        - think: A thought process or reasoning behind the user's intent extraction.
        - query: A refined user query.
        - action_type: The type of action to be performed, which is always "parse_user_query".
        - timeframe: The timeframe for the event search, represented as a datetime object in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        - city: The city where the user wants to find events, represented as a string.
        - location: The location where the user wants to find events, represented as a string.
        - keywords: A list of keywords, strictly related to the users query.

        **Your task has two phases:**
        1.  **Parse Phase:** Analyze the user's raw text to extract structured information (keywords, location, timeframe).
        2.  **Synthesize Phase:** Use the structured information you just extracted to construct a new, ideal 'semantic_search_query'.
        
        ---
        **PHASE 1: PARSING RULES:**
        *    **Date Extraction Rules:**
            * No date/time mentioned -> start_date = current_date; end_date = current_date + 14 days.
            * Only month is mentioned:
                - If current month matches -> start_date = current_date; end_date = last day of the month.
                - If next month is mentioned -> start_date = first day of that month; end_date = last day of that month.
            * Specific date mentioned -> start_date = that date.
            * If a timeframe is mentioned (e.g., "next week", "in two weeks", "this weekend", "next weekend"):
                - start_date = start date of that time frame; end_date = end date of that time frame.
            * If a date range is mentioned -> extract both start and end date.

            Examples (assuming current_date is 2025-07-14):
                - "zajęcia z badmintona" → start: 2025-07-14, end: None
                - "zajęcia z badmintona w lipcu" → start: 2025-07-14, end: 2025-07-31
                - "zajęcia z badmintona w sierpniu" → start: 2025-08-01, end: 2025-08-31
                - "zajęcia z badmintona 20 lipca" → start: 2025-07-20, end: None
            
        *    **City Extraction Rules:**
            * If no city is mentioned -> city = Warsaw.

        *    **Keyword Extraction Rules:**
            *  **Refine the Query:** First, create a 'query_refined'. This should be a clear, self-contained natural language sentence that captures the user's full intent.
            *  **Extract Core Keywords:** From the query, identify the main keywords. **Crucially, you MUST return these keywords in their base, nominative (Mianownik) Polish form.** For example, if the user says "warsztatów kreatywnych", the keyword should be "warsztaty kreatywne" or "kreatywność", not the inflected form.
            *  **Expand with Synonyms ('expanded_keywords'):** Brainstorm 2-3 additional keywords that are synonymous or conceptually very close to the core keywords. This is for broadening the search. For "warsztaty rysunku" (drawing workshop), good expansions would be "zajęcia plastyczne" (art classes) or "kurs malarstwa" (painting course).
            *  **Prioritise Specifics:** A specific entity (e.g., a band name like "Kult" or a venue like "Stadion Narodowy") is always a better keyword than a general term ("koncert", "stadion").

        ---
        **PHASE 2: SYNTHESIS RULES FOR 'semantic_search_query'**

        Your goal is to create the most descriptive and unambiguous query possible. Combine the structured elements into a natural, flowing sentence.

        *   **Structure:** Follow a pattern like: '[Adjectives/Keywords] [Event Type] dla [Audience] w lokalizacji [Specific Location], [City] w okresie [Descriptive Timeframe]'.
        *   **Be Descriptive:**
            *   Instead of just the keyword "pilates", use "darmowe zajęcia pilates".
            *   Instead of just "lipiec", use a more descriptive phrase like "w miesiącu lipcu 2025" or "w weekend 5-6 lipca 2025".
            *   Explicitly mention the city and district.
        *   **The query should be a complete thought that fully captures the user's request.**

        ---
        **EXAMPLES:**

        **Example 1:**
        *   **User Input:** "darmowe pilates ursynów w lipcu 2025"
        *   **Parsed Data:** keywords=["pilates", "darmowe"], location="Ursynów", timeframe="2025-07-01 to 2025-07-31"
        *   **Ideal 'semantic_search_query':** "darmowe zajęcia pilates dla dorosłych w dzielnicy Ursynów w Warszawie w miesiącu lipcu 2025"

        **Example 2:**
        *   **User Input:** "koncerty dla dzieci w ten weekend"
        *   **Parsed Data:** keywords=["koncerty", "dla dzieci"], timeframe="2025-07-19 to 2025-07-20" (assuming today is before that weekend)
        *   **Ideal 'semantic_search_query':** "koncerty muzyczne dla dzieci i rodzin w Warszawie w weekend 19-20 lipca 2025"

        ---
        All responses **MUST** be in Polish language.
        """
        user_query = state.original_query
        
        intent = deps.client.chat.completions.create(
            model=deps.parsing_model,
            response_model=UserIntent,
            max_retries=deps.max_retries,
            messages=[
                {"role":"system", "content": SYSTEM_PROMPT_INTENT_EXTRACTION},
                {"role": "user", "content": user_query}
            ],
            temperature=0.0)
        state.user_intent = intent
        return intent
    
    def summarise(self, result: UserIntent) -> str:
        """Create action summary"""
        if not result:
            return {"error": "No user intent extracted. Please check the user query.",
                    "suggested_action": "Try parsing the user query again with 'parse_user_query'"}
        
        summary = f"Think: {result.think}\n"
        summary += f"Parsed user query:\n"
        summary += f"Timeframe: {result.timeframe.start_date.isoformat()}-{result.timeframe.end_date.isoformat() if result.timeframe.end_date else 'N/A'}\n"
        summary += f"City: {result.city}\n"
        summary += f"Location: {result.location}\n"
        summary += f"Refined User Query: {result.query_refined}\n"
        summary += f"Core Keywords: {', '.join([k.keyword for k in result.keywords])}\n"
        if result.expanded_keywords:
            summary += f"Expanded Keywords: {', '.join(result.expanded_keywords)}\n"
        summary += "The user's requirements have been successfully extracted. Your required next action is 'search_event_pages' to find a list of potential events."

        return summary


#################################
#### SEARCH TTILE EMBEDDINGS ####
#################################

class EventTitleResult(BaseModel):
    """Represents a single search result for an event page."""
    page_id: str = Field(description="Unique identifier for the event page")
    title: str = Field(description="Title of the event page")
    distance: float = Field(description="Distance score of the similarity embedding search")

class SearchEventPageTitlesTool(BaseModel):
    """
    Searches the database for event pages based on the parsed user intent.
    This should be used immediately after 'parse_user_query' to get an initial list of events.
    Do NOT use this tool if a list of search results already exists. If results are present, your job is to process them using 'select_best_event_file'.
    """
    think: str = Field(description="Why is this search needed and what information is sought")
    action_type: Literal["search_event_pages"] = "search_event_pages"

    def execute(self, state: StateManager, deps: DependencyManager) -> List[EventTitleResult]:

        if not state.user_intent:
            return {"error": "User intent not found in state. Please extract user intent first using parse_user_query tool.",
                    "suggested_action": "parse_user_query"}
        elif not state.user_intent.keywords:
            return {"error": "No keywords found in user intent. Please ensure the user intent extraction included keywords.",
                    "suggested_action": "parse_user_query"}

        results_dict={}

        query_results = deps.collection.query(
            query_texts=[state.user_intent.semantic_search_query],
            n_results=20)

        for i in range(len(query_results['ids'][0])):
            page_id = query_results['ids'][0][i]

            if page_id not in results_dict:
                results_dict[page_id] = EventTitleResult(
                    page_id=page_id,
                    title=query_results['metadatas'][0][i]['title'],
                    distance=query_results['distances'][0][i]
                )

        final_results=list(results_dict.values())
        final_results.sort(key=lambda x: x.distance)

        state.search_title_results = final_results

        return final_results
    
    def summarise(self, results: List[EventTitleResult]) -> str:
        """summarise the search results."""

        if isinstance(results, dict) and "error" in results:
            return f"Error: {results['error']}. Suggested action: {results['suggested_action']}"

        summary = f"Searched using a refined query and expanded keywords, finding {len(results)} unique potential results."
        summary += " You must now begin processing this list. Your required next action is 'select_best_event_file' to pick the most promising event."
                    
        return summary
    

###########################################
############ GET EVENT DETAILS ############
###########################################

class EventDetailsStart(BaseModel):
    """Represents the start date and time of an event."""
    date: datetime = Field(description="The event start date/time in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the datetime extraction (0-1).")

class EventDetailsEnd(BaseModel):
    """Represents the end date and time of an event."""
    date: datetime = Field(description="The event end date/time in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the datetime extraction (0-1).")

class RecurringDates(BaseModel):
    """Represents recurring dates for an event."""
    all_dates : List[datetime] = Field(description="List of all recurring dates for the event in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).")

class EventDetails(BaseModel):
    """Structured event information extracted from file"""
    title: str = Field(description="Event title")
    event_type: str = Field(description="Type of event (concert, exhibition, theater, festival, workshop, etc.)")
    
    start_datetime: EventDetailsStart
    end_datetime: EventDetailsEnd
    recurring_dates: Optional[RecurringDates] = Field(description="List of all recurring dates for the event, if applicable",
                                                default=None)
    
    location: str = Field(description="Venue or specific location")
    city: str = Field(description="City where event takes place")
    district: Optional[str] = Field(description="District or neighborhood if mentioned")
    
    description: str = Field(description="Full event description")
    summary: str = Field(description="Brief 1-2 sentence summary of the event")
    
    target_audience: Optional[str] = Field(description="Who this event is for (if specified)")
    price_info: Optional[str] = Field(description="Ticket/entry price information")
    
    source_url: Optional[str] = Field(None, description="Event website or ticket link")


##########################################
########### READ FILE CONTENTS ###########
##########################################

class SelectionDecision(BaseModel):
    """"Represents the decision whether to continue searching or stop"""
    think: str = Field(description="The reasoning behind the decision to continue or stop")
    decision: Literal["continue", "stop"] = Field(description="The final decision")
    page_id: Optional[str] = Field(description="The page_id of the next file to process, if the decision is 'continue'",
    default=None)

class SelectEventFileTool(BaseModel):
    """
    Analyzes the current search state and decides whether to continue processing files or to stop.
    If continuing, it selects the single best, unread event file from the existing search results list.
    Use this to begin processing a search list OR to get the next event after a previous one was not a match.
    If stopping, it provides the reason.
    This is the core decision-making tool for iterating through search results.lts.
    """
    think: str = Field(description="My thought process on why I need to make a continue/stop decision now.")
    action_type: Literal["select_best_event_file"] = "select_best_event_file"

    def execute(self, state: StateManager, deps: DependencyManager) -> SelectionDecision:
        if not state.search_title_results:
            return {"error": "No search results found. Please perform a search first using search_event_pages.",
                    "suggested_action": "search_event_pages"}

        unread_events = [res for res in state.search_title_results if res.page_id not in state.read_event_page_ids]

        if not unread_events:
            return SelectionDecision(
                think="There are no more files in the search results to process. The search is complete",
                decision="stop")

        decision_prompt = self._build_decision_prompt(state, unread_events)

        decision = deps.client.chat.completions.create(
            model=deps.main_model,
            response_model=SelectionDecision,
            messages=[
                {"role":"system", "content":"You are a meticulous and strategic research assistant. Your goal is to find all highly relevant results, not just the first one. You stop only when the potential for finding better results diminishes significantly."},
                {"role":"user", "content":decision_prompt}
            ],
            max_retries=deps.max_retries,
            temperature=0.0)

        if decision.decision=="continue":
            best_next_event = min(unread_events, key=lambda x: x.distance).page_id
            decision.page_id = best_next_event
            state.selected_page_id = best_next_event

        return decision        
    
    def summarise(self, result: SelectionDecision) -> str:
        """Create action summary based on the decision"""
        if "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"

        if not result:
            return f"Error: {result['error']}\nThere are no more events to check from the search list. You must now provide a final answer based on what you have found so far.\nThe required next action is 'final_answer'."
        
        if result.decision=="stop":
            summary = f"Decision: Stop searching. Reason: {result.think}\n"
            summary += "I have concluded that further searching will not be fruitful. Your required next action is 'final_answer'."
            return summary
        
        elif result.decision=="continue" and result.page_id:
            summary = f"Decision: Continue searching. Reason: {result.think}\n"
            summary += f"Selected next event file with page_id: {result.page_id}\n"
            summary += "The next required action is to use the 'read_event_file_contents' tool to get the details of this file."
            return summary

        return "An unexpected decision was made. Stopping search to be safe. Required next action is 'final_answer'."
    

    def _build_decision_prompt(self, state: StateManager, unread_events: List[EventTitleResult]) -> str:

        history_summary = "No events have been evaluated yet."
        min_distance_of_good_match = float('inf')

        if state.evaluation_history:
            sorted_evals = sorted(state.evaluation_history, key=lambda e: e.get('match_confidence', 0), reverse=True)
            good_matches = [e for e in sorted_evals if e.get('match_confidence', 0) >= 0.7]

            if good_matches:
                best_match_page_id = good_matches[0]['page_id']
                for res in state.search_title_results:
                    if res.page_id == best_match_page_id:
                        min_distance_of_good_match = res.distance
                        break
            
            summary_lines = [f"So far, you have evaluated {len(sorted_evals)} event(s)."]
            if good_matches:
                summary_lines.append(f"\nFound {len(good_matches)} high-confidence matches (confidence >= 0.7).")
                summary_lines.append(f"The best so far had a search distance of {min_distance_of_good_match:.2f}.")
            
            summary_lines.append("\nTop evaluated events:")
            for e in sorted_evals[:5]:
                summary_lines.append(f"- '{e['title']}' (Confidence: {e.get('match_confidence', 0):.2f}, Reason: {e['overall_reasoning']})")

            history_summary = "\n".join(summary_lines)

        preview_limit = 5
        upcoming_summary = "Here are the next best search results to consider:\n"
        for event in sorted(unread_events, key=lambda x: x.distance)[:preview_limit]:
            upcoming_summary += f"- '{event.title}' (search distance: {event.distance:.2f})\n"
        
        prompt = f"""
        You must make a strategic decision: should you continue processing search results or stop now?
        Your goal is to find ALL highly relevant events, not just the first one.

        **CONTEXT:**
        Your original query was: "{state.original_query}"

        **SEARCH HISTORY**
        {history_summary}

        **UPCOMING RESULTS:**
        {upcoming_summary}

        **YOUR TASK: Apply the principle of diminishing returns by analyzing the search distances.**
        Weigh the quality of events already found against the potential of upcoming events.
        The "search distance" measures relevance (lower is better). Your primary task is to identify a "jump" in this distance, which signals that the remaining results are of lower quality.

        - **You SHOULD CONTINUE if:**
            - You haven't found any high-confidence (>=0.7) matches yet, and the upcoming results still have low search distances and relatively close to each other.
            - You HAVE found good matches, but the next unread event has a search distance that is **similar to or better than** the best one you've already found (e.g., best found was 0.25, next is 0.28). This indicates it could be another excellent match.

        - **You SHOULD STOP if:**
            - You have found at least one high-confidence match, AND the search distances of the upcoming events are **significantly worse** than your best match's distance (e.g., best found was 0.25, next is 0.6). This indicates diminishing returns.
            - The titles of the upcoming events are clearly and completely unrelated to the user's query, even if their distance is low.

        Based on this analysis, decide whether to 'continue' or 'stop' and provide your reasoning.
        """
        return prompt

class ReadEventFileTool(BaseModel):
    """
    Reads and parses the full contents of a SINGLE event file previously chosen by 'select_best_event_file'.
    Its direct and only follow-up action is 'evaluate_event_details_against_user_query'. Do not use any other tool after this one.
    """
    action_type: Literal["read_event_file_contents"] = "read_event_file_contents"
    think: str = Field(description="Why reading this specific event file")
    
    def execute(self, state: StateManager, deps: DependencyManager) -> Dict[str, Any]:
        if not state.selected_page_id:
            return {"error": "No page_id selected. Please select a file first using select_best_event_file.",
                    "suggested_action": "select_best_event_file"}
        
        page_id = state.selected_page_id
        
        try:
            file_path = os.path.join(deps.event_dir, f"{page_id}.md")
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            extraction_prompt = f"""You are a world-class data extraction engine. Your primary goal is to parse unstructured text and populate a structured JSON object based on the following principles.

                                    **CORE EXTRACTION PRINCIPLES:**

                                    1.  **Comprehensive Field Extraction**: You must extract information for ALL fields required by the target JSON schema. This includes event type, location, summary, pricing, and audience, in addition to all date/time information.

                                    2.  **Intelligent Date & Time Parsing**:
                                        - **Format-Agnostic**: Recognize any common date or time format (e.g., 'YYYY-MM-DD', 'DD.MM.YYYY', 'Month Day, Year') and convert it to the required ISO 8601 format.
                                        - **Handle Messy Text**: The source text may be poorly formatted, with dates and times run together. Use your intelligence to identify all distinct event occurrences.
                                        - **Define 'start_datetime'**: This is the start time of the **first session** relevant to the user's query timeframe.
                                        - **Define 'end_datetime'**: This is the end time of the **VERY LAST session** in the entire list of dates. For single events, this is the event's own end time.
                                        - **Define 'recurring_dates'**: This is a complete list of the start times for **EVERY session** found in the text. If there is only one date, this list will contain that single date.

                                    3.  **Detailed Content Extraction**:
                                        - **Event Type**: Identify the 'event_type' from the content (e.g., concert, exhibition, workshop, festival, sports).
                                        - **Location**: Identify the specific 'location' (venue name), 'city', and 'district' if mentioned.
                                        - **Summary**: Create a brief, 1-2 sentence 'summary' of the event's purpose.
                                        - **Pricing & Audience**: Extract any 'price_info' (including "free" or "bezpłatne") and the intended 'target_audience'.
                                        - **Source URL**: Extract the source URL if it is present in the text.

                                    4.  **General Rules**:
                                        - **Use Your Knowledge**: Infer reasonable missing information where appropriate (e.g., if a park is mentioned, you can infer the city).
                                        - **Language**: All text-based fields in your output (like summary, description, etc.) **MUST** be in the Polish language.

                                    ---
                                    **TASK DATA:**

                                    **User Requirements:**
                                    - Timeframe: {state.user_intent.timeframe.start_date.isoformat()} - {state.user_intent.timeframe.end_date.isoformat() if state.user_intent.timeframe.end_date else 'N/A'}

                                    **Event File Content to Parse:**
                                    {raw_content}

                                    Now, apply all of these principles to extract the information from the provided content into the required structured format.
                                    """

            parsed_event = deps.client.chat.completions.create(
                model=deps.reading_model,
                response_model=EventDetails,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting and structuring event information. Use your knowledge of cities, venues, and event types to provide complete information."},
                    {"role": "user", "content": extraction_prompt}
                ],
                max_retries=deps.max_retries,
                temperature=0.0
            )
            event_dict = parsed_event.model_dump()
            event_dict["start_datetime"]["date"] = event_dict["start_datetime"]["date"].isoformat()
            event_dict["end_datetime"]["date"] = event_dict["end_datetime"]["date"].isoformat()
            if event_dict.get("recurring_dates"):
                event_dict["recurring_dates"] = [_date.isoformat() for _date in event_dict["recurring_dates"].get("all_dates", [])]

            state.event_details = {
                "page_id": page_id,
                "raw_content": raw_content,
                "parsed": event_dict,
                "file_path": file_path
            }
            
            location_str = f"{event_dict['location']}, {event_dict['city']}"
            if event_dict.get('district'):
                location_str += f" ({event_dict['district']})"
            
            summary_data = {
                "page_id": page_id,
                "summary": {
                    "title": event_dict['title'],
                    "type": event_dict['event_type'],
                    "date": event_dict['start_datetime']['date'],
                    "location": location_str,
                    "brief": event_dict.get('summary', event_dict['description'] + "...")
                }
            }
            if event_dict.get('price_info'):
                summary_data["summary"]["price"] = event_dict['price_info']
            if event_dict.get('target_audience'):
                summary_data["summary"]["audience"] = event_dict['target_audience']

            state.read_event_page_ids.add(page_id)
            state.read_event_pages_content_dict[page_id] = event_dict
            
            return summary_data
            
        except FileNotFoundError:
            return {"error": f"Event file not found: {page_id}.md",
                    "suggeste_action":"Select another file using select_best_event_file"}
        except Exception as e:
            return {"error": f"Failed to read or parse event file: {str(e)}",
                    "suggested_action": "No idea mate, think of something"} ##TODO correct this XD 
    
    def summarise(self, result: Dict[str, Any]) -> str:
        """Create conversation summary"""
        if "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"
        
        summary = result["summary"]
        
        parts = [
            f"Read event '{summary['title']}'",
            f"({summary['type']})",
            f"on {summary['date']}",
            f"at {summary['location']},",
            f"Summary: {summary['brief'][:100]}"
        ]

        if summary.get('price'):
            parts.append(f"- {summary['price']}")
        
        summary_str = " ".join(parts)

        summary_str += f"\nThe details for event '{summary['title']}' have been read and parsed. Your required next action is 'evaluate_event_details_against_user_query' to check if it's a match."
        return summary_str

class EventMatches(BaseModel):
    """Represents whether the event matches the user's requirements."""
    matches: bool = Field(description="Whether the event matches the user's requirements")
    think: str = Field(description="Thought process behind the match determination")

class EventEvaluation(BaseModel):
    """Structured evaluation result"""
    matches: EventMatches = Field(description="Overall match determination")
    page_id: Optional[str] = None
    title: Optional[str] = None
    match_confidence: float = Field(ge=0, le=1, 
                                    description="""Confidence score 0-1 based on:
                                    1.0 = Perfect match (all criteria met)
                                    0.8-0.9 = Strong match (minor differences like date off by a few days)
                                    0.6-0.7 = Good match (one criterion off, like different district)
                                    0.4-0.5 = Partial match (multiple criteria differ but same category)
                                    0.2-0.3 = Weak match (same city but wrong type/date)
                                    0.0-0.1 = No match (completely different)""")
    
    theme_evaluation: str = Field(description="Evaluation of the event's core subject matter or theme.")
    theme_proximity: Literal["perfect", "closely_related", "broadly_related", "unrelated"] = Field(
        description="How close the event's theme is to the user's request."
    )
    
    date_evaluation: str = Field(description="Evaluation of date match")
    date_proximity: Literal["perfect", "within_a_week", "within_a_month", "outside_timespan"] = Field(
        description="How close the event's date is to the user's timeframe."
    )
    
    location_evaluation: str = Field(description="Evaluation of location match")
    location_proximity: Literal["perfect", "adjacent_district", "different_district", "different_city"] = Field(
        description="How close the event's location is to the user's request."
    )
    
    type_evaluation: str = Field(description="Evaluation of event type/keywords match")
    type_matches: bool
    
    overall_reasoning: str = Field(description="Overall reasoning for the decision")
    recommendation: str = Field(description="What to do next - try another event or provide this one")

class EvaluateEventTool(BaseModel):
    """
    Compares the most recently read event against the user's intent to determine if it's a good match.
    This tool is the core decision-making step. It MUST be used after 'read_event_file_contents'.
    The result of this tool dictates whether to 'final_answer' (on a match) or 'select_best_event_file' (on a mismatch).
    """
    action_type: Literal["evaluate_event_details_against_user_query"] = "evaluate_event_details_against_user_query"
    think: str = Field(description="What aspects need careful evaluation")
    
    def execute(self, state: StateManager, deps: DependencyManager) -> EventEvaluation:
        if not state.event_details:
            return {"error": "No event details found. Please read an event file first.", 
                    "suggested_action": "read_event_file_contents"}
        if not state.user_intent:
            return {"error": "No user intent found. Cannot evaluate without requirements.",
                    "suggested_action": "parse_user_query"}
        
        evaluation_prompt = f"""Evaluate if this event matches the user's requirements.
                                User Requirements:
                                - Query: {state.user_intent.query_refined}
                                - Looking for: {', '.join([k.keyword for k in state.user_intent.keywords])}
                                - City: {state.user_intent.city}
                                - Desired Location/District: {state.user_intent.location or 'Any'}
                                - Start Date: {state.user_intent.timeframe.start_date.isoformat()}
                                - End Date: {state.user_intent.timeframe.end_date.isoformat() if state.user_intent.timeframe.end_date else 'N/A'}
                                

                                Event Details:
                                - Title: {state.event_details['parsed']['title']}
                                - Date: {state.event_details['parsed']['start_datetime']['date']}
                                - Recurring Dates (if any): {', '.join(state.event_details['parsed'].get('recurring_dates', [])) if state.event_details['parsed'].get('recurring_dates') else 'N/A'}
                                - Location: {state.event_details['parsed']['location']}
                                - City: {state.event_details['parsed']['city']}
                                - District: {state.event_details['parsed'].get('district', 'N/A')}
                                - Description: {state.event_details['parsed']['description']}                        

                                **INSTRUCTIONS**

                                Based on the information above, you must fill out the following evaluation fields. Be strict but fair in your judgment.

                                1.  **Theme Proximity**: Classify how well the event's core subject matter matches the user's keywords.
                                    - 'perfect': An exact match (e.g., user wants "pottery workshop", event is "pottery workshop").
                                    - 'closely_related': A very similar activity (e.g., user wants "pottery", event is "ceramics glazing class").
                                    - 'broadly_related': Same general category but different specifics (e.g., user wants "rock concert", event is "music festival").
                                    - 'unrelated': Different subjects (e.g., user wants "rock concert", event is "jazz concert").

                                2.  **Date Proximity**: Classify how well the event's date fits the user's timeframe.
                                    - 'perfect': The event date falls within the user's start and end dates.
                                    - 'within_a_week': The event is within 7 days before the start or after the end of the user's timeframe.
                                    - 'within_a_month': The event is within the same month but more than a week off.
                                    - 'outside_timespan': The event is in a completely different month or year.

                                3.  **Location Proximity**: Use your geographic knowledge to classify the location match.
                                    - 'perfect': The event is in the exact city and district/location requested.
                                    - 'adjacent_district': The event is in a neighboring district within the same city (e.g., Ursynów vs. Mokotów).
                                    - 'different_district': The event is in the correct city but a non-adjacent, distant district.
                                    - 'different_city': The event is in a different city.

                                4.  **Overall Match ('matches.matches')**: This is a final boolean decision. To make it, follow this simple checklist. You will set 'matches.matches' to 'True' if **ALL** of the following conditions are met. Otherwise, set it to 'False'.
                                    - 'theme_proximity' MUST be 'perfect' OR 'closely_related'.
                                    - 'date_proximity' MUST be 'perfect' OR 'within_a_week'.
                                    - 'location_proximity' MUST be 'perfect' OR 'adjacent_district'.

                                    Think step-by-step in the 'matches.think' field, explicitly checking each of the three conditions above before concluding with 'True' or 'False'.

                                5.  **Confidence Score ('match_confidence')**: Provide a float from 0.0 to 1.0 representing your overall confidence, considering all factors. A perfect match should be 1.0. A closely related theme in an adjacent district might be 0.8. An unrelated theme should be close to 0.0.

                                6.  **Reasoning ('overall_reasoning')**: Write a brief, one-sentence explanation for your decision, highlighting the key matching and mismatching points. For example: "This event is a perfect theme match and takes place in the right month, but it is in a different district."

                                Be somewhat flexible but not overly permissive.
                                Always provide answers in Polish language"""
        try:
            evaluation = deps.client.chat.completions.create(
                model=deps.evaluation_model,
                response_model=EventEvaluation,
                messages=[
                    {"role": "system", "content": "You are evaluating if events match user requirements. Use your knowledge of geography, dates, and event types."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_retries=deps.max_retries,
                temperature=0.1)

            eval_dict = evaluation.model_dump()
            eval_dict["page_id"] = state.event_details["page_id"]
            eval_dict["title"] = state.event_details['parsed']['title']
            
            state.last_evaluation = eval_dict
            state.evaluated_page_ids.append(state.event_details["page_id"])
            state.evaluation_history.append(eval_dict)

            evaluation.page_id = state.event_details["page_id"]
            evaluation.title = state.event_details['parsed']['title']
            return evaluation
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}",
                    "suggested_action": "Try a different approach or select another file."}
    
    def summarise(self, result: Union[EventEvaluation, Dict[str, str]]) -> str:
        """Create conversation summary"""
        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"
        
        if result.matches.matches:
            summary = (f"Event '{getattr(result, 'title', 'N/A')}' is a match! "
                    f"({result.match_confidence:.0%} confident). I have saved this result. "
                    f"I will now check for other potential matches from the search list.")
            summary += ("\nYour required next action is 'select_best_event_file' to continue processing the list.")
            return summary
        
        else:
            summary = f"Event '{getattr(result, 'title', 'N/A')}' doesn't match - {result.overall_reasoning}. I will try the next event."
            summary += "\nYou must now try the next event from your search results. Your required next action is 'select_best_event_file'. Do not search again."
            return summary
        
########################
##### FINAL ACTION #####
########################
class FinalAction(BaseModel):
    """
    This is the final tool that concludes the task. Use this to provide the answer to the user.
    This tool should ONLY be used in two cases:
    1. An event has been evaluated and confirmed as a match.
    2. You have exhausted all other options (e.g., search returned no results, or all results were evaluated as not a match) and you need to inform the user.
    """
    action_type: Literal["final_answer"] = "final_answer"
    think: str = Field(description="Summarize the findings and the reasoning for the final answer.")

    def execute(self, state: StateManager, deps: DependencyManager) -> str:
        """Generate a comprehensive, human-readable answer from the collected state."""
        
        if not state.evaluation_history:
            intent = state.user_intent
            answer = "I'm sorry, but my search for events matching your request came up empty.\n"
            if intent:
                keywords = [kw.keyword for kw in intent.keywords]
                answer+=f"I was looking for: An event related to '{', '.join(keywords)}' in {intent.city} around {intent.timeframe.start_date.strftime('%B %Y')} and {state.user_intent.timeframe.end_date.isoformat() if state.user_intent.timeframe.end_date else 'N/A'}."
                
            answer+="\nThere may be no events of this type listed, or you could try searching with different keywords."
            return answer


        matching_events = [e for e in state.evaluation_history if e.get("matches", {}).get("matches")]

        if matching_events:            
            matching_events.sort(key=lambda e: (-e['match_confidence'],
                                                state.read_event_pages_content_dict[e['page_id']]['start_datetime']['date'])
                                 )

            answer = f"After reviewing the options, I found {len(matching_events)} event(s) that match your request!\n\n"
            
            for i, event_eval in enumerate(matching_events):
                user_start = datetime.fromisoformat(state.user_intent.timeframe.start_date.isoformat()).date()
                user_end = datetime.fromisoformat(state.user_intent.timeframe.end_date.isoformat()).date() if state.user_intent.timeframe.end_date else user_start

                page_id = event_eval['page_id']
                event_details = state.read_event_pages_content_dict[page_id]

                answer += f"\t--- Event {i+1} ---\n"
                answer += f"\tTitle:    {event_details['title']}\n"
                
                if event_details.get("recurring_dates"):
                    relevant_dates = [dt_str for dt_str in event_details["recurring_dates"] 
                                    if user_start <= datetime.fromisoformat(dt_str).date() <= user_end + timedelta(days=14)]

                    if relevant_dates:
                        answer+=f"\tUpcoming Dates in {user_start.strftime('%B %Y')}:\n"
                        for dt_str in relevant_dates:
                            answer += f"\t\t- {self._format_date(dt_str)}\n"                
                else:
                    answer += f"\tStart Date:     {self._format_date(event_details['start_datetime']['date'])}\n"
                    answer += f"\tEnd Date:       {self._format_date(event_details['end_datetime']['date'])}\n"

                if event_details.get('price_info'):
                    answer += f"\tPrice:         {event_details['price_info']}\n"
                answer += f"\tLocation:      {event_details['location']}\n"
                answer += f"\tSummary: {event_details['summary']}\n"
                answer += f"\tMore details: {event_details['source_url']}\n\n"
                answer +="\n"
            
            return answer


        non_perfect_evals = [e for e in state.evaluation_history if not e.get("matches", {}).get("matches")]

        close_alternatives = [
            e for e in non_perfect_evals
            if e.get('theme_proximity') in ['perfect', 'closely_related'] and
            (e.get('date_proximity') in ['perfect', 'within_a_week', 'within_a_month'] or e.get('location_proximity') in ['perfect', 'adjacent_district'])]

        if close_alternatives:
            close_alternatives.sort(key=lambda e: -e['match_confidence'])
            answer = "I couldn't find a perfect match. However, I found these very close alternatives:\n\n"

            close_alternatives = close_alternatives[:3]
            for i, event_eval in enumerate(close_alternatives):
                user_start = datetime.fromisoformat(state.user_intent.timeframe.start_date.isoformat()).date()
                user_end = datetime.fromisoformat(state.user_intent.timeframe.end_date.isoformat()).date() if state.user_intent.timeframe.end_date else user_start
                page_id = event_eval['page_id']
                event_details = state.read_event_pages_content_dict[page_id]

                answer += f"\t--- Event {i+1} ---\n"
                answer += f"\tTitle:    {event_details['title']}\n"

                if event_details.get("recurring_dates"):
                    relevant_dates = [dt_str for dt_str in event_details["recurring_dates"] 
                                    if user_start <= datetime.fromisoformat(dt_str).date() <= user_end + timedelta(days=14)
                                    ]

                    if relevant_dates:
                        answer+=f"\tUpcoming Dates in {user_start.strftime('%B %Y')}:\n"
                        for dt_str in relevant_dates:
                            answer += f"\t\t- {self._format_date(dt_str)}\n"
                else:
                    answer += f"\tStart Date:     {self._format_date(event_details['start_datetime']['date'])}\n"
                    answer += f"\tEnd Date:       {self._format_date(event_details['end_datetime']['date'])}\n"

                if event_details.get('price_info'):
                    answer += f"\tPrice:         {event_details['price_info']}\n"
                answer += f"\tLocation:      {event_details['location']}\n"
                answer += f"\tSummary: {event_details['summary']}\n"
                answer += f"\tMore details: {event_details['source_url']}\n\n"
                answer +="\n"
                answer += f"\tReasoning: {event_eval['overall_reasoning']}\n\n"
            return answer
                                                   
        theme_only_alternatives = [e for e in non_perfect_evals
                                    if e.get('theme_proximity') in ['perfect', 'closely_related']]
        
        if theme_only_alternatives:
            theme_only_alternatives.sort(key=lambda e: -e['match_confidence'])
            answer = "While I couldn't find a close match, these events have the right theme but differ in other aspects like date or location:\n\n"

            theme_only_alternatives = theme_only_alternatives[:3]
            for i, event_eval in enumerate(theme_only_alternatives):
                user_start = datetime.fromisoformat(state.user_intent.timeframe.start_date.isoformat()).date()
                user_end = datetime.fromisoformat(state.user_intent.timeframe.end_date.isoformat()).date() if state.user_intent.timeframe.end_date else user_start
                page_id = event_eval['page_id']
                event_details = state.read_event_pages_content_dict[page_id]
                answer += f"\t--- Event {i+1} ---\n"
                answer += f"\tTitle:    {event_details['title']}\n"

                if event_details.get("recurring_dates"):
                    relevant_dates = [dt_str for dt_str in event_details["recurring_dates"] 
                                    if user_start <= datetime.fromisoformat(dt_str).date() <= user_end + timedelta(days=14)]

                    if relevant_dates:
                        answer+=f"\tUpcoming Dates in {user_start.strftime('%B %Y')}:\n"
                        for dt_str in relevant_dates:
                            answer += f"\t\t- {self._format_date(dt_str)}\n"                
                else:
                    answer += f"\tStart Date:     {self._format_date(event_details['start_datetime']['date'])}\n"
                    answer += f"\tEnd Date:       {self._format_date(event_details['end_datetime']['date'])}\n"

                if event_details.get('price_info'):
                    answer += f"\tPrice:         {event_details['price_info']}\n"
                answer += f"\tLocation:      {event_details['location']}\n"
                answer += f"\tSummary: {event_details['summary']}\n"
                answer += f"\tMore details: {event_details['source_url']}\n\n"
                answer +="\n"
                answer += f"\tReasoning: {event_eval['overall_reasoning']}\n\n"
            return answer

        if state.evaluation_history:
            best_overall_event = sorted(state.evaluation_history, key=lambda e: e['match_confidence'], reverse=True)[0]
            page_id = best_overall_event['page_id']
            event_details = state.read_event_pages_content_dict[page_id]
            answer = "I couldn't find any close matches for your request. After reviewing all options, the single closest event I found is:\n\n"
            answer += f"\tTitle:    {event_details['title']}\n"

            if event_details.get("recurring_dates"):
                    user_start = datetime.fromisoformat(state.user_intent.timeframe.start_date.isoformat()).date()
                    user_end = datetime.fromisoformat(state.user_intent.timeframe.end_date.isoformat()).date() if state.user_intent.timeframe.end_date else user_start
                    
                    relevant_dates = [dt_str for dt_str in event_details["recurring_dates"] 
                                    if user_start <= datetime.fromisoformat(dt_str).date() <= user_end + timedelta(days=14)]

                    if relevant_dates:
                        answer+=f"\tUpcoming Dates in {user_start.strftime('%B %Y')}:\n"
                        for dt_str in relevant_dates:
                            answer += f"\t\t- {self._format_date(dt_str)}\n"                
            else:
                answer += f"\tStart Date:     {self._format_date(event_details['start_datetime']['date'])}\n"
                answer += f"\tEnd Date:       {self._format_date(event_details['end_datetime']['date'])}\n"

            if event_details.get('price_info'):
                answer += f"\tPrice:         {event_details['price_info']}\n"
            answer += f"\tLocation:      {event_details['location']}\n"
            answer += f"\tSummary: {event_details['summary']}\n"
            answer += f"\tMore details: {event_details['source_url']}\n\n"
            answer +="\n"
            answer += f"\tReasoning: {best_overall_event['overall_reasoning']}\n\n"
            return answer

        return "I am sorry, but I was unable to find any relevant events after a thorough search."
    

    def _format_date(self, date_str: str) -> str:
        """Format date nicely for display."""
        try:
            from datetime import datetime
            dt_obj = datetime.fromisoformat(date_str)
            return dt_obj.strftime("%A, %B %d, %Y at %I:%M %p")
        except (ValueError, TypeError):
            return date_str

    def summarise(self, result: str) -> str:
        """The result is the final answer. This summary signals the end of the task."""
        return "A final answer has been generated and provided to the user. The task is now complete."    


AgentActions = Union[ParseUserQueryTool,SearchEventPageTitlesTool, SelectEventFileTool,
                     ReadEventFileTool, EvaluateEventTool, FinalAction]


### Rebuild the damn models
StateManager.model_rebuild()
DependencyManager.model_rebuild()
UserIntentDateTime.model_rebuild()
UserIntentKeyWord.model_rebuild()
UserIntent.model_rebuild()
ParseUserQueryTool.model_rebuild()
EventTitleResult.model_rebuild()
SearchEventPageTitlesTool.model_rebuild()
EventDetailsStart.model_rebuild()
EventDetailsEnd.model_rebuild()
EventDetails.model_rebuild()
SelectEventFileTool.model_rebuild()
ReadEventFileTool.model_rebuild()
EventEvaluation.model_rebuild()
EvaluateEventTool.model_rebuild()


####################################################################
###################### AGENT CLASS DEFINITION ######################
####################################################################


SYSTEM_PROMPT = f"""You are an AI assistant specialized in finding local events that match user requirements.

Today is {datetime.now().strftime('%A, %B %d, %Y')}.

You have access to these tools:
1. parse_user_query - Extract what the user is looking for (keywords, location, dates)
   Use this FIRST to understand the request.
   Only re-run if you exhaust all other options.

2. search_event_pages - Search event database using keyword embeddings
   Returns multiple results ranked by relevance

3. select_best_event_file - Select the next most relevant event from search results
   Automatically picks the best unreviewed option

4. read_event_file_contents - Read full details of the selected event
   Extracts structured information including dates, location, type

5. evaluate_event_details_against_user_query - Evaluate whether the event details match the user requirements

6. final_answer - Provide the final response to the user
   Can report success, no matches, or partial matches

Each tool requires:
- think: Your reasoning for this action
- action_type: The exact tool name from above

WORKFLOW GUIDANCE:
- Always start by parsing user query with parse_user_query
- Search for the best matching event using search_event_pages, load and evaluate them with evaluate_event_details_against_user_query
- If an event doesn't match, try the next one
- The tools handle complex matching (districts within cities, date flexibility, event type synonyms)
- Continue until you find a match or exhaust reasonable options
- Always provide answers in Polish language

IMPORTANT:
- Trust the evaluation tool's judgment about matches
- Errors will guide you if prerequisites are missing
- Focus on finding what the user actually wants, not just the closest keyword match
"""


class MyAgent:
    """A simple AI agent that can search for event pages."""
    
    def __init__(self, 
                 main_model: str = "gpt-4.1-mini",
                 reading_model: str = "gpt-4.1-mini",
                 parsing_model: str = "gpt-4.1-mini" , 
                 evaluation_model: str = "gpt-4.1-mini",
                 verbose: bool = True):

        self.verbose = verbose
        self.state = StateManager()

        self.deps = DependencyManager(
            client=instructor.from_openai(OpenAI()),
            max_retries=5,
            collection=init_collection(),
            main_model=main_model,
            parsing_model=parsing_model,
            evaluation_model=evaluation_model,
            reading_model=reading_model,
            
            event_dir=EVENT_DIR)
        
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.action_history = []        
        
    def _log(self, message: str):
        """Print if verbose is True."""
        if self.verbose:
            print(message)


    def step(self, user_query: str, max_steps: int = 20) -> str:
        """Process user query through multiple reasoning steps.
        Make independent decisions and use tools autonomously to gather information
        Necessary to answer the user's query.
        
        Args:
            user_query (str): The user's query or request to process.
            max_steps (int): Maximum number of actions to take.
            
        Returns:
            Final answer string
            """

        self.state.original_query = user_query
        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })

        for step_num in range(max_steps):
            self._log(f"\n----- Step {step_num + 1} -----\n")
            try:
                action = self.deps.client.chat.completions.create(
                    model=self.deps.main_model,
                    response_model=AgentActions,
                    messages= self.conversation_history,
                    max_retries=self.deps.max_retries,
                    max_tokens=2048,
                    temperature=0.1
                )
                results = action.execute(state=self.state, deps=self.deps)

                summary = action.summarise(results)
                action_summary = f"Action: {action.action_type}\n\nThink: {action.think}\n\nResult: {summary}\n"
                self._log(action_summary)
                # print(" ")
                # print("="*50)
                # print("STATE")
                # print(self.state.model_dump_json(indent=2))
                # print(" ")
                # print("USER INTENT")
                # print(self.state.user_intent)
                # print(" ")
                # pprint(self.state.search_title_results)
                # print(self.state.selected_page_id)
                # print(" ")
                # print(" READ PAGES")
                # print(self.state.read_event_page_ids)

                self.conversation_history.append({
                    "role": "assistant",
                    "content": action_summary
                })

                self.action_history.append({
                    "step": step_num + 1,
                    "think": action.think,
                    "action_type": action.action_type,
                    "result": results,
                    "summary": summary
                })

                if isinstance(action, FinalAction):
                    answer = action.execute(state=self.state, deps=self.deps)
                    return answer
                
            except Exception as e:                
                self._log(f"Error during action generation: {e}")
                self.conversation_history.append({
                    "role": "assistant",
                    "content": f"Error occured: {str(e)}."
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": f"Error: {str(e)}. Try a different appraoch"
                })
        self._log("\nReached maximum steps.\nRequesting final answer...\n")            
        self.conversation_history.append({
            "role": "user",
            "content": "You've reached the maximum number of steps. Please provide a final answer based on the information gathered."
        })

        try:
            final_action = self.deps.client.chat.completions.create(
                model=self.deps.main_model,
                response_model=FinalAction,
                messages=self.conversation_history,
                max_retries=self.deps.max_retries,
                max_tokens=4096
            )
            answer = final_action.execute(state=self.state, deps=self.deps)
            self._log(f"\nFinal Thought: {final_action.think}")
            return answer
        
        except Exception as e:
            self._log(f"Error during final action generation: {e}")
            return "I apologise, but I encountered an error while trying to provide a final answer. Please try rephrasing your request or being more specific about what you're looking for."

    def get_action_summary(self) -> str:
        """Get a summary of all actions taken."""
        summary_parts = ["Action Summary:\n"]
        
        for action in self.action_history:
            summary_parts.append(f"\nStep {action['step']}: {action['action_type']}")
            summary_parts.append(f"  Thought: {action['think']}")
            summary_parts.append(f"  Result: {action['summary']}\n")
        
        return "\n".join(summary_parts)
    
if __name__ == "__main__":
    
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        try:
            if user_query.lower() == 'exit':
                break
            
            agent = MyAgent(main_model="gpt-4.1-mini", verbose=True)
            final_answer = agent.step(user_query)
            print(f"\nFinal Answer: {final_answer}")
            print("\n")
            # print(agent.get_action_summary())
        except KeyboardInterrupt:
            print("\nExiting...")
            break    