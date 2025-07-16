from __future__ import annotations
import os
import json
from datetime import datetime, date
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
                                              min_length=1)

class ParseUserQueryTool(BaseModel):
    """Parse user query extracting user requirements."""
    think: str = Field(description="Why is this extraction needed and what information is sought")
    action_type: Literal["parse_user_query"] = "parse_user_query"

    def execute(self, state: StateManager, deps: DependencyManager) -> UserIntent:
        """Extract user intent from the user query."""

        SYSTEM_PROMPT_INTENT_EXTRACTION = f"""You are a world-class expert at extracting user intent from the user query in a form of unstructured text.

        The current_date is {date.today().isoformat()}.

        Returns:
        - think: A thought process or reasoning behind the user's intent extraction.
        - query: A refined user query.
        - action_type: The type of action to be performed, which is always "parse_user_query".
        - timeframe: The timeframe for the event search, represented as a datetime object in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        - city: The city where the user wants to find events, represented as a string.
        - location: The location where the user wants to find events, represented as a string.
        - keywords: A list of keywords, strictly related to the users query.

        Date Extraction Rules:
        1. No date/time mentioned -> start_date = current_date; end_date = current_date + 14 days.
        2. Only month is mentioned:
            - If current month matches -> start_date = current_date; end_date = last day of the month.
            - If next month is mentioned -> start_date = first day of that month; end_date = last day of that month.
        3. Specific date mentioned -> start_date = that date.
        4. If a timeframe is mentioned (e.g., "next week", "in two weeks", "this weekend", "next weekend"):
            - start_date = start date of that time frame; end_date = end date of that time frame.
        5. If a date range is mentioned -> extract both start and end date.

        Examples (assuming current_date is 2025-07-14):
            - "zajęcia z badmintona" → start: 2025-07-14, end: None
            - "zajęcia z badmintona w lipcu" → start: 2025-07-14, end: 2025-07-31
            - "zajęcia z badmintona w sierpniu" → start: 2025-08-01, end: 2025-08-31
            - "zajęcia z badmintona 20 lipca" → start: 2025-07-20, end: None
        
        City Extraction Rules:
        1. If no city is mentioned -> city = Warsaw.

        Keyword Extraction Rules:
        1. Prioritise keywords relating to specific entities over general activities:
            - A band name should be prioritised over a general activity like "concert".

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
        summary += f"Extracted user intent:\n"
        summary += f"Timeframe: {result.timeframe.start_date.isoformat()}-{result.timeframe.end_date.isoformat() if result.timeframe.end_date else 'N/A'}\n"
        summary += f"City: {result.city}\n"
        summary += f"Location: {result.location}\n"
        summary += f"Refined User Query: {result.query_refined}\n"
        summary += "Now that the user's requirements are understood, the next step is to use 'search_event_pages' to find relevant events based on the extracted keywords."

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
    """Search for top 10 relevant event pages using title embedding similarity. Use this to get an initial list of potential events. 
       DO NOT use this if you have already selected a specific file and need to read its contents."""
    think: str = Field(description="Why is this search needed and what information is sought")
    action_type: Literal["search_event_pages"] = "search_event_pages"

    def execute(self, state: StateManager, deps: DependencyManager) -> List[EventTitleResult]:

        if not state.user_intent:
            return {"error": "User intent not found in state. Please extract user intent first using parse_user_query tool.",
                    "suggested_action": "parse_user_query"}
        elif not state.user_intent.keywords:
            return {"error": "No keywords found in user intent. Please ensure the user intent extraction included keywords.",
                    "suggested_action": "parse_user_query"}
        keywords = [k.keyword.lower() for k in state.user_intent.keywords]
        final_dict = {}
        results = []

        for keyword in keywords:
            # kw_dict = {}
            kw_results = deps.collection.query(
                query_texts=[keyword],
                n_results=10)

            output = []
            for i in range(len(kw_results['ids'][0])):
                results.append(EventTitleResult(
                    page_id=kw_results['ids'][0][i],
                    title=kw_results['metadatas'][0][i]['title'],
                    distance=kw_results['distances'][0][i]
                ))
            
            # kw_dict["results"] = output
            # kw_dict["min_distance"] = min([res.distance for res in output])


            # final_dict["_".join(keyword.split(" "))] = kw_dict

        state.search_title_results=results

        return results
    
    def summarise(self, results: List[EventTitleResult]) -> str:
        """summarise the search results."""
        # keywords = list(results.keys())

        if "error" in results:
            return f"Error: {results['error']}. Suggested action: {results['suggested_action']}"

        summary = f'Searched for event pages most relevant to the user query'
        summary += "A list of potential events has been found. The next step is to use 'select_best_event_file' to pick the single most promising event to investigate further."
                    
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

class SelectEventFileTool(BaseModel):
    """Select the event file with the smallest distance measure."""
    think: str = Field(description="I should select the event file based on the smallest distance measure.")
    action_type: Literal["select_best_event_file"] = "select_best_event_file"

    def execute(self, state: StateManager, deps: DependencyManager) -> str:
        if not state.search_title_results:
            return {"error": "No search results found. Please perform a search first using search_event_pages.",
                    "suggested_action": "search_event_pages"}

        state.selected_page_id = self._get_page_id(state)
        if not state.selected_page_id:
            return {"error":"All event files have already been read or no suitable files found.",
                    "suggested_action": "parse_user_query to identify new keywords"}
            
        return state.selected_page_id
    
    def summarise(self, result: str) -> str:
        """Create action summary"""
        if "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"
        
        return f"Selected event file with page_id: {result}\nThe next logical step is to use the 'read_event_file_contents' tool to get the details of this file"
    
    def _get_page_id(self, state: StateManager) -> str:
        """Get the page_id of the selected event file."""
        try:
            page_id = min([res for res in state.search_title_results if res.page_id not in state.read_event_page_ids],
                        key=lambda x: x.distance).page_id

            return page_id
        # empty list -> attribute rror return none and use new keyword
        except AttributeError:
            return None

class ReadEventFileTool(BaseModel):
    """Use this tool to read the full contents of a specific event file AFTER it has been chosen by 'select_best_event_file'."""

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

            extraction_prompt = f"""Extract structured event information from this file.

                                    User Requirements:
                                    - timeframe: {state.user_intent.timeframe.start_date.isoformat()} - {state.user_intent.timeframe.end_date.isoformat() if state.user_intent.timeframe.end_date else 'N/A'}

                                    Event File Content:
                                    {raw_content}

                                    Instructions:
                                    - Identify the event type based on the content (concert, exhibition, theater, workshop, festival, etc.)
                                    - Extract all date/time information
                                    - Identify the specific venue/location and city
                                    - Note the district/neighborhood if mentioned
                                    - Create a brief 1-2 sentence summary
                                    - Extract any pricing, audience, or registration information
                                    - Use your knowledge to infer missing information when reasonable
                                    - Always provide answers in Polish language

                                    Date Extraction Rules:
                                    1. If an event has recurring dates -> start_date = the date closest to user- defined timeframe.
                                    2. If the event doesn't have recurring dates, but has a specific date mentioned -> start_date = that date.
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

        summary_str += f"\nThe details for event '{summary['title']}' have been read.\nNow, these details must be compared against the user's original request using the 'evaluate_event_details_against_user_query' tool."
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
    theme_matches: bool
    
    date_evaluation: str = Field(description="Evaluation of date match")
    date_matches: bool
    
    location_evaluation: str = Field(description="Evaluation of location match")
    location_matches: bool
    
    type_evaluation: str = Field(description="Evaluation of event type/keywords match")
    type_matches: bool
    
    overall_reasoning: str = Field(description="Overall reasoning for the decision")
    recommendation: str = Field(description="What to do next - try another event or provide this one")

class EvaluateEventTool(BaseModel):
    """Use this AFTER reading an event's contents with 'read_event_file_contents' to determine if it's a good match for the user.
    Compares the most recently read event (from the state) against the user's initial requirements (also from the state).

    Returns:
        EventEvaluation: The evaluation result containing match confidence, reasoning, and recommendations.
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
                                - Start Date: {state.user_intent.timeframe.start_date.isoformat()}
                                - End Date: {state.user_intent.timeframe.end_date.isoformat() if state.user_intent.timeframe.end_date else 'N/A'}
                                

                                Event Details:
                                - Title: {state.event_details['parsed']['title']}
                                - Date: {state.event_details['parsed']['start_datetime']['date']}
                                - Recurring Dates (if any): {', '.join(state.event_details['parsed'].get('recurring_dates', [])) if state.event_details['parsed'].get('recurring_dates') else 'N/A'}
                                - Location: {state.event_details['parsed']['location']}
                                - City: {state.event_details['parsed']['city']}
                                - Description: {state.event_details['parsed']['description']}                        

                                Consider:
                                1. Theme/Subject Matter: First evaluate the core subject. Is the event about the user's topic of interest?
                                   For example, if the user wants a **"pottery making workshop"**:
                                    - An event called **"Ceramics Glazing Class"** is a strong **THEME match**, as it's about the same craft.
                                    - An event called **"Beginner's Weaving Workshop"** is a **THEME mismatch**, despite being the same event type.
                                2. Geographic knowledge (e.g., districts within cities)
                                3. Date flexibility (e.g., "next few weeks" from user's specified date)
                                4. Event type synonyms and related concepts
                                5. User's likely intent even if not explicitly stated
                                6. Overall Match: An event can be a good overall match even if the type is different, as long as the theme is correct.

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
            summary = f"Event '{getattr(result, 'title', 'N/A')}' matches! ({result.match_confidence:.0%} confident) - {result.overall_reasoning}"
            summary += " The next step is to use 'final_answer' to present this result to the user."
            return summary
        
        else:
            summary = f"Event '{getattr(result, 'title', 'N/A')}' doesn't match - {result.overall_reasoning}."
            summary += "\nThe next step is to use 'select_best_event_file' to pick the next best event from the search results and repeat the process."
            return summary
        
########################
##### FINAL ACTION #####
########################
class FinalAction(BaseModel):
    """
    Provide the final, comprehensive, human-readable answer to the user based on all gathered information.
    This tool  synthesizes the results to construct the best possible response.
    If no perfect match is found, it will suggest the best available alternative.
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
    
        sorting_key=lambda e: (e.get('theme_matches', False),
                               e.get('location_matches', False),
                               e.get('date_matches', False),                               
                               e.get('type_matches', False),
                               e.get('confidence', 0))

        best_event_overall = sorted(state.evaluation_history, key=sorting_key, reverse=True)[0]

        if best_event_overall["matches"].get('matches', False):
            best_page_id = best_event_overall['page_id']
            best_event_details = state.read_event_pages_content_dict[best_page_id]
            answer="After reviewing the options, I found an event that's a great match for your request!\n\n\n"
            answer+=f"\tTitle:  {best_event_details['title']}\n\n"
            answer+=f"\tDate:  {self._format_date(best_event_details['start_datetime']['date'])}\n\n"
            answer+=f"\tEnd Date: {self._format_date(best_event_details['end_datetime']['date'])}\n\n"
            if best_event_details.get('price_info'):
                answer+=f"\tPrice: {best_event_details['price_info']}\n\n"
            answer+=f"\tLocation:  {best_event_details['location']}\n\n"
            answer+=f"\tBrief Summary: {best_event_details['summary']}\n\n"
            answer+=f"\tMore details can be found here: {best_event_details['source_url']}\n\n"
            answer+=f"Why this is a good match:\n{best_event_overall['overall_reasoning']}"

            return answer
        
        else:
            best_page_id = best_event_overall['page_id']
            best_event_details = state.read_event_pages_content_dict[best_page_id]
            answer="I couldn't find a perfect match for your request. However, after reviewing all the options, here is the closest alternative I found:\n\n\n"
            answer+=f"\tTitle: {best_event_details['title']}\n\n"
            answer+=f"\tStart Date: {self._format_date(best_event_details['start_datetime']['date'])}\n\n"
            answer+=f"\tEnd Date: {self._format_date(best_event_details['end_datetime']['date'])}\n\n"
            if best_event_details.get('price_info'):
                answer+=f"\tPrice: {best_event_details['price_info']}\n\n"
            answer+=f"\tLocation: {best_event_details['location']}\n\n"
            answer+=f"\tBrief Summary: {best_event_details['summary']}\n\n"
            answer+=f"\tMore details can be found here: {best_event_details['source_url']}\n\n"
            answer+=f"Please note why this isn't a perfect match:\n{best_event_overall['overall_reasoning']}\n"
            answer+="This might still be of interest to you. If not, you could try rephrasing your request with a different date or keywords."

            return answer

    def _format_date(self, date_str: str) -> str:
        """Format date nicely for display."""
        try:
            from datetime import datetime
            dt_obj = datetime.fromisoformat(date_str)
            return dt_obj.strftime("%A, %B %d, %Y at %I:%M %p")
        except (ValueError, TypeError):
            return date_str

    def summarise(self, result: str) -> str:
        """The result is the final answer, so we just summarize that an answer was provided."""
        return "A final answer has been generated and provided to the user."


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


    def step(self, user_query: str, max_steps: int = 30) -> str:
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
                    max_tokens=4096,
                    temperature=0.1
                )
                results = action.execute(state=self.state, deps=self.deps)

                summary = action.summarise(results)
                action_summary = f"Action: {action.action_type}\n\nThink: {action.think}\n\nResult: {summary}\n"
                self._log(action_summary)
                # print(self.state.model_dump_json(indent=2))
                # print(self.state.user_intent)
                pprint(self.state.search_title_results)
                print(self.state.selected_page_id)
                print(self.state.read_event_page_ids)

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