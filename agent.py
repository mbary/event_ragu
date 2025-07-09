from __future__ import annotations
import os
import json
from datetime import datetime, date
from typing import Union, List, Literal, Optional, Dict, Any, Set
from collections import OrderedDict
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


# collection = init_collection()

###################################################################
################### TOOL AND OUTPUT DEFINITIONS ###################
###################################################################

###########################################################
######### STATE MANAGER AND COLLECTION DEFINITION #########
###########################################################

class StateManager(BaseModel):
    """State manager to keep track of the conversation, extracted values and actions taken."""
    # Returned by UserIntent
    
    original_query: Optional[str] = Field(description="The original user query that initiated the conversation"
                                          , default=None)

    user_intent: Optional[UserIntent] = Field(description="The user's intent extracted from the original query", default=None)
    
    # Returned by SearchEventPageTitlesTool
    search_title_results: Dict = Field(description="List of event pages found using title embedding similarity search",
                                       default_factory=dict)
    current_search_keyword: Optional[str] = Field(description="The current keyword being searched for in the event pages"
                                                  , default=None)
    exhausted_search_keywords: Set[str] = Field(description="List of keywords that have been searched for and exhausted",
                                                 default_factory=set)
    selected_page_id: Optional[str] = Field(description="The current page_id being processed"
                                            , default=None)

    read_event_pages: Set = Field(description="List of event page_ids that have been read", default_factory=set)
    event_details: Optional[Dict[str, Any]] = Field(description="Structured event information extracted from the file"
                                                    , default=None)
    evaluated_page_ids: List = Field(description="List of event pages that have been evaluated for relevance against the user intent",
                                        default_factory=list)
    last_evaluation: EventEvaluation = Field(description="The last evaluation result of the event against user intent"
                                                       , default=None)
    evaluation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All evaluated events with details and confidence scores"
    )



class DependencyManager(BaseModel):
    """A class to manage shared dependencies and configurations for the agent."""
    client: Any = Field(description="The instructor client used for LLM interactions")
    collection: chromadb.Collection = Field(description="ChromaDB collection for storing event pages and their embeddings")
    model: str = Field(description="The LLM model to use for interactions")
    event_dir: Path = Field(description="Directory where event files are stored")
    max_retries: int = Field(default=5, description="Maximum number of retries for LLM calls")


#########################
###### User Intent ######
#########################

class UserIntentDateTime(BaseModel):
    """Class represents the datetime information extracted from the user query."""
    timeframe: date = Field(
            description="The datetime extracted from the user query in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).",
            examples=["2025-10-01", "2025-10-01T18:00:00Z", "2028-04-26"])
    confidence: float = Field(ge=0, le=1,
                                description="Confidence score of the datetime extraction (0-1).")
    
class UserIntentKeyWord(BaseModel):
    """Class represents a keyword and associated confidence extracted from the user query."""
    keyword: str = Field(description="A keyword extracted from the user query.")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the keyword extraction (0-1).")
    
class UserIntent(BaseModel):
    """Represents the user's intent for the event search."""
    think: str = Field(
        description="A thought process or reasoning behind the user's intent extraction.",
        examples=["What does the user intend to do?"])
    
    # action_type: Literal["parse_user_query"] = "parse_user_query"
    query_refined: str = Field(description="A refined user query.")

    timeframe: UserIntentDateTime = Field(description="The timeframe for the event search", 
                                          examples=["2025-10-01", "2025-10-01T18:00:00Z", "2028-04-26"])

    city: str = Field(description="The city where the user wants to find events."
                    #   , examples=["Krakow", "Gdansk","Warsaw"]
                      )

    location: Optional[str] = Field(description="The location where the user wants to find events.", 
                                    example=["Ursus", "Stadion Narodowy", 
                                            "Centrum Nauki Kopernik",
                                            "Park","Teatr","Opera"])

    keywords: List[UserIntentKeyWord] = Field(description="A list of keywords, specifically related to the event, to refine the search.",
                                              examples=["concert", "exhibition", "theater", "art", "music"], 
                                              max_length=5, 
                                              min_length=1)

class ParseUserQueryTool(BaseModel):
    """Parse user query extracting user requirements.
    
    Args:
        user_query (str): The user's query to extract intent from.
    
    Returns:
        UserIntent: The extracted user requirements containing keywords, timeframe, city, and location.

    Example:
        "Gdzie mogę najwczesniej oddac krew w warszawie?" -> {"timeframe": "2025-10-01T00:00:00Z", "city": "Warszawa", "keywords": ["oddac krew", "warszawa"]}
    """
    think: str = Field(description="Why is this extraction needed and what information is sought")
    action_type: Literal["parse_user_query"] = "parse_user_query"

    def execute(self, state: StateManager, deps: DependencyManager) -> UserIntent:
        """Extract user intent from the user query."""

        SYSTEM_PROMPT_INTENT_EXTRACTION = f"""You are a world-class expert at extracting user intent from the user query in a form of unstructured text.

        Current date is {date.today().isoformat()}.

        Returns:
        - think: A thought process or reasoning behind the user's intent extraction.
        - query: A refined user query.
        - action_type: The type of action to be performed, which is always "parse_user_query".
        - timeframe: The timeframe for the event search, represented as a datetime object in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
        - city: The city where the user wants to find events, represented as a string.
        - location: The location where the user wants to find events, represented as a string.
        - keywords: A list of keywords, strictly related to the users query.

        If no the specified datetime is vague, always relate it to the current date.
        All your responses **MUST** be in Polish language.
        You should always think step by step.
        """

        user_query = state.original_query
        
        intent = deps.client.chat.completions.create(
            model="gpt-4.1-mini",
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
            return {"error": "No user intent extracted. Please check the user query."}
        
        summary = f"Think: {result.think}\n"
        summary += f"Extracted user intent:\n"
        summary += f"Timeframe: {result.timeframe.timeframe.isoformat()}\n"
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
       DO NOT use this if you have already selected a specific file and need to read its contents.

    Args:
        keywords (List[str]): List of keywords to search for in event titles.
    Returns:
        Dict[str, Dict[str, Union[float, List[EventTitleResult]]]]: A dictionary with keywords as keys and a dictionary of results as values.
    Example:
        {'bieg': {'min_distance': 0.396332323551178,
                    'results': [EventPageResult(page_id='biegaj_z_team_zabieganedni_00128', title='biegaj z team zabieganedni', distance=0.396332323551178),
                                EventPageResult(page_id='bieg_po_nowe_życie_00239', title='bieg po nowe życie', distance=0.43145501613616943)
                                ]
                    }
        }
    """
    think: str = Field(description="Why is this search needed abd what information is sought")
    action_type: Literal["search_event_pages"] = "search_event_pages"

    def execute(self, state: StateManager, deps: DependencyManager) -> Dict[str, Dict[str, Union[float, List[EventTitleResult]]]]:

        if not state.user_intent:
            return {"error": "User intent not found in state. Please extract user intent first using parse_user_query tool.",
                    "suggested_action": "parse_user_query"}
        elif not state.user_intent.keywords:
            return {"error": "No keywords found in user intent. Please ensure the user intent extraction included keywords.",
                    "suggested_action": "parse_user_query"}
        keywords = [k.keyword for k in state.user_intent.keywords]
        # print(keywords)
        final_dict = {}
        # print("="*30)
        # print("KEYWORD TYPE KURWA")
        # print(type(keywords))
        for keyword in keywords:
            print(f"Searching for keyword: {keyword}")
            kw_dict = {}
            kw_results = deps.collection.query(
                query_texts=[keyword],
                n_results=10)

            output = []
            for i in range(len(kw_results['ids'][0])):
                output.append(EventTitleResult(
                    page_id=kw_results['ids'][0][i],
                    title=kw_results['metadatas'][0][i]['title'],
                    distance=kw_results['distances'][0][i]
                ))
            
            kw_dict["results"] = output
            kw_dict["min_distance"] = min([res.distance for res in output])


            final_dict["_".join(keyword.split(" "))] = kw_dict
        # print("="*30)
        # print("FINAL DICT")
        # pprint(final_dict)

        # Update current stete
        state.search_title_results.update(final_dict)

        return final_dict
    
    def summarise(self, results: Dict[str, Dict[str, Union[float, List[EventTitleResult]]]]) -> str:
        """summarise the search results."""
        keywords = list(results.keys())

        if "error" in results:
            return f"Error: {results['error']}. Suggested action: {results['suggested_action']}"

        summary = f'To search for event pages most relevant to the user query, I used the following keywords: {", ".join(keywords)}.\n'
        summary += "Each keyword yielded the following results:\n"
        for keyword in keywords:
            summary += f"Keyword: {keyword}\n"
            summary += f"Minimum Distance: {results[keyword]['min_distance']}\n"
            summary += f"File count: {len(results[keyword]['results'])}\n"

        summary += "A list of potential events has been found. The next step is to use 'select_best_event_file' to pick the single most promising event to investigate further."
                    
        return summary
    

###########################################
############ GET EVENT DETAILS ############
###########################################

class EventDetailsStart(BaseModel):
    """Represents the start date and time of an event."""
    date: datetime = Field(description="The datetime extracted from the event file in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).",
                        examples=["2025-10-01", "2025-10-01T18:00:00Z", "2028-04-26"])
    confidence: float = Field(ge=0, le=1, description="Confidence score of the datetime extraction (0-1).")

class EventDetailsEnd(BaseModel):
    """Represents the end date and time of an event."""
    date: datetime = Field(description="The datetime extracted from the event file in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).",
                        examples=["2025-10-01", "2025-10-01T18:00:00Z", "2028-04-26"])
    confidence: float = Field(ge=0, le=1, description="Confidence score of the datetime extraction (0-1).")

class EventDetails(BaseModel):
    """Structured event information extracted from file"""
    title: str = Field(description="Event title")
    event_type: str = Field(description="Type of event (concert, exhibition, theater, festival, workshop, etc.)")
    
    start_datetime: EventDetailsStart
    end_datetime: EventDetailsEnd
    
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
    """Select the event file with the smallest distance measure.
    
    Args:
        results_dict (Dict[str, Union[float, SearchEventPagesOutput]]): Dictionary containing the results of the search.
    
    Returns:
        page_id (str): The page_id of the selected event file.
    """
    think: str = Field(description="I should select the event file based on the smallest distance measure.")
    action_type: Literal["select_best_event_file"] = "select_best_event_file"

    def execute(self, state: StateManager, deps: DependencyManager) -> str:
        print("="*30)
        print("Selecting event file based on smallest distance measure.")

        if not state.search_title_results:
            return {"error": "No search results found. Please perform a search first using search_event_pages.",
                    "suggested_action": "search_event_pages"}
        
        if not state.current_search_keyword:
            state.current_search_keyword = min(state.search_title_results, key=lambda x: state.search_title_results[x]["min_distance"])

        state.selected_page_id = self._get_page_id(state.current_search_keyword, state)

        if not state.selected_page_id:
            state.exhausted_search_keywords.append(state.current_search_keyword)
            state.current_search_keyword = self._get_next_keyword(state)

            if not state.current_search_keyword:
                return {"error": "No more keywords to search. Generate a new query or proceed to provide the final answer.",
                        "suggested_action": "final_action or parse_user_query"}

        return state.selected_page_id
    
    def summarise(self, result: str) -> str:
        """Create action summary"""
        if "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"
        
        return f"Selected event file with page_id: {result}\nThe next logical step is to use the 'read_event_file_contents' tool to get the details of this file"
    
    def _get_page_id(self, keyword: str, state: StateManager) -> str:
        """Get the page_id of the selected event file."""
        try:
            page_id = min([res for res in state.search_title_results[keyword]["results"] if res.page_id not in state.read_event_pages],
                        key=lambda x: x.distance).page_id

            return page_id
        # empty list -> attribute rror return none and use new keyword
        except AttributeError:
            return None
        
    def _get_next_keyword(self, state: StateManager) -> str:
        state.exhausted_search_keywords.add(state.current_search_keyword)
        keyword = min((key for key in state.search_title_results if key not in state.exhausted_search_keywords), 
                                  key=lambda x: state.search_title_results[x]["min_distance"])

        return keyword


class ReadEventFileTool(BaseModel):
    """Use this tool to read the full contents of a specific event file AFTER it has been chosen by 'select_best_event_file'.

    Args:
        page_id (str): The page_id of the event file selected by select_best_event_file.

    Returns:
        Dict[str, Any]: A dictionary containing the structured event information extracted from the file.

    Example:
        "concert_00123"->   {
                                "page_id": "concert_00123",
                                "raw_content": "Full content of the event file",
                                "parsed": {
                                    "title": "Concert Title",
                                    "event_type": "concert",
                                    "start_datetime": {"date": "2025-10-01T18:00:00Z", "confidence": 0.95},
                                    "end_datetime": {"date": "2025-10-01T20:00:00Z", "confidence": 0.90},
                                    "location": "Venue Name",
                                    "city": "Warsaw",
                                    "district": "Mokotów",
                                    "description": "Detailed description of the event.",
                                    "summary": "Brief summary of the event.",
                                    "target_audience": "General public",
                                    "price_info": "$20 - $50",
                                    "source_url": None
                                },
                                "file_path": "/path/to/event_file.md"
                            }
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

            extraction_prompt = f"""Extract structured event information from this file.

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
                                    """

            parsed_event = deps.client.chat.completions.create(
                model="gpt-4.1-mini",
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
                    "brief": event_dict.get('summary', event_dict['description'][:100] + "...")
                }
            }
            if event_dict.get('price_info'):
                summary_data["summary"]["price"] = event_dict['price_info']
            if event_dict.get('target_audience'):
                summary_data["summary"]["audience"] = event_dict['target_audience']

            state.read_event_pages.add(page_id)
            
            return summary_data
            
        except FileNotFoundError:
            return {"error": f"Event file not found: {page_id}.md",
                    "suggeste_action":"Select another file using select_best_event_file"}
        except Exception as e:
            # raise e
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
            f"Summary: {summary['brief']}"
        ]

        if summary.get('price'):
            parts.append(f"- {summary['price']}")
        
        summary_str = " ".join(parts)

        summary_str += f"\nThe details for event '{summary['title']}' have been read.\nNow, these details must be compared against the user's original request using the 'evaluate_event_details_against_user_query' tool."
        return summary_str


class EventEvaluation(BaseModel):
    """Structured evaluation result"""
    matches: bool = Field(description="Overall match determination")
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
    
    Args:
        user_requirements (UserIntent): The user's requirements extracted from the original query.
        event_details (Dict[str, Any]): The structured event information extracted from the file.

    Returns:
        EventEvaluation: The evaluation result containing match confidence, reasoning, and recommendations.
    
    Example:
        { "matches": True,
          "match_confidence": 0.85,
          "theme_evaluation": "The event theme aligns with the user's interests.",
          "theme_matches": True,
          "date_evaluation": "The event date is within the user's specified timeframe.",
          "date_matches": True,
          "location_evaluation": "The event is in the user's specified city.",
          "location_matches": True,
          "type_evaluation": "The event type matches the user's interests.",
          "type_matches": True,
          "overall_reasoning": "The event matches the user's requirements based on date, location, and type.",
          "recommendation": "Proceed with this event as it matches the user's requirements."
        }
    """
    action_type: Literal["evaluate_event_details_against_user_query"] = "evaluate_event_details_against_user_query"
    think: str = Field(description="What aspects need careful evaluation")
    
    def execute(self, state: StateManager, deps: DependencyManager) -> Dict[str, Any]:
        # Validation
        if not state.event_details:
            return {"error": "No event details found. Please read an event file first.", 
                    "suggested_action": "read_event_file_contents"}
        if not state.user_intent:
            return {"error": "No user intent found. Cannot evaluate without requirements.",
                    "suggested_action": "parse_user_query"}
        
        # Prepare context for LLM
        evaluation_prompt = f"""Evaluate if this event matches the user's requirements.
                                User Requirements:
                                - Query: {state.user_intent.query_refined}
                                - Looking for: {', '.join([k.keyword for k in state.user_intent.keywords])}
                                - City: {state.user_intent.city}
                                - Date: {state.user_intent.timeframe.timeframe.isoformat()}

                                Event Details:
                                - Title: {state.event_details['parsed']['title']}
                                - Date: {state.event_details['parsed']['start_datetime']['date']}
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

                                Be somewhat flexible but not overly permissive."""

        try:
            evaluation = deps.client.chat.completions.create(
                model="gpt-4.1-mini",
                response_model=EventEvaluation,
                messages=[
                    {"role": "system", "content": "You are evaluating if events match user requirements. Use your knowledge of geography, dates, and event types."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_retries=deps.max_retries,
                temperature=0.1)
            
            result = evaluation.model_dump()
            result["page_id"] = state.event_details["page_id"]
            result["event_title"] = state.event_details['parsed']['title']

            state.last_evaluation = result
            state.evaluated_page_ids.append(state.event_details["page_id"])
            state.evaluation_history.append({
                                            "page_id": state.event_details["page_id"],
                                            "title": state.event_details["parsed"]["title"],
                                            "confidence": result["match_confidence"],
                                            "reasons": result["overall_reasoning"],
                                            "date": state.event_details["parsed"]["start_datetime"]["date"],
                                            "location": f"{state.event_details['parsed']['location']}, {state.event_details['parsed']['city']}",
                                            "matches": result["matches"],
                                            "theme_matches": result["theme_matches"],
                                            "type_matches": result["type_matches"],
                                            "date_matches": result["date_matches"],
                                            "location_matches": result["location_matches"],
                                            "type_matches": result["type_matches"],
                                            })
            return result 
            
        except Exception as e:
            # raise e
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def summarise(self, result: Dict[str, Any]) -> str:
        """Create conversation summary"""
        if "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"
        
        if result["matches"]:
            summary = f"Event '{result['event_title']}' matches! ({result['match_confidence']:.0%} confident) - {result['overall_reasoning']}"
            summary += " The next step is to use 'final_answer' to present this result to the user."
            return summary
        
        else:
            summary = f"Event '{result['event_title']}' doesn't match - {result['overall_reasoning']}."
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
                answer+=f"**I was looking for:** An event related to '{', '.join(keywords)}' in {intent.city} around {intent.timeframe.timeframe.strftime('%B %Y')}."
                
            answer+="\nThere may be no events of this type listed, or you could try searching with different keywords."
            return answer
    
        sorting_key=lambda e: (e.get('theme_matches', False),
                               e.get('location_matches', False),
                               e.get('date_matches', False),                               
                               e.get('type_matches', False),
                               e.get('confidence', 0))

        best_event_overall = sorted(state.evaluation_history, key=sorting_key, reverse=True)[0]

        if best_event_overall.get('matches', False):
            answer="After reviewing the options, I found an event that's a great match for your request!\n"
            answer+=f"**{best_event_overall['title']}**\n"
            answer+=f"**Date:** {self._format_date(best_event_overall['date'])}"
            answer+=f"**Location:** {best_event_overall['location']}\n"
            answer+=f"More details can be found here: {best_event_overall['source_url']}\n\n"
            answer+=f"**Why this is a good match:**\n*{best_event_overall['reasons']}*"

            return answer
        
        else:
            answer="I couldn't find a perfect match for your request. However, after reviewing all the options, here is the closest alternative I found:\n"
            answer+=f"**{best_event_overall['title']}**\n"
            answer+=f"**Date:** {self._format_date(best_event_overall['date'])}"
            answer+=f"**Location:** {best_event_overall['location']}\n"
            answer+=f"More details can be found here: {best_event_overall['source_url']}\n\n"
            answer+=f"**Please note why this isn't a perfect match:**\n*{best_event_overall['reasons']}*\n"
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


keys = ["user_intent","current_search_keyword","exhausted_search_keywords","selected_page_id","read_event_pages","event_details","last_evaluation","evaluated_page_ids","evaluation_history"]

# keys = ["user_intent","current_search_keyword","exhausted_search_keywords","selected_page_id","read_event_pages","last_evaluation","evaluated_page_ids","evaluation_history"]

def get_subset(chuj, keys):
    data = chuj.model_dump()
    return {k: data[k] for k in keys}



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

WORKFLOW GUIDANCE:
- Always start by parsing user query with parse_user_query
- Search for the best matching event using search_event_pages, load and evaluate them with evaluate_event_details_against_user_query
- If an event doesn't match, try the next one
- The tools handle complex matching (districts within cities, date flexibility, event type synonyms)
- Continue until you find a match or exhaust reasonable options
- Always reason and provide answers in Polish

IMPORTANT:
- Trust the evaluation tool's judgment about matches
- Errors will guide you if prerequisites are missing
- Focus on finding what the user actually wants, not just the closest keyword match
"""


class MyAgent:
    """A simple AI agent that can search for event pages."""
    
    def __init__(self, model: str = "gpt-4.1-mini", verbose: bool = True):

        self.verbose = verbose
        self.state = StateManager()

        self.deps = DependencyManager(
            client=instructor.from_openai(OpenAI()),
            max_retries=5,
            collection=init_collection(),
            model=model,
            event_dir=EVENT_DIR)
        
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.action_history = []        
        
    def _log(self, message: str):
        """Print if verbose is True."""
        if self.verbose:
            print(message)


    def step(self, user_query: str, max_steps: int = 15) -> str:
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
                    model=self.deps.model,
                    response_model=AgentActions,
                    messages= self.conversation_history,
                    max_retries=self.deps.max_retries,
                    max_tokens=4096,
                    temperature=0.1
                )

                self._log(f"Action: {action.action_type}\n")
                self._log(f"Thought: {action.think}")

                results = action.execute(state=self.state, deps=self.deps)

                summary = action.summarise(results)
                action_summary = f"Action: {action.action_type}\nThink: {action.think}\nResult: {summary}"
                # action_summary = f"Action: {action.action_type}\nResult: {summary}"
                print(" ")
                print("="*30)
                print("ACTION SUMMARY")
                print(f"\n\n{action_summary}\n\n")
                print("="*30)
                print(" ")
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
                    print("="*30)
                    # self._log(f"\nFinal Answer: {answer}")
                    self._log(f"Thought: {action.think} ")
                    print("="*30)
                    return answer

                pprint(get_subset(self.state,keys))
            except Exception as e:
                self._log(f"Error during action generation: {e}")
                # raise e
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
                model=self.deps.model,
                response_model=FinalAction,
                messages=self.conversation_history,
                max_retries=self.deps.max_retries,
                max_tokens=4096
            )
            answer = final_action.execute(state=self.state, deps=self.deps)
            self._log(f"\nFinal Thought: {final_action.think}")
            # self._log(f"Final Answer: {answer}")
            return answer
        
        except Exception as e:
            self._log(f"Error during final action generation: {e}")
            # raise e
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
            
            agent = MyAgent(model="gpt-4.1-mini", verbose=True)
            final_answer = agent.step(user_query)
            print(f"\nFinal Answer: {final_answer}")
            print("\n")
            # print(agent.get_action_summary())
        except KeyboardInterrupt:
            print("\nExiting...")
            break    