""""
This file contains the agent class and tools for processing user queries related to events.
Tools include:
- UserIntent: Extracts user intent from the query, including keywords, timeframe, city, and location.
- SearchEventPageTitlesTool: Searches for event pages using title embedding similarity.
- ReadEventFileContentsTool: Reads the contents of an event file based on the page_id.
- FinalAction: Provides the final answer to the user based on the event details.

Tools to be implemented:
- EvaluateEventRelevanceTool: Evaluates the relevance of an event to the user's intent.
- SelectEventFileTool: Selects the event file based on the page_id with the smallest distance measure. (this will be called within another tool, not by the agent directly)
- StateManager: Manages the state of the conversation, including user intent, search results,
  and read event pages and all other actions taken by the agent.
- AgentActions: Union of all tools that the agent can use to process the user query. 
    - Update with new tools as they are implemented.
- DependencyManager: Contains shared dependencies and configurations for the agent.
    - chromadb: Database for storing event pages and their embeddings.
    - instructor: Library for interacting with the OpenAI API and managing chat completions.
    - init_collection: Function to initialize the ChromaDB collection for event pages.
    - SYSTEM_PROMPT: System prompt for the agent, providing context and instructions for processing user queries.
        - Other prompts that might be used for specific tools or actions.


Update Current Tools with:
- Execute functions for each tool that will be called by the agent:
    - each function will take state and shared dependencies as arguments
      and use them accordingly (select relevant data from the state, call the database, etc.)
    - include error handling for cases when the state is not set up correctly which ensures that the agent
        can handle unexpected situations gracefully.

- Summarise function:
    - function whose purpose is to summarise the results of the tool execution
      which will then be added to the conversation history so that the agent is aware of the results
      and can use it in the next step.
            # After execution
            result = action.execute(state=self.state)

            # Use the tool's own summarise method
            summary = f"Action: {action.action_type}
            Think: {action.think}
            Result: {action.summarise(result)}"

            self.conversation_history.append({
                "role": "assistant",
                "content": summary
            })
    - include error handling 
"""



import os
import json
from datetime import datetime, date
from typing import Union, List, Literal, Optional, Dict, Any, Set
from collections import OrderedDict

import chromadb
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
from setup_db import init_collection


EVENT_DIR = "data/events"


# collection = init_collection()

###################################################################
################### TOOL AND OUTPUT DEFINITIONS ###################
###################################################################

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
    
    action_type: Literal["extract_user_intent"] = "extract_user_intent"
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

class ExtractUserIntentTool(BaseModel):
    """Extract user intent from the user query.
    
    Args:
        user_query (str): The user's query to extract intent from.
    
    Returns:
        UserIntent: The extracted user intent containing keywords, timeframe, city, and location.
    """
    think: str = Field(description="Why is this extraction needed and what information is sought")
    action_type: Literal["extract_user_intent"] = "extract_user_intent"

    def execute(self, state: StateManager, deps: DependencyManager) -> UserIntent:
        """Extract user intent from the user query."""

        SYSTEM_PROMPT_INTENT_EXTRACTION = f"""You are a world-class expert at extracting user intent from the user query in a form of unstructured text.

        Current date is {date.today().isoformat()}.

        Returns:
        - think: A thought process or reasoning behind the user's intent extraction.
        - query: A refined user query.
        - action_type: The type of action to be performed, which is always "extract_user_intent".
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
    """Execute the search and return for 10 results using title embedding similarity.
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
            return {"error": "User intent not found in state. Please extract user intent first using extract_user_intent tool.",
                    "suggested_action": "extract_user_intent"}
        elif not state.user_intent.keywords:
            return {"error": "No keywords found in user intent. Please ensure the user intent extraction included keywords.",
                    "suggested_action": "extract_user_intent"}
        keywords = state.user_intent.keywords
        final_dict = {}

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
    """Execute the selection of the event file based on the page_id with the smallest distance measure.
    
    Args:
        results_dict (Dict[str, Union[float, SearchEventPagesOutput]]): Dictionary containing the results of the search.
    
    Returns:
        str: The page_id of the selected event file.
    """
    think: str = Field(description="I should select the event file based on the smallest distance measure.")
    action_type: Literal["select_event_file"] = "select_event_file"

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
                        "suggested_action": "final_action or extract_user_intent"}

        return state.selected_page_id
    
    def summarise(self, result: str) -> str:
        """Create action summary"""
        if "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"
        
        return f"Selected event file with page_id: {result}"
    
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
    """Read event file contents and extract structured information using AI.
    
    Prerequisites: Must have selected a page_id using select_event_file.
    After this: Use evaluate_event to check if it matches requirements.
    """
    action_type: Literal["read_event_file"] = "read_event_file"
    think: str = Field(description="Why reading this specific event file")
    
    def execute(self, state: StateManager, deps: DependencyManager) -> Dict[str, Any]:
        if not state.selected_page_id:
            return {"error": "No page_id selected. Please select a file first using select_event_file.",
                    "suggested_action": "select_event_file"}
        
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
                    "suggeste_action":"Select another file using select_event_file"}
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
            f"Summary: {summary['brief']}"
        ]

        if summary.get('price'):
            parts.append(f"- {summary['price']}")
        
        return " ".join(parts)


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
    
    date_evaluation: str = Field(description="Evaluation of date match")
    date_matches: bool
    
    location_evaluation: str = Field(description="Evaluation of location match")
    location_matches: bool
    
    type_evaluation: str = Field(description="Evaluation of event type/keywords match")
    type_matches: bool
    
    overall_reasoning: str = Field(description="Overall reasoning for the decision")
    recommendation: str = Field(description="What to do next - try another event or provide this one")

class EvaluateEventTool(BaseModel):
    """Evaluate if the current event matches user requirements using LLM intelligence.
    
    Prerequisites: Must have read event details using read_event_file.
    
    Args:
        user_intent (UserIntent): The user's intent extracted from the query.
        event_details (EventDetails): The details of the event to evaluate.

    Returns:
        EventEvaluation: The evaluation result containing match confidence, reasoning, and recommendations.
    
    Example:
        { "matches": True,
          "match_confidence": 0.85,
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
    action_type: Literal["evaluate_event"] = "evaluate_event"
    think: str = Field(description="What aspects need careful evaluation")
    
    def execute(self, state: StateManager, deps: DependencyManager) -> Dict[str, Any]:
        # Validation
        if not state.event_details:
            return {"error": "No event details found. Please read an event file first.", 
                    "suggested_action": "read_event_file"}
        if not state.user_intent:
            return {"error": "No user intent found. Cannot evaluate without requirements.",
                    "suggested_action": "extract_user_intent"}
        
        # Prepare context for LLM
        evaluation_prompt = f"""Evaluate if this event matches the user's requirements.
                                User Requirements:
                                - Query: {state.user_intent.get('query', 'Not specified')}
                                - Looking for: {', '.join([kw['keyword'] for kw in state.user_intent.get('keywords', [])])}
                                - City: {state.user_intent.get('city', 'Not specified')}
                                - Date: {state.user_intent.get('timeframe', {}).get('timeframe', 'Not specified')}

                                Event Details:
                                - Title: {state.event_details['parsed']['title']}
                                - Date: {state.event_details['parsed']['start_datetime']['date']}
                                - Location: {state.event_details['parsed']['location']}
                                - City: {state.event_details['parsed']['city']}
                                - Description: {state.event_details['parsed']['description']}

                                Consider:
                                1. Geographic knowledge (e.g., districts within cities)
                                2. Date flexibility (e.g., "next few weeks" from user's specified date)
                                3. Event type synonyms and related concepts
                                4. User's likely intent even if not explicitly stated

                                Be somewhat flexible but not overly permissive."""

        try:
            evaluation = deps.client.chat.completions.create(
                model="gpt-4.1-mini",
                response_model=EventEvaluation,
                messages=[
                    {"role": "system", "content": "You are evaluating if events match user requirements. Use your knowledge of geography, dates, and event types."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1)
            
            result = evaluation.model_dump()
            result["page_id"] = state.event_details["page_id"]
            result["event_title"] = state.event_details['parsed']['title']

            state.last_evaluation = result
            state.evaluated_page_ids.append(state.event_details["page_id"])
            state.evaluation_history.append({
                                            "page_id": state.event_details["page_id"],
                                            "title": event["title"],
                                            "matches": result["matches"],
                                            "confidence": result["confidence"],
                                            "reasons": result.get("reasons", []),
                                            "date": event["start_datetime"]["date"],
                                            "location": f"{event['location']}, {event['city']}"
                                            })
            return result
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def summarise(self, result: Dict[str, Any]) -> str:
        """Create conversation summary"""
        if "error" in result:
            return f"Error: {result['error']}\nSuggested Action: {result['suggested_action']}"
        
        if result["matches"]:
            return f"Event '{result['event_title']}' matches! ({result['confidence']:.0%} confident) - {result['overall_reasoning']}"
        else:
            return f"Event '{result['event_title']}' doesn't match - {result['overall_reasoning']}"
    

# class ReadEventFileContentsTool(ReadEventFileContentsInput):
#     """Read the contents of an event file based on the page_id.
#     Returns:
#         ReadEventFileContentsOutput: Output containing the contents of the event file."""
    
#     def execute(self) -> EventDetails:
#         """Execute the reading of the event file contents.
        
#         Returns:
#             ReadEventFileContentsOutput: Output containing the contents of the event file.
#         """
#         print("="*30)
#         print(f"Reading event file for page_id: {self.page_id}")
#         file_path = os.path.join(EVENT_DIR, f"{self.page_id}.md")
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Event file {file_path} does not exist.")
        
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = f.read()
        
#         return data
#         # return ReadEventFileContentsOutput(
#         #     page_id=self.page_id,
#         #     contents=data
#         # )



########################
##### FINAL ACTION #####
########################

# class FinalAction(BaseModel):
#     """Provide the final answer to the user
#     Returns:
#         EventDetails: The final comprehensive answer to the user's query, including event details like title, dates, location, and description.
#     """
#     action_type: Literal["finish"] = "finish"
#     think: str = Field(description="Final reasoning before providing the answer")
#     answer: str = Field(description="Final comprehensive answer to the user's query")
#     # answer: str = Field(description="Final comprehensive answer to the user's query")
#     confidence: float = Field(ge=0, le=1, description="Confidence score of the final answer (0-1)")

#     def execute(self) -> str:
#         """Execute the final action and return the answer.
        
#         Returns:
#             str: Final answer to the user's query.
#         """
#         return self.answer

class FinalAction(BaseModel):
    """Provide the final answer based on all gathered information.
    
    Can handle:
    - Successfully found matching event(s)
    - No matches found (with explanation)
    - Partial matches with caveats
    - Multiple matching events
    """
    action_type: Literal["final_answer"] = "final_answer"
    think: str = Field(description="Reasoning about what type of answer to provide")
    answer_type: Literal["found", "not_found", "partial_match", "multiple_matches"] = Field(
        description="Type of answer to generate based on search results"
    )
    include_alternatives: bool = Field(
        default=False,
        description="Whether to mention other events that were close matches"
    )
    
    def execute(self, state: StateManager, deps: DependencyManager) -> Dict[str, Any]:
        """Generate comprehensive answer from collected state."""
        
        has_match = state.last_evaluation and state.last_evaluation.get("matches", False)
        has_event_details = state.event_details is not None
        num_evaluated = len(state.evaluated_page_ids)
        
        if self.answer_type == "found" and not has_match:
            return {"error": "Cannot provide 'found' answer without a matching event"}
        
        if self.answer_type == "found" and not has_event_details:
            return {"error": "Cannot provide 'found' answer without event details"}
        
        if self.answer_type == "found":
            answer = self._build_found_answer(state)
            confidence = state.last_evaluation.get("confidence", 0.8)
            
        elif self.answer_type == "not_found":
            answer = self._build_not_found_answer(state, num_evaluated)
            confidence = 0.9  
            
        elif self.answer_type == "partial_match":
            if not has_event_details:
                answer = self._build_fallback_answer(state)
                confidence = 0.3
            else:
                answer = self._build_partial_match_answer(state)
                confidence = state.last_evaluation.get("confidence", 0.5) if state.last_evaluation else 0.5
            
        elif self.answer_type == "multiple_matches":
            answer = self._build_multiple_matches_answer(state)
            confidence = 0.85
            
        else:
            answer = self._build_fallback_answer(state)
            confidence = 0.3
        
        return {
            "final_answer": answer,
            "answer_type": self.answer_type,
            "search_completeness": confidence,
            "events_evaluated": num_evaluated,
            "has_match": has_match,
            "search_keywords": [kw["keyword"] for kw in state.user_intent.get("keywords", [])] if state.user_intent else []
        }
    
    def _build_found_answer(self, state: StateManager) -> str:
        """Build answer for successfully found event."""
        event = state.event_details["parsed"]
        evaluation = state.last_evaluation
        
        answer_parts = [f"I found a great match for you!\n\n",
                        f"**{event['title']}**\n",
                        f"Date: {self._format_date(event['start_datetime']['date'])}\n",
                        f"Location: {event['location']}, {event['city']}"]
        
        if event.get('district'):
            answer_parts.append(f" ({event['district']})")
        answer_parts.append("\n")
        
        answer_parts.append(f"Type: {event['event_type']}\n")
    
        if event.get('price_info'):
            answer_parts.append(f"Price: {event['price_info']}\n")
        
        answer_parts.append(f"\nDescription:\n{event.get('summary', event['description'][:200])}...\n")
        
        if evaluation:
            answer_parts.append(f"\nWhy this matches your request:\n{evaluation.get('overall_reasoning', 'Matches your criteria')}\n")
        
        if event.get('source_url'):
            answer_parts.append(f"\nMore info: {event['source_url']}")
        
        return "".join(answer_parts)
    
    def _build_not_found_answer(self, state: StateManager, num_evaluated: int) -> str:
        """Build answer when no matches found."""
        intent = state.user_intent
        
        answer_parts = [
            "I couldn't find any events that match your specific requirements.\n\n",
            f"What I was looking for:\n"
        ]
        
        if intent:
            keywords = [kw["keyword"] for kw in intent.get("keywords", [])]
            answer_parts.append(f"- Keywords: {', '.join(keywords)}\n")
            answer_parts.append(f"- Location: {intent.get('city', 'Not specified')}\n")
            
            timeframe = intent.get('timeframe', {})
            if isinstance(timeframe, dict) and 'timeframe' in timeframe:
                answer_parts.append(f"- Date: Around {timeframe['timeframe']}\n")
            else:
                answer_parts.append(f"- Date: Not specified\n")
        
        answer_parts.append(f"\nI checked {num_evaluated} events but none matched all your criteria.\n")
        
        # Add common mismatch reasons if we have evaluation history
        if state.last_evaluation and not state.last_evaluation.get("matches"):
            answer_parts.append(f"\nLast event didn't match because:\n")
            answer_parts.append(f"{state.last_evaluation.get('overall_reasoning', 'Criteria mismatch')}\n")
        
        answer_parts.append("\nSuggestions:\n")
        answer_parts.append("- Try broader keywords or different terms\n")
        answer_parts.append("- Consider nearby dates or flexible timing\n")
        answer_parts.append("- Check neighboring cities or districts\n")

        if self.include_alternatives:
            alternatives = self._get_top_alternatives(state, limit=3, min_confidence=0.2)
            if alternatives:
                answer_parts.append("\n📋 Closest matches (though not ideal):\n")
                for i, alt in enumerate(alternatives, 1):
                    answer_parts.append(
                        f"{i}. **{alt['title']}** (confidence: {alt['confidence']:.0%})\n"
                        f"    {alt['date']} 📍 {alt['location']}\n"
                        f"    {alt['match_summary']}\n\n"
                    )
            else:
                answer_parts.append("\n(No alternatives met even the minimum criteria)\n")
 
        
        return "".join(answer_parts)
    
    def _build_partial_match_answer(self, state: StateManager) -> str:
        """Build answer for partial matches."""
        event = state.event_details["parsed"]
        evaluation = state.last_evaluation or {}
        
        answer_parts = [
            "I found an event that partially matches your requirements:\n\n",
            f"**{event['title']}**\n",
            f"{self._format_date(event['start_datetime']['date'])}\n",
            f"{event['location']}, {event['city']}"
        ]
        
        if event.get('district'):
            answer_parts.append(f" ({event['district']})")
        answer_parts.append("\n\n")
        
        answer_parts.append(f"Match Details:\n")
        
        if evaluation:
            # Show what matched and what didn't
            if evaluation.get('date_matches'):
                answer_parts.append("✅ Date matches your request\n")
            else:
                answer_parts.append("❌ Date doesn't match perfectly\n")
                
            if evaluation.get('location_matches'):
                answer_parts.append("✅ Location is correct\n")
            else:
                answer_parts.append("❌ Different location than requested\n")
                
            if evaluation.get('type_matches'):
                answer_parts.append("✅ Event type matches\n")
            else:
                answer_parts.append("❌ Different type of event\n")
            
            answer_parts.append(f"\nOverall Assessment:\n{evaluation.get('overall_reasoning', 'Partial match to your criteria')}\n")
            answer_parts.append(f"Confidence: {evaluation.get('confidence', 0.5):.0%}\n")
        
        if event.get('price_info'):
            answer_parts.append(f"\nPrice: {event['price_info']}\n")
            
        answer_parts.append(f"\nAbout: {event.get('summary', event['description'][:150])}...\n")

        if self.include_alternatives:
            # Get alternatives with similar or better confidence
            current_confidence = state.last_evaluation.get("confidence", 0.5)
            alternatives = self._get_top_alternatives(
                state, 
                limit=2, 
                min_confidence=current_confidence - 0.2 
            )
            
            if alternatives:
                answer_parts.append("\n📋 Similar events to consider:\n")
                for alt in alternatives:
                    comparison = "better" if alt['confidence'] > current_confidence else "similar"
                    answer_parts.append(
                        f"- **{alt['title']}** ({comparison} match: {alt['confidence']:.0%})\n"
                        f"  {alt['date']} at {alt['location']}\n"
                    )
            
        return "".join(answer_parts)
    
    def _get_top_alternatives(self, state: SharedState, limit: int = 3, min_confidence: float = 0.3) -> List[Dict]:
        """Get top alternative events based on evaluation confidence scores."""
        
        if not state.evaluation_history:
            return []
        
        alternatives = []
        for eval_record in state.evaluation_history:
            # Skip if it was already selected as the main result
            if state.event_details and eval_record["page_id"] == state.event_details["page_id"]:
                continue
                
            # Only include if above minimum confidence threshold
            if eval_record["confidence"] >= min_confidence:
                alternatives.append({
                    "page_id": eval_record["page_id"],
                    "title": eval_record["title"],
                    "confidence": eval_record["confidence"],
                    "date": eval_record.get("date", "Date TBD"),
                    "location": eval_record.get("location", "Location TBD"),
                    "match_summary": self._summarize_match_quality(eval_record)
                })
        
        alternatives.sort(key=lambda x: x["confidence"], reverse=True)
        
        return alternatives[:limit]
    
    def _summarize_match_quality(self, eval_record: Dict) -> str:
        """Create a brief summary of why this is a good/poor match."""
        confidence = eval_record["confidence"]
        reasons = eval_record.get("reasons", [])
        
        if confidence >= 0.8:
            return "Strong match - minor differences only"
        elif confidence >= 0.6:
            if reasons:
                return f"Good match - {reasons[0]}"
            return "Good match with some differences"
        elif confidence >= 0.4:
            if reasons:
                return f"Partial match - {'; '.join(reasons[:2])}"
            return "Partial match"
        else:
            return "Weak match - significantly different"

    def _build_multiple_matches_answer(self, state: StateManager) -> str:
        """Build answer when multiple events match."""
        # This would need enhanced state tracking for multiple matches
        # For now, provide a template that could be expanded
        answer_parts = [ ##TODO to be finished
            "I found multiple events that match your criteria! 🎊\n\n"
        ]
        
        # If we tracked multiple matches in state, we'd list them here
        if state.event_details:
            event = state.event_details["parsed"]
            answer_parts.append(f"Here's one great option:\n\n")
            answer_parts.append(f"**{event['title']}**\n")
            answer_parts.append(f"{self._format_date(event['start_datetime']['date'])}\n")
            answer_parts.append(f"{event['location']}, {event['city']}\n")
            
            answer_parts.append(f"\n(Additional matching events would be listed here)\n")
        
        answer_parts.append("\nWould you like details on any specific event?")
        
        return "".join(answer_parts)
    
    def _build_fallback_answer(self, state: StateManager) -> str:
        """Fallback for unexpected states."""
        parts = [
            "I encountered an issue while searching for events.\n\n"
        ]
        
        if state.user_intent:
            keywords = [kw["keyword"] for kw in state.user_intent.get("keywords", [])]
            parts.append(f"I was searching for: {', '.join(keywords)}\n")
        
        if state.evaluated_page_ids:
            parts.append(f"I checked {len(state.evaluated_page_ids)} events\n")
        
        parts.append("\nPlease try rephrasing your request or being more specific about what you're looking for.")
        
        return "".join(parts)
    
    def _format_date(self, date_str: str) -> str:
        """Format date nicely for display."""
        try:
            from datetime import datetime
            date = datetime.fromisoformat(date_str)
            if date.hour != 0 or date.minute != 0:
                return date.strftime("%A, %B %d, %Y at %I:%M %p")
            else:
                return date.strftime("%A, %B %d, %Y")
        except:
            return date_str
    
    def summarize(self, result: Dict[str, Any]) -> str:
        """Create conversation summary."""
        answer_type = result.get("answer_type", "unknown")
        confidence = result.get("confidence", 0)
        has_match = result.get("has_match", False)
        
        if answer_type == "found":
            return f"✅ Provided matching event (confidence: {confidence:.0%})"
        elif answer_type == "not_found":
            return f"❌ No matches found after checking {result.get('events_evaluated', 0)} events"
        elif answer_type == "partial_match":
            return f"⚠️ Provided partial match (confidence: {confidence:.0%})"
        elif answer_type == "multiple_matches":
            return f"🎊 Found multiple matching events"
        else:
            return "Provided final answer"


AgentActions = Union[ExtractUserIntentTool,SearchEventPageTitlesTool, SelectEventFileTool,
                     ReadEventFileTool, ReadEventFileTool, EvaluateEventTool, FinalAction]



###########################################################
######### STATE MANAGER AND COLLECTION DEFINITION #########
###########################################################

class StateManager(BaseModel):
    """State manager to keep track of the conversation, extracted values and actions taken."""
    # Returned by UserIntent
    
    original_query: Optional[str] = Field(description="The original user query that initiated the conversation")

    user_intent: Optional[UserIntent] = Field(description="The user's intent extracted from the original query")
    
    # Returned by SearchEventPageTitlesTool
    search_title_results: Dict = Field(description="List of event pages found using title embedding similarity search",
                                       default_factory=dict)
    current_search_keyword: Optional[str] = Field(description="The current keyword being searched for in the event pages")
    exhausted_search_keywords: Set[str] = Field(description="List of keywords that have been searched for and exhausted",
                                                 default_factory=set)
    selected_page_id: Optional[str] = Field(description="The current page_id being processed")

    read_event_pages: Set = Field(description="List of event page_ids that have been read", default_factory=set)
    event_details: Optional[Dict[str, Any]] = Field(description="Structured event information extracted from the file")
    evaulated_event_pages: List = Field(description="List of event pages that have been evaluated for relevance against the user intent",
                                        default_factory=list)
    last_evaluation: Optional[EventEvaluation] = Field(description="The last evaluation result of the event against user intent")
    evaluation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All evaluated events with details and confidence scores"
    )



class DependencyManager(BaseModel):
    """A class to manage shared dependencies and configurations for the agent."""
    client: instructor.Client = Field(description="The instructor client used for LLM interactions")
    collection: chromadb.Collection = Field(description="ChromaDB collection for storing event pages and their embeddings")

####################################################################
###################### AGENT CLASS DEFINITION ######################
####################################################################

SYSTEM_PROMPT = f"""You are a helpful assistant that helps users find information about events.

You have access to the following tools:
1. search_event_pages: Find event pages using embedding similarity search.
2. read_event_file_contents: Read the contents of an event file using the page_id with the smalles distance measure returned from the search_event_pages tool.
3. finish: Provide the final answer with all event details.

Today is {datetime.now().today()}.

You may make up to 10 tool calls before giving your final answer.
"""
"""
In each turn, respond in the following format:
<think>
[your thoughts here]
</think>
<tool>
 

When you have found the answer, respond in the following format:
<think>
[your thoughts here]
</think>
<answer>
[final answer here]
</answer>
"""


class MyAgent:
    """A simple AI agent that can search for event pages."""
    
    def __init__(self, model: str = "gpt-4.1-mini", verbose: bool = True):
        self.model=model
        self.verbose = verbose
        self.action_history = []

        
        
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = instructor.from_openai(
            OpenAI(api_key=os.environ.get("OPENAI_API_KEY") ),
            mode=instructor.Mode.TOOLS_STRICT,
            temperature=0.5
            )
        
    def _log(self, message: str):
        """Print if verbose is True."""
        if self.verbose:
            pprint(message)


    def step(self, user_query: str, max_steps: int = 3) -> str:
        """Process user query through multiple reasoning steps.
        
        Args:
            user_query (str): The user's query or request to process.
            max_steps (int): Maximum number of actions to take.
            
        Returns:
            Final answer string
            """
        
        self.user_intent = extract_user_intent(user_query=user_query, client=self.client)
        self.refined_query = self.user_intent.query
        self.ordered_title_keywords = sorted(self.user_intent.keywords, key=lambda x: x.confidence, reverse=True)

        # print("="*30)
        # print("USER KEYWORDS")
        # pprint(self.ordered_title_keywords)

        self.conversation_history.append({
            "role": "user",
            "content": self.refined_query
        })

        for step_num in range(max_steps):
            self._log(f"\n----- Step {step_num + 1} -----\n")

            try:
                action = self.client.chat.completions.create(
                    model=self.model,
                    response_model=AgentActions,
                    messages= self.conversation_history,
                    max_tokens=4096
                )
                # pprint(action.model_dump())
            except Exception as e:
                self._log(f"Error during action generation: {e}")
                return "An error occurred while processing your request."
            
            self._log(f"Thought: {action.think}")
            self._log(f"Action: {action.action_type}")
            # pprint(action.model_dump())

            if action.action_type == "search_event_pages":
                result = action.execute(keywords=[kw.keyword for kw in self.ordered_title_keywords])
                self.event_pages = result
                print("="*30)
                print("ECENT PAGES")
                pprint(self.event_pages)
                
            elif action.action_type == "read_event_file_contents":
                page_id = min(
                    self.event_pages, 
                    key=lambda x: self.event_pages[x]["min_distance"]
                )
                result = action.execute(page_id=page_id, 
                                        client=self.client, 
                                        model=self.model
                                        )
                print("="*30)
                print("Reading event file contents for page_id:", page_id)
                pprint(result.model_dump())
            else: 
                result = action.execute()
                print("="*30)
                self._log(f"Result: {result.model_dump()}")

            # Add to memory
            self.action_history.append({
                "step": step_num + 1,
                "think": action.think,
                "action_type": action.action_type,
                "result": result
            })

            action_summary = f"Action: {action.action_type}\nThink: {action.think}\nResult: {result}"

            self.conversation_history.append({
                "role": "assistant",
                "content": action_summary + f"Result: {result}"
            })

            if isinstance(action, FinalAction):
            # if isinstance(action, FinalActionTool):
                self._log(f"\nFinal Answer: {action.answer}")
                self._log(f"Confidence: {action.confidence}")
                # return action.answer.model_dump()
                return action.answer
            
            self.conversation_history.append({
                "role": "user",
                "content": "based on this result, what should we do next? If you have enough information, provide a final answer."
            })
        self._log("Reached maximum steps without final answer.")
        self.conversation_history.append({
            "role": "user",
            "content": "You've reached the maximum number of steps. Please provide a final answer based on the information gathered."
        })
        final_action = self.client.chat.completions.create(
            model=self.model,
            response_model=FinalAction,
            messages=self.conversation_history,
            max_tokens=4096
        )
        print(f"Final Thought: {final_action.think}")
        # return final_action.answer.model_dump()
        return final_action.answer
    
    def get_action_summary(self) -> str: ##TODO replace this with the custom summarise() method in each tool
        """Get a summary of all actions taken"""
        summary = "Action Summary:\n"
        for action in self.action_history:
            summary += f"\nStep {action['step']}: {action['action_type']}\n"
            summary += f"  Thought: {action['think']}\n"
            summary += f"  Result: {action['result']}\n"
        return summary
    
if __name__ == "__main__":
    
    collection = init_collection()
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        try:
            if user_query.lower() == 'exit':
                break
            
            agent = MyAgent(model="gpt-4.1-mini", verbose=True)
            final_answer = agent.step(user_query)
            print(f"\nFinal Answer: {final_answer}")
            print("\n")
            print(agent.get_action_summary())
        except KeyboardInterrupt:
            print("\nExiting...")
            break    