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
- SharedDependencies: Contains shared dependencies and configurations for the agent.
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

            # Use the tool's own summarize method
            summary = f"Action: {action.action_type}
            Think: {action.think}
            Result: {action.summarize(result)}"

            self.conversation_history.append({
                "role": "assistant",
                "content": summary
            })
    - include error handling 
"""



import os
import json
from datetime import datetime, date
from typing import Union, List, Literal, Optional, Dict, Any
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


collection = init_collection()

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
    
    query: str = Field(description="A refined user query.")

    action_type: Literal["extract_user_intent"] = "extract_user_intent"

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




def extract_user_intent(user_query: str, client: instructor, 
                        state: StateManager) -> UserIntent:
    """Extract user intent from the user query."""

    intent = client.chat.completions.create(
        model="gpt-4.1-mini",
    response_model=UserIntent,
    messages=[
        {"role":"system", "content": SYSTEM_PROMPT_INTENT_EXTRACTION},
        {"role": "user", "content": user_query}
    ],
    temperature=0.0)
    return intent


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
            kw_results = collection.query(
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
        state.search_title_results.append(final_dict)

        return final_dict
    
    def summarise(self, results: Dict[str, Dict[str, Union[float, List[EventTitleResult]]]]) -> str:
        """Summarize the search results."""
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

class SelectEventFileInput(BaseModel):
    """Represents the selection of an event file based on the page_id."""
    action_type: Literal["select_event_file"] = "select_event_file"
    page_id: str = Field(description="Unique identifier for the event page",
                         examples=["event1", "event2", "event3"])
    think: str = Field(description="I should select the event file based on the smallest distance measure.")


class SelectEventFileTool(SelectEventFileInput):
    """Execute the selection of the event file based on the page_id with the smallest distance measure.
    
    Args:
        results_dict (Dict[str, Union[float, SearchEventPagesOutput]]): Dictionary containing the results of the search.
    
    Returns:
        str: The page_id of the selected event file.
    """

    def execute(self, results_dict: Dict[str, Union[float, SearchEventPagesOutput]]) -> str:
        print("="*30)
        print("Selecting event file based on smallest distance measure.")
        page_id = min(
            results_dict, 
            key=lambda x: results_dict[x]["min_distance"]
        )
        print(f"Selected page_id: {page_id}")
        return page_id
## TODO just implemtnt finding the next file with smallest distance measure


class ReadEventFileContentsTool(BaseModel):
    """Read the contents of an event file using the page_id with the smalles distance measure returned from the search_event_pages tool.
    Args:
        page_id (str): Unique identifier for the event page, with smallest distance measure.
    
    Returns:
        event_details: Output containing the extracted structured details of the event.
    Example:
    """
    think: str = Field(description="Why is this reading needed and what information is sought")
    action_type: Literal["read_event_file_contents"] = "read_event_file_contents"
    event_details: EventDetails = Field(description="Details of the read event")
    def execute(self, page_id:str, client: instructor, model: str):
        with open(os.path.join(EVENT_DIR, f"{page_id}.md"), 'r', encoding='utf-8') as f:
            data = f.read()

        result = client.chat.completions.create(
            model=model,
            response_model=EventDetails,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that reads event files and extracts structured details."},
                {"role": "user", "content": f"Read the event file with page_id: {page_id} and extract structured details."},
                {"role": "assistant", "content": data}
            ],
            temperature=0.0
        )
        return ReadEventFileContentsOutput(
            page_id=page_id,
            contents=result
        )


class EventEvaluation(BaseModel):
    """Structured evaluation result"""
    matches: bool = Field(description="Overall match determination")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")
    
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
    Uses AI to understand nuanced requirements and fuzzy matching.
    """
    action_type: Literal["evaluate_event"] = "evaluate_event"
    think: str = Field(description="What aspects need careful evaluation")
    
    def execute(self, state: SharedState, deps: SharedDependencies) -> Dict[str, Any]:
        # Validation
        if not state.event_details:
            return {"error": "No event details found. Please read an event file first."}
        if not state.user_intent:
            return {"error": "No user intent found. Cannot evaluate without requirements."}
        
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
            evaluation = deps.openai_client.chat.completions.create(
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
            
            # Update state
            state.last_evaluation = result
            if not result["matches"]:
                state.evaluated_page_ids.append(state.event_details["page_id"])
            
            return result
            
        except Exception as e:
            return {"error": f"Evaluation failed: {str(e)}"}
    
    def summarize(self, result: Dict[str, Any]) -> str:
        """Create conversation summary"""
        if "error" in result:
            return f"Error: {result['error']}"
        
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

class FinalAction(BaseModel):
    """Provide the final answer to the user
    Returns:
        EventDetails: The final comprehensive answer to the user's query, including event details like title, dates, location, and description.
    """
    action_type: Literal["finish"] = "finish"
    think: str = Field(description="Final reasoning before providing the answer")
    answer: str = Field(description="Final comprehensive answer to the user's query")
    # answer: str = Field(description="Final comprehensive answer to the user's query")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the final answer (0-1)")

    def execute(self) -> str:
        """Execute the final action and return the answer.
        
        Returns:
            str: Final answer to the user's query.
        """
        return self.answer


AgentActions = Union[SearchEventPageTitlesTool, ReadEventFileContentsTool ,FinalAction]



###########################################################
######### STATE MANAGER AND COLLECTION DEFINITION #########
###########################################################

class StateManager(BaseModel):
    """State manager to keep track of the conversation, extracted values and actions taken."""
    # Returned by UserIntent
    user_intent: Optional[UserIntent] = Field(description="The user's intent extracted from the original query")
    
    # Returned by SearchEventPageTitlesTool
    search_title_results: List = Field(description="List of event pages found using title embedding similarity search",
                                       default_factory=list)
    
    read_event_pages: List = Field(description="List of event pages that have been read", default_factory=list)
    evaulated_event_pages: List = Field(description="List of event pages that have been evaluated for relevance against the user intent",
                                        default_factory=list)



class DependencyManager(BaseModel):
    pass

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
    
    def get_action_summary(self) -> str:
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