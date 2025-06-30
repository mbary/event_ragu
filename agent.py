import os
import json
from datetime import datetime, date
from typing import Union, List, Literal, Optional, Dict, Any

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
# db_client = chromadb.PersistentClient(path=".chroma_db")
# print(db_client.get_collection("event_titles").count())
# print(db_client.list_collections())

###################################
####### Structured Outputs ########
###################################

class EventPageResult(BaseModel):
    """Represents a single search result for an event page."""
    page_id: str = Field(description="Unique identifier for the event page")
    title: str = Field(description="Title of the event page")

    
class SearchEventPagesOutput(BaseModel):
    """Output containing top 10 relevant event pages."""
    results: List[EventPageResult] = Field(
        description="List of top 10 relevant event pages",
        examples=[
            {"page_id": "event1", "title": "Concert in the Park"},
            {"page_id": "event2", "title": "Art Exhibition Opening"}
        ])
    
class ReadEventDetailsInput(BaseModel):
    """Input for reading event page details"""
    action_type: Literal["read_event_details"] = "read_event_details"
    think: str = Field(description="Why this event file needs to be read and what information to extract")
    page_id_list: List[str] = Field(description="List of The IDs of the pages to read",
                         examples=[["zajęcia_sportowe_dla_wilanowskich_seniorów_00078",
                                   "memoriał_bohdana_bartosiewicza_00012"]])

class SearchEventPagesInput(BaseModel):
    """Input for searching event pages"""
    action_type: Literal["search_event_pages"] = "search_event_pages"
    think: str = Field(description="Why is this search needed abd what information is sought")
    query: str = Field(description="Search query to find relevant event pages",
                        examples=["What's shown in muzeums?", "What's played at the cinema?"
                                  , "What concerts are there?"])
    

class EventDetails(BaseModel):
    """Representes the event details for the selected event"""
    title: str = Field(..., description="The main title of the event.")

    start_datetime: datetime = Field(description="Start date and time of the event",
                            examples=["2023-10-01 18:00:00", "2023-10-01 18:00:00"])
    
    end_datetime: datetime = Field(description="End date and time of the event",
                          examples=["2023-10-01 20:00:00", "2023-10-01 20:00:00"])
    
    location: str = Field(description="Location of the event",
                            examples=["Zamek Królewski", "Teatr Narodowy"])
    
    description: str = Field(description="Summary description of the event",
                             examples=["A concert by the Warsaw Philharmonic Orchestra",
                                       "An exhibition of modern art at the National Museum"])
    
    source_url: str = Field(None, description="The source URL of the event page.")
    # confidence: float = Field(ge=0, le=1, description="Confidence score of the final answer (0-1)")


###################################
############## Tools ##############
###################################


    
class SearchEventPageTitlesTool(SearchEventPagesInput):
    """Search for top 10 relevant event pages using title embedding similarity.
    Returns:
        SearchEventPagesOutput: Output containing top 10 relevant event pages."""

    def execute(self) -> SearchEventPagesOutput:
        """Execute the search and return for 10 results using title embedding similarity.
        Returns:
            SearchEventPagesOutput: Output containing top 10 relevant event pages.
        """
        results = collection.query(
            query_texts=[self.query],
            n_results=10)

        output = []
        for i in range(len(results['ids'][0])):
            output.append(EventPageResult(
                page_id=results['ids'][0][i],
                title=results['metadatas'][0][i]['title']
            ))
        return output

class ReadEventDetailsTool(ReadEventDetailsInput):
    """Read the markdown content of an event file.
    
    Returns:
        str: The full markdown content of the event file, which contains event details like title, dates, location, and description.
    """
    
    def execute(self) -> str:
        """Read and return the markdown content of the event file.
        
        Returns:
            str: The full markdown content of the event file.
        """
        # Construct file path
        file_paths = [os.path.join(EVENT_DIR, f"{page_id}.md") for page_id in self.page_id_list]
        
        for file_path in file_paths:
            
            # Check if file exists
            if not os.path.exists(file_path):
                return f"Error: Event file not found for page_id: {file_paths}"
            
            # Read and return the markdown content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Return with a helpful header for the agent
                return f"Event file content for {file_paths}:\n\n{content}"
                
            except Exception as e:
                return f"Error reading file {file_path}: {str(e)}"



class FinalAction(BaseModel):
    """Provide the final answer to the user
    Returns:
        EventDetails: The final comprehensive answer to the user's query, including event details like title, dates, location, and description.
    """
    action_type: Literal["finish"] = "finish"
    think: str = Field(description="Final reasoning before providing the answer")
    answer: EventDetails = Field(description="Final comprehensive answer to the user's query")
    # answer: str = Field(description="Final comprehensive answer to the user's query")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the final answer (0-1)")

    def execute(self) -> EventDetails:
        """Execute the final action and return the answer.
        
        Returns:
            str: Final answer to the user's query.
        """
        return self.answer

# AgentActions = Union[SearchEventPageTitlesTool, FinalActionTool, ReadPageDetailsTool]
AgentActions = Union[SearchEventPageTitlesTool, FinalAction, ReadEventDetailsTool]


SYSTEM_PROMPT = f"""You are a helpful AI agent that answers question by breaking them down into steps.
Think step by step, and use the tools available to you to gather information.

You have access to the following tools:
1. search_event_pages: Search for events by title/topic
2. read_event_details: Read the full markdown content of a specific event page
3. finish: Provide the final answer

Today is {datetime.now().today()}.

For each step:
1. Think about the information you need
2. Choose the most appropriate action
3. Use the result to inform your next step
4. Always read the event page content before providing a final answer.
5. Always read all relevant event pages before providing a final answer.
6. When you have enough information, and are confident in your answer, provide a final response.

Always think before taking an action, and explain why you are taking it.
Always explain your reasoning in the 'think' field of the action.
"""##TODO add verifiers and parsing tools into prompt



# SYSTEM_PROMPT = """You are a helpful AI agent that answers question by breaking them down into steps.
# Think step by step, and use the tools available to you to gather information.
# You can use the following tools:
# {tool_descriptions}
# For each step:
# 1. Think about the information you need
# 2. Choose the most appropriate action
# 3. Use the result to inform your next step
# 4. When you have enough information, and are confident in your answer, provide a final response.
# You can use the following tools:


# Always think before taking an action, and explain why you are taking it.
# Always explain your reasoning in the 'think' field of the action.
# """##TODO add verifiers and parsing tools into prompt

##TODO Add more tools (read event page, search web)
##TODO Add confidence scoring to the final answer (and manybe tools too)



class MyAgent:
    """A simple AI agent that can search for event pages."""
    
    def __init__(self, model: str = "gpt-4.1-mini", verbose: bool = True):
        self.model=model
        self.verbose = verbose
        self.action_history = []
        
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = instructor.from_openai(
            OpenAI(api_key=os.environ.get("OPENAI_API_KEY") ),
            mode=instructor.Mode.TOOLS_STRICT)
        
    def _log(self, message: str):
        """Print if verbose is True."""
        if self.verbose:
            print(message)

    def step(self, user_query: str, max_steps: int = 10) -> str:
        """Process user query through multiple reasoning steps.
        
        Args:
            user_query (str): The user's query or request to process.
            max_steps (int): Maximum number of actions to take.
            
        Returns:
            Final answer string
            """
        
        self._log(f"\n{'='*60}")
        self._log(f"USER QUERY: {user_query}")
        self._log(f"{'='*60}\n")

        self.conversation_history.append({
            "role": "user",
            "content": user_query
        })

        for step_num in range(max_steps):
            self._log(f"\n--- Step {step_num + 1} ---")

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
            result = action.execute()
            self._log(f"Result: {result}")

            self.action_history.append({
                "step": step_num + 1,
                "think": action.think,
                "action_type": action.action_type,
                "result": result
            })

            action_summary = f"Action: {action.action_type}\nThink: {action.think}\nResult: {result}"

            self.conversation_history.append({
                "role": "assistant",
                "content": action_summary
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
            agent.get_action_summary()
            print(f"\nFinal Answer: {final_answer}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break    