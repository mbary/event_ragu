import os
import json
from datetime import datetime
from typing import Union, List, Literal, Optional


import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from setup_db import init_collection


client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.TOOLS
)


collection = init_collection()

###################################
####### Structured Outputs ########
###################################

class EventPageResult(BaseModel):
    """Represents a single search result for an event page."""
    page_id: str = Field(description="Unique identifier for the event page")
    title: str = Field(description="Title of the event page")


###################################
############## Tools ##############
###################################

class SearchEventPageTitlesTool(BaseModel):
    """Search for top 10 relevant event pages using title embedding similarity."""

    action_type: Literal["search_event_pages"] = "search_event_pages"
    think: str = Field(description="Why is this search needed and what information is sought")
    query: str = Field(
        description="Search query to find relevant event pages",
        examples=["Muzeum", "Kino", "Koncert"]
    )
    result: List[EventPageResult] = Field(description="List of top 10 relevant event pages")

    def execute(self) -> List[EventPageResult]:
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

class FinalAction(BaseModel):
    """Provide the final answer to the user"""
    action_type: Literal["finish"] = "finish"
    think: str = Field(description="Final reasoning before providing the answer")
    answer: str = Field(description="Final comprehensive answer to the user's query")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the final answer (0-1)")

    def execute(self) -> str:
        """Execute the final action and return the answer.
        
        Returns:
            str: Final answer to the user's query.
        """
        return self.answer


AgentActions = Union[SearchEventPageTitlesTool, FinalAction]



a = SearchEventPageTitlesTool(
    think="I need to find event pages related to the query",
    query="rower"
)

a.execute()


SYSTEM_PROMPT = """You are a helpful AI agent that answers question by breaking them down into steps.
For each step:
1. Think about the information you need
2. Choose the most appropriate action
3. Use the result to inform your next step
4. When you have enough information, and are confident in your answer, provide a final response.
You can use the following tools:


Always think before taking an action, and explain why you are taking it.
Always explain your reasoning in the 'think' field of the action.
"""##TODO add verifiers and parsing tools into prompt

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
            mode=instructor.Mode.TOOLS)
        
    def _log(self, message: str):
        """Print if verbose is True."""
        if self.verbose:
            print(message)

    def step(self, user_query: str, max_steps: int = 5) -> str:
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

        for step_num in range(max_steps):
            self._log(f"\n--- Step {step_num + 1} ---")

            try:
                action = self.client.chat.completions.create(
                    model=self.model,
                    response_model=AgentActions,
                    messages= self.conversation_history,
                    max_tokens=4096
                )
            except Exception as e:
                self._log(f"Error during action generation: {e}")
                return "An error occurred while processing your request."
            
            self._log(f"Thought: {action.think}")
            self._log(f"Action: {action.action_type}")

            result = action.execute()
            self._log(f"Result: {result}")

            self.action_history.append({
                "step": step_num + 1,
                "think": action.think,
                "action": action.action_type,
                "result": result
            })

            action_summary = f"Action: {action.action_type}\nThink: {action.think}\nResult: {result}"

            self.conversation_history.append({
                "role": "assistant",
                "content": action_summary
            })

            if isinstance(action, FinalAction):
                self._log(f"\nFinal Answer: {action.answer}")
                self._log(f"Confidence: {action.confidence}")
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
        return final_action.answer
    
