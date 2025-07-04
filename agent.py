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

###################################################################
################### TOOL AND OUTPUT DEFINITIONS ###################
###################################################################

###################
### User Intent ###
###################

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

    location: Optional[str] = Field(description="The location where the user wants to find events.", example=["Ursus", "Stadion Narodowy", 
                                                                                                              "Centrum Nauki Kopernik",
                                                                                                              "Park","Teatr","Opera"])

    keywords: List[UserIntentKeyWord] = Field(description="A list of keywords, specifically related to the event, to refine the search.",
                                              examples=["concert", "exhibition", "theater", "art", "music"], 
                                              max_length=5, 
                                              min_length=2)



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




def extract_user_intent(user_query: str, client: instructor) -> UserIntent:
    """Extract user intent from the user query."""

    intent = client.chat.completions.create(
        model="gpt-4.1-mini",
    response_model=UserIntent,
    messages=[
        {"role":"system", "content": SYSTEM_PROMPT_INTENT_EXTRACTION},
        {"role": "user", "content": query}
    ],
    temperature=0.0)
    return intent






####################################################################
###################### AGENT CLASS DEFINITION ######################
####################################################################


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


    def step(self, user_query: str, max_steps: int = 10) -> str:
        """Process user query through multiple reasoning steps.
        
        Args:
            user_query (str): The user's query or request to process.
            max_steps (int): Maximum number of actions to take.
            
        Returns:
            Final answer string
            """
        
        self.user_intent = extract_user_intent(user_query= user_query, client=self.client)

        self.title_keywords = sorted(self.user_intent.keywords, key=lambda x: x.confidence, reverse=True)

        
