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


class EventPageResult(BaseModel):
    """Represents a single search result for an event page."""
    page_id: str = Field(description="Unique identifier for the event page")
    title: str = Field(description="Title of the event page")


class SearchEventPageTitlesTool(BaseModel):
    """Search for top 10 relevant event pages using title embedding similarity."""

    action_type: Literal["search_event_pages"] = "search_event_pages"
    think: str = Field(description="Why is this search needed and what information is sought")
    query: str = Field(
        description="Search query to find relevant event pages",
        examples=["Muzeum", "Kino", "Koncert"]
    )

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
    
a = SearchEventPageTitlesTool(
    think="I need to find event pages related to the query",
    query="rower"
)

a.execute()