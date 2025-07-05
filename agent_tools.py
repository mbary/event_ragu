import os
import json
from datetime import datetime, date, timedelta
from typing import Union, List, Literal, Optional, Dict, Any

import chromadb
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field, field_validator, ValidationError
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from pprint import pprint
client = instructor.from_openai(OpenAI())

#######################################################################################################################################
#######################################################################################################################################
##################################################### EVENT DETAILS EXTRACTION ########################################################
#######################################################################################################################################
#######################################################################################################################################

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
    """Represents the structured details of a single event."""
    title: str = Field(description="The main title of the event.")
    start_datetime: EventDetailsStart = Field(description="The start date and time of the event in ISO 8601 format.")
    end_datetime: EventDetailsEnd = Field(description="The end date and time of the event in ISO 8601 format.")
    location: str = Field(description="The physical location, venue, or address of the event.")
    city: str = Field(description="The city where the event is taking place.")
    description: str = Field(description="A detailed summary or description of the event's content.")
    source_url: Optional[str] = Field(None, description="The source URL of the event page, if available.")


extraction_client = instructor.from_openai(OpenAI())

event  = """
# URL: https://kultura.um.warszawa.pl/-/studio-premiera-alphago_lee
# Alphago\_Lee. Teoria poświęcenia
27.09.2025 19:00 - 27.09.2025 21:00
Teatr Studio
teatrpremierateatr studio
Teatr Studio
Teatr Studio zaprasza na premierę "Alphago\_Lee. Teoria poświęcenia" w reż. Natalii Korczakowskiej.
To międzynarodowy projekt performatywny.
Rok 2016. W nowo wybudowanym hotelu Four Seasons w Seulu rozegrał się historyczny pojedynek – koreański mistrz gry Go, Lee Sedol, stanął do walki z programem AlphaGo. Dziewięć lat później, w momencie, kiedy AI staje się naszą codziennością, wracamy do tej historii. Ten spektakl to nie tylko teatralna rekonstrukcja tamtego meczu. To pytanie o przyszłość. O społeczne i środowiskowe koszty rozwoju technologii.
Więcej na stronie teatru
 Wydarzenie zapewnia następujące udogodnienia

"""

event="""
# URL: https://ursus.um.warszawa.pl/-/oddaj-krew-w-ursus-4
# Oddaj krew w Ursusie
07.08.2025 09:00 - 07.08.2025 14:00
Zapraszamy do włączenia się w akcję krwiodawstwa w Ursusie.
Mobilny punkt pobrań zorganizowany przez Klub HDK PCK Ursus stanie 7 sierpnia od 9:00 do 14:00 przed budynkiem urzędu w Ursusie na pl. Czerwca 1976 roku nr 1.
Krwiodawcą może być zdrowa osoba między 18. a 65. rokiem życia.
 Wydarzenie zapewnia następujące udogodnienia
"""

event_details = extraction_client.chat.completions.create(
                model="gpt-4.1-mini",
                response_model=EventDetails,
                messages=[
                    {"role": "system", "content": "You are a world-class expert at extracting structured event information from unstructured markdown text. Extract the details accurately. Parse the date into human-readable format."},
                    {"role": "user", "content": f"Please extract the event details from the following text:\n\n{event}"}
                ]
            )

pprint(event_details.model_dump())

#######################################################################################################################################
#######################################################################################################################################
##################################################### USER INTENT EXTRACTION ##########################################################
#######################################################################################################################################
#######################################################################################################################################



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
    
    # Ensure thar the user specifies a city in their query
    @field_validator("city") # God knows whether that's working, I'm affraig te model is hallucinating a popular ciy
    @classmethod             # Rendering this pointless, most likely this is better for user input validation
    def ensure_city(cls, value):
        print("Validating city: ", value)
        if not value:
            raise ValueError("City must be provided.")
        return value

SYSTEM_PROMPT_INTENT_EXTRACTION = f"""You are a world-class expert at extracting user intent from the user query in a form of unstructured text.
Extract the user's intent, datetime, location, and keywords accurately.
Current date is {date.today().isoformat()}.
The datetime should be in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
If no the specified datetime is vague, always relate it to the current date.
All your responses **MUST** be in Polish language.
You should always think step by step.
"""

"""
When you have found the answer, respond in the following format:
<think>
[your thoughts here]
</think>
<answer>
[final answer here]
</answer>"""


SYSTEM_PROMPT_INTENT_EXTRACTION = f"""You are a world-class expert at extracting user intent from the user query in a form of unstructured text.

Current date is {date.today().isoformat()}.

Returns:
- think: A thought process or reasoning behind the user's intent extraction.
- query: A refined user query.
- action_type: The type of action to be performed, which is always "extract_user_intent".
- timeframe: The timeframe for the event search, represented as a datetime object in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).
- city: The city where the user wants to find events, represented as a string.
- location: The location where the user wants to find events, represented as a string.
- keywords: A list of keywords, strictly related to the users query and doesn't contain the city name.

If no the specified datetime is vague, always relate it to the current date.
All your responses **MUST** be in Polish language.
You should always think step by step.
"""





query = "Gdzie w Warszawie mogę znaleźć jakieś dobre pierogi, zupę pomidorową bądź inne smaczne jedzenie pierwszego stycznia 2026? najlepiej w jakimś barze mlecznym?"
query = "Gdzie  mogę znaleźć jakieś dobre pierogi pierwszego stycznia 2026?"
query = "Bardzo lubie wpierdalać pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi pierogi i mam na nie bardzo dużą ochotę. Gdzie mogę znaleźć jakieś dobre pierrogi pierwszego stycznia 2026? "

query = "Słuchaj no kurwa mordo bo powiem ci tak, że w życiu chodzi o to żeby było miło i przyjemnie, a nie żeby się pierdolić w jakieś pierdoły. Dlatego wazna" \
"jest rodzina, przyjaciele i dobre kurwa twoja mać jedzenie. Cenie sobie dobre jedzenie i no kurwa mogę wpierdlać od rana do wieczora." \
"Wszystko, no powiem Ci kurwa, że wszystko jest dobre, i wszystko praktycznie mógłbym wpieralać non stop. takie na przygład spaghetti bolognese. no kurwa gdzie ja kurwa znajde dobre pierogi w tej przeklętej Warszawie, bo ja pierdolę nie wiem gdzie szukać." \
"kocham spaghetti bolognese. no co jak kurwa co ale ci jebani wlosi potrafia robic bolognese, jezu kocham spaghetti bolognese. to tez mogę wpierdalać codziennie i czasem robie sobie bolognese na 2 tygodnie i wpierdalam spaghetti bolognese trzy razy dziennie jem bolognese, ale nie tylko, bo lubię też pomidorówkę, no kurwa takiej pomidorówki jak moja mamusia robi to kurwa nigdzie nie ma. NIGDZIE MÓWIE CI KURWA NIGDZIE! No mógłbym wpierdalać pomidorówkę codziennie, ale nie tylko pomidorówkę, bo lubię też inne rzeczy. NO za dobrą pomidorówkę to mógłbym zajebać człowieka i całą jego rodzinę na jego oczach, gdyby mi kto obiecał miskę gorącej, pomidorówki, kocham pomidorówkę sialalalala POMIDORÓWKA!!!! albo taka pizza, jezusie kochany pizza to jest jednak pizza i nie ma prostszej rzeczy ktora sprawia mi radosc" \
"kocham tez pizze i mógłbym nawet zjesc pizze teraz i mam trochę ochote, ale nie, pizza kiedy indziej " \
"Na przykład lubię też chuja wpierdalac, no kurwa chuje to jest coś co mogę żreź non stop. I powiem ja Ci, że mam kurwa ochotę na chuje teraz więc poweidz no mi kochaniutki" \
" "
query = "Gdzie i kiedy mogę w warszawie oddać krew?"
intent = client.chat.completions.create(
    model="gpt-4.1-mini",
    response_model=UserIntent,
    messages=[
        {"role":"system", "content": SYSTEM_PROMPT_INTENT_EXTRACTION},
        {"role": "user", "content": query}
    ],
    temperature=0.0)
    
pprint(intent.model_dump(),)
type(intent.keywords)
intent['keywords']
print("")
print("")
print("")

intent.city
type(intent.city)
for keyword in intent.keywords:
    print(f"Keyword: {keyword.keyword}, Confidence: {keyword.confidence:.2f}")

#######################################################################################################################################
#######################################################################################################################################
#################################################### Event Relevancy Evaluation #######################################################
#######################################################################################################################################
#######################################################################################################################################


class CompareDates(BaseModel):
    """Class represents the comparison of two dates."""
    date_intent: datetime = Field(description="The first date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).")
    date_file: datetime = Field(description="The second date in ISO 8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ).")
    
class EventFileRelevance(BaseModel):
    """Class represents the relevancy of an event to the user's intent."""
    think: str = Field(description="I will evaluate how relevant is the event described in the file to the event described in the user intent.")
    action_type: Literal["evaluate_event_relevancy"] = "evaluate_event_relevancy"
    page_id:str

def evaluate_relevancy(self, user_intent: UserIntent, event_details: EventDetails):
    pass


"""



Pesudo code:

1) search documents 
if read documents
    get event details


"""


client = OpenAI()
ans = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role":"user","content":"WHAT IS LOVEEEEE?"}])

pprint(dir(ans))

pprint(dir(ans.choices[0]))

ans.choices[0]







