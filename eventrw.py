import os
import datetime
from typing import Any, List, Optional, Union
from googleapiclient.discovery import build
from langchain.docstore.document import Document as LCDocument
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar"]

class GoogleCalendarReader:
    def __init__(self):
        self.credentials = self._get_credentials()

    def load_data(
        self,
        number_of_results: Optional[int] = 100,
        start_date: Optional[Union[str, datetime.date]] = None,
    ) -> List[LCDocument]:
        service = build("calendar", "v3", credentials=self.credentials)

        if start_date is None:
            start_date = datetime.date.today()
        elif isinstance(start_date, str):
            start_date = datetime.date.fromisoformat(start_date)

        start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
        end_datetime = datetime.datetime.combine(start_date, datetime.time.max)
        start_datetime_utc = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        end_datetime_utc = end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=start_datetime_utc,
                timeMax=end_datetime_utc,
                maxResults=number_of_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = events_result.get("items", [])
        results = []

        for event in events:
            start_time = event["start"].get("dateTime", event["start"].get("date"))
            end_time = event["end"].get("dateTime", event["end"].get("date"))
            summary = event.get("summary", "No Title")
            description = event.get("description", "")
            location = event.get("location", "No Location")

            event_string = (
                f"Event: {summary}\n"
                f"Start: {start_time}\n"
                f"End: {end_time}\n"
                f"Description: {description}\n"
                f"Location: {location}\n"
            )

            results.append(LCDocument(page_content=event_string))

        return results

    def _get_credentials(self) -> Any:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=3030)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return creds

    def create_event(self, summary: str, location: str, description: str, start_time: str, end_time: str, attendees: List[str]):
        service = build("calendar", "v3", credentials=self.credentials)
        event = {
            "summary": summary,
            "location": location,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": "Asia/Kolkata"
            },
            "end": {
                "dateTime": end_time,
                "timeZone": "Asia/Kolkata"
            },
            "attendees": [{"email": email} for email in attendees]
        }
        event = service.events().insert(calendarId="primary", body=event).execute()
        return event.get('htmlLink')

# Instantiate the GoogleCalendarReader and load data
loader = GoogleCalendarReader()
documents = loader.load_data(start_date="2024-07-11", number_of_results=10)

# Convert documents to LangChain format
formatted_documents: List[LCDocument] = documents

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_documents = text_splitter.split_documents(formatted_documents)
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(split_documents, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0.7, model_name="gpt-4"),
    vector_store.as_retriever(),
    memory=memory
)

query = "Create a summary for what am I doing on the day: 2024-07-11"
result = qa({"question": query})
print(result['answer'])

# Function to parse natural language input and create events
def parse_and_create_event(input_text: str):
    tokens = input_text.split("'")
    if len(tokens) >= 9:
        summary = tokens[1]
        location = tokens[3]
        date = tokens[5]
        start_time = tokens[7]
        end_time = tokens[9]
        attendees = tokens[11].split(", ")
        start_datetime = f"{date}T{start_time}:00+05:30"
        end_datetime = f"{date}T{end_time}:00+05:30"
        event_link = loader.create_event(summary, location, summary, start_datetime, end_datetime, attendees)
        print(f"Event Created: {event_link}")
    else:
        print("Invalid input format.")


input_text = "Create an event titled 'Lunch' at 'xyzrestaurant' on '2024-07-21' from '10:00' to '11:00' with attendees 'email3@example.com, email2@example.com'"
parse_and_create_event(input_text)
