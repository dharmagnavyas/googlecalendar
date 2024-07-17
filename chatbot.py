import os
import datetime
from typing import Any, List, Optional, Union
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document as LCDocument
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/calendar.readonly"]

class GoogleCalendarReader:
    def __init__(self):
        self.credentials = self._get_credentials()

    def load_data(
        self,
        number_of_results: Optional[int] = 100,
        start_date: Optional[Union[str, datetime.date]] = None,
    ) -> List[str]:
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

            results.append(event_string)

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

# Initialize the LLM and GoogleCalendarReader
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
google_calendar_reader = GoogleCalendarReader()

# Define the prompt template
prompt_template = """
You are an AI assistant helping to schedule a meeting. Please ask the user for the following details one by one:
1. Event Name
2. Event Location
3. Event Description
4. Event Date (YYYY-MM-DD)
5. Event Start Time (HH:MM)
6. Event End Time (HH:MM)
7. Attendees' email addresses (separated by commas)
"""

# MessageHistory class to manage chat history
class MessageHistory:
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def add_messages(self, messages):
        self.messages.extend(messages)

    def get_messages(self):
        return self.messages

session_history = {}

def get_session_history(session_id):
    if session_id not in session_history:
        session_history[session_id] = MessageHistory()
    return session_history[session_id]

def set_session_history(session_id, history):
    session_history[session_id] = history

# Function to run the interactive event creation chatbot
def interactive_chatbot():
    session_id = "unique_session_id_1"  # You can generate or set this as needed

    conversation = RunnableWithMessageHistory(
        runnable=llm,
        prompt=PromptTemplate.from_template(prompt_template),
        get_session_history=lambda session_id: get_session_history(session_id).get_messages(),
        set_session_history=lambda session_id, history: set_session_history(session_id, history)
    )

    print("📆 Welcome to the Google Calendar Event Creator Chatbot! 💭")

    details = {}
    questions = [
        "Event Name",
        "Event Location",
        "Event Description",
        "Event Date (YYYY-MM-DD)",
        "Event Start Time (HH:MM)",
        "Event End Time (HH:MM)",
        "Attendees' email addresses (separated by commas)"
    ]

    for question in questions:
        print(f"🤖Asking: {question}")
        print(f"📆 Please provide the {question.lower()}: ", end="")
        user_input = input("👤 ")
        details[question] = user_input

    # Parse the collected details
    event_name = details["Event Name"]
    event_location = details["Event Location"]
    event_description = details["Event Description"]
    event_date = details["Event Date (YYYY-MM-DD)"]
    start_time = details["Event Start Time (HH:MM)"]
    end_time = details["Event End Time (HH:MM)"]
    attendees_input = details["Attendees' email addresses (separated by commas)"]
    attendees = [email.strip() for email in attendees_input.split(",")]

    # Create the event
    start_datetime = f"{event_date}T{start_time}:00+05:30"
    end_datetime = f"{event_date}T{end_time}:00+05:30"
    event_link = google_calendar_reader.create_event(event_name, event_location, event_description, start_datetime, end_datetime, attendees)
    
    print(f"📅 Event Created: {event_link}")

    # Ask for the date to summarize events
    print(f"🤖 Please enter the date you want a summary for (YYYY-MM-DD): ", end="")
    summary_date = input("👤 ")

    # Generate the summary for the given date
    documents = google_calendar_reader.load_data(start_date=summary_date, number_of_results=10)

    # Convert documents to LangChain format
    formatted_documents: List[LCDocument] = [LCDocument(page_content=doc) for doc in documents]

    # OpenAIEmbeddings uses text-embedding-ada-002
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(formatted_documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(split_documents, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(),
        memory=memory
    )

    query = f"Create a summary for what I am doing on the day: {summary_date}"
    result = qa.invoke({"question": query})
    print(f"📆 Summary for {summary_date}: {result['answer']}")

# Run the chatbot
interactive_chatbot()
