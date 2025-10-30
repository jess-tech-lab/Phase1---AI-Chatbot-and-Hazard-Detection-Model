import traceback
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import base64
from langchain_core.messages import HumanMessage
import whisper

# Import services
from mycalendar import CalendarService
from convo import Conversation
from maindelegator import delegation

import os
from dotenv import load_dotenv

from build_qdrant import get_delegator_client


# This line MUST run before any code that initializes the LLM
load_dotenv()

CONF_THRESHOLD = 0.8  # tweak between 0.7–0.9 depending on embedding scale
 # Can be False if created/built already
retriever_client = get_delegator_client(recreate=False, rebuild=True)

class Processor:
    """Handle different types of LLM actions through a pipeline system."""
    
    def __init__(self):

        #model = whisper.load_model("base")
        # Initialize the LLM (Gemini for testing)
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Defined actions for the LLM
        #print("we're initialising")
        print("JH - Pipeline -> Handle action delegation")
        self.actions = {
            'calendar': CalendarService(self.llm).handle_calendar,
            'pill': delegation.delegate,
            'call': delegation.delegate, 
            'personal': delegation.delegate
        }
    
    async def process_audio(self, audio: bytes, mime_type: str = 'audio/wav') -> Dict[str, Any]:
        model = whisper.load_model("base")
        """Transcribe raw audio to text and process it through the LLM action pipeline."""
        try:
            #print("srsly bruhhhhhhhhhh")
            # Convert audio to base64
            #audio_base64 = base64.b64encode(audio).decode('utf-8')
            #transcript="Appointment for tomorrow"
            #print(f"Transcript: {transcript}")
            # Create a message with the audio
            """
            message = HumanMessage(
                content=[
                    {"type": "text",  "text": "Transcribe the audio and if you can't just output the following: I would like an appointment for tomorrow"},
                    {"type": "media", "data": audio, "mime_type": mime_type},
                ]
            )
            
            # Get the transcript from Gemini
            response = await self.llm.acall([message])
            transcript = response.content.strip()
            print("We've go a transcript yall")
            print(f"Transcript: {transcript}")
        
            """
            result = model.transcribe(audio)
            print("Transcription:", result["text"])
            transcript = result["text"]
            
            # Route the transcript to the appropriate handler
            return await self.process_command(transcript)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}"
            }
    
    async def process_command(self, command: str) -> Dict[str, Any]:
        """Process a voice command through the LLM action pipeline."""
        try:
            # Determine the action type from the command
            action_type = await self._determine_action_type(command)
            print("Action Type: " + action_type)
            
            # Get the function for the determined action type
            handler = self.actions.get(action_type)

            # Check if handler is defined
            if not handler:
                return {
                    "status": "error",
                    "message": f"No handler for action type: {action_type}"
                }

            # ***************************************************************
            # ** MODIFIED RAG Step: Check if action type requires context **
            RAG_ACTIONS = {'pill', 'call', 'personal'}
            context = ""
            
            # ** Check if the action type is in the set of RAG actions **
            print(f"Checking if action type '{action_type}' requires RAG context...")
            if action_type in RAG_ACTIONS and retriever_client:
                print(f"Action type '{action_type}' requires RAG context. Retrieving from Qdrant...")
                try:
                    retrieved_results = retriever_client.query(prompt=command, k=2)
                    print(f"Retrieved {retrieved_results} results from Qdrant.")

                    # Get best category and score
                    if retrieved_results:
                        best_cat = max(retrieved_results, key=retrieved_results.get)
                        best_score = retrieved_results[best_cat]
                    else:
                        best_cat = None
                        best_score = 0.0

                    if best_score < CONF_THRESHOLD:
                        print(f"Low confidence ({best_score:.2f}) — ignoring Qdrant result.")
                        context = ""  # treat as generic
                    else:
                        context = best_cat  # pass persona key
                        print(f"Qdrant context retrieved:\n{context}")
                except Exception as e:
                    print(f"Error retrieving from Qdrant: {str(e)}")
                    import traceback
                    traceback.print_exc()

                return await handler(command, context=context)

            elif action_type == "calendar":
                print(f"Handling calendar action — no RAG context needed.")
                return await handler(command)
            # ***************************************************************
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}"
            }
    
    async def _determine_action_type(self, command: str) -> str:
        """Determine action type from command text"""

        print("Printing the command" + str(command))
        
        prompt = f"""
        Determine the action type based on the command: "{command}"
        Possible action types are: calendar, pill, call, personal. 
        Reply with a one word action type in lowercase.
        """
        try:
            # Call the LLM to determine the correct action type
            response = await self.llm.ainvoke(prompt)
            print(response.content.strip())
            return response.content.strip()
        except Exception as e:
            print("Error determining action type: " + str(e))
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}"
            }