import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Any # <-- Add this import at the top
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))



# This prints every model your key can use!
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

# Load the API key from your .env file

# Configure the Gemini AI
# Using the flash model because it is incredibly fast (best for voice UI)
model = genai.GenerativeModel('gemini-2.5-flash') 

app = FastAPI()

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # <--- The "*"" means ALLOW ALL PORTS (Fixes the 8080 error!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    context_data: Optional[Any] = None

@app.get("/")
def read_root():
    return {"status": "Auto Vision AI Backend is Online"}

@app.post("/api/chat")
async def process_voice_command(request: ChatRequest):
    print(f"🎙️ User asked: {request.prompt}")
    
    # 2. Inject the live React data directly into the AI's brain!
    system_context = f"""
    You are 'Auto', an advanced AI fleet management assistant for 'Auto Vision'. 
    You are talking to the fleet manager out loud, so keep answers VERY concise (1 or 2 short sentences).
    
    CRITICAL CONTEXT: Here is the live, real-time JSON data from the user's fleet dashboard right now:
    {request.context_data}
    
    If the user asks about their vehicles, status, or sensors, use the JSON data above to give a specific, accurate answer.
    """
    
    full_prompt = f"{system_context}\n\nUser query: {request.prompt}"

    try:
        model = genai.GenerativeModel('gemini-2.5-flash') 
        response = model.generate_content(full_prompt)
        ai_text = response.text.replace('*', '') 
        
        print(f"🤖 AI Answered: {ai_text}")
        return {"answer": ai_text}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"answer": "I'm sorry, I am having trouble connecting to my neural network right now."}
