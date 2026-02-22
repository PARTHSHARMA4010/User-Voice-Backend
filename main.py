import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai


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
    allow_origins=["http://localhost:8080"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"status": "Auto Vision AI Backend is Online"}

@app.post("/api/chat")
async def process_voice_command(request: ChatRequest):
    print(f"🎙️ User asked: {request.prompt}")
    
    # 🧠 THE SYSTEM PROMPT
    # This gives the AI its personality and context before answering!
    system_context = """
    You are 'Auto', an advanced AI fleet management assistant for a dashboard called 'Auto Vision'. 
    The user is a fleet manager. 
    Keep your answers VERY concise, professional, and conversational. 
    Limit your response to 1 or 2 short sentences, as your text will be read out loud by a text-to-speech engine.
    """
    
    full_prompt = f"{system_context}\n\nUser query: {request.prompt}"

    try:
        # Call the Gemini AI
        response = model.generate_content(full_prompt)
        ai_text = response.text.replace('*', '') # Strip markdown asterisks so TTS doesn't read them
        
        print(f"🤖 AI Answered: {ai_text}")
        return {"answer": ai_text}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"answer": "I'm sorry, I am having trouble connecting to my neural network right now."}