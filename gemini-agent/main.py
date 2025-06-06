from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("instructions.txt", "r") as f:
    instructions_template = f.read()

class ValidationRequest(BaseModel):
    response: str

@app.post("/validate")
async def validate(data: ValidationRequest):
    prompt = instructions_template.replace("{response}", data.response)
    response = model.generate_content(prompt)
    return {"evaluation": response.text.strip()}

if __name__ == "__main__":
    uvicorn.run("main:app", port=3000, reload=True)
