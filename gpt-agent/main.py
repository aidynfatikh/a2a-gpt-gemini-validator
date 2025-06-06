import os
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

load_dotenv()
llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key=os.getenv("OPENAI_API_KEY"))

prompt = PromptTemplate(
    input_variables=["question"],
    template="You are a helpful assistant. Answer the question:\n{question}"
)

chain = prompt | llm

def ask_gpt(question: str) -> str:
    return chain.invoke({"question": question}).strip()

def validate_with_gemini(response: str) -> str:
    try:
        res = requests.post("http://localhost:3000/validate", json={"response": response})
        return res.json().get("evaluation", "No evaluation received.")
    except Exception as e:
        return f"Validation failed: {str(e)}"

def main():
    print("🤖 GPT + Gemini Validator (A2A)\n")
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            gpt_output = ask_gpt(user_input)
            print(f"\n🧠 GPT: {gpt_output}")
            evaluation = validate_with_gemini(gpt_output)
            print(f"🧠 Gemini Evaluation: {evaluation}\n")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
