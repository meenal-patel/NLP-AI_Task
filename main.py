from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import torch
import uvicorn
import nest_asyncio
import csv
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

nest_asyncio.apply()

app = FastAPI()

tokenizer_cls = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_cls = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

tokenizer_qa = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model_qa = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cls.to(device)
model_qa.to(device)

# Database connection or setup

DATABASE_URL = "postgresql://user:password@localhost/db_name"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# News API configuration
NEWS_API_URL = "https://api.example.com/news"
NEWS_API_KEY = "YOUR_API_KEY"

@app.post("/predict")
async def predict(question: str):
    # Task 1: Analyze and categorize the question
    inputs_cls = tokenizer_cls.encode_plus(question, add_special_tokens=True, return_tensors="pt")
    inputs_cls = inputs_cls.to(device)
    outputs_cls = model_cls(**inputs_cls)
    predicted_label = torch.argmax(outputs_cls.logits, dim=1).item()

    # Define the possible categories
    categories = ["company department", "employee role", "company news"]

    # Task 2: Retrieve relevant information based on the category
    category = categories[predicted_label]
    if category == "company department":
        # Retrieve department details from a CSV file
        with open("department.csv", "r") as csv_file:
            reader = csv.DictReader(csv_file)
            department_data = [row for row in reader]

        # Process department details and return the relevant information
        # ...

    elif category == "employee role":
        # Query the SQL database for employee details
        session = SessionLocal()
        # Perform database queries or operations
        # ...

        session.close()

    elif category == "company news":
        # Make a request to the news API to fetch company news
        headers = {"Authorization": f"Bearer {NEWS_API_KEY}"}
        params = {"query": "company news"}
        response = requests.get(NEWS_API_URL, headers=headers, params=params)
        news_data = response.json()

        # Process news data and return the relevant information
        # ...

    return {"question": question, "category": category, "answer": "relevant information"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
