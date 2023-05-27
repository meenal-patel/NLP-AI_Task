import requests

# API endpoint URL
url = "http://localhost:8000/predict"

# Read the input JSON file
with open("test.json", "r") as f:
    json_data = f.read()

# Send the POST request
response = requests.post(url, json=json_data)

# Get the response JSON
output = response.json()
print(output)

# Extract relevant information from the response
question = output["question"]
category = output["category"]
answer = output["answer"]

# Print the results
print("Question:", question)
print("Category:", category)
print("Answer:", answer)
