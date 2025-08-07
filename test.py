import requests

url = "http://127.0.0.1:8000/upload/"
data = {
    "title": "Sample Document",
    "content": "This is the content of the document."
}

response = requests.post(url, json=data)

print(response.status_code)  # Should print 200 if the request is successful
print(response.json())  # Print the response data
