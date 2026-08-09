import requests
import os

url = 'http://127.0.0.1:5000/api/import'
file_path = os.path.join(os.path.dirname(__file__), 'sample_input.csv')
with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
    print('Status code:', response.status_code)
    print('Response JSON:', response.json())
