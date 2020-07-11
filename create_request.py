import requests
import base64
import json

filename=""

f = open(filename, "rb").read()

r = response = requests.post('http://localhost:5000/api/predict', json={
    "temple id":"410010",
    "image type":"PNG",
    "image": str(base64.b64encode(f).decode('utf-8'))
})
# "image":
# print(str(base64.b64encode(f).decode('utf-8')))
print(response.status_code)
print(response.text)


# print({
#     "image": str(base64.encodebytes(f))
# })