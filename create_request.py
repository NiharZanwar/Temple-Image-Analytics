import requests
import base64
import random
import os
import json
from imutils import paths

request_save_data_flag=False
request_make_model_flag=True
request_predict_flag=False

def request_save_data(filename,category):
    #filename = "E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\411010\\Training data\\Door closed\\410010_CHB001_170420_003928_101_003928.jpg"

    f = open(filename, "rb").read()

    r = response = requests.post('http://localhost:5000/api/save_data', json={
        "temple_id": "410010",
        "image_type": "jpg",
        "train_test":"test",
        "category":category,
        "image_name":"image"+str(random.randint(1,10000000000)),
        "image": str(base64.b64encode(f).decode('utf-8'))
    })
    # "image":
    # print(str(base64.b64encode(f).decode('utf-8')))
    print("For save_data")
    print(response.status_code)
    print(response.text)


if request_save_data_flag==True:
    imagepaths=list(paths.list_images("E:\\PS1 SMARTi electronics\\Programs and Data\\CNN_Categorisation_test9"))
    for imagepath in imagepaths:
        category=imagepath.split(os.path.sep)[-2]
        request_save_data(imagepath,category)



if request_make_model_flag:
    model_id="410010"

    r = response = requests.post('http://localhost:5000/api/make_model', json={
        "temple_id": "410010",
        "forceful":True
    })
    # "image":
    # print(str(base64.b64encode(f).decode('utf-8')))
    print("For make_model")
    print(response.status_code)
    print(response.json())



if(request_predict_flag):
    filename=""

    f = open(filename, "rb").read()

    r = response = requests.post('http://localhost:5000/api/predict', json={
        "temple id":"410010",
        "image type":"PNG",
        "image": str(base64.b64encode(f).decode('utf-8'))
    })
    # "image":
    # print(str(base64.b64encode(f).decode('utf-8')))
    print("For prediction")
    print(response.status_code)
    print(response.text)



# print({
#     "image": str(base64.encodebytes(f))
# })