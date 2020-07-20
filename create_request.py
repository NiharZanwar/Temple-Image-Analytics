import requests
import base64
import random
import os
import json
from imutils import paths

request_save_data_flag=False
request_make_model_flag=False
request_predict_flag=True

def request_save_data(filename,category):
    #filename = "E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\411010\\Training data\\Door closed\\410010_CHB001_170420_003928_101_003928.jpg"

    f = open(filename, "rb").read()

    r = response = requests.post('http://localhost:5000/api/save_data', json={
        "temple_id": "410010",
        "image_type": "jpg",
        "train_test":"train",
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
    imagepaths=list(paths.list_images("E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\411010\\Training data"))
    for imagepath in imagepaths:
        category=imagepath.split(os.path.sep)[-2]
        request_save_data(imagepath,category)



if request_make_model_flag:
    model_id="410012"

    r = response = requests.post('http://localhost:5000/api/make_model', json={
        "temple_id": model_id,
        "forceful":True
    })
    # "image":
    # print(str(base64.b64encode(f).decode('utf-8')))
    print("For make_model")
    print(response.status_code)
    print(response.json())



if(request_predict_flag):
    filename="E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\410012\\2020-04-06\\410012_CHB001_060420_092648_101_092648.jpg"

    f = open(filename, "rb").read()

    r = response = requests.post('http://localhost:5000/api/predict', json={
        "temple id":"410013",
        "image type":"jpg",
        "image_name":filename.split(os.path.sep)[-1],
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