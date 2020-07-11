import json


create_config_file=True
create_query_file=True

if create_config_file==True:
    data={}
    data["training data path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\Temple Original Images\\411010\\Training data"
    #data["hyperparameters"]={}
    data["log file path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\FlaskTest1\\log_file.txt"
    data["testing data path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\CNN_Categorisation_test9"
    data["save model path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\FlaskTest1"
    data["temple id"]="410010"

    with open('config.txt', 'w') as outfile:
        json.dump(data, outfile,indent=4)

if create_query_file==True:
    data={}
    data["path to models"]="E:\\PS1 SMARTi electronics\\Programs and Data\\FlaskTest1"
    data["query images"]="E:\\PS1 SMARTi electronics\\Programs and Data\\FlaskTest1\\query images"
    data["temple id"]="410010"
    data["log file path"] = "E:\\PS1 SMARTi electronics\\Programs and Data\\FlaskTest1\\log_file.txt"

    with open('query.txt', 'w') as outfile:
        json.dump(data, outfile,indent=4)

