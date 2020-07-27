import json
import os


create_config_file=True
create_query_file=False

if create_config_file==True:
    data={}
    data["training_data_path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\DockerDirectoryStructure\\training data"
    #data["hyperparameters"]={}
    data["logs_path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\DockerDirectoryStructure\\logs"
    data["testing_data_path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\DockerDirectoryStructure\\testing data"
    data["models_path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\DockerDirectoryStructure\\models"
    data["app_path"]="E:\\PS1 SMARTi electronics\\Programs and Data\\DockerDirectoryStructure\\app"
    data["config_path"] = "E:\\PS1 SMARTi electronics\\Programs and Data\\DockerDirectoryStructure\\config"

    os.chdir(data["config_path"])
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

