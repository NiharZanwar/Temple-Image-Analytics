from flask import Flask,render_template,request,abort
import TempleImagesNN
import json
import os
import sys

app = Flask(__name__)



@app.route('/training', methods=['GET','POST'])
def get_training_data():
    if (request.method=='GET'):
        return render_template("training_form.html")

    elif (request.method == 'POST'):
        form=request.form

        print(form['config_file'])
        trainer=TempleImagesNN.TempleNNTrainer(str(form['config_file']))

        return render_template("training_done.html",trainer=trainer)


@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if (request.method=='GET'):
        return render_template("get_query.html")

    elif (request.method=='POST'):
        form=request.form

        print("Query file is",form['query_file'])
        predictor=TempleImagesNN.TempleImagesPredictor()
        predictor.parse_query_file(str(form['query_file']))
        response=predictor.predict()

        return render_template('prediction_done.html',classes=response)


@app.route('/api/predict',methods=['GET','POST'])
def predict_json_request():
    if(request.method=='POST'):
        predictor=TempleImagesNN.TempleImagesPredictor()
        predictor.set_attributes(path_to_models="E:\\PS1 SMARTi electronics\\Programs and Data\\FlaskTest1")
        predictor.parse_query_json(request.get_json())
        response=predictor.predict()
        if response["error_msg"]!="All OK":
            return(json.dumps(str(response)),400)
        else:
            return(json.dumps(str(response)))


config={}

def parse_config_json(config_file_path):
    global config
    # Opening and reading contents of file as json
    with open(config_file_path, 'r') as config_file:
        config_json = json.load(config_file)

    # Now we have the json file. We'll set the attributes accordingly
    config["training_data_path"]=config_json["training_data_path"]
    config["testing_data_path"]=config_json["testing_data_path"]
    config["models_path"]=config_json["models_path"]
    config["logs_path"]=config_json["logs_path"]

if __name__=='__main__':
    path_to_config=""
    #Check if config file exists in specified path
    if not os.path.isfile(path_to_config):
        print("Config file doesnt exist. Exiting")
        sys.exit()

    else:
        parse_config_json(path_to_config)





    app.run(debug=True)