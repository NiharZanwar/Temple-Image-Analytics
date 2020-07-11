from flask import Flask,render_template,request,abort
import TempleImagesNN
import json

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






# @app.route('/', methods=['GET'])
# def hello():
#     return "Hello"
#
#
# if __name__ == '__main__':
#     json_config = get_config('config.json')
#
#     upload_path = json_config['default_parameters']['upload_directory']
#     detected_path = json_config['default_parameters']['detected_directory']
#     # checking if directories for uploaded and detected images exist if not create them
#     check_dirs([upload_path, detected_path])
#
#     # loading the model before start of server
#     t0 = time.perf_counter()
#     detector = load_model(json_config['model_file']['default'], json_config['default_parameters']['default_speed'])
#     t1 = time.perf_counter()
#
#     print("Time elapsed in loading model:", t1 - t0)
#
#     app.run(debug=True, host=json_config['host'], port=json_config['port'])

if __name__=='__main__':
    app.run(debug=True)