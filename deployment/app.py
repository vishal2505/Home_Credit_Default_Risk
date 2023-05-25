from flask import Flask, jsonify, request
import pickle
import sqlite3
import flask
import pandas as pd
import os
import prediction_pipeline as pp

app = Flask(__name__)

#home page
@app.route('/', methods = ['GET'])
def home_page():
    return flask.render_template('index.html')

#prediction page
@app.route('/predict', methods = ['POST', 'GET'])
def predict():
	#return flask.render_template('predict.html')
    customer_ids = get_customer_ids_from_csv()    
    return flask.render_template('prediction.html', customer_ids=customer_ids)

@app.route('/customer_details')
def customer_details():
    customer_ids = get_customer_ids_from_csv()
    customer_id = request.args.get('customer_id')
    customer_details = get_customer_details_from_csv(customer_id)
    print(customer_details)
    return flask.render_template('prediction.html', customer_ids=customer_ids, customer_details=customer_details, customer_id=customer_id)

#results page
@app.route('/result', methods = ['POST', 'GET'])
def result():
     #customer_id = request.args.get('customer_id')
     customer_id = '100013'
     prediction = pp.predict_class(customer_id)
     return flask.render_template('result.html', prediction=prediction)


def read_csv_file():
    #return pd.read_csv("./deployment/data/application_test.csv").head(100)
    return pd.read_csv("./deployment/data/application_test.csv").iloc[:350 , :20]

def get_customer_ids_from_csv():
    print(os.getcwd())
    df = read_csv_file()
    sk_curr_ids = df['SK_ID_CURR'].tolist()
    return sk_curr_ids

def get_customer_details_from_csv(sk_id_curr):
    df = read_csv_file()
    print(sk_id_curr)
    if  int(sk_id_curr) in df.values:
        return {
            'headers': df.columns.values.tolist(),
            'values': df[df['SK_ID_CURR'] == int(sk_id_curr)].values.flatten().tolist()
        }
    else:
        return None

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0', port=8888)