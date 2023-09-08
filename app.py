import pickle
import os
from flask import Flask,request,app,jsonify,url_for,render_template
import pandas as pd
import numpy as np


app = Flask(__name__)

model=pickle.load(open("reg_model.pkl","rb"))
scalar=pickle.load(open("scalling.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    output_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(output_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True,port=5002)