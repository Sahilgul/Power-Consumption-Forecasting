from crypt import methods
import os
# from turtle import title
import pandas as pd
import matplotlib.pyplot as plt
# import xgboost as xgb
from flask import Flask, render_template,redirect,url_for,request
import pickle
from flask import session



def max_load():
    with open('static/files/new_data' , 'rb') as i:
        new_data = pickle.load(i)
    new_data1 = new_data.loc[(new_data['MW_Prediction'] >= 2571) & (new_data['MW_Prediction'] <= 2600)]
    return new_data1

app=Flask(__name__)

app.config["UPLOAD_PATH"] = "static/files"

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('Login.html')

@app.route("/upload_file", methods=["GET", "POST"]) 
def upload_file():
    if request.method == 'POST':
        for i in request.files.getlist('file_name'):
            i.save(os.path.join(app.config['UPLOAD_PATH'],i.filename))
        return render_template('Index.html',msg='File Uploaded Successfully!')
    return render_template("upload.html", msg="Please Upload Dataset for Next Process!")


@app.route('/index', methods=['GET','POST'])
def index():
    return render_template('Index.html')

@app.route('/lstm_model', methods=['GET','POST'])
def lstm_model():    
    return render_template('lstm_model.html')

@app.route('/xgb_model', methods=['GET','POST'])
def xgb_model():
    return render_template('xgb_model.html')


@app.route('/show_xgb', methods=['GET','POST'])
def show_xgb():
    with open('static/files/new_data' , 'rb') as i:
        new_data = pickle.load(i)
    plt.ioff()
    fig =plt.figure(figsize=(16,8))
    plt.plot(new_data['MW_Prediction'],color='green')
    plt.xlabel('Date')
    plt.ylabel('Normalized power consumption scale kWh')
    plt.title('30 Days Future Predictions made by XGBoost Model')
    plt.savefig('static/charts_temp/chart1.png')
    return render_template('XGB-Gr1.html')

@app.route('/show_max',methods=['GET','POST'])
def show_max():
    new_data1 = max_load()
    plt.ioff()
    fig =plt.figure(figsize=(16,8))
    plt.plot(new_data1['MW_Prediction'],color='green',lw=5)
    plt.xlabel('Date')
    plt.ylabel('Normalized power consumption scale kWh')
    plt.title('Maximum Demand')
    plt.savefig('static/charts_temp/chart3.png')
    return render_template('XGB-Gr2.html')

@app.route('/forecast_lstm',methods=['GET','POST'])
def forecast_lstm():
    with open('static/files/next_predic' , 'rb') as i:
        next_predic = pickle.load(i)
    # new_data1 = max_load()
    plt.ioff()
    plt.figure(figsize=(20,10))
    plt.plot(next_predic,color='blue')
    plt.title('Next 30 Days Future Predictions [01/08/2018 - 01/09/2018]')
    plt.xlabel('Dates',fontsize=20)
    # plt.xticks(fontsize=15,rotation=45)
    # plt.yticks(fontsize=15,rotation=45)
    plt.ylabel('Prediction_kWh',fontsize=20)
    plt.savefig('static/charts_temp/chart4.png')
    return render_template('lstm-Gr2.html')

@app.route('/max_lstm',methods=['GET','POST'])
def max_lstm():
    with open('static/files/next_predic' , 'rb') as i:
        next_predic = pickle.load(i)
    max_data = next_predic.loc[(next_predic['next_predicted_days_value'] >= 2571) & (next_predic['next_predicted_days_value'] <= 3500)]
    plt.ioff()
    plt.figure(figsize=(16,8))
    plt.plot(max_data,color='blue',lw=5)
    plt.title('Next 30 Days Future Predictions [01/08/2018 - 01/09/2018]',fontsize=20)
    plt.xlabel('Dates',fontsize=20)
    plt.ylim([0,3000])
    plt.xticks(fontsize=18,rotation=45)
    plt.yticks(fontsize=18,rotation=45)
    plt.ylabel('Prediction_kWh',fontsize=20)
    plt.savefig('static/charts_temp/chart5.png')
    return render_template('lstm-Gr3.html')


if __name__ =='__main__':
    app.run(debug=True)
