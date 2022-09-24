from distutils.command.upload import upload
import imp
import os
import base64
import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import warnings
import xgboost as xgb

from io import BytesIO
from curses import flash
from fileinput import filename
from flask import Flask, render_template,redirect,url_for,request
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import FileField,SubmitField
from wtforms.validators import InputRequired
from xgboost import plot_importance, plot_tree
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def xgboost_clean(df):
    split_date = '01-Jan-2015'
    df_train = df.loc[df.index <= split_date].copy()
    df_test = df.loc[df.index > split_date].copy()

    def create_features(df, label=None):
        """
        Creates time series features from datetime index
        """
        df['date'] = df.index
        df['hour'] = df['date'].dt.hour
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['dayofyear'] = df['date'].dt.dayofyear
        df['dayofmonth'] = df['date'].dt.day
        df['weekofyear'] = df['date'].dt.weekofyear
        
        X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear']]
        if label:
            y = df[label]
            return X, y
        return X

    X_train, y_train = create_features(df_train, label='Production_kWh')
    X_test, y_test = create_features(df_test, label='Production_kWh')

    reg = xgb.XGBRegressor(n_estimators=1000,objective='reg:linear',max_depth=3,learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=True
            )

    df_test['MW_Prediction'] = reg.predict(X_test)
    df_all = pd.concat([df_test, df_train], sort=False)

    futures = pd.date_range('2018-08-03','2018-09-03', freq='1h')
    future_df = pd.DataFrame(index=futures)
    future_df['isFuture'] = True


    new_data = create_features(future_df)
    new_data['MW_Prediction'] = reg.predict(new_data)
    return new_data


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
        return render_template('upload.html',msg='File Uploaded Successfully!')
    return render_template("upload.html", msg="Not Uploaded!")


@app.route('/index', methods=['GET','POST'])
def index():
    return render_template('Index.html')

@app.route('/lstm_model', methods=['GET','POST'])
def lstm_model():    
    return render_template('lstm_model.html')

@app.route('/xgb_model', methods=['GET','POST'])
def xgb_model():
    return render_template('xgb_model.html')

@app.route('/show_lstm', methods=['GET','POST'])
def show_lstm():
    lstm_predictions = pd.read_csv('static/files/lstm_pred.csv')
    y_test = pd.read_csv('static/files/y_test.csv')
    plt.ioff()
    fig =plt.figure(figsize=(16,8))
    plt.plot(y_test,color='blue',label='Actual power consumption data')
    plt.plot(lstm_predictions, alpha=0.7, color='orange',label='Predicted power consumption data')
    plt.xlabel('Time')
    plt.ylabel('Normalized power consumption scale')
    plt.title('Predictions made by LSTM model')
    plt.legend()
    plt.savefig('static/charts_temp/chart2.png')
    return render_template('show_lstm.html')

@app.route('/show_xgb', methods=['GET','POST'])
def show_xgb():
    df = pd.read_csv('static/files/Malaysia_hourly.csv', index_col=[0], parse_dates=[0])
    new_data = xgboost_clean(df)
    plt.ioff()
    fig =plt.figure(figsize=(16,8))
    plt.plot(new_data['MW_Prediction'],color='blue')
    plt.title('Future Predictions')
    plt.savefig('static/charts_temp/chart1.png')
    return render_template('show_xgbost.html')
if __name__ =='__main__':
    app.run(debug=True)
    
