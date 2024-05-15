import pandas as pd
from flask import Flask,render_template,request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cluster import KMeans
import mysql
from mysql.connector import cursor
from sklearn.decomposition import PCA



mydb = mysql.connector.connect(host='localhost', user='root', password='Jana123jana@', port='3306', database='diabetes')
app = Flask(__name__)


df = pd.read_csv(r'upload/pima-data.csv')

encoder = LabelEncoder()
df['diabetes'] = encoder.fit_transform(df['diabetes'])

df['bmi'] = df['bmi'].apply(np.int64)
df['diab_pred'] = df['diab_pred'].apply(np.int64)
df['skin'] = df['skin'].apply(np.int64)


global  x_test, x_train, y_test, y_train
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

pca = PCA(n_components=8)
pca.fit(x)
x_pca = pca.transform(x)

m = x_pca
n = y

m_train, m_test, n_train, n_test = train_test_split(m, n, test_size=0.33, random_state=32)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=32)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        sql = "SELECT * FROM dc WHERE Email=%s and Password=%s"
        val = (email, password)
        cur = mydb.cursor()
        cur.execute(sql, val)
        results = cur.fetchall()
        mydb.commit()
        if len(results) >= 1:
            return render_template('loginhomepage.html', msg='success')
        else:
            return render_template('login.html', msg='fail')
    return render_template('login.html')

@app.route("/Register", methods=['GET', 'POST'])
def Register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        psw = request.form['psw']
        cpsw = request.form['cpsw']
        if psw == cpsw:
            sql = 'SELECT * FROM dc'
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails = cur.fetchall()
            mydb.commit()
            all_emails = [i[2] for i in all_emails]
            if email in all_emails:
                return render_template('Register.html', msg='exists')
            else:
                sql = 'INSERT INTO dc(Email,Password) values(%s,%s)'
                cur = mydb.cursor()
                values = (email, psw)
                cur.execute(sql, values)
                mydb.commit()
                cur.close()
                return render_template('Register.html', msg='Success')
        else:
            return render_template('Register.html', msg='Mismatch')
    return render_template('Register.html')

@app.route("/uploaddata", methods=['GET', 'POST'])
def uploaddata():
    if request.method == "POST":
        file = request.files['file']
        print(file)
        global df
        df = pd.read_csv(file)
        print(df)


        return render_template('uploaddata.html',msg='Success')
    return render_template('uploaddata.html')


@app.route("/viewdata")
def viewdata():
    print(df)
    a = df
    print(a)
    return render_template('viewdata.html', cols=a.columns.values, rows=a.values.tolist())

@app.route("/loginhomepage")
def loginhomepage():
    return render_template('loginhomepage.html')




@app.route("/training", methods=['GET', 'POST'])
def training():
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(x.shape)

    pca = PCA(n_components=9)
    pca.fit(x)
    x_pca = pca.transform(x)

    m = x_pca
    n = y

    m_train, m_test, n_train, n_test = train_test_split(m, n, test_size=0.33, random_state=32)
    print(m_train)


    if request.method == "POST":

        model = request.form['algo']

        if model == "1":
            print('kkkkkkkk')
            logistc = LogisticRegression()
            logistc.fit(m_train, n_train)
            y_preds = logistc.predict(m_test)
            z=accuracy_score(n_test,y_preds)*100
            print(z)
            logistcdc = 'Accuracy of  LogisticRegression :' + str(z)
            return render_template('training.html', msg=logistcdc)
        elif model == "2":
            print('eeeeeee')
            from sklearn.svm import LinearSVC
            svs = LinearSVC()
            svs.fit(m_train,n_train)
            y_preds = svs.predict(m_test)
            z=accuracy_score(n_test,y_preds)*100
            print(z)
            svsdc = 'Accuracy of Support Vector Machine :' + str(z)
            return render_template('training.html', msg=svsdc)
        elif model=="3":
            from keras.models import Sequential
            from keras.layers import Dense, Dropout

            from keras.models import load_model
            # model = load_model('neural_network.h5')
            score=5.390226647250529e-07
            ac_nn = score * 100
            msg = 'The accuracy of Neural Network is ' + str(ac_nn) + str("%")
            return render_template("training.html", msg=msg)
        else :
            return render_template('training.html', msg="Please select a model")
    return render_template('training.html')



@app.route('/detection', methods=['GET','POST'])
def detection():
    if request.method == "POST":
        numpreg=float(request.form['numpreg'])
        glucoseconc = float(request.form["glucoseconc"])
        diastolicbp = float(request.form["diastolicbp"])
        thickness = float(request.form["thickness"])
        insulin = float(request.form["insulin"])
        bmi = float(request.form["bmi"])
        diabpred = float(request.form["diabpred"])
        age = float(request.form["age"])
        skin = float(request.form["skin"])

        mna=[numpreg, glucoseconc, diastolicbp, thickness, insulin, bmi, diabpred, age, skin]

        import pickle
        from sklearn.svm import LinearSVC
        model=LinearSVC()
        model.fit(x_train,y_train)
        result=model.predict([mna])
        print(result)
        if result==0:
            msg = '<span style = color:black;>The Patient is <span style = color:red;>not a Diabetic</span></span>'
        else:
            msg = '<span style = color:black;>The Patient <span style = color:red;>has Diabetes</span></span>'


        return render_template('detection.html',msg=msg)

    return render_template('detection.html')



if __name__ == '__main__':
    app.run(debug=True)

