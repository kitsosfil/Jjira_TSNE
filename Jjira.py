import numpy as np 
import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
import joblib

from flask import Flask, render_template, request, escape
from flask_debugtoolbar import DebugToolbarExtension

 
app = Flask(__name__)
app.debug = True
app.secret_key = 'development key'
toolbar = DebugToolbarExtension(app)


@app.route('/', methods = ['POST', 'GET'])
def main():
    classifier = joblib.load('classifier.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    if request.method == 'GET':
       return render_template('index.html', name = None)

    if request.method == 'POST':
        new_entry = request.form['entry']
        new_entry = [new_entry]
        v_entry = vectorizer.transform(new_entry)
       
        answer = classifier.predict(v_entry.toarray())
        return render_template('index.html', answer= answer)
            
if __name__ == "__main__":
    app.run(debug=True)