import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
import requests
import joblib
from pickle import *
from flask import *
app = Flask(__name__)
v=joblib.load('vectorizer.pkl')
model=joblib.load('news.pkl')
port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
def translate(source_text,source):
    source_lang = source
    target_lang = 'en'
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={source_text}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        result_text = data[0][0][0]
        return result_text
    else:
        return 'could not translate'
def ttranslate(source_text,source):
    source_lang = 'en'
    target_lang = source
    url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={source_lang}&tl={target_lang}&dt=t&q={source_text}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        result_text = data[0][0][0]
        return result_text
    else:
        return 'could not translate'
@app.route('/')
def hello():
    return render_template('index.html',pred='')
@app.route('/', methods=['POST'])
def predict():
    print(request.form)
    fea = [str(x) for x in request.form.values()]
    s=translate(fea[1],fea[0])
    s=stemming(s)
    s=[s]
    s=v.transform(s)
    y=model.predict(s)[0]
    if y==1:
        return render_template('index.html',pred=ttranslate('The news is fake',fea[0]))
    return render_template('index.html',pred=ttranslate('The news is real',fea[0]))

if __name__ == '__main__':
    app.run(debug=True, port=8080)