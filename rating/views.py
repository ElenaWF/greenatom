from django.shortcuts import render

import pickle
import os
import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
STOP_WORDS = set(stopwords.words('english'))

RANDOM_STATE = 42


with open(os.path.join(os.path.dirname(__file__), 'files/count_status.pkl'), "rb") as f:
    count = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'files/model_status.sav'), "rb") as f:
    model_status = pickle.load(f)

with open(os.path.join(os.path.dirname(__file__), 'files/model_rating.sav'), "rb") as f:
    model_rating = pickle.load(f)


def decontract(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text


def strip_all_entities(text):
    text = text.replace('<br />', ' ').replace('\r', '').replace('\n', ' ').lower() #remove <br />, \n and \r and lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text) #remove links and mentions
    text = re.sub(r'[^\x00-\x7f]',r'', text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = [word for word in text.split() if word not in STOP_WORDS]
    text = ' '.join(text)
    return text


def remove_mult_spaces(text):
    return re.sub("\s\s+" , " ", text)


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words, get_wordnet_pos(words)) for words in tokenized])


def deep_clean(text):
    text = decontract(text)
    text = strip_all_entities(text)
    text = remove_mult_spaces(text)
    text = lemmatize(text)
    return text


def prep_text(text):
    text = deep_clean(text)
    text = pd.DataFrame({'as': [text], })
    count_tf_idf = count.transform(text['as'])
    return count_tf_idf


def index(request):
    return render(request, "index.html")


def postuser(request):
    # получаем из данных запроса POST, отправленные через форму данные
    name = request.POST.get("name", "Undefined")
    predicted_status = model_status.predict(prep_text(name))
    predicted_rat = model_rating.predict(prep_text(name))[0]

    if predicted_status == 1:
        pred = 'Позитивный'
    else:
        pred = 'Негативный'

    return render(request, 'result.html', locals())