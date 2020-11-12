import requests
import tflearn
import tensorflow as tf
import numpy as np
import re

from langdetect import detect
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer
from collections import Counter
from .models import *
from django.shortcuts import render

VOCAB_SIZE = 5000
stemer = RussianStemmer()
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}
stem_count = Counter()
tokenizer = TweetTokenizer()
f = open('text.txt', 'r')
vocab1 = f.read()
f.close()
vocab = vocab1.split()
token_2_idx = {vocab[i]: i for i in range(VOCAB_SIZE - 2)}


def build_model(learning_rate=0.1):
    tf.reset_default_graph()
    net = tflearn.input_data([None, VOCAB_SIZE])
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net,
        optimizer='sgd',
        learning_rate=learning_rate,
        loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model


def get_stem(token):
    stem = stem_cache.get(token, None)
    if stem:
        return stem
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem


def article_to_vector(article, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(article):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif show_unknowns:
            print("Unknown token: {}".format(token))
    return vector


def test_tweet(article):
    tweet_vector = article_to_vector(article, False)
    positive_prob = model.predict([tweet_vector])[0][1]
    if 0.35 < positive_prob < 0.66:
        return "нейтрально"
    elif positive_prob > 0.65:
        return "позитивно  " + str(int(((positive_prob * 100 - 65) / 35) * 100))
    else:
        return "негативно  " + str(int(((0.36 - positive_prob) / 0.36) * (-100)))


def test_article_better(title, description):
    title_vector = article_to_vector(title, True)
    description_vector = article_to_vector(description, True)
    title_positive_prob = model.predict([title_vector])[0][1]
    description_positive_prob = model.predict([description_vector])[0][1]

    positive_prob = (title_positive_prob + description_positive_prob) / 2

    if 0.35 < positive_prob < 0.66:
        return "нейтрально"
    elif positive_prob > 0.65:
        return "позитивно  " + str(int(((positive_prob * 100 - 65) / 35) * 100))
    else:
        return "негативно  " + str(int(((0.36 - positive_prob) / 0.36) * (-100)))


model = build_model(learning_rate=0.75)
model.load("model")


def find(request):
    if request.method == 'GET':
        just_counter = 0
        them = request.GET['them']
        url = "http://newsapi.org/v2/everything?q=" + them + "&sortBy=publishedAt&apiKey=02293a96d6814c4783965bf628c2d2ee"
        file = requests.get(url)
        datas = json.loads(file.text)
        dataa = []
        for data in datas['articles']:
            description_of_article = [data['description']]
            title_of_article = [data['title']]
            all_of_article = [data['title'] + data['description']]
            flag = 1
            for data1 in dataa:
                if data['title'] == data1['title']:
                    flag = 0
            if flag != 0:
                tone = test_tweet(str(title_of_article))
                tone2 = test_tweet(str(description_of_article))
                tone3 = test_tweet(str(all_of_article))
                tone4 = test_article_better(str(title_of_article), str(description_of_article))

                # for tweet in article_for_testing:
                #   tone = (test_tweet(tweet))

                if detect(data['description']) == 'ru' and request.GET.get('q'):
                    just_counter += 1
                    dataa.append({"title": data['title'], "description": data['description'], "url": data['url'],
                                  "toneTitle": ("заголовок : " + tone),
                                  "toneDescription": ("описание : " + tone2),
                                  "toneAll": ("в общем : " + tone3),
                                  "toneAve": ("среднее : " + tone4), "counter": just_counter})
                elif not request.GET.get('q'):
                    just_counter += 1
                    dataa.append({"title": data['title'], "description": data['description'], "url": data['url'],
                                  "toneTitle": ("заголовок : " + tone),
                                  "toneDescription": ("описание : " + tone2),
                                  "toneAll": ("в общем : " + tone3),
                                  "toneAve": ("среднее : " + tone4), "counter": just_counter})

        data = {"message": dataa, "theme": them}
        return render(request, "mainSite/showThems.html", context=data)


def hello(request):
    return render(request, 'mainSite/index.html', context={})
