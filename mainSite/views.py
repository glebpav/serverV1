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

VOCAB_SIZE = 6500
weight_of_words = 0.25

stemer = RussianStemmer()
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}
stem_count = Counter()
tokenizer = TweetTokenizer()

f = open('vocab.txt', 'r')
f1 = open('word_tone.txt', 'r')
vocab = f.read().split()
word_tone = f1.read().split()

token_2_idx = {vocab[i].replace(';', ''): i for i in range(VOCAB_SIZE)}
word_token = {}

for word in word_tone:
    mas = word.split(";")
    dict = {mas[0]: mas[1]}
    word_token.update(dict)

f.close()
f1.close()


def build_model(learning_rate=0.1):
    tf.reset_default_graph()
    net = tflearn.input_data([None, VOCAB_SIZE])
    net = tflearn.fully_connected(net, 300, activation='ReLU')
    net = tflearn.fully_connected(net, 50, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net,
        optimizer='Adam',
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
    print(positive_prob, " -- ", article)

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


def multiple_answer(tone_from_nn, tone_from_unknown_words):
    return tone_from_nn * (1 - weight_of_words) + tone_from_unknown_words * weight_of_words


def article_to_vector_clearly(article, count_unknowns=True):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    unknown_word_count = 0
    unknown_word_sum = 0
    for token in tokenizer.tokenize(article):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif count_unknowns:
            tone = word_token.get(stem, None)
            if tone:
                unknown_word_count += 1
                if tone == 'NEUT':
                    unknown_word_sum += 0.5
                elif tone == 'PSTV':
                    unknown_word_sum += 1
                elif tone == 'NGTV':
                    unknown_word_sum += 0
    if unknown_word_count == 0:
        return [vector, None]
    print(unknown_word_sum / unknown_word_count)
    return [vector, unknown_word_sum / unknown_word_count]


def test_article_the_best(title, description):
    title_vector = article_to_vector_clearly(title, True)
    description_vector = article_to_vector_clearly(description, True)
    title_positive_prob = model.predict([title_vector[0]])[0][1]
    description_positive_prob = model.predict([description_vector[0]])[0][1]

    positive_prob = (title_positive_prob + description_positive_prob) / 2

    if title_vector[1] is not None and description_vector[1] is not None:
        positive_prob = multiple_answer((title_positive_prob + description_positive_prob) / 2,
                                        (title_vector[1] + description_vector[1]) / 2)
    elif title_vector[1] is not None and description_vector[1] is None:
        positive_prob = multiple_answer((title_positive_prob + description_positive_prob) / 2,
                                        (title_vector[1]) / 2)
    elif title_vector[1] is None and description_vector[1] is not None:
        positive_prob = multiple_answer((title_positive_prob + description_positive_prob) / 2,
                                        (description_vector[1]) / 2)

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
        them = stemer.stem(request.GET['them'])

        url = "http://newsapi.org/v2/everything?q=" + them + "&sortBy=publishedAt&apiKey=02293a96d6814c4783965bf628c2d2ee"
        file = requests.get(url)
        datas = json.loads(file.text)
        dataa = []
        for data in datas['articles']:
            try:
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
                    tone5 = test_article_the_best(str(title_of_article), str(description_of_article))

                    if detect(data['description']) == 'ru' and request.GET.get('q'):
                        just_counter += 1
                        dataa.append({"title": data['title'], "description": data['description'], "url": data['url'],
                                      "toneTitle": ("заголовок : " + tone),
                                      "toneDescription": ("описание : " + tone2),
                                      "toneAll": ("в общем : " + tone3),
                                      "toneAve": ("тональность : " + tone4),
                                      "toneAveBest": ("тональность : " + tone5),
                                      "counter": just_counter})
                    elif not request.GET.get('q'):
                        just_counter += 1
                        dataa.append({"title": data['title'], "description": data['description'], "url": data['url'],
                                      "toneTitle": ("заголовок : " + tone),
                                      "toneDescription": ("описание : " + tone2),
                                      "toneAll": ("в общем : " + tone3),
                                      "toneAve": ("тональность : " + tone4),
                                      "toneAveBest": ("тональность : " + tone5),
                                      "counter": just_counter})
            except Exception:
                print("nope")

        data = {"message": dataa, "theme": them}
        return render(request, "mainSite/showThems.html", context=data)


def hello(request):
    return render(request, 'mainSite/index.html', context={})
