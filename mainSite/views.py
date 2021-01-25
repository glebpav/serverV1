import requests
import tflearn
import tensorflow as tf
import numpy as np
import re

from langdetect import detect
from nltk.stem.snowball import RussianStemmer
from nltk.tokenize import TweetTokenizer
from collections import Counter
from googleapi import google
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
from GoogleNews import GoogleNews

from .serializers import *
from .models import *

VOCAB_SIZE = 6500
weight_of_words = 0.15
num_page = 1

sites = {"лента ру", "рбк", "риа", "медуза"}

stemer = RussianStemmer()
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}
stem_count = Counter()
tokenizer = TweetTokenizer()

f = open('vocab.txt', 'r')
f1 = open('word_tone.txt', 'r')
fileBadWords = open('badWordsVocab.txt', 'r')

vocab = f.read().split()
word_tone = f1.read().split()
badWordsVocab = fileBadWords.read().split()

token_2_idx = {vocab[i].replace(';', ''): i for i in range(VOCAB_SIZE)}
word_token = {}
badWordsDict = {}

for word in word_tone:
    mas = word.split(";")
    dict = {mas[0]: mas[1]}
    word_token.update(dict)

for word in badWordsVocab:
    badWordsDict.update({word:word})

f.close()
f1.close()
fileBadWords.close()

print(badWordsDict)

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
            vector[idx] = 1

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


def multiple_answer(tone_from_nn, tone_from_unknown_words):
    return tone_from_nn * (1 - weight_of_words) + tone_from_unknown_words * weight_of_words


def article_to_vector_clearly(article, count_unknowns=True):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    unknown_word_count = 0
    unknown_word_sum = 0
    for token in tokenizer.tokenize(article):
        stem = get_stem(token)
        badWord = badWordsDict.get(stem, None)
        idx = token_2_idx.get(stem, None)

        if badWord is not None:
            print(badWord)

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
    return [vector, unknown_word_sum / unknown_word_count]


def test_article_the_best(title, description):
    title_vector = article_to_vector_clearly(title, True)
    description_vector = article_to_vector_clearly(description, True)
    title_positive_prob = model.predict([title_vector[0]])[0][1]
    description_positive_prob = model.predict([description_vector[0]])[0][1]
    #print(title_positive_prob)
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
        return ["нейтрально", "info"]
    elif positive_prob > 0.65:
        return ["позитивно  " + str(int(((positive_prob * 100 - 65) / 35) * 100)), "success"]
    else:
        return ["негативно  " + str(int(((0.36 - positive_prob) / 0.36) * (-100))), "danger"]


def test_article_the_best_api(title, description):
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

    return positive_prob



model = build_model(learning_rate=0.75)
model.load("model")


def find(request):
    if request.method == 'GET':
        just_counter = 0
        theme = request.GET['them']

        dataa = []
        googlenews = GoogleNews()

        for site in sites:
            googlenews.clear()
            googlenews = GoogleNews(lang='ru')
            googlenews.search(theme + " " + site)
            googlenews.get_page(1)
            for result in googlenews.results():
                try:

                    description_of_article = [result["desc"]]
                    title_of_article = [result["title"]]
                    all_of_article = [result["title"] + result["desc"]]
                    media = result["media"]

                    if media != "Lenta" and media != "РИА Новости" and media != "РБК" and media != "Meduza":
                        continue;

                    #print(media)

                    tone = test_tweet(str(title_of_article))
                    tone2 = test_tweet(str(description_of_article))
                    tone3 = test_tweet(str(all_of_article))
                    tone4 = test_article_better(str(title_of_article), str(description_of_article))
                    tone5 = test_article_the_best(str(title_of_article), str(description_of_article))

                    if detect(result["desc"]) == 'ru' and request.GET.get('q'):
                        just_counter += 1
                        dataa.append({"title": title_of_article[0], "description": description_of_article[0],
                                      "url": result["link"],
                                      "toneTitle": ("заголовок : " + tone),
                                      "toneDescription": ("описание : " + tone2),
                                      "toneAll": ("в общем : " + tone3),
                                      "toneAve": ("тональность : " + tone4),
                                      "toneAveBest": ("тональность : " + tone5[0]),
                                      "typeBox": (tone5[1]),
                                      "counter": just_counter})
                    elif not request.GET.get('q'):
                        just_counter += 1
                        dataa.append({"title": title_of_article[0], "description": description_of_article[0],
                                      "url": result["link"],
                                      "toneTitle": ("заголовок : " + tone),
                                      "toneDescription": ("описание : " + tone2),
                                      "toneAll": ("в общем : " + tone3),
                                      "toneAve": ("тональность : " + tone4),
                                      "toneAveBest": ("тональность : " + tone5[0]),
                                      "typeBox": (tone5[1]),
                                      "counter": just_counter})
                except Exception:
                    1

        data = {"message": dataa, "theme": theme}
        if len(dataa) == 0:
            return render(request, "mainSite/not_found_page.html", context=data)

        return render(request, "mainSite/showThems.html", context=data)


def hello(request):
    return render(request, 'mainSite/index.html', context={})


class News_view(APIView):

    def get(self, request, theme):
        news = []
        for site in sites:

            googlenews.clear()
            googlenews = GoogleNews(lang='ru')
            googlenews.search(theme + " " + site)

            for result in googlenews.get_page(1):
                item = News()

                description_of_article = [result["desc"]]
                title_of_article = [result["title"]]
                media = result["media"]
                #print(media)

                """tone = test_tweet(str(title_of_article))
                tone2 = test_tweet(str(description_of_article))
                tone3 = test_tweet(str(all_of_article))
                tone4 = test_article_better(str(title_of_article), str(description_of_article))"""
                tone5 = test_article_the_best_api(str(title_of_article), str(description_of_article))

                item.title = title_of_article[0]
                item.body = description_of_article[0]
                item.url = result["link"]
                item.rating = tone5



                news.append(item)

        serializer = NewsSerializer(news, many=True)
        return Response({"status":"ok","news": serializer.data})

    def post(self, request):
        article = request.data.get('article')
        # Create an article from the above data
        serializer = NewsSerializer(data=article)
        if serializer.is_valid(raise_exception=True):
            article_saved = serializer.save()
        return Response({"success": "Article '{}' created successfully".format(article_saved.title)})


class User_check(APIView):
    def post(self, request):
        found_user_with_login = False
        checking_user = request.data.get("user")
        users = Person.objects.all()

        for user in users:
            if user.login == checking_user["login"]:
                found_user_with_login = True
                if user.password == checking_user["password"]:
                    serializer = UserSerializer(user)
                    return Response({"status":"ok","user": serializer.data})

        if found_user_with_login:
            return Response({"status":"bad request","trouble": "password is incorrect"})
        else:
            return Response({"status":"bad request","trouble": "no user with such login"})


class User_view(APIView):

    def get(self, request):
        users = Person.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response({"users": serializer.data})

    def post(self, request):
        user_checking = request.data.get("user")
        serializer = UserSerializer(data=user_checking)
        if serializer.is_valid(raise_exception=True):
            users = Person.objects.all()
            for user in users:
                if user.login == user_checking["login"]:
                    return Response({"status": "bad response", "trouble": "User with such login is already exists"})

        serializer.save()
        return Response({"status": "ok"})

    def put(self, request, pk):
        saved_article = get_object_or_404(Person.objects.all(), pk=pk)
        data = request.data.get('user')
        serializer = UserSerializer(instance=saved_article, data=data, partial=True)
        if serializer.is_valid(raise_exception=True):
            user_saved = serializer.save()
        return Response({
            "status":"ok","user": serializer.data
        })

    def delete(self, request, pk):
        # Get object with this pk
        article = get_object_or_404(Person.objects.all(), pk=pk)
        article.delete()
        return Response({
            "message": "Person with id `{}` has been deleted.".format(pk)
        }, status=204)
