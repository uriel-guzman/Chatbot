import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

print(data)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words.extend(nltk.word_tokenize(pattern))
        docs.append(pattern)

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

