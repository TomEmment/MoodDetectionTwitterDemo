import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn import svm
from tensorflow import keras
from keras import models
from keras.utils import to_categorical
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
import joblib
import pathlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

cwd = os.getcwd()

ListWord = ["NANA"]
modelNeuralNet = keras.models.load_model("LSTMeuralNetModelAngryVsNone.tf")


for filename in os.listdir(cwd + "/Angry"):
    Final = cwd + "/Angry/" + filename
    f = open(Final, "r")
    try:
        Text = f.read()
        Sentance = Text.split(" ")
        for word in Sentance:
            if word in ListWord:
                Yes = True
                
            else:
                ListWord.append(word)

    except:
        print("Encoding error")



for filename in os.listdir(cwd + "/None"):
    Final = cwd + "/None/" + filename
    f = open(Final, "r")
    try:
        Text = f.read()
        Sentance = Text.split(" ")
        for word in Sentance:
            if word in ListWord:
                Yes = True
                
            else:
                ListWord.append(word)

    except:
        print("Encoding error")







text = " "


while text != "exit":




    text = input("Input a message to test: ")
    Sentance = text.split(" ")
    Vector = [0] * 60
    Position = 1
    for word in Sentance:
        Index = ListWord.index(word)
        Vector[Position] = Index
        Position = Position + 1




    Vector = np.array([Vector])
    Vector = Vector.reshape(Vector.shape[0],Vector.shape[1], 1)

    Vector = Vector.astype(np.float32)   
    predictions = modelNeuralNet.predict(Vector)
    print("Neural Net Predicts:                  ", predictions)


