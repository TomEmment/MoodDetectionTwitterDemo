import string
import tweepy
import csv
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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
import time
dataframe = pd.read_csv("ALL.csv")

stop_words = set(stopwords.words('english'))
####input your credentials here
consumer_key = 'NGeDyT0HtMgKMZBkJuC9icsKj'
consumer_secret = 'IYqpeBiUR4tHhp94Zmw9CF9nBAlwBUKThbIOfclSS88JkPOb9w'
access_token = '1318227698762874884-xRjxzFz3yaLRzgMsy25sOm0pBvtU5n'
access_token_secret = '6PtBMQ9Yc9sKyoOzrWnuCNfm4MPwysy0T1yyEtMBzxoLR'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
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

y = dataframe["Label"]
x = dataframe["Text"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=23)

cv = CountVectorizer()

features = cv.fit_transform(x_train.astype('U').values)
features_test = cv.transform(x_test.astype('U').values)
text = ""

modelSVM = joblib.load("SVMHappyVsSadVsNonelinear0.1.pkl")

SeenTweets = []
with open('Seen.txt', 'r') as fp:
    line = fp.readline()
    while line:
        SeenTweets.append(line[:-1])
        line = fp.readline()
while True:
    Alltweet = api.mentions_timeline(1)    
    fp = open('Seen.txt', 'a')
    for Message in Alltweet:
        tweet = Message.text
        if Message.text not in SeenTweets:
            print(Message.text)
            Sentance = tweet.split(" ")
            Sentance.remove("@DetectionMood")
            fp.write(Message.text+"\n")
            Vector = [0] * 60
            Position = 1
            for word in Sentance:
                try:
                    Index = ListWord.index(word)
                    Vector[Position] = Index
                    Position = Position + 1
                except:
                    Temp = True
            Sentance = ", ".join(Sentance)
            Final = cv.transform([Sentance])
            print("SVM predicition: ", modelSVM.predict(Final),"Probability:",modelSVM.predict_proba(Final))
            Vector = np.array([Vector])
            Vector = Vector.reshape(Vector.shape[0],Vector.shape[1], 1)
            Vector = Vector.astype(np.float32)   
            predictions = modelNeuralNet.predict(Vector)
            print("Neural Net Predicts:                  ", predictions)
            HappyVsSad = modelSVM.predict_proba(Final)
            AngryVsNone = predictions
            if HappyVsSad[0][0] > 0.65:
                Reply = "Your Tweet: " + Message.text + " Was found to be showing the emotion: Happy"
                api.send_direct_message(Message.user.id, Reply)
                print("Tweeting back Happy")
            elif HappyVsSad[0][2] > 0.65:
                Reply = "Your Tweet: " + Message.text + " Was found to be showing the emotion: Sad"
                api.send_direct_message(Message.user.id, Reply)
                print("Tweeting back Sad")        
            elif HappyVsSad[0][1] > 0.60:
                Reply = "Your Tweet: " + Message.text + " Was found to be showing no emotion"
                api.send_direct_message(Message.user.id, Reply)
                print("Tweeting back None")
            else:
                if AngryVsNone[0][0] > 0.7 or AngryVsNone[0][0] < 0.3:
                    Reply = "Your Tweet: " + Message.text + " Was found to be showing the emotion: Anger"
                    api.send_direct_message(Message.user.id, Reply)
                    print("Tweeting back Anger")
    time.sleep(300)
    fp.close
    
