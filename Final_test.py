# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 19:20:46 2019

@author: ssn
"""
#importing packages
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
import pickle 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download("wordnet")

#importing data set
traindata = pd.read_csv("train.csv",header = 0)
testdata = pd.read_csv("test.csv",header = 0)
sample= pd.read_csv("sample_submission.csv",header = 0)

#spliting the data
list_tweets = traindata["tweet"]

#finding the word count
traindata['word_count'] = traindata['tweet'].apply(lambda x: len(str(x).split(" ")))
traindata[['tweet','word_count']].head()

#finding the character count
traindata['char_count'] = traindata['tweet'].str.len() ## this also includes spaces
traindata[['tweet','char_count']].head()

#average words
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

traindata['avg_word'] = traindata['tweet'].apply(lambda x: avg_word(x))
traindata[['tweet','avg_word']].head()

#cleaning the tweets
clean_tweets = []

for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    words = nltk.word_tokenize(tweet)
    newwords = [word for word in words if word not in stopwords.words('english')]
    tweet = ' '.join(newwords)
    clean_tweets.append(tweet)

for i in range(len(clean_tweets)):
    clean_tweets[i] = re.sub(r"user"," ",clean_tweets[i])
    clean_tweets[i] = re.sub(r"^\s+","",clean_tweets[i])
    clean_tweets[i] = re.sub(r"ð","",clean_tweets[i])
    clean_tweets[i] = re.sub(r"\s+"," ",clean_tweets[i])
    

#converting to base form using WordNetLemmatizer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

for i in range(len(clean_tweets)):
    words = nltk.word_tokenize(clean_tweets[i])
    newwords = [lemmatizer.lemmatize(word) for word in words]
    clean_tweets[i] = ' '.join(newwords)

#using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

tfidf_vectorizer = CountVectorizer(max_features = 3000,
                                   min_df = 3, 
                                   max_df = 0.6, 
                                   stop_words = stopwords.words("english"))

X_tfidf = tfidf_vectorizer.fit_transform(clean_tweets).toarray()

#spliting into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, traindata['label'], test_size=0.3, random_state=1)

#applying MultinomialNB algorithm
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
reg = MultinomialNB().fit(X_train, y_train)
reg.fit(X_train, y_train)

#predict the values
predicted= reg.predict(X_test)

#finding the metrics
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

from sklearn.metrics import precision_recall_fscore_support
all=precision_recall_fscore_support(y_test, predicted, average='macro')
print('Precision score=',all[0]*100)
print('Recall score=',all[1]*100)
print('F1 score=',all[2]*100)


#applying the same for test data
new_tweets = testdata["tweet"]

prediction_tweets = []

for tweet in new_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet = re.sub(r"\W"," ",tweet)
    tweet = re.sub(r"\d"," ",tweet)
    tweet = re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+[a-z]$"," ",tweet)
    tweet = re.sub(r"^[a-z]\s+"," ",tweet)
    tweet = re.sub(r"\s+"," ",tweet)
    words = nltk.word_tokenize(tweet)
    newwords = [word for word in words if word not in stopwords.words('english')]
    tweet = ' '.join(newwords)
    prediction_tweets.append(tweet)
 
for i in range(len(prediction_tweets)):
    prediction_tweets[i] = re.sub(r"user"," ",prediction_tweets[i])
    prediction_tweets[i] = re.sub(r"^\s+","",prediction_tweets[i])
    prediction_tweets[i] = re.sub(r"\s+"," ",prediction_tweets[i])
    prediction_tweets[i] = re.sub(r"ð","",prediction_tweets[i])


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

for i in range(len(prediction_tweets)):
    words = nltk.word_tokenize(prediction_tweets[i])
    newwords = [lemmatizer.lemmatize(word) for word in words]
    prediction_tweets[i] = ' '.join(newwords)


tweet_tfidf = tfidf_vectorizer.transform(prediction_tweets)
tweet_tfidf = tweet_tfidf.toarray()
prediction = reg.predict(tweet_tfidf)
sample['label'] = prediction
#print(sample["label"].value_counts())
sample.to_csv('prediction_4.csv',index=False)
