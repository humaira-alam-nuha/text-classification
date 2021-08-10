# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 21:58:36 2020

@author: Humaira Alam Nuha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
nltk.download('stopwords')

#importing Dataset
reviews = load_files('txt_sentoken/')
X,y = reviews.data, reviews.target

#storing as Pickle Files
with open('X.pickle', 'wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle', 'wb') as f:
    pickle.dump(y,f)
    
# Unpickling the Dataset: this process is efficient to load larger amount of data
with open('X.pickle', 'rb') as f:
    pickle.load(f)
with open('y.pickle', 'rb') as f:
    pickle.load(f)
    
# Creating the corpus
corpus = []
for i in range(0, len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    review = re.sub(r'^[a-z]\s+', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

#BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

#to convert into tf-idf model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

#Train-Test
from sklearn.model_selection import train_test_split
text_train, text_test, sent_train, sent_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Logistic Regression
from sklearn import svm
classifier = svm.SVC(kernel='linear')
classifier.fit(text_train, sent_train)

sent_pred = classifier.predict(text_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(sent_test, sent_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(sent_pred, sent_test)
print("Accuracy Score: ", accuracy)

#Pickling Classifier
with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)
    
#Pickling the vectorizer
with open('tfidfmodel.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)


# Unpickling classifier and vectorizer
with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)
    
with open('tfidfmodel.pickle', 'rb') as f:
    tfidf = pickle.load(f)
    
sample = ["Morning Traffic is annoying!"]
sample = tfidf.transform(sample).toarray()
print("Result: ", clf.predict(sample))