#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:09:50 2022

@author: arpanrajpurohit
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer as suffix_stripper
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

DATABASE_PATH = "dataset/Sarcasm_Headlines_Dataset.json"
IGNORE_WORD_LIST = "stopwords"
ROW_IN_SHAPE = 0
DATABASE_COLUMN_HEADLINE_NAME = "headline"
DATABASE_OUTPUT_COLUMN = 2
LANGUAGE = 'english'
TEST_SIZE_IN_PER = 0.20
EXCLAMATION_IN_SYMBOL = '!'
EXCLAMATION_IN_WORD = ' exclamation'
QUESTION_IN_SYMBOL = '?'
QUESTION_IN_WORD   = ' question'
QUOTATION_IN_SYMBOL = r'\'(.+?)\''
QUOTATION_IN_WORD   = ' quotation'
NOT_LOWERCASE_LETTERS = '[^a-z]'
SPACE = ' '

iterations    = 100000
learning_rate = 0.01
LABEL_ERROR    = 'Error'
LABEL_ACCURACY = 'Accuracy'
LABEL_PRECISION = 'Precision'
LABEL_RECALL    = 'Recall'

#database import
database =  pd.read_json(DATABASE_PATH, lines = True)

#Data preprocessing
nltk.download(IGNORE_WORD_LIST)
from nltk.corpus import stopwords as sw
headlines = []
output = database.iloc[:, DATABASE_OUTPUT_COLUMN]
total_lines = database.shape[ROW_IN_SHAPE]

for index in range(0, total_lines):
    edit = re.sub(EXCLAMATION_IN_SYMBOL, EXCLAMATION_IN_WORD, database[DATABASE_COLUMN_HEADLINE_NAME][index])
    edit = edit.replace(QUESTION_IN_SYMBOL, QUESTION_IN_WORD)
    found = re.findall(QUOTATION_IN_SYMBOL, edit)
    if found:
        edit += QUOTATION_IN_WORD
    edit = re.sub(NOT_LOWERCASE_LETTERS, SPACE, edit)
    edit = edit.split()
    ss = suffix_stripper()
    edit = [ss.stem(word) for word in edit if not word in sw.words(LANGUAGE)]
    edit = SPACE.join(edit)
    headlines.append(edit)

train_input, test_input, train_output, test_output = train_test_split(headlines,output,test_size=TEST_SIZE_IN_PER,random_state=0)
cv = CountVectorizer(ngram_range=(1,3))
cv_train_input  = cv.fit_transform(train_input)
cv_test_input  = cv.transform(test_input)

#from scratch model implementation froze my pc due to big size of cv_train_input
#implementation can be found below

logistic_regression = LogisticRegression()
logistic_regression.fit(cv_train_input, train_output)

test_output_prediction = logistic_regression.predict(cv_test_input)

conf_mat = confusion_matrix(test_output, test_output_prediction)
true_positive  = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative  = conf_mat[1][1]
total_conf     = true_positive+true_negative+false_positive+false_negative

error_rate = (false_positive+false_negative)/total_conf
accuracy   = (true_positive + true_negative)/total_conf
precision  = true_positive / (true_positive + false_positive)
recall     = true_positive / (true_positive + false_negative)
print(accuracy)
print(error_rate)
print(precision)
print(recall)
# def intercept(input_dataset):
#     input_dataset = input_dataset.todense()
#     intercept = np.ones((input_dataset.shape[0], 1))
#     return np.concatenate((intercept, input_dataset), axis=1)

# def sigmoid(layer_input):
#     return 1 / (1 + np.exp(layer_input))

# def calculate_hiddenlayer_input(input_dataset, theta):
#     layer_input = np.dot(input_dataset, theta)
#     hidden_layer = sigmoid(layer_input)
#     return hidden_layer

# int_cv_train_input = intercept(cv_train_input)
# theta = np.zeros(int_cv_train_input.shape[1])
# train_output_size = train_output.size


# costs = []

# for i in range(iterations):
#     hidden_layer = calculate_hiddenlayer_input(int_cv_train_input, theta)
    
#     cost = (-train_output * np.log(hidden_layer) - (1 - train_output) * np.log(1 - hidden_layer)).mean()
#     costs.append(cost)
    
#     gradient = np.dot(int_cv_train_input.T, (hidden_layer - train_output)) / train_output_size
#     theta -= learning_rate * gradient  # gradient descent

# #predict for test data
# int_cv_test_input = intercept(cv_test_input)
# test_output_prob  = calculate_hiddenlayer_input(int_cv_test_input, theta)
# test_output_predictions = test_output_prob.round()

# conf_mat = confusion_matrix(test_output, test_output_prediction)
# true_positive  = conf_mat[0][0]
# false_positive = conf_mat[0][1]
# false_negative = conf_mat[1][0]
# true_negative  = conf_mat[1][1]
# total_conf     = true_positive+true_negative+false_positive+false_negative

# error_rate = (false_positive+false_negative)/total_conf
# accuracy   = (true_positive + true_negative)/total_conf
# precision  = true_positive / (true_positive + false_positive)
# recall     = true_positive / (true_positive + false_negative)