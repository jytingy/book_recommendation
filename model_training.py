import random       #for randomizing bot responses
import json        
import pickle       #for serialization
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer     #reduce words to the 'stem' so as not to compromise performance 
                                            #(eg work, working, worked, works), treats them all as one word
import tensorflow as tf

lemmatizer = WordNetLemmatizer()

goals = json.loads(open('goals.json').read())   #read contents of json file as text. using loads creates a python dictionary object

keywords = []
ignore_chars = ['!', '?', '.', ',']
categories = []
reasoning = []

#training data
for goal in goals['goals']:
    for reason in goal['text_input']:
        wordList = nltk.word_tokenize(reason)  #tokenize = split into individual words from collection of words
        keywords.extend(wordList)
        categories.append((wordList, goal['reason']))
        if goal['reason'] not in reasoning:
            reasoning.append(goal['reason'])
keywords = [lemmatizer.lemmatize(keyword) for keyword in keywords if keyword not in ignore_chars]
keywords = set(keywords)

pickle.dump(keywords, open('keywords.pkl', 'wb'))
pickle.dump(reasoning, open('reasoning.pkl', 'wb')) #wb = write only file in binary code

#neural network needs numerical values, so we need to represent our list of words/classes as a numerical value
#set individual words to 0 or 1 depending on if the word occurs in that specific reason category
training_data = []
output_empty = [0] * len(reasoning)
 
for category in categories:
    bag = []
    keyword_reasoning = category[0]
    keyword_reasoning = [lemmatizer.lemmatize(keyword.lower()) for keyword in keyword_reasoning]
    for keyword in keywords:
        bag.append(1) if keyword in keyword_reasoning else bag.append(0)
    
    output_row = list(output_empty)
    output_row[reasoning.index(category[1])] = 1
    training_data.append([bag, output_row])

