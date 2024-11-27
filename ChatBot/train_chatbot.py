import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json
import pickle

intents_file = open('intents.json').read()
intents = json.loads(intents_file) 

# pre-processing
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each work
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add document in the corpus
        documents.append((word, intent['tag']))
        #add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
#print (len(documents), "documents")
# classes = intents
#print (len(classes), "classes", classes)
# words = all words, vocabulary
#print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create the training data
training = []
# create empty array for the output
output_empty = [0] * len(classes)

for doc in documents:
    #initializing bag of words
    bag = []
    #list of tokenized words for the pattern
    word_patterns = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # create the bag of words array with 1, if word is found in current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])


