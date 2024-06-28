import json
import pickle
import numpy
import random
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

k=nltk.download('wordnet')
read=open("intents.json").read()
# json.loads("intents.json")
intents=json.loads(read)
# print("TYPE:",type(intents))
words=[];classes=[];documents=[]
ignore_letters=["?",".","!","'",","]
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if(intent['tag'] not in classes):
            classes.append(intent['tag'])
# print(word_list)
# print(words)
# print(documents)
lemmatize=WordNetLemmatizer()
word=[lemmatize.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))
# print(words)
classes=sorted(set(classes))
print(classes)
wordpickle=open('word.pkl','wb')
# classpickle=open('words.pkl','wb')
cls=pickle.dump(words,wordpickle)
# wrd=pickle.dump(classes,classpickle)
classpickle=open('words.pkl','wb')
# cls=pickle.dump(words,wordpickle)
word=pickle.dump(classes,classpickle)
training=[]
output_empty=[0]*len(classes)
for document in documents:
    bag=[]
    word_pattern=document[0]
    word_pattern=[lemmatize.lemmatize(word.lower())
                  for word in word_pattern]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)
    output_raw=list(output_empty)
    output_raw[classes.index(document[1])]=1
    training.append([bag,output_raw])
print(training)