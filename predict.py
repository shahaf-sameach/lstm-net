'''Example script to generate text from Nietzsche's writings.
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from sklearn.cross_validation import train_test_split
import numpy as np
import random
import sys
import math
import pdb
import time

# path = get_file('test.txt')
text = open("test.txt").read()
test_text = open('test.txt').read()
print('corpus length:', len(text))

chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '\",-?!.")
print('total chars:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# sys.exit(0)
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

sentences = sentences[:100]
next_chars = next_chars[:100]

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences[:1000]):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# sys.exit(0)

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
model.fit(X, y, batch_size=128, nb_epoch=1)
acc = 0
cross_entropy = 0

print("predicting")
test_text = test_text[:1000]
iterations = range(1,len(test_text) - maxlen + 1)
sentence = test_text[:maxlen]
t0 = time.time()
for i in iterations:
    
    y = np.zeros((1, maxlen, len(chars)))    
    for t, char in enumerate(sentence):
        y[0, t, char_indices[char]] = 1.

    preds = model.predict(y, verbose=0)[0]
    next_index = sample(preds)
    predict_next_char = indices_char[next_index]
    actual_next_char = test_text[i + maxlen - 1]
    # pdb.set_trace()
    if predict_next_char == actual_next_char: 
        acc = acc + 1
    
    cross_entropy = cross_entropy - math.log(preds[char_indices[actual_next_char]],2)
    sentence = test_text[i:maxlen + i]

cross_entropy = cross_entropy / (len(iterations))
acc = float(acc) / (len(iterations))

print("acc=%s" % acc)
print("cross_entropy=%s" % cross_entropy)
t1 = time.time()
print("took %s" %(t1-t0))

