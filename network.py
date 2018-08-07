from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import math

class Network(object):

  def __init__(self, maxlen=40):
    self.maxlen = maxlen
    self.chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '\",-?!:*.")
    self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
    self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(self.maxlen, len(self.chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(self.chars)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    self.model = model

  def train(self, text, nb_epoch=60, batch_size=128, weight_file="weights.hdf5", nb_sentences=None):
    self.nb_sentences = nb_sentences
    # Callback for model saving:
    checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1)

    sentences ,next_chars = self.__create_sequntel_sentances(text)
    train_x, train_y = self.__vectorize(sentences, next_chars)
    
    # Training
    self.model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, callbacks=[checkpointer])
 
  def load_weights(self, weight_file):
    self.model.load_weights(weight_file)

  def evaluate(self, text):
    iterations = range(1,len(text) - self.maxlen + 1)
    sentence = text[:self.maxlen]
    
    cross_entropy = 0
    acc = 0

    for i in iterations:
      if i % 1000 == 0:
        print("iteration: %s/%s" %(i,len(iterations)))
      
      x = np.zeros((1, self.maxlen, len(self.chars)))    
      for t, char in enumerate(sentence):
        x[0, t, self.char_indices[char]] = 1.

      preds = self.model.predict(x, verbose=0)[0]
      next_index = self.__sample(preds, 0.2)
      predict_next_char = self.indices_char[next_index]
      actual_next_char = text[i + self.maxlen - 1]
      if predict_next_char == actual_next_char: 
        acc = acc + 1
      
      cross_entropy = cross_entropy - math.log(preds[self.char_indices[actual_next_char]],2)
      sentence = text[i:self.maxlen + i]

    cross_entropy = cross_entropy / (len(iterations))
    acc = float(acc) / (len(iterations))

    return (acc, cross_entropy)

  def predict(self, sentence):
    for diversity in [0.2, 0.5, 1.0, 1.2]:
      print()
      print('----- diversity:', diversity)

      generated = ''
      generated += sentence
      print('----- Generating with seed: "' + sentence + '"')
      sys.stdout.write(generated)

      next_char = ""
      while(next_char != '.'):
        x = np.zeros((1, self.maxlen, len(self.chars)))
        for t, char in enumerate(sentence):
            x[0, t, self.char_indices[char]] = 1.

        preds = self.model.predict(x, verbose=0)[0]
        next_index = self.__sample(preds, diversity)
        next_char = self.indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

  def __create_sequntel_sentances(self, text, step=1):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - self.maxlen, step):
      sentences.append(text[i: i + self.maxlen])
      next_chars.append(text[i + self.maxlen])
    
    print('nb sequences:', len(sentences))

    if self.nb_sentences:
      sentences = sentences[:self.nb_sentences]
      next_chars = next_chars[:self.nb_sentences]
    
    return (sentences, next_chars)

  def __vectorize(self, sentences, next_chars=None):
    print('Vectorization...')
    X = np.zeros((len(sentences), self.maxlen, len(self.chars)), dtype=np.bool)
    Y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, self.char_indices[char]] = 1
        Y[i, self.char_indices[next_chars[i]]] = 1

    return (X,Y)

  def __sample(self, a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

if __name__ == '__main__':
  a = Network()
  model = a.get_model()

