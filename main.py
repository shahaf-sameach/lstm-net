from __future__ import print_function
from network import Network
from utils import random_seed

weight_file = "weights.hdf5"
maxlen = 40

print("loading training data...")
text = open("train.txt").read()

# build network
print("building model...")
lstm = Network(maxlen=maxlen)

# train
print("training...")
lstm.train(text, nb_epoch=60, nb_sentences=30000, weight_file=weight_file)

# loading model from file
print("loading weights..")
lstm.load_weights(weight_file)

# predict
question = "how are you?"
print("generating text using random seed: %s" % question)
lstm.predict("how many minutes in an hour?")

# # evaluate
# print("loading test data...")
# test_text = open("test.txt").read()

# print("evaluating model...")
# score = lstm.evaluate(test_text)
# print("acc=%s cross entropy=%s" %score )


