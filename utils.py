
import random

def random_seed(text, sentence_len):
  start_index = random.randint(0, len(text) - sentence_len - 1)
  return text[start_index: start_index + sentence_len]
