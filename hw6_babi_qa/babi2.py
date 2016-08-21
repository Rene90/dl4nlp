#!/usr/bin/python
#
# author:
# 
# date: 
# description:
#

'''Trains a memory network on the bAbI dataset.

References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question a1ing: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698

- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Activation, Dense, Merge, Permute, Dropout
from keras.layers import LSTM, SimpleRNN, Input
from keras.layers.core import Flatten
from keras.utils.data_utils import get_file

from functools import reduce
import tarfile

from data import get_stories, vectorize_stories

path = 'babi-tasks-v1-2.tar.gz'
#origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [a1]) for story, q, a1 in train_stories + test_stories)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, a1):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, a1s_train = vectorize_stories(train_stories, word_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, a1s_test = vectorize_stories(test_stories, word_idx, story_maxlen, query_maxlen)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('a1s: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('a1s_train shape:', a1s_train.shape)
print('a1s_test shape:', a1s_test.shape)
print('-')
print('Compiling...')

print(inputs_train.shape)
print(queries_train.shape)

X = Input(shape=(story_maxlen,), dtype="int32")
Q = Input(shape=(query_maxlen,), dtype="int32")

embedding_dim = story_maxlen

# embed the input sequence into a sequence of vectors
m1 = Sequential()
m1.add(Embedding(input_dim=vocab_size,
                 output_dim=embedding_dim,
                 input_length=story_maxlen)(X))
# output: (samples, story_maxlen, embedding_dim)
# embed the question into a sequence of vectors
u1 = Sequential()
u1.add(Embedding(input_dim=vocab_size,
                 output_dim=embedding_dim,
                 input_length=query_maxlen)(Q))
# output: (samples, query_maxlen, embedding_dim)
# compute a 'w1' between input sequence elements (which are vectors)
# and the question vector sequence
w1 = Sequential()
w1.add(Merge([m1, u1], mode='dot', dot_axes=[2, 2]))
#w1.add(Activation('softmax'))
# output: (samples, story_maxlen, query_maxlen)
# embed the input into a single vector with size = story_maxlen:
c1 = Sequential()
c1.add(Embedding(input_dim=vocab_size,
                 output_dim=query_maxlen,
                 input_length=story_maxlen)(X))
# output: (samples, story_maxlen, query_maxlen)
# sum the w1 vector with the input vector:
o1 = Sequential()
o1.add(Merge([w1, c1], mode='sum'))
# output: (samples, story_maxlen, query_maxlen)
o1.add(Permute((2, 1)))  # output: (samples, query_maxlen, story_maxlen)


#u2 = Sequential()
#u2.add(Merge([o1, u1], mode='sum'))

#m2 = Sequential()
#m2.add(Embedding(input_dim=vocab_size,
                 #output_dim=embedding_dim,
                 #input_length=story_maxlen))

#w2 = Sequential()
#w2.add(Merge([m2, u2], mode='dot', dot_axes=[2, 2]))

#c2 = Sequential()
#c2.add(Embedding(input_dim=vocab_size,
                 #output_dim=query_maxlen,
                 #input_length=story_maxlen))

#o2 = Sequential()
#o2.add(Merge([w2, c2], mode='sum'))
#o2.add(Permute((2, 1)))

# concatenate the w1 vector with the question vector,
# and do logistic regression on top
a1 = Sequential()
a1.add(Merge([o1, u1], mode='sum'))
a1.add(Flatten()) # why not in original format?
# one regularization layer -- more would probably be needed.
a1.add(Dense(vocab_size))
# we output a probability distribution over the vocabulary
a1.add(Activation('softmax'))

a1.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['accuracy'])
# Note: you could use a Graph model to avoid repeat the input twice
a1.fit([inputs_train, queries_train], a1s_train,
           batch_size=512,
           nb_epoch=10,
           validation_data=([inputs_test, queries_test], a1s_test))

from keras.utils.visualize_util import plot

if __name__ == "__main__" and False:
    plot(a1, to_file='model.png')

    json_model = a1.to_json()
    with open("model.json", "w") as fh:
        fh.write(json_model)

    a1.save_weights("rnn_weights.h5")
