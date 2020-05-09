# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:19:11 2020

@author: Shobhit
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pandas as pd
import tensorflow as tf1
import numpy as np
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def softmax(x): 
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


#========= loading data
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
MAX_VOCAB_SIZE = 30000
marker_pad = '<pad>'

tf.reset_default_graph()


data = pd.read_csv("/content/drive/My Drive/chat_bot/chat_csv.csv").values

target_counter = Counter()
input_counter = Counter()

input_texts = []
target_texts = []

c = 0

for i in data:
    # for input
    line = i[0]
    line = [w.lower() for w in nltk.word_tokenize(line)]
    
    if len(line) > MAX_TARGET_SEQ_LENGTH:
        line = line[0:MAX_TARGET_SEQ_LENGTH]
    
    input_texts.append(line)
    for w in line:
        input_counter[w] += 1
        
    # for output
    line = i[1]
    line = [w.lower() for w in nltk.word_tokenize(line)]
    
    if len(line) > MAX_TARGET_SEQ_LENGTH:
        line = line[0:MAX_TARGET_SEQ_LENGTH]
        
    for w in line:
        target_counter[w] += 1
    target_texts.append(line)
    
# =============================================================================
#     if c > 100:
#         break
#     c+=1
# =============================================================================
    
input_w2i, input_i2w, target_w2i, target_i2w = {},{},{},{}
input_w2i[marker_pad] = 0
target_w2i[marker_pad] = 0

#================== input ===============================================
## we will create dictionaries to provide a unique integer for each word.
for idx, word in enumerate(input_counter.most_common(MAX_VOCAB_SIZE)):
    input_w2i[word[0]] = idx+1

# inverse dictionary for vocab_to_int.
input_i2w = dict([(idx, word) for word, idx in input_w2i.items()])

# ==== output ===========

## we will create dictionaries to provide a unique integer for each word.
for idx, word in enumerate(target_counter.most_common(MAX_VOCAB_SIZE)):
    target_w2i[word[0]] = idx+1

# inverse dictionary for vocab_to_int.
target_i2w = dict([(idx, word) for word, idx in target_w2i.items()])
#======================================================================

x = [[input_w2i.get(word, 0) for word in sentence] for sentence in input_texts]
y = [[target_w2i.get(word, 0) for word in sentence] for sentence in target_texts]

inputVocabLen = len(input_w2i)
targetVocabLen = len(target_w2i)
    
#============ paddinng and splitting =================
input_seq_len = 15
output_seq_len = 15

for i in range(len(x)):
    for k in range(input_seq_len - len(x[i])):
        x[i] = x[i] + [input_w2i[marker_pad]]
    
    for k in range(output_seq_len - len(y[i])):
        y[i] = y[i] + [target_w2i[marker_pad]]

X_train,  X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.05)
 
# =============================================================================
# x = tf.placeholder(dtype=tf.float32, shape = [None])
# y = tf.nn.softmax(x,name="sigmoid")
# 
# with tf.Session() as sess:
#     res = sess.run(y, feed_dict = {x:[1.0, -0.5, 3.4, -2.1, 0.0, -6.5]})
#     print(res)
#     
# =============================================================================
                
                    
#======== model helper function =======================

# feed data into placeholders
def feed_dict(x, y, batch_size = 64):
    feed = {}
    
    idxes = np.random.choice(len(x), size = batch_size, replace = False)
    
    for i in range(input_seq_len):
        feed[encoder_inputs[i].name] = np.array([x[j][i] for j in idxes], dtype = np.int32)
        
    for i in range(output_seq_len):
        feed[decoder_inputs[i].name] = np.array([y[j][i] for j in idxes], dtype = np.int32)
        
    feed[targets[len(targets)-1].name] = np.full(shape = [batch_size], fill_value = target_w2i[marker_pad], dtype = np.int32)
    
    for i in range(output_seq_len-1):
        batch_weights = np.ones(batch_size, dtype = np.float32)
        target = feed[decoder_inputs[i+1].name]
        for j in range(batch_size):
            if target[j] == target_w2i[marker_pad]:
                batch_weights[j] = 0.0
        feed[target_weights[i].name] = batch_weights
        
    feed[target_weights[output_seq_len-1].name] = np.zeros(batch_size, dtype = np.float32)
    
    return feed

# define our loss function

# sampled softmax loss - returns: A batch_size 1-D tensor of per-example sampled softmax losses
def sampled_loss(labels, logits):
    return tf.nn.sampled_softmax_loss(
                        weights = w_t,
                        biases = b,
                        labels = tf.reshape(labels, [-1, 1]),
                        inputs = logits,
                        num_sampled = 512,
                        num_classes = targetVocabLen)

# decode output sequence
def decode_output(output_seq):
    words = []
    for i in range(output_seq_len):
        smax = softmax(output_seq[i])
        idx = np.argmax(smax)
        words.append(target_i2w[idx])
    return words

# =============================================================================
# model building   
# =============================================================================
encoder_inputs = [tf.placeholder(dtype=tf.int32, shape=[None], name="encoder{}".format(i)) for i in range(input_seq_len)]

decoder_inputs = [tf.placeholder(dtype= tf.int32, shape=[None], name="decoder{}".format(i)) for i in range(output_seq_len)]

targets = [decoder_inputs[i+1] for i in range(output_seq_len-1)]
targets.append(tf.placeholder(dtype= tf.int32, shape=[None], name="last_target"))       
#target weight

target_weights = [tf.placeholder(dtype= tf.float32, shape = [None], name="target_w{}".format(i)) for i in range(output_seq_len)]

size = 512
w_t = tf.get_variable('proj_w', [targetVocabLen, size], tf.float32)
b = tf.get_variable('proj_b', [targetVocabLen], tf.float32)
w = tf.transpose(w_t)
output_projection = (w,b)

def train_model(): 
  outputs, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                                              encoder_inputs,
                                              decoder_inputs,
                                              tf.contrib.rnn.BasicLSTMCell(size),
                                              num_encoder_symbols = inputVocabLen,
                                              num_decoder_symbols = targetVocabLen,
                                              embedding_size = 100,
                                              feed_previous = False,
                                              output_projection = output_projection,
                                              dtype = tf.float32)

  # Weighted cross-entropy loss for a sequence of logits
  loss = tf.contrib.legacy_seq2seq.sequence_loss(outputs, targets, target_weights, softmax_loss_function = sampled_loss)

  # =============================================================================
  # training and plots
  # =============================================================================

  # ops and hyperparameters
  learning_rate = 7e-3
  batch_size = 96
  #steps = 25501
  steps = 2000

  # ops for projecting outputs
  outputs_proj = [tf.matmul(outputs[i], output_projection[0]) + output_projection[1] for i in range(output_seq_len)]

  # training op
  optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
  # tf.train.RMSPropOptimizer

  # init op
  init = tf.global_variables_initializer()


  # Loss values appended to plot diagram
  losses = []

  # Save checkpoint to restore the model later 
  saver = tf.train.Saver()

  with tf.Session() as sess:
      sess.run(init)
      
      for step in range(steps):
          feed = feed_dict(X_train, Y_train, batch_size)
          sess.run(optimizer, feed_dict = feed)
          
          #if step % 500 == 0:
          if step % 100 == 0:
              loss_value = sess.run(loss, feed_dict = feed)
              print('step: {}, loss: {}'.format(step, loss_value))
              losses.append(loss_value)
              
      sf = saver.save(sess, 'model.ckpt')#, global_step=step)
      print('Checkpoint is saved')

  import matplotlib.pyplot as plt 
  # plot the losses
  with plt.style.context('fivethirtyeight'):
      plt.plot(losses, linewidth = 1)
      plt.xlabel('Steps')
      plt.ylabel('Losses')
  plt.show()
  
train_model()
