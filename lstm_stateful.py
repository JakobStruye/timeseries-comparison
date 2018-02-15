import tensorflow as tf
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, Flatten, GRU
#from recurrent import LSTM,  GRU
from keras.optimizers import adam, rmsprop

from tqdm import tqdm
#
# def get_state_variables(batch_size, cell):
#     # For each layer, get the initial state and make a variable out of it
#     # to enable updating its value.
#     (state_c, state_h) = cell.zero_state(batch_size, tf.float32)
#     state_variables = tf.contrib.rnn.LSTMStateTuple(
#         tf.Variable(state_c, trainable=False),
#         tf.Variable(state_h, trainable=False))
#
#     return state_variables
#
#
# def get_state_update_op(state_variable, new_state):
#     # Add an operation to update the train states with the last state tensors
#
#     # Assign the new state to the state variables on this layer
#     update_ops = [state_variable[0].assign(new_state[0]),
#                        state_variable[1].assign(new_state[1])]
#     # Return a tuple in order to combine all update_ops into a single operation.
#     # The tuple's actual value should not be used.
#     return update_ops
#
# batch_size = 11
# max_length = 1
# frame_size = 1
# data = tf.placeholder(tf.float32, (batch_size, max_length, frame_size))
# cell = tf.nn.rnn_cell.LSTMCell(20, state_is_tuple=True)
# #cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
# # For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
# states = get_state_variables(batch_size, cell)
#
# # Unroll the LSTM
# outputs, new_states = tf.nn.dynamic_rnn(cell, data, initial_state=states)
#
# dense = tf.layers.dense(outputs, 1)
#
# # Add an operation to update the train states with the last state tensors.
# update_op = get_state_update_op(states, new_states)
#
# pred = tf.reshape(dense, (tf.shape(dense)[0], 1))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# target = np.reshape(np.array([2,3,3,2,1,1,2,3,3,2,1]), (11,1))
# cost = tf.reduce_mean(tf.abs(target - pred))
# minimize = optimizer.minimize(cost)
#
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for i in range(1000):
#     minim, out, upd, final = sess.run([minimize, outputs, update_op, dense], {data: np.reshape(np.array([1,2,3,3,2,1,1,2,3,3,2]), (batch_size, max_length, frame_size))})
# print final

lookback = 1
epochs = 10000

data = tf.placeholder(tf.float32, [None, 1, lookback])  # Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 1])

nodes = 200

#
# step_tf = tf.placeholder_with_default(tf.constant(-1, dtype=tf.int64), shape=[], name="step")
# is_train = tf.placeholder_with_default(True, shape=())
#
#
# zero_state = tf.zeros(shape=[1, nodes])
# c_state = tf.Variable(zero_state, trainable=False, validate_shape=False)
# h_state = tf.Variable(zero_state, trainable=False)
# init_encoder = tf.cond(tf.equal(step_tf, 0),
#                        true_fn=lambda: tf.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state),
#                        false_fn=lambda: tf.nn.rnn_cell.LSTMStateTuple(c_state, h_state))
#
#
# cell = tf.nn.rnn_cell.LSTMCell(nodes)
#
# val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)#, initial_state=init_encoder)
#
# update_ops = [c_state.assign(state.c, use_locking=True),
#               h_state.assign(state.h, use_locking=True)]
#
# dense = tf.layers.dense(val, 1)
#
# pred = tf.reshape(dense, (tf.shape(dense)[0], 1))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
# #cost = tf.losses.mean_squared_error(target, pred)
# cost = tf.reduce_mean(tf.square(target - pred))
# minimize = optimizer.minimize(cost)

ins = [1,2,3,4,2,6]
outs =[0.7,0.6,0.5,0.4,0.1,0.2]

rnn = Sequential()
lstm = GRU(nodes, input_shape=(1, 1), batch_size=1, stateful=True)
rnn.add(lstm)
#rnn.add(Dropout(0.5))
rnn.add(Dense(1))
opt = adam(lr=0.003, decay=0.0)  # 1e-3)
rnn.compile(loss='mae', optimizer=opt)
print rnn.summary()

for epoch in tqdm(range(epochs)):
    #for i in range(len(ins)):
    rnn.fit(np.reshape(np.array(ins), (6,1,1)), np.reshape(np.array(outs), (6,1)), epochs=1, batch_size=1, shuffle=False, verbose=2 if epoch >= epochs - 2 else 0)
    lstm.reset_states()

# old_rnn = rnn
# rnn = Sequential()
# lstm = GRU(nodes, batch_input_shape=(1,1, 1), stateful=True)
# rnn.add(lstm)
# rnn.add(Dropout(0.5))
# rnn.add(Dense(1))
# old_weights = old_rnn.get_weights()
# opt = adam(lr=0.003, decay=0.0)  # 1e-3)
# rnn.compile(loss='mae', optimizer=opt)
# rnn.set_weights(old_weights)


results = []
for i in range(len(ins)):
    #lstm.reset_states()
    result = rnn.predict(np.reshape(ins[i], (1,1,1)), batch_size=1)
    print ins[i], ":", result[0][0]
    results.append(result)
    import keras.backend as K
results = np.reshape(results, (6,))
target = np.reshape(outs, (6,))
print np.mean(np.abs(np.subtract(np.array(results), target)))




# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     for epoch in range(epochs):
#         for i in range(len(ins)):
#             the_cost, _, the_outs, the_state = sess.run([cost, minimize, dense, state], feed_dict={data: np.reshape(ins[i], (1,1,1)), target: np.reshape(outs[i], (1,1)), is_train: True, step_tf: epoch+i})
#         print the_cost
#         print the_state[0].shape
#     step = 0
#     for input in ins:
#         result, the_state = sess.run([dense, state], feed_dict={data: np.reshape(input, (1,1,1)), is_train: False, step_tf: step})
#         step += 1
#         print result
#         print the_state[0].shape

