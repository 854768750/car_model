import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time 

start_time = time.time()
iterations = 100001
hidden_neuron_num1 = 400
hidden_neuron_num2 = 200
predict_size = 150
train_size = 150
test_size = 150

plot = True

'''
load data from the second line, the first line of the txt is data name
array delta_ori stores the difference between two orientations and is limited between -2pi~2pi
command_data includes time step and torque
input_data includes command data and previous state
state_data includes current state
state_data does not include init_state

'''
f = open("/home/lion/car_model/state_command_data_3.txt", "r")
next(f)
result = []
for line in f:
    result.append(map(float,line.split(' ')))
for i in range(0, len(result)-1):
    result[i][0] = result[i+1][0] - result[i][0]
delta_ori = np.zeros((len(result)-1, 1))
for i in range(0, len(delta_ori)):
    delta_ori[i][0] = result[i+1][3] - result[i][3]
    if delta_ori[i][0]>math.pi:
        delta_ori[i][0] = delta_ori[i][0] - 2 * math.pi
    elif delta_ori[i][0]<-math.pi:
        delta_ori[i][0] = delta_ori[i][0] + 2 * math.pi
    #print(delta_ori[i][0])
result = np.array(result)
input_data = np.concatenate((result[0:-1,0:1],result[0:-1,6:8],result[0:-1,1:6]),axis=1)
command_data = np.concatenate((result[0:-1,0:1],result[0:-1,6:8]),axis=1)
state_data = result[:,1:6]
init_state = state_data[0,:].reshape([-1,5])
state_data = np.delete(state_data, 0, 0)

x_data = input_data[0:train_size,:]
y_data = np.concatenate((state_data[0:train_size,0:2],delta_ori[0:train_size,:],state_data[0:train_size,3:5]),axis=1)

test_x_data = input_data[0:test_size,:]
test_y_data = np.concatenate((state_data[0:test_size,0:2],delta_ori[0:test_size,:],state_data[0:test_size,3:5]),axis=1)

xs = tf.placeholder(tf.float32, [None, 8])
ys = tf.placeholder(tf.float32, [None, 5])

weights1 = tf.Variable(tf.random_normal([8,hidden_neuron_num1],seed=1))
biases1 = tf.Variable(tf.random_normal([1,hidden_neuron_num1],seed=1))
Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
output1 = tf.nn.relu(Wx_plus_b1)

weights2 = tf.Variable(tf.random_normal([hidden_neuron_num1,5],seed=1))
biases2 = tf.Variable(tf.random_normal([1,5],seed=1))
Wx_plus_b2 = tf.matmul(output1, weights2) + biases2
output2 = (Wx_plus_b2)
prediction = output2

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.03, global_step, 1000, 0.99, staircase = False)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)
saver=tf.train.Saver(max_to_keep=1)

for i in range(iterations): 
    x_feed = x_data 
    y_feed = y_data
    sess.run(train_step, feed_dict={xs: x_feed, ys: y_feed})
    if i % (1000) == 0: 
        loss_data = sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data})
        print("step: ", i, loss_data, "learning_rate: ", sess.run(learning_rate))
        #saver.save(sess, '/home/lion/car_model/ckpt/mnist.ckpt', global_step=i+1)
        #plt.plot(i, loss_data, marker='*', color='red')
        #plt.pause(0.0001)

if plot:
    plt.ion()
    plt.show()
    plt.pause(0.0001)

current_state = init_state
predict_state = np.zeros((predict_size,5))
for i in range(predict_size):
    current_input = np.concatenate((command_data[i,:].reshape([-1, 3]), current_state),axis=1)
    current_output = sess.run(prediction, feed_dict={xs: current_input})
    current_ori = current_state[0, 2] + current_output[0, 2]
    #print(i, current_output[0,2], y_data[i, 2])
    if current_ori>math.pi:
        current_ori = current_ori - 2 * math.pi
    elif current_ori<-math.pi:
        current_ori = current_ori + 2 * math.pi
    current_state = current_output
    current_state[0, 2] = current_ori
    predict_state[i,:] = current_state
    print("y_data:", y_data[i])
    print("prediction:", predict_state[i,:])

if plot:
    plt.plot(y_data[:, 0], y_data[:, 1], marker='*', color='red')
    plt.plot(predict_state[:, 0], predict_state[:, 1], marker='*', color='green')
    plt.pause(0.000001)

end_time = time.time()
print("cost_time:",end_time-start_time)
raw_input("press any button to exit")

