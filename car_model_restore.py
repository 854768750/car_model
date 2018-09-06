import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

iterations = 10000
'''
load data from the second line, the first line of the txt is data name
array delta_ori stores the difference between two orientations and is limited between -2pi~2pi
command_data includes time step and torque
input_data includes command data and previous state
state_data includes current state
state_data does not include init_state

'''
f = open("/home/lion/car_model/state_command_data.txt", "r")
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

x_data = input_data
y_data = np.concatenate((state_data[:,0:2],delta_ori,state_data[:,3:5]),axis=1)

xs = tf.placeholder(tf.float32, [None, 8])
ys = tf.placeholder(tf.float32, [None, 5])

weights1 = tf.Variable(tf.random_normal([8,20],seed=1))
biases1 = tf.Variable(tf.random_normal([1,20],seed=1))
Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
output1 = tf.nn.relu(Wx_plus_b1)

weights2 = tf.Variable(tf.random_normal([20,5],seed=1))
biases2 = tf.Variable(tf.random_normal([1,5],seed=1))
Wx_plus_b2 = tf.matmul(Wx_plus_b1, weights2) + biases2
output2 = Wx_plus_b2
prediction = output2

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.03).minimize(loss)

saver=tf.train.Saver()
sess = tf.InteractiveSession()
model_file=tf.train.latest_checkpoint('/home/lion/car_model/ckpt/')
saver.restore(sess,model_file)
for i in range(iterations): 
    x_feed = x_data
    y_feed = y_data
    sess.run(train_step, feed_dict={xs: x_feed, ys: y_feed})
    if i % (iterations/10) == 0: 
        print(sess.run(loss, feed_dict={xs: x_feed, ys: y_feed}))

plt.ion()
plt.show()
plt.pause(0.0001)

current_state = init_state
for i in range(50):
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
    #print("y_data:", y_data[i])
    #print("prediction:", current_state)
    plt.plot(y_data[i, 0], y_data[i, 1], marker='*', color='red')
    plt.plot(current_state[0, 0], current_state[0, 1], marker='*', color='green')
    plt.pause(0.0001)

raw_input("press any button to exit")

