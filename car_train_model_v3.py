import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time

start_time = time.time()
loss_data = 0
iterations = 100001
hidden_neuron_num1 = 100
hidden_neuron_num2 = 500
train_size = 300
test_size = 50
predict_size = train_size + test_size
plot = True
train = 2 # 1 for first time train, 2 for loading parameter and not training, 3 for loading and keep training
layer_num = 2
project_dir = "/home/lion/car_model"
print("iterations:",iterations)
lowest_loss = 100

'''
load data from the second line, the first line of the txt is data name
array delta_ori stores the difference between two orientations and is limited between -2pi~2pi
command_data includes time step and torque
input_data includes command data and previous state
state_data includes current state
state_data does not include init_state

'''
f = open(project_dir+"/state_command_data_4.txt", "r")
next(f)
result = []
for line in f:
    result.append(map(float,line.split(' ')))
for i in range(0, len(result)-1):
    result[i][0] = result[i+1][0] - result[i][0]
result = np.array(result)
input_data = np.concatenate((result[0:-1,0:1],result[0:-1,9:11],result[0:-1,1:3],result[0:-1,5:9]),axis=1)
command_data = np.concatenate((result[0:-1,0:1],result[0:-1,9:11]),axis=1)
state_data = np.concatenate((result[:,1:3],result[:,5:9]),axis=1)
init_state = state_data[0,:].reshape([-1,6])
state_data = np.delete(state_data, 0, 0)

x_data = input_data[0:train_size,:]
y_data = state_data[0:train_size,:]

test_x_data = input_data[train_size:(train_size+test_size),:]
test_y_data = state_data[train_size:(train_size+test_size),:]

xs = tf.placeholder(tf.float32, [None, 9])
ys = tf.placeholder(tf.float32, [None, 6])

weights1 = tf.Variable(tf.random_normal([9,6],seed=1))
biases1 = tf.Variable(tf.random_normal([1,6],seed=1))
Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
output1 = (Wx_plus_b1)

prediction = output1

if train==1:
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.999, staircase = False)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver=tf.train.Saver(max_to_keep=1)

    for i in range(1,iterations): 
        x_feed = x_data 
        y_feed = y_data
        sess.run(train_step, feed_dict={xs: x_feed, ys: y_feed})
        loss_data = sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data})
        if loss_data<lowest_loss:
            lowest_loss = loss_data
            print("lowest loss:", lowest_loss)
            saver.save(sess, project_dir+'model.ckpt', global_step=global_step)
        if i % (1000) == 0: 
            print("step: ", sess.run(global_step), loss_data, "learning_rate: ", sess.run(learning_rate))

elif train==2:
    global_step = tf.Variable(0)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.999, staircase = False)
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    saver=tf.train.Saver()
    sess = tf.InteractiveSession()
    model_file=tf.train.latest_checkpoint(project_dir+'/ckpt_v3')
    saver.restore(sess,model_file)

elif train==3:
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.999, staircase = False)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    saver=tf.train.Saver()
    sess = tf.InteractiveSession()
    model_file=tf.train.latest_checkpoint(project_dir+'/ckpt_v3')
    saver.restore(sess,model_file)
    saver=tf.train.Saver(max_to_keep=1)

    for i in range(iterations): 
        x_feed = x_data 
        y_feed = y_data
        sess.run(train_step, feed_dict={xs: x_feed, ys: y_feed})
        loss_data = sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data})
        if loss_data<lowest_loss:
            lowest_loss = loss_data
            print("lowest loss:", lowest_loss)
            saver.save(sess, project_dir+'model.ckpt', global_step=global_step)
        if i % (1000) == 0: 
            print("step: ", sess.run(global_step), loss_data, "learning_rate: ", sess.run(learning_rate))

if plot:
    plt.ion()
    plt.show()
    plt.pause(0.0001)

current_state = init_state.reshape([-1,6])
predict_state = np.zeros((predict_size,6))
for i in range(0,predict_size,1):
    current_input = np.concatenate((command_data[i,:].reshape([-1, 3]), current_state),axis=1)
    current_output = sess.run(prediction, feed_dict={xs: current_input})
    current_state = current_output
    #print("current_input:", current_input)
    predict_state[i,:] = current_state
    #print("state_data:", state_data[i])
    #print("prediction:", predict_state[i-train_size])
    
if plot:
    plt.plot(state_data[0:predict_size, 0], state_data[0:predict_size, 1], marker='*', color='red')
    plt.plot(predict_state[0:train_size, 0], predict_state[0:train_size, 1], marker='*', color='green')
    plt.plot(predict_state[train_size:predict_size, 0], predict_state[train_size:predict_size, 1], marker='*', color='yellow')
    plt.pause(0.000001)

print(sess.run(weights1))
print(sess.run(biases1))
end_time = time.time()
print("cost_time:",end_time-start_time)
print("train_size:",train_size)
print("test_size:",test_size,"loss_data:",sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data}))
print("predict_size:",predict_size)
raw_input("press any button to exit")

