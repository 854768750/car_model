import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time

start_time = time.time()
loss_data = 0
iterations = 100001
hidden_neuron_num1 = 200
hidden_neuron_num2 = 50
train_size = 20
test_size = 1
predict_size = train_size + test_size
plot = True
train = 2 # 1 for first time train, 2 for loading parameter and not training, 3 for loading and keep training
layer_num = 2
project_dir = "/home/lion/car_model/"
prev_state_size = 5
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
f = open(project_dir+"state_command_data_4.txt", "r")
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
init_state = state_data[range(prev_state_size),:].reshape([-1,6])
state_data = np.delete(state_data, range(prev_state_size), 0)

con_data = np.zeros((len(result)-prev_state_size, prev_state_size*9))
for i in range(len(result)-prev_state_size):
   con_data[i,:] = np.concatenate((input_data[i,:],input_data[i+1,:],input_data[i+2,:],input_data[i+3,:],input_data[i+4,:]), axis = 0).reshape([1,prev_state_size*9])

print(con_data.shape, state_data.shape)
x_data = con_data[0:train_size,:]
y_data = state_data[0:train_size,:]

test_x_data = con_data[train_size:(train_size+test_size),:]
test_y_data = state_data[train_size:(train_size+test_size),:]

xs = tf.placeholder(tf.float32, [None, prev_state_size*9])
ys = tf.placeholder(tf.float32, [None, 6])

weights1 = tf.Variable(tf.random_normal([prev_state_size*9,6],seed=1))
biases1 = tf.Variable(tf.random_normal([1,6],seed=1))
Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
output = (Wx_plus_b1)

prediction = output

global_step = tf.Variable(0)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
learning_rate = tf.train.exponential_decay(0.01, global_step, 1000, 0.999, staircase = False)
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
saver=tf.train.Saver()
sess = tf.InteractiveSession()
model_file=tf.train.latest_checkpoint(project_dir+'ckpt_v4/')
saver.restore(sess,model_file)

if plot:
    plt.ion()
    plt.show()
    plt.pause(0.0001)

'''total_show_points = 20
start = 0
end = 20
step = 1
predict_state = np.zeros((end-start,6))
for j in range(start,end,1):
    current_state = input_data[j:j+prev_state_size,3:9].reshape([-1,6])
    for i in range(0,step,1):
        current_input = np.concatenate((command_data[j+i:j+prev_state_size+i,:].reshape([-1, 3]), current_state),axis=1).reshape([1,prev_state_size*9])
        current_output = sess.run(prediction, feed_dict={xs: current_input})
        current_state = np.concatenate((current_state[1:prev_state_size,:], current_output),axis=0)
        #print("current_input:", current_input)
        predict_state[j-start,:] = current_output
    #print("state_data:", state_data[j])
    #print("prediction:", predict_state[i])
    
if plot:
    plt.plot(state_data[0:total_show_points, 0], state_data[0:total_show_points, 1], marker='*', color='red')
    for i in range(start,end,1):
        plt.plot(predict_state[i-start, 0], predict_state[i-start, 1], marker='*', color='green')
        plt.pause(0.000001)

aver_error = 0
for i in range(start,end,1):
    aver_error += math.sqrt(pow(predict_state[i-start, 0]-state_data[i,0],2) + pow(predict_state[i-start,1]-state_data[i,1],2))
aver_error /= (end-start)
print("aver_error:",aver_error)
'''
total_show_points = 20
start = 8
end = 9
step = 6
predict_state = np.zeros((end-start,step,6))
for i in range(start,end,1):
    current_state = input_data[i:i+prev_state_size,3:9].reshape([-1,6])
    for j in range(0,step,1):
        current_input = np.concatenate((command_data[i+j:i+prev_state_size+j,:].reshape([-1, 3]), current_state),axis=1).reshape([1,prev_state_size*9])
        current_output = sess.run(prediction, feed_dict={xs: current_input})
        current_state = np.concatenate((current_state[1:prev_state_size,:], current_output),axis=0)
        #print("current_input:", current_input)
        predict_state[i-start,j,:] = current_output
    #print("state_data:", state_data[j])
    #print("prediction:", predict_state[i])
    
if plot:
    plt.plot(state_data[0:total_show_points, 0], state_data[0:total_show_points, 1], marker='*', color='red')
    for i in range(start,end,1):
        plt.plot(predict_state[i-start,:, 0], predict_state[i-start,:, 1], marker='*', color='green')
        plt.pause(0.000001)

aver_error = 0
for i in range(start,end,1):
    for j in range(0,step,1):
        aver_error += math.sqrt(pow(predict_state[i-start,j, 0]-state_data[i+j,0],2) + pow(predict_state[i-start,j,1]-state_data[i+j,1],2))
aver_error /= (end-start)*step
print("aver_error:",aver_error)
print(sess.run(weights1))
print(sess.run(biases1))

end_time = time.time()
print("cost_time:",end_time-start_time)
print("train_size:",train_size)
print("test_size:",test_size,"loss_data:",sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data}))
print("predict_size",predict_size)
raw_input("press any button to exit")

