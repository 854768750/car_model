import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv

#line 4
def linearized_sigmoid(x):
    #the sigmoid is divided into 3 segments, Mx is inf, x1 is -1/slope, x2 is 1/slope
    slope = 0.4
    x1 = -1/slope
    x2 = 1/slope
    if x<x1:
        y = 0
    elif x>x2:
        y = 1
    else:
        y = x/2*slope+0.5
    up = y + 0.1
    return y, up

def sigmoid_2d(x):
    y = np.zeros(x.shape)
    shape = x.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            y[i][j] = 1/(1+math.exp(-x[i][j]))
    return y

start_time = time.time()
loss_data = 0
iterations = 10001
hidden_neuron_num1 = 50
hidden_neuron_num2 = 500
train_size = 17000
test_size = 1000
predict_size = 20
plot = True
train = 2 # 1 for first time train, 2 for loading parameter and not training, 3 for loading and keep training
layer_num = 2
project_dir = "/home/lion/car_model"
print("iterations:",iterations)
lowest_loss = 100
batch_size  = 100

'''
load data from the second line, the first line of the txt is data name
array delta_ori stores the difference between two orientations and is limited between -2pi~2pi
command_data includes time step and torque
input_data includes command data and previous state
state_data includes current state
state_data does not include init_state

'''
f = open(project_dir+"/state_command_data_7.txt", "r")
next(f)
result = []
for line in f:
    result.append(map(float,line.split(' ')))
for i in range(0, len(result)-1):
    result[i][0] = result[i+1][0] - result[i][0]
result = np.array(result)

command_data = result[0:-1,9:11].reshape([-1,2])
quaternion_t = result[0:-1,5:7].reshape([-1,2])
v_t = result[0:-1,7:8].reshape([-1,1])
theta_t = np.zeros((len(result)-1,1))
for i in range(0, len(result)-1):
    theta_t[i][0] = math.atan2(2*quaternion_t[i][0]*quaternion_t[i][1], pow(quaternion_t[i][1],2)-pow(quaternion_t[i][0],2))
input_data = np.concatenate((theta_t,v_t,command_data[:,:].reshape([-1,2])),axis=1)

delta_x = np.zeros((len(result)-1,1))
delta_y = np.zeros((len(result)-1,1))
delta_quaternion = np.zeros((len(result)-1,2))
v_t_1 = np.delete(result[:,7:8].reshape([-1,1]), 0, 0)
for i in range(0, len(result)-1):
    delta_x[i][0] = result[i+1][1] - result[i][1]
    delta_y[i][0] = result[i+1][2] - result[i][2]
    delta_quaternion[i,:] = result[i+1,5:7] - result[i,5:7]
output_data = np.concatenate((delta_x,delta_y),axis=1)

x_data = input_data[0:train_size,:]
y_data = output_data[0:train_size,:]

test_x_data = x_data#input_data[train_size:train_size+test_size,:]
test_y_data = y_data#output_data[train_size:train_size+test_size,:]

input_size = 4
output_size = 2
xs = tf.placeholder(tf.float32, [None, input_size])
ys = tf.placeholder(tf.float32, [None, output_size])


weights1 = tf.Variable(tf.random_normal([input_size,hidden_neuron_num1],seed=1))
biases1 = tf.Variable(tf.random_normal([1,hidden_neuron_num1],seed=1))
Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
output1 = tf.nn.sigmoid(Wx_plus_b1)

weights2 = tf.Variable(tf.random_normal([hidden_neuron_num1,output_size],seed=1))
biases2 = tf.Variable(tf.random_normal([1,output_size],seed=1))
Wx_plus_b2 = tf.matmul(output1, weights2) + biases2
output2 = (Wx_plus_b2)

prediction = output2

if train==2:
    #global_step = tf.Variable(0)
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    #learning_rate = tf.train.exponential_decay(0.05, global_step, 100, 0.999, staircase = False)
    #train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
    saver=tf.train.Saver()
    sess = tf.InteractiveSession()
    model_file=tf.train.latest_checkpoint(project_dir+'/ckpt_v5')
    saver.restore(sess,model_file)

if plot:
    plt.ion()
    plt.show()
    plt.pause(0.0001)


start = 20000
current_state = result[start,1:3].reshape([-1,2])
predict_state = np.zeros((predict_size,2))
for i in range(0,predict_size,1):
    if(i%500==0):
        current_state = result[start+i,1:3].reshape([-1,2])
    current_input = np.concatenate((theta_t[start+i,:].reshape([-1,1]), v_t[start+i,:].reshape([-1,1]), command_data[start+i,:].reshape([-1,2])),axis=1)
    current_output = sess.run(prediction, feed_dict={xs: current_input})
    current_state = current_output + current_state#result[start+i,1:3].reshape([-1,2])
    #print("current_input:", current_input)
    predict_state[i,:] = current_state
    #print("state_data:", state_data[i+start,0:6])
    #print("prediction:", predict_state[i,0:6])
    
if plot:
    l1 = plt.plot(result[start, 1], result[start, 2], marker='*', color='black')
    l2 = plt.plot(result[start+1:start+1+predict_size, 1], result[start+1:start+1+predict_size, 2], marker='*', color='red', label='ground truth')
    l3 = plt.plot(predict_state[0:predict_size, 0], predict_state[0:predict_size, 1], marker='*', color='green', label='predict')
    plt.pause(0.000001)


save_data = False
if save_data:
    #out = open('plot_data.csv','a', newline='')
    #csv_write = csv.writer(out,dialect='excel')
    #csv_write.writerow(stu1)
    np.savetxt('/home/lion/plot_data.csv', np.concatenate((quaternion_t[start+1:start+1+predict_size, 0:2],predict_state[0:predict_size, 0:2]),axis=1), delimiter = ' ')
    print ("write over")


#print(sess.run(weights1))
#print(sess.run(biases1))
end_time = time.time()
print("cost_time:",end_time-start_time)
print("train_size:",train_size)
print("test_size:",test_size,"loss_data:",sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data}))
print("predict_size:",predict_size)



#line 2
velocity_value = np.array(range(30,100,1))*0.1#[3, 4, 5, 6, 7, 8, 9, 10]#0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
angle_value = np.array(range(-20,20,1))*0.01#[-0.2, -0.18, -0.16, -0.14, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]#
current_input = np.zeros([len(velocity_value)*len(angle_value),4])
for i in range(len(velocity_value)):
    for j in range(len(angle_value)):
        current_input[i*len(angle_value)+j,0] = theta_t[start]
        current_input[i*len(angle_value)+j,1] = v_t[start]
        current_input[i*len(angle_value)+j,2] = velocity_value[i]   
        current_input[i*len(angle_value)+j,3] = angle_value[j]
#line 3
input_size = 4
output_size = 2
xs = tf.placeholder(tf.float32, [None, input_size])
ys = tf.placeholder(tf.float32, [None, output_size])
Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
output1 = tf.nn.sigmoid(Wx_plus_b1)
Wx_plus_b2 = tf.matmul(output1, weights2) + biases2
output2 = (Wx_plus_b2)
prediction = output2  
current_output = sess.run(prediction, feed_dict={xs: current_input})  
#print(current_input)
#print(current_output)
print("ground truth:", np.concatenate((theta_t[start,:].reshape([-1, 1]), v_t[start,:].reshape([-1, 1]), command_data[start,:].reshape([-1,2])),axis=1))
current_output = current_output + result[start,1:3].reshape([-1,2]) 
#print(current_output)  
if plot:
    l4 = plt.plot(current_output[:, 0], current_output[:, 1], marker='.', color='yellow', label='reachable state', alpha=0.5)
    plt.pause(0.000001)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.legend()

'''
#line 2
velocity_limit = [3, 10]
angle_limit = [-0.2, 0.2]
velocity = (np.random.rand(1,1)*(velocity_limit[1]-velocity_limit[0]) + velocity_limit[0])[0][0]
angle = 0

#line 3 
current_input = tf.Variable(np.array([0, 0.2,velocity,angle]).reshape([-1,4]),dtype=tf.float32)
current_input_cap = tf.Variable(np.array([0, 0.2,velocity,angle]).reshape([-1,4]),dtype=tf.float32)
sess.run(current_input.initializer)
sess.run(current_input_cap.initializer)
print(sess.run(current_input))
weights1 = weights1.eval()
biases1 = biases1.eval()
weights2 = weights2.eval()
biases2 = biases2.eval()
Wx_plus_b1 = tf.matmul(current_input, weights1) + biases1
output1 = tf.nn.sigmoid(Wx_plus_b1)
Wx_plus_b2 = tf.matmul(output1, weights2) + biases2
output2 = (Wx_plus_b2)
prediction = output2  
radius = tf.sqrt(tf.square(prediction[0])+tf.square(prediction[0]))
print(sess.run(radius))
current_output = sess.run(prediction)
print(current_output)
#print(tf.trainable_variables())

#line 5
terminate = False

#line 6
while not terminate:
    #line 7
    [grads] = tf.gradients(prediction,[current_input])
    step_size = 0.5
    gradient = sess.run(grads)
    gradient[0][0] = 0
    gradient[0][1] = 0
    gradient[0][3] = 0
    print("gradient:",gradient)
    current_input_assign = current_input_cap.assign(current_input + step_size * gradient)
    print("current_input_cap:",sess.run(current_input_assign))
    print("current_input:",sess.run(current_input))
    current_output_cap = sess.run(prediction)
    print("current_output_cap:",current_output_cap)

    #line 8
    delta = 0.001
    current_output = current_output_cap + np.array([[delta, 0]])
    print("current_output:",current_output)
    #line 9
    feas = False
    temp_velocity = np.array(range((int)(velocity_limit[0]*10000),(int)(velocity_limit[1]*10000)))*0.0001
    for i in range(len(temp_velocity)):
        temp_input = np.array([0, 0.2, temp_velocity[i], angle])
        temp_output = np.dot(sigmoid_2d(np.dot(temp_input,weights1)+biases1),weights2)+biases2
        if temp_output[0][0]>current_output[0][0]:
            current_input_ = temp_input
            current_output_ = temp_output
            feas = True
            break
    if feas==True:
        current_input_assign = current_input.assign(np.array([current_input_]))
        current_input_assign.eval()
        current_output = current_output_
    else:
        terminate = True

    print(current_input.eval())
    print(current_output)
'''

np.savetxt('/home/lion/weights1.csv', np.array(weights1.eval()), delimiter = ' ')
np.savetxt('/home/lion/biases1.csv', np.array(biases1.eval()), delimiter = ' ')
np.savetxt('/home/lion/weights2.csv', np.array(weights2.eval()), delimiter = ' ')
np.savetxt('/home/lion/biases2.csv', np.array(biases2.eval()), delimiter = ' ')
raw_input("press any button to exit")