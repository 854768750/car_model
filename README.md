# Simulated Racecar Modeling

## Steps

### 1) Clone source code into your workspace from github

Follow the instruction from https://github.com/BU-DEPEND-Lab/Racecar/tree/master/F1tenth-Simulation, this includes gazebo model and other necessary files for the simulation. Clone the codes to your workspace and build them.

Then clone the codes from [https://github.com/854768750/car_model](https://github.com/854768750/car_model), these are for automatically controlling the car in gazebo, collecting data of the car and building model of the car.
### 2) Run the simulated car in gazebo
Comment the keyboard control node in the following file as we want the car to be controlled by the program but not the keyboard.

In ${workspace}/src/F1tenth-Simulation/racecar_simulator/racecar_control/launch/racecar_control.launch modify the following line 

    <!--node pkg="racecar_control" type="keyboard_teleop.py" name="keyboard_teleop" output="screen" launch-prefix="xterm -e"></node-->
Launch the file by

    source ~/catkin_ws/devel/setup.bash
    roslaunch racecar_gazebo racecar.launch
and gazebo environment with a car model is shown in the window.

### 3) Publish control command and let the car move automatically

    python {path to car_model}/publish_command_v1.py

After this python script begins, the car will move according to the command it receives. Here I make the car move at a low speed and a small steering angle so it will not drift. The command torque is [1:1:10] and the steering angle is [-0.2:0.02:0.2].
### 4) Collect data from gazebo simulation environment

    python {path to car_model}/car_data_collect_v2.py
This script will subscribe to two topics /vesc/odom and /drive_parameters. The first one is odometry information of the car, including pose and velocity. The second one includes control command received by the car.
### 5) Train dynamic model of the car

    python {path to car_model}/car_train_model_v5.py
The first step is to set parameter of the model, such as iterations, training size, test size, etc.

    #parameters setting
    start_time = time.time()
    loss_data = 0
    iterations = 200001
    hidden_neuron_num1 = 100
    hidden_neuron_num2 = 100
    train_size = 90000
    test_size = 5000
    predict_size = 93000
    plot = True
    train = 1 # 1 for first time train, 2 for loading parameter and not training, 3 for loading variables and keep training
    project_dir = "/home/lion/car_model"
    print("iterations:",iterations)
    lowest_loss = 100
    batch_size  = 100
Next step is to load the collected data into an numpy array, the data set can be specified by modifying parameter in open() function.

    #load data from txt and store in array result
    f = open(project_dir+"/state_command_data_7.txt", "r")
    next(f)
    result = []
    for line in f:
    result.append(map(float,line.split(' ')))
    for i in range(0, len(result)-1):
    result[i][0] = result[i+1][0] - result[i][0]
    result = np.array(result)
Then generate input data of the model, it includes car orientation and velocity, as well as input control command.

    #generate input_data
    command_data = result[0:-1,9:11].reshape([-1,2])
    quaternion_t = result[0:-1,5:7].reshape([-1,2])
    v_t = result[0:-1,7:8].reshape([-1,1])
    theta_t = np.zeros((len(result)-1,1))
    for i in range(0, len(result)-1):
	    theta_t[i][0] = math.atan2(2*quaternion_t[i][0]*quaternion_t[i][1], pow(quaternion_t[i][1],2)-pow(quaternion_t[i][0],2))
	input_data = np.concatenate((theta_t,v_t,command_data[:,:].reshape([-1,2])),axis=1)

Output data includes difference of car position between current and previous step.

    #generate output_data
    delta_x = np.zeros((len(result)-1,1))
    delta_y = np.zeros((len(result)-1,1))
    delta_quaternion = np.zeros((len(result)-1,2))
    v_t_1 = np.delete(result[:,7:8].reshape([-1,1]), 0, 0)
    for i in range(0, len(result)-1):
	    delta_x[i][0] = result[i+1][1] - result[i][1]
	    delta_y[i][0] = result[i+1][2] - result[i][2]
	    delta_quaternion[i,:] = result[i+1,5:7] - result[i,5:7]
	output_data = np.concatenate((delta_x,delta_y),axis=1)
Following is the NN structure, including 2 hidden layers. The number of hidden neurons can be set at the beginning.

    #structure of the car dynamic model
    weights1 = tf.Variable(tf.random_normal([input_size,hidden_neuron_num1],seed=1))
    biases1 = tf.Variable(tf.random_normal([1,hidden_neuron_num1],seed=1))
    Wx_plus_b1 = tf.matmul(xs, weights1) + biases1
    output1 = tf.nn.relu(Wx_plus_b1)
    
    weights2 = tf.Variable(tf.random_normal([hidden_neuron_num1,hidden_neuron_num2],seed=1))
    biases2 = tf.Variable(tf.random_normal([1,hidden_neuron_num2],seed=1))
    Wx_plus_b2 = tf.matmul(output1, weights2) + biases2
    output2 = tf.nn.sigmoid(Wx_plus_b2)
    
    weights3 = tf.Variable(tf.random_normal([hidden_neuron_num2,output_size],seed=1))
    biases3 = tf.Variable(tf.random_normal([1,output_size],seed=1)
    Wx_plus_b3 = tf.matmul(output2, weights3) + biases3
    output3 = (Wx_plus_b3)
    prediction = output3

if train equals to 1, a new model will be trained with global step initialized at 0

    #initialize all variables of NN
    if train==1:
	    global_step = tf.Variable(0)
	    learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.999, staircase = False)
	    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
	    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
	    sess = tf.InteractiveSession()
	    init = tf.global_variables_initializer()
	    sess.run(init)
	    saver=tf.train.Saver(max_to_keep=1)
	    #train for ietrations
	    for i in range(1,iterations):
		    #feed batch_size samples at one time
		    feed_start = (i*batch_size)%train_size
		    feed_end = ((i+1)*batch_size)%train_size
		    x_feed = x_data
		    y_feed = y_data
		    sess.run(train_step, feed_dict={xs: x_feed, ys: y_feed})
		    loss_data = sess.run(loss, feed_dict={xs: test_x_data, ys: test_y_data})
		    #store the variables with the lowest loss
		    if loss_data<lowest_loss:
			 lowest_loss = loss_data
			 print("lowest loss:", lowest_loss)
			 saver.save(sess, project_dir+'/ckpt_v5/model.ckpt', global_step=global_step)
			 if i % (1000) == 0:
			     print("step: ", sess.run(global_step), loss_data, "learning_rate: ", sess.run(learning_rate))
if train is 2, the model is restored and not trained any more

    #just load variables of the NN and don not train the NN
    elif train==2:
	    global_step = tf.Variable(0)
	    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
	    learning_rate = tf.train.exponential_decay(0.05, global_step, 100, 0.999, staircase = False)
	    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
	    saver=tf.train.Saver()
	    sess = tf.InteractiveSession()
	    model_file=tf.train.latest_checkpoint(project_dir+'/ckpt_v5')
	    saver.restore(sess,model_file)
if train is 3, the model is loaded and trained.

    #load variables and continue to train the NN
    elif train==3:
	    global_step = tf.Variable(0)
	    learning_rate = tf.train.exponential_decay(0.05, global_step, 100, 0.999, staircase = False)
	    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
	    train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, global_step = global_step)
	    saver=tf.train.Saver()
	    sess = tf.InteractiveSession()
	    model_file=tf.train.latest_checkpoint(project_dir+'/ckpt_v5')
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
			    saver.save(sess, project_dir+'/ckpt_v5/model.ckpt', global_step=global_step)
			if i % (1000) == 0:
				print("step: ", sess.run(global_step), loss_data, "learning_rate: ", sess.run(learning_rate))
This part predicts the output of every step and adds them together. As the error becomes larger when steps increase, you can reset the accumulated error to zero every n steps by modifying **if(i%5000==0).**

    start = 0
    current_state = result[start,1:3].reshape([-1,2])
    predict_state = np.zeros((predict_size,2))
    for i in range(0,predict_size,1):
        if(i%5000==0):
            current_state = result[start+i,1:3].reshape([-1,2])
        current_input = np.concatenate((theta_t[start+i,:].reshape([-1,1]),v_t[start+i,:].reshape([-1,1]), command_data[start+i,:].reshape([-1,2])),axis=1)
        current_output = sess.run(prediction, feed_dict={xs: current_input})
        current_state = current_output + current_state#result[start+i,1:3].reshape([-1,2])
        #print("current_input:", current_input)
        predict_state[i,:] = current_state
        #print("state_data:", state_data[i+start,0:6])
        #print("prediction:", predict_state[i,0:6])
Plot the ground truth and predicted trajectory using matplot.

    if plot:
	    plt.plot(result[start, 1], result[start, 2], marker='*', color='black')
	    plt.plot(result[start+1:start+1+predict_size, 1], result[start+1:start+1+predict_size, 2], marker='*', color='red')
	    plt.plot(predict_state[0:predict_size, 0], predict_state[0:predict_size, 1], marker='*', color='green')
	    #plt.plot(predict_state[train_size:predict_size, 0], predict_state[train_size:predict_size, 1], marker='*', color='yellow')
	    plt.pause(0.000001)
if you want to save the real and predicted trajectory, set the save_data to True.

    save_data = True
    if save_data:
	    #out = open('plot_data.csv','a', newline='')
	    #csv_write = csv.writer(out,dialect='excel')
	    #csv_write.writerow(stu1)
	    np.savetxt('/home/lion/plot_data.csv', np.concatenate((result[start+1:start+1+predict_size, 1:3],predict_state[0:predict_size, 0:2]),axis=1), delimiter = ' ')
	    print ("write over")
