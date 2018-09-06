#coding:utf-8
import numpy as np
import tensorflow as tf
import math
import rospy
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from racecar_control.msg import drive_param
import tf as t_f
import time
from std_srvs.srv import Empty

# 超参数
H = 50 # number of hidden layer neurons
batch_size = 25 # every how many episodes to do a param update?
learning_rate = 1e-1 # feel free to play with this to train faster or more stably.
gamma = 0.99 # discount factor for reward
D = 1 # input dimensionality

tf.reset_default_graph()

# 神经网络的输入环境的状态，并且输出左/右的概率
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations,W1))
W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1,W2)
probability = tf.nn.sigmoid(score)

# 定义其他部分
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")

# 定义损失函数
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages) 
newGrads = tf.gradients(loss,tvars)

# 为了减少奖励函数中的噪声，我们累积一系列的梯度之后才会更新神经网络的参数
adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # Our optimizer
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # Placeholders to send the final gradients through when we update.
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


xs,drs,ys = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.initialize_all_variables()

pos_x = 0.0
pos_y = 0.0
pos_phi = 0.0
linear_velocity = 0.0
angular_velocity = 0.0
print_data = True
observation = 0.0
new_observation = False
done = 0.0
reward = 0.0
observation_num = 0
observation_sum = 0

def callback_pose(data):
    global pos_x
    global pos_y
    global pos_phi
    global linear_velocity
    global angular_velocity
    global new_observation

    pos_x = data.pose.pose.position.x
    pos_y = data.pose.pose.position.y
    (roll, pitch, pos_phi) = t_f.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    linear_velocity = data.twist.twist.linear.x 
    angular_velocity = data.twist.twist.angular.z
    new_observation = True

rospy.init_node("car_controller")
rospy.Subscriber("/odom", Odometry, callback_pose)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
reset_turtlebot = rospy.ServiceProxy('/reset_turtlebot', Empty)


# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
        
    goal = 2*math.pi*(np.random.rand()-0.5)
    print("goal:", goal)

    while episode_number <= total_episodes and not rospy.is_shutdown():
        
        #observation is limited between -pi ~ +pi
        observation = goal - pos_phi
        if observation > math.pi:
            observation -= 2*math.pi
        elif observation < -math.pi:
            observation += 2*math.pi
        observation = pow(observation,1)/pow(math.pi,1)
        reward = goal - abs(observation)
        observation_sum += observation
        if abs(observation) > math.pi:
            done = 1
        else:
            done = 0

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
        reward_sum += reward

        x = np.reshape(observation,[1,D])
        tfprob = sess.run(probability,feed_dict={observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        xs.append(x) # observation
        y = 1 if action == 0 else 0 # a "fake label"
        ys.append(y)

        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = (tfprob[0][0] - 0.5)*100
        msg.angular.z = max(min(msg.angular.z,0.7),-0.7)
        pub.publish(msg)

        if new_observation or done==1:
            print(observation, msg.angular.z, reward, episode_number, done)
            new_observation = False

        
        # 批量更新
        if done: 
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs,drs,ys = [],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)

            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0: 
                sess.run(updateGrads,feed_dict={W1Grad: gradBuffer[0],W2Grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0

                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print 'Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size)
                reward_sum = 0

            reset_turtlebot()

print episode_number,'Episodes completed.'