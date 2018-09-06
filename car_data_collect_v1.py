import rospy
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from racecar_control.msg import drive_param
import tf
import math
import numpy as np
import time
from tf import TransformListener

f = open("/home/lion/car_model/data.txt", "w")
v = 0.0
angle = 0.0 
print_data = True

def callback_pose(data):
    global v
    global angle
    global print_data
    
    time = data.header.stamp.secs + data.header.stamp.nsecs/1000000000.0
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    (roll, pitch, phi) = tf.transformations.euler_from_quaternion([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    linear_velocity = data.twist.twist.linear.x 
    angular_velocity = data.twist.twist.angular.z
    
    if print_data:
        print "time", time
        print "pose", x, y, phi
        print "twist", linear_velocity, angular_velocity
        print "command", v, angle
        
    
    f.write(str(time)+" "+str(x)+" "+str(y)+" "+str(phi)+" "+str(linear_velocity)+" "+str(angular_velocity)+" "+str(v)+" "+str(angle)+"\n")

def callback_command(data):
    global v
    global angle
    #print "received command"
    v = data.velocity
    angle = data.angle
    
# Start the node
if __name__ == '__main__':
    rospy.init_node("car_data_collect")
    rospy.Subscriber("/vesc/odom", Odometry, callback_pose)
    rospy.Subscriber("/drive_parameters", drive_param, callback_command)
    f.write("time   x    y    orientation    linear_velocity    angular_velocity    command_velocity command_angle\n")
    
    while not rospy.is_shutdown():
        pass
    f.close()
    
    
