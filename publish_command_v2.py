import rospy
from racecar_control.msg import drive_param
from geometry_msgs.msg import Twist


velocity_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
angle_value = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
msg = Twist()
sleep_time = 1.0
    
# Start the node
if __name__ == '__main__':
    rospy.init_node("publish_command")
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
    
    while not rospy.is_shutdown():
        for i in range(len(velocity_value)):
            msg.linear.x = velocity_value[i]
            for j in range(len(angle_value)):
                msg.angular.z = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angular.z = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in range(len(angle_value)):
                msg.angular.z = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angular.z = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
        for i in reversed(range(len(velocity_value))):
            msg.linear.x = velocity_value[i]
            for j in range(len(angle_value)):
                msg.angular.z = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angular.z = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in range(len(angle_value)):
                msg.angular.z = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angular.z = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)


    
    

