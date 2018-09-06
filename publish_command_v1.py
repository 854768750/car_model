import rospy
from racecar_control.msg import drive_param

velocity_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
angle_value = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
msg = drive_param()
sleep_time = 2.0
    
# Start the node
if __name__ == '__main__':
    rospy.init_node("publish_command")
    pub = rospy.Publisher("/drive_parameters", drive_param, queue_size = 1)
    
    while not rospy.is_shutdown():
        for i in range(len(velocity_value)):
            msg.velocity = velocity_value[i]
            for j in range(len(angle_value)):
                msg.angle = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angle = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in range(len(angle_value)):
                msg.angle = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angle = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
        for i in reversed(range(len(velocity_value))):
            msg.velocity = velocity_value[i]
            for j in range(len(angle_value)):
                msg.angle = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angle = angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in range(len(angle_value)):
                msg.angle = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)
            for j in reversed(range(len(angle_value))):
                msg.angle = -angle_value[j]
                pub.publish(msg)
                rospy.sleep(sleep_time)


    
    

