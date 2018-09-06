import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import csv

''' 
5 steps         result_4.csv    0.00453769747743502
10 steps        result_5.csv    0.008221225444975364
20 steps        result_6.csv    0.015603677119444612
50 steps        result_7.csv    0.03768528783022336
100 steps       result_8.csv    0.07390170998743287
200 steps       result_9.csv    0.14286185671802556
500 steps       result_10.csv   0.3268168467220772
1000 steps      result_11.csv   0.5577196141555744
2000 steps      result_12.csv   0.8748957303544963
5000 steps      result_13.csv   1.3828190335404236
10000 steps     result_14.csv   2.425366796868158
20000 steps     result_15.csv   4.203267851272511
50000 steps     result_16.csv   4.776225310643097
'''

start_time = time.time()

project_dir = "/home/lion/car_model/trajectory"
f = open(project_dir+"/result_6.csv", "r")
result = []
for line in f:
    result.append(map(float,line.split(' ')))
result = np.array(result)


ground_truth = result[:,0:2]
predict_state = result[:,2:4]

plt.plot(ground_truth[:, 0], ground_truth[:, 1], marker='*', color='red')
plt.plot(predict_state[:, 0], predict_state[:, 1], marker='*', color='green')
plt.pause(0.000001)

sess = tf.InteractiveSession()
print(sess.run(tf.reduce_mean(tf.sqrt(tf.square(ground_truth[:, 0] - predict_state[:, 0])  \
    + tf.square(ground_truth[:, 1] - predict_state[:, 1])),reduction_indices=[0])))

end_time = time.time()
print("cost_time:",end_time-start_time)
raw_input("press any button to exit")

