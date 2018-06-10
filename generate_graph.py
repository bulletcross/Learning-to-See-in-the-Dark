# Sample script for generating the graph,
# for debugging purpose

import tensorflow as tf
import numpy as np
import train_Sony as M

def generate_unet():
    print('Generating graph file for debigging')
    # Create a default session, and construct graph in-it
    sess=tf.Session()
    in_image=tf.placeholder(tf.float32,[None,None,None,4])
    gt_image=tf.placeholder(tf.float32,[None,None,None,3])
    out_image=M.network(in_image)
    # Define loss function
    G_loss=tf.reduce_mean(tf.abs(out_image - gt_image))
    # Obtain all trainable variables
    t_vars=tf.trainable_variables()
    # Define the back-prop flow in model graph
    # For some reason, leaving out weights involved in up-smapling??
    vars_to_train = [var for var in t_vars if var.name.startswith('g_')]
    vars_not_to_train = [var for var in t_vars if not var.name.startswith('g_')]
    print('Trainable variables:')
    for v in vars_to_train:
        print(v.name)
    print('Non-trainable variables:')
    for v in vars_not_to_train:
        print(v.name)
    G_opt=tf.train.AdamOptimizer(0.001).minimize(G_loss,var_list = vars_to_train)
    # Writing out the graph
    writer = tf.summary.FileWriter('tf_graph', sess.graph)
    writer.close()

if __name__ == '__main__':
    generate_unet()
