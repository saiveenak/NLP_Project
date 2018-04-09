import tensorflow as tf
import numpy as np
from wordEmbedding3D import get3Dmatrix
matrix3D, maxCaptionLength = get3Dmatrix()

filterSizes = [2, 3, 4]
numFilters = 2
sequenceLength = maxCaptionLength
embeddingSize = 100


pooled_outputs = []
for i, filter_size in enumerate(filterSizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        filterShape = [filter_size, embeddingSize, 1, numFilters]

        W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[numFilters]), name="b")

        numrows = len(matrix3D[0])  # 3 rows in your example
        numcols = len(matrix3D[0][0])
        text = np.reshape(matrix3D[0], (-1,numrows,numcols,1))

        input = tf.convert_to_tensor(text,dtype=tf.float32)
        conv = tf.nn.conv2d(
            input,
            W,
            strides=[1, 1, 1, 1],
            name="conv",
            padding = "SAME")
        
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequenceLength - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)

init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(pooled_outputs))